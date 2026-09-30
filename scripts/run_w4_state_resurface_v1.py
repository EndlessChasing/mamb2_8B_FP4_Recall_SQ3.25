#!/usr/bin/env python3
"""Full W4 per-token S16/SQ controls and five-arm fresh Resurface evaluation."""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, resurface_data as data, resurface_native as native
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from mamba2_recall.state_quant import StateQuant
from state_ppl_codec_v10 import StatePPLQuantV10
from prepare_state_first_v5 import tensor_sha
from evaluate_quant_first import FrozenBase, compare_pair, check_restoration
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from run_statequant import save_json
from run_state_resurface_v11 import classify_prediction, summarize_mk
import run_state_ppl_v10 as v10
import train_w4_state_resurface_v1 as trainer
import w4_state_binding_v1 as shared
need, sha, read_json = shared.need, shared.sha, shared.read_json
PROTOCOL = shared.PROTOCOL_SHA
NUMERIC_PROTOCOL = shared.NUMERIC_PROTOCOL_SHA
TRAIN_MANIFEST = shared.TRAIN_MANIFEST_SHA
CACHE_BYTES = shared.CACHE_BYTES
ADAPTER_BYTES = shared.ADAPTER_BYTES
S16_CACHE_BYTES = shared.S16_CACHE_BYTES
ARMS = ('w4_s16','w4_sq325','w4_sq325_resurface','restored_w4_sq325','restored_w4_s16')
FORMAT = 'W4_STATE_RESURFACE_EVAL_V1'
COMPARE_FORMAT = 'W4_STATE_RESURFACE_COMPARISON_V1'


def code_hashes():
    return shared.code_hashes(extra=['scripts/run_w4_state_resurface_v1.py',
                                     'scripts/train_w4_state_resurface_v1.py'])


def validate_training(path,parent_binding):
    """Check final training/export/checkpoint evidence before any model load."""
    report=read_json(path)
    need(report.get('format')=='W4_STATE_RESURFACE_TRAIN_V1' and report.get('complete') is True
         and report.get('mode')=='formal' and report.get('successful_updates')==1536
         and 1536<=report.get('attempts',-1)<=1544 and 'error' not in report,
         'Completed formal fresh1536-update training required')
    need(report.get('code_sha256')==trainer.code_hashes(),'Training source inventory differs')
    binding=report['binding']
    required=dict(parent_binding,protocol_sha256=PROTOCOL,numeric_protocol_sha256=NUMERIC_PROTOCOL,
        source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,tokenizer_sha256=runtime.TOKENIZER_SHA256,
        train_manifest_sha256=TRAIN_MANIFEST,prose_manifest_sha256=trainer.PROSE_MANIFEST_SHA,
        prose_tokens_sha256=trainer.PROSE_TOKENS_SHA,successful_updates=1536,adapter=native.FORMAT,
        initial_adapter='fresh V=0,g=1,w=0,b=-4; no pretrained adapter or checkpoint',
        fresh_initialization=True,prior_adapter_loaded=False,checkpoint_loaded=False,
        teacher='separate unadapted W4 package, S16 per-token carry',v2_trainer_sha256=trainer.V2_TRAINER_SHA,
        state_mode='v10_32_32_64; exact packed stored-scale forward; 64-live STE backward')
    need(all(binding.get(k)==v for k,v in required.items()),'Fresh training binding differs')
    fresh=report.get('fresh_initialization',{})
    need(all(fresh.get(k) is True for k in ('all_224_masters_exact_fresh_values','optimizer_state_empty','scaler_exact_initial'))
         and fresh.get('prior_adapter_loaded') is False and fresh.get('checkpoint_loaded') is False
         and fresh.get('masters_dtype')=='float32' and fresh.get('master_tensors')==224
         and fresh.get('parameters')==1154104 and fresh.get('optimizer_steps')==0
         and fresh.get('scaler')==dict(scale=1024.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=0),
         'Exact fresh masters/optimizer/scaler proof missing')
    need(report.get('frozen_base_check',{}).get('identity_version_gradients_unchanged') is True
         and report.get('frozen_state_calibration_check') is True
         and report.get('teacher_base_parameters_frozen') is True and report.get('selected_table_bytes_unchanged') is True,
         'Frozen source/teacher/table proof missing')
    for key,flag in (('initialization_check','fresh_identity_matches_packed_bitwise'),
                     ('deployed_export_check','packed_training_forward_bitwise_equal')):
        proof=report.get(key,{})
        need(proof.get(flag) is True and proof.get('probe_tokens')==128
             and [row.get('probe_tokens') for row in proof.get('probes',[])]==[128,512],
             'Both128/512-token parity proofs required: '+key)
        for row in proof['probes']:
            need(row.get('packed_training_forward_bitwise_equal') is True and row.get('packed_probe_finite') is True
                 and row['cache']['total_bytes']==CACHE_BYTES,'Invalid exact packed parity proof')
    checkpoint_check=report.get('final_checkpoint_export_check',{})
    need(all(checkpoint_check.get(k) is True for k in ('all_master_casts_equal_export','optimizer_exact','scaler_exact'))
         and checkpoint_check.get('master_tensors')==224,'Final checkpoint export proof missing')
    ordered=torch.randperm(1536,generator=torch.Generator().manual_seed(2026092803)).tolist()
    successful=overflows=0
    need(len(report['history'])==report['attempts'],'Training history is incomplete')
    for attempt,row in enumerate(report['history'],1):
        need(successful<1536 and row['attempt']==attempt and row['update_index']==successful
             and row['schedule_entry']==ordered[successful] and type(row['overflow']) is bool,
             'Successful-update/retry schedule differs')
        successful+=int(not row['overflow']);overflows+=int(row['overflow'])
        need(row['successful_updates']==successful,'Successful-update counter differs')
    need(successful==1536 and overflows==report['overflows']<=8,'Final training/overflow count differs')
    export=report['adapter']
    need(Path(export['file']).name==export['file'],'Unsafe adapter path')
    adapter_path=path.parent/export['file']
    need(data.sha_file(adapter_path)==export['sha256'] and adapter_path.stat().st_size==export['bytes']
         and export.get('roundtrip_bitwise_equal') is True and export.get('parameters')==1154104
         and export.get('payload_bytes')==ADAPTER_BYTES and export.get('gate_mode')=='soft','FP16 export differs')
    tensors=native.read_fp16(adapter_path,expected_binding=binding)['tensors']
    need(len(tensors)==224 and sum(v.numel()*v.element_size() for v in tensors.values())==ADAPTER_BYTES
         and {k:native.tensor_hash(v) for k,v in tensors.items()}==export['tensor_sha256'],'FP16 tensor inventory differs')
    final=report['final_checkpoint'];need(Path(final['file']).name==final['file'],'Unsafe checkpoint path')
    cp_path=path.parent/final['file']
    need(data.sha_file(cp_path)==final['sha256'] and cp_path.stat().st_size==final['bytes']
         and checkpoint_check['final_checkpoint_sha256']==final['sha256'],'Final checkpoint receipt differs')
    checkpoint=torch.load(cp_path,map_location='cpu',weights_only=True)
    need(checkpoint['format']=='W4_STATE_RESURFACE_CHECKPOINT_V1' and checkpoint['binding']==binding
         and checkpoint['successful_updates']==1536 and checkpoint['attempts']==report['attempts'],
         'Final checkpoint identity/count differs')
    masters=checkpoint['masters']
    need(set(masters)==set(tensors) and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all())
         and torch.equal(v.half(),tensors[k]) for k,v in masters.items()),'Final master FP16 casts differ')
    need(len(checkpoint['optimizer']['state'])==224 and all(int(x['step'])==1536
         for x in checkpoint['optimizer']['state'].values()),'Final optimizer steps differ')
    evidence=report['probe_evidence'];need(Path(evidence['file']).name==evidence['file'],'Unsafe probe evidence path')
    evidence_path=path.parent/evidence['file']
    need(data.sha_file(evidence_path)==evidence['sha256'] and evidence_path.stat().st_size==evidence['bytes'],
         'Raw training/export probe evidence differs')
    proof=torch.load(evidence_path,map_location='cpu',weights_only=True)
    need(proof['format']=='W4_STATE_RESURFACE_TRAIN_PROOF_V1' and proof['binding']==binding
         and set(proof['masters'])==set(masters) and all(torch.equal(v,proof['masters'][k]) for k,v in masters.items()),
         'Raw probe/master binding differs')
    for stage,key in (('initialization','initialization_check'),('export','deployed_export_check')):
        need(set(proof[stage])=={'128','512'},'Raw probe population differs')
        for row in report[key]['probes']:
            length=row['probe_tokens'];entry=proof[stage][str(length)]
            hidden=entry['packed_hidden'];other=entry['training_hidden']
            need(hidden.dtype==other.dtype==torch.float16 and tuple(hidden.shape)==tuple(other.shape)==(1,length,4096)
                 and bool(torch.isfinite(hidden).all()) and torch.equal(hidden,other)
                 and native.tensor_hash(hidden)==row['packed_hidden_sha256']==row['training_hidden_sha256']
                 and tuple(entry['token_ids'].shape)==(1,length) and entry['token_ids'].dtype==torch.int64
                 and runtime.token_digest(entry['token_ids'].numpy())==row['probe_token_sha256'],
                 'Raw exact128/512-token proof differs')
    smoke=report.get('discarded_smoke_check',{})
    need(smoke.get('discarded') is True and smoke.get('successful_updates')==1
         and smoke.get('formal_reinitializes_masters_optimizer_scaler') is True
         and smoke.get('backend_policy_exact') is True,'Formal run lacks discarded-smoke/reinitialization proof')
    return report,adapter_path


def require_cpu_audit(path, report_path, stage):
    need(path is not None, 'Independent CPU audit path required')
    audit = read_json(path)
    need(audit.get('format') == 'W4_STATE_AUDIT_V1' and audit.get('complete') is True
         and audit.get('passed') is True and audit.get('cuda_initialized') is False
         and audit.get('stage') == stage and audit.get('protocol_sha256') == PROTOCOL
         and audit.get('input_report_sha256') == sha(report_path)
         and audit.get('source_sha256') == sha(ROOT/'scripts/audit_w4_state_v1.py'),
         'Passing independent CPU audit with exact input/source binding required')
    return audit


def validate_pretrain(path, audit_path, binding):
    report = read_json(path)
    need(report.get('format') == 'W4_STATE_PRETRAIN_V1' and report.get('complete') is True
         and report.get('protocol_sha256') == PROTOCOL
         and report.get('w4_manifest_sha256') == shared.W4_MANIFEST_SHA
         and report.get('calibration_sha256') == binding['calibration_sha256']
         and report.get('code_sha256') == code_hashes()
         and set(report.get('reports',{})) == set(ARMS[:2]), 'Pretraining control binding differs')
    raw = {}
    for arm,item in report['reports'].items():
        need(Path(item['file']).name == item['file'], 'Unsafe pretraining report path')
        raw_path = path.parent/item['file']
        need(sha(raw_path) == item['sha256'], 'Pretraining control raw file changed')
        raw[arm] = read_json(raw_path)
        need(raw[arm].get('complete') is True and raw[arm].get('arm') == arm
             and raw[arm]['ppl']['ppl'] == item['ppl'], 'Pretraining raw control incomplete')
    audit = require_cpu_audit(audit_path,path,'pretrain')
    return report,raw,audit


def validate_w4_training(path, audit_path, binding, pretrain_path, pretrain_audit_path, w4_dir):
    report,adapter_path = validate_training(path,binding)
    audit = require_cpu_audit(audit_path,path,'training')
    actual = report['binding']
    pretrain = read_json(pretrain_path)
    need(actual.get('pretrain_report_sha256') == sha(pretrain_path)
         and actual.get('pretrain_audit_sha256') == sha(pretrain_audit_path)
         and actual.get('pretrain_raw_report_sha256') == {k:v['sha256'] for k,v in pretrain['reports'].items()}
         and actual.get('teacher_w4_manifest_sha256') == shared.W4_MANIFEST_SHA
         and actual.get('training_code_sha256') == trainer.code_hashes()
         and actual.get('upstream_kernel_proof') == shared.validate_upstream_kernels(),
         'W4 training must bind exact pretraining controls, teacher, source, and kernel proof')
    expected = shared.expected_base_hashes(w4_dir)
    for key in ('initial_student_base_check','initial_teacher_base_check','frozen_base_check','final_teacher_base_check'):
        proof = report.get(key,{})
        need(proof.get('actual_content_checked') is True
             and proof.get('decoded_tensor_sha256') == expected
             and proof.get('loaded_parameter_bytes') == 16473999360,
             'Actual initial/final W4 student/teacher content proof differs: '+key)
    for key in ('student','teacher'):
        proof = report.get('native_restoration',{}).get(key,{})
        need(proof.get('native_forwards_restored') is True and proof.get('adapter_hooks_removed') is True
             and proof.get('layers') == 56, 'Training native forward/hook restoration differs')
    return report,adapter_path,audit


def native_snapshot(model):
    return [(module,module.forward,dict(module._forward_pre_hooks),dict(module._forward_hooks))
            for layer in model.backbone.layers for module in (layer.mixer,layer.mixer.norm)]


def check_native_snapshot(snapshot):
    need(all(module.forward == forward and dict(module._forward_pre_hooks) == pre
             and dict(module._forward_hooks) == post for module,forward,pre,post in snapshot),
         'State controller did not restore native forwards/hooks')
    return dict(complete=True,native_mixer_forwards_restored=True,native_hooks_restored=True)


def is_s16(arm):
    return arm in ('w4_s16','restored_w4_s16')


@contextlib.contextmanager
def execution_for(model,table,arm):
    snapshot = native_snapshot(model)
    execution = StateQuant(model,'s16') if is_s16(arm) else StatePPLQuantV10(model,table,layout=shared.LAYOUT)
    digest = tensor_sha(table)
    try:
        with execution:
            try:
                yield execution
            finally:
                if not is_s16(arm):
                    need(native.tensor_hash(execution.permutations) == digest, 'Actual runtime table changed')
    finally:
        check_native_snapshot(snapshot)


def spec_for(arm):
    if is_s16(arm):
        return dict(candidate_id=arm,candidate_name='s16',diagnostic='s16',deployable=False,
                    scale_mode=None,int4_clip=None,layout='s16',variant='s16')
    spec = v10.candidate_spec(shared.LAYOUT,dict(candidate_order=list(v10.LAYOUTS)))
    spec.update(candidate_id=arm,candidate_name='selected_w4_sq325')
    return spec


def storage_descriptor(execution,arm):
    if not is_s16(arm):
        result = execution.storage_descriptor()
        v10.validate_storage_descriptor(result,shared.LAYOUT)
        return result
    layers = []
    for row in execution._cache:
        state = row.state.tensors
        need(set(state) == {'state'}, 'S16 controller has extra persistent state')
        value = state['state']
        need(list(value.shape) == [1,128,64,128] and value.dtype == torch.float16
             and value.untyped_storage().nbytes() == 2097152
             and list(row.conv.shape) == [1,10240,4] and row.conv.dtype == torch.float16
             and row.conv.untyped_storage().nbytes() == 81920, 'S16 physical cache layout differs')
        layers.append(dict(state_shape=list(value.shape),state_bytes=value.untyped_storage().nbytes(),
            conv_shape=list(row.conv.shape),conv_storage_bytes=row.conv.untyped_storage().nbytes(),
            tensors=dict(state=dict(shape=list(value.shape),dtype=str(value.dtype),storage_bytes=value.untyped_storage().nbytes()))))
    need(len(layers) == 56 and execution.permutations is None, 'S16 layer/table inventory differs')
    return dict(layout='s16',layers=layers)


@torch.inference_mode()
def evaluate_ppl(model,table,windows,path,common):
    arm = common['arm']; spec = spec_for(arm)
    result = dict(common,**spec,complete=False,ppl=dict(windows=[]))
    save_json(path,result)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with execution_for(model,table,arm) as execution:
        probe = windows[0][1][:128].cuda()[None]
        first = execution.backbone(probe,reset=True)
        if not bool(torch.isfinite(first).all()): raise v10.v6.CandidateInvalid('Nonfinite PPL reset probe')
        zero_scales = v10.finite_cache(execution)
        identity = v10.cache_identity(execution)
        first_sha = native.tensor_hash(first)
        second = execution.backbone(probe,reset=True)
        if not bool(torch.isfinite(second).all()): raise v10.v6.CandidateInvalid('Nonfinite repeated PPL probe')
        v10.finite_cache(execution)
        need(torch.equal(first,second) and identity == v10.cache_identity(execution), 'Repeated reset output/cache changed')
        result['repeated_reset_probe'] = dict(tokens=128,hidden_sha256=first_sha,hidden_and_cache_exact=True,
            cache_tensor_sha256=identity,token_sha256_int64le=runtime.token_digest(probe.cpu().numpy()),
            cache=v10.require_cache(execution,spec))
        result['storage_descriptor_probe'] = storage_descriptor(execution,arm)
        del first,second,probe,identity
        total = 0.; count = 0
        for index,(start,window) in enumerate(windows):
            tokens = window.cuda()
            hidden = execution.backbone(tokens[:-1][None],reset=True)
            if not bool(torch.isfinite(hidden).all()): raise v10.v6.CandidateInvalid('Nonfinite PPL hidden')
            zero_scales += v10.finite_cache(execution)
            loss_sum = 0.
            for pos in range(0,hidden.shape[1],64):
                end = min(pos+64,hidden.shape[1])
                logits = model.lm_head(hidden[:,pos:end]).float()
                loss = F.cross_entropy(logits.reshape(-1,256000),tokens[pos+1:end+1],reduction='sum')
                loss_sum += float(loss)
                del logits,loss
            if not math.isfinite(loss_sum): raise v10.v6.CandidateInvalid('Nonfinite PPL loss')
            targets = len(window)-1; total += loss_sum; count += targets
            result['ppl']['windows'].append(dict(start=start,target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()),nll=loss_sum,ppl=v10.finite_exp(loss_sum/targets)))
            result['ppl'].update(nll=total,target_tokens=count,ppl=v10.finite_exp(total/count))
            result['cache'] = v10.require_cache(execution,spec)
            result['storage_descriptor'] = storage_descriptor(execution,arm)
            result['ppl_end_cache_tensor_sha256'] = v10.cache_identity(execution)
            save_json(path,result)
            if index == 0 or (index+1)%8 == 0 or index+1 == len(windows):
                print(f'[{arm} PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.6f}',flush=True)
            del hidden,tokens
        result['ppl_cache'] = copy.deepcopy(result['cache'])
    result.update(complete=True,ppl_complete=True,runtime_table_unchanged=True,
        persistent_float_finite_checks_passed=True,allocation_storage_validated=True,
        controller_restoration=dict(complete=True,native_mixer_forwards_restored=True,native_hooks_restored=True),
        zero_scale_observations=zero_scales,gpu_memory=runtime.gpu_memory_receipt(),elapsed_seconds=time.time()-started)
    v10.v6.check_ppl(result,v10.v6.window_identity(windows))
    return result


@torch.inference_mode()
def evaluate_mk(model,tokenizer,table,cases,path,result):
    arm = result['arm']; spec = spec_for(arm)
    result.update(complete=False,mk=dict(rows=[]))
    save_json(path,result)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with execution_for(model,table,arm) as execution:
        for index,case in enumerate(cases):
            encoded = tokenizer.encode(case['prompt'])
            hidden = execution.backbone(torch.tensor(encoded,device='cuda',dtype=torch.long)[None],reset=True)[:,-1:]
            generated = []
            for step in range(12):
                logits = model.lm_head(hidden)
                if not bool(torch.isfinite(logits).all()): raise v10.v6.CandidateInvalid('Nonfinite MK logits')
                token = int(logits.argmax(-1).item()); generated.append(token)
                if token == tokenizer.eos_token_id or step == 11: break
                hidden = execution.backbone(torch.tensor([[token]],device='cuda'))
            v10.finite_cache(execution)
            output = tokenizer.decode(generated); match = re.search(r'(?<!\d)\d{6}(?!\d)',output)
            prediction = match.group() if match else None
            row = dict(case,prompt_tokens=len(encoded),prompt_token_sha256_int64le=runtime.token_digest(encoded),
                generated_ids=generated,output=output,prediction=prediction,correct=prediction==case['answer'])
            row['prediction_category'] = classify_prediction(row)
            result['mk']['rows'].append(row)
            result['mk_cache'] = v10.require_cache(execution,spec)
            if index == 0 or (index+1)%8 == 0:
                save_json(path,result); print(f'[{arm} MK] {index+1}/{len(cases)}',flush=True)
        result['storage_descriptor_mk'] = storage_descriptor(execution,arm)
        result['mk_end_cache_tensor_sha256'] = v10.cache_identity(execution)
    result['mk']['summary'],result['mk']['strata'] = summarize_mk(result['mk']['rows'])
    result['mk']['elapsed_seconds'] = time.time()-started
    result.update(complete=True,mk_complete=True,mk_persistent_float_finite_checks_passed=True,
        mk_runtime_table_unchanged=True,gpu_memory_mk=runtime.gpu_memory_receipt(),
        mk_controller_restoration=dict(complete=True,native_mixer_forwards_restored=True,native_hooks_restored=True))
    return result


def adapter_storage(bank,export,arm):
    cache_bytes = S16_CACHE_BYTES if is_s16(arm) else CACHE_BYTES
    if bank is None:
        return dict(loaded=False,tensors=0,parameters=0,fp16_payload_bytes=0,resident_storage_bytes=0,
            persistent_buffer_bytes=0,additional_recurrent_cache_bytes=0,ema_enabled=False,
            cache_plus_adapter_bytes=cache_bytes)
    values = bank.masters
    need(len(values) == 224 and all(v.dtype == torch.float16 and v.is_cuda and not v.requires_grad
         and v.grad is None for v in values.values()), 'Installed adapter geometry/dtype differs')
    hashes = {k:native.tensor_hash(v) for k,v in values.items()}
    need(hashes == export['tensor_sha256'], 'Installed adapter tensors differ')
    storages = {v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in values.values()}
    need(not list(bank.adapters.named_buffers()) and not bank._frames and not bank._requests,
         'Unexpected persistent adapter buffers/active frames')
    payload = sum(v.numel()*v.element_size() for v in values.values()); resident = sum(storages.values())
    need(payload == resident == ADAPTER_BYTES, 'Installed adapter bytes differ')
    return dict(loaded=True,tensors=224,parameters=1154104,fp16_payload_bytes=payload,
        resident_storage_bytes=resident,persistent_buffer_bytes=0,additional_recurrent_cache_bytes=0,
        ema_enabled=False,cache_plus_adapter_bytes=cache_bytes+resident,serialized_file_bytes=export['bytes'],
        adapter_sha256=export['sha256'],tensor_sha256=hashes,tensors_unchanged=True)


def exact_ppl(before,after):
    need(before.get('ppl_complete') is True and after.get('ppl_complete') is True,
         'Exact replay requires completed PPL')
    for field in ('ppl','repeated_reset_probe','cache','ppl_cache','storage_descriptor_probe',
                  'storage_descriptor','ppl_end_cache_tensor_sha256'):
        need(before[field] == after[field], 'Exact PPL replay differs: '+field)
    return dict(complete=True,per_window_nll_exact=True,reset_output_and_cache_exact=True,
        physical_cache_exact=True,ppl_end_tensor_hashes_exact=True,windows=130,target_tokens=264764)


def exact_removal(before,after):
    proof = check_restoration(before,after)
    ppl = exact_ppl(before,after)
    for field in ('mk_cache','storage_descriptor_mk','mk_end_cache_tensor_sha256'):
        need(before[field] == after[field], 'Restored MK cache differs: '+field)
    need(before['mk']['summary'] == after['mk']['summary'] and before['mk']['strata'] == after['mk']['strata'],
         'Restored MK summary/strata differs')
    return dict(proof,ppl=ppl,mk_end_cache_exact=True,mk_end_tensor_hashes_exact=True,
                all_storage_descriptors_exact=True)


def quality_checks(results,replays,restorations,training_audit):
    repair = compare_pair(results['w4_sq325'],results['w4_sq325_resurface'])
    adapted_ppl = repair['candidate_ppl']
    checks = dict(ppl_no_worse_than_w4_sq325=adapted_ppl <= repair['control_ppl'],
        normal_mk_increases=repair['normal_mk_correct_delta'] > 0,
        paired95_lower_positive=repair['normal_mk_paired_bootstrap_95ci'][0] > 0,
        exact_pretrain_replay=all(row['complete'] for row in replays.values()),
        exact_both_restorations=all(row['complete'] for row in restorations.values()),
        cache_same_budget=all(row[field]['total_bytes'] == (S16_CACHE_BYTES if is_s16(arm) else CACHE_BYTES)
            for arm,row in results.items() for field in ('ppl_cache','mk_cache')),
        adapter_storage_verified=results['w4_sq325_resurface']['adapter_storage']['resident_storage_bytes'] == ADAPTER_BYTES,
        all_integrity_checks_passed=all(row.get('candidate_table_unchanged') is True
            and row.get('frozen_source',{}).get('identity_version_gradients_unchanged') is True
            and row.get('adapter_hooks_removed_after_arm') is True for row in results.values()),
        training_audit_passed=training_audit['passed'])
    separate = dict(ppl_no_worse_than_fresh_w4_s16=adapted_ppl <= results['w4_s16']['ppl']['ppl'],
                    ppl_strictly_below_8p25=adapted_ppl < 8.25)
    return repair,checks,separate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('pretrain','full'),required=True)
    for name in ('w4-dir','source-dir','calibration','out-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('data-root','prose-tokens','pretrain-report','pretrain-audit','training-report','training-audit'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    shared.check_protocol(); kernel = shared.validate_upstream_kernels()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(),'Fresh output directory required')
    selected,receipt,binding = shared.load_selection(args.calibration)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == runtime.TOKENIZER_SHA256,'Tokenizer differs')
    if args.prose_tokens is not None:
        need(sha(args.prose_tokens) == shared.TRAIN_SHA,'TRAIN file changed')
    if args.data_root is not None:
        need(sha(args.data_root/'train/manifest.json') == TRAIN_MANIFEST,'Numeric TRAIN manifest changed')
    training = adapter_path = training_audit = pretrain = pretrain_raw = pretrain_audit = None
    if args.stage == 'full':
        need(all(getattr(args,k) is not None for k in ('pretrain_report','pretrain_audit','training_report','training_audit')),
             'Full evaluation requires pretrain/training reports and independent audits')
        pretrain,pretrain_raw,pretrain_audit = validate_pretrain(args.pretrain_report,args.pretrain_audit,binding)
        training,adapter_path,training_audit = validate_w4_training(args.training_report,args.training_audit,binding,
            args.pretrain_report,args.pretrain_audit,args.w4_dir)
    ids,dataset = load_wikitext_tokens(tokenizer,'validation')
    windows = ppl_windows(ids,2048)
    need(len(windows) == 130 and sum(len(w)-1 for _,w in windows) == 264764
         and dataset['token_stream_sha256_int64le'] == shared.VALIDATION_TOKENS_SHA,'Full PPL population differs')
    # CONFIRM is not opened, generated, or scored before the full stage.
    cases = None
    if args.stage == 'full':
        cases = data.generate_cases('confirm'); case_proof = data.validate_cases(cases,'confirm')
        need(len(cases) == 768 and sum(c['condition']=='normal' for c in cases) == 384
             and data.sha_bytes(data.canonical_bytes(cases)) == shared.CONFIRM_CASE_SHA,'Frozen CONFIRM population differs')
    torch.set_num_threads(8); torch.manual_seed(20260929); torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest'); policy = pin_replay_backend()
    if training is not None:
        need(training['binding']['backend_policy'] == policy,'Training/evaluation backend differs')
    hashes = code_hashes(); args.out_dir.mkdir(parents=True)
    model = shared.load_w4(args.w4_dir); frozen = FrozenBase(model)
    original_snapshot = native_snapshot(model)
    expected_base = shared.expected_base_hashes(args.w4_dir)
    table = selected['permutations']; table_sha = tensor_sha(table)
    common = dict(format=FORMAT,stage=args.stage,protocol_sha256=PROTOCOL,
        w4_manifest_sha256=shared.W4_MANIFEST_SHA,calibration_sha256=sha(args.calibration),
        calibration_receipt_sha256=sha(args.calibration.with_suffix('.json')),binding=binding,
        candidate_table_sha256=table_sha,tokenizer_sha256=tokenizer.sha256,
        train_file_sha256=shared.TRAIN_SHA,train_manifest_sha256=TRAIN_MANIFEST,
        numeric_protocol_sha256=NUMERIC_PROTOCOL,dataset=dataset,upstream_kernel_proof=kernel,
        w4_loader_receipt=model._package_receipt,backend_policy=policy,code_sha256=hashes,code_hashes=hashes,
        environment=runtime.environment_receipt(),heldout_used=True,heldout_used_for_selection=False,
        mk_used=args.stage == 'full',quality_scope='Historically exposed validation and CONFIRM families; no untouched generalization claim')
    if training is not None:
        common.update(training_report_sha256=sha(args.training_report),training_audit_sha256=sha(args.training_audit),
            pretrain_report_sha256=sha(args.pretrain_report),pretrain_audit_sha256=sha(args.pretrain_audit),
            training_binding=training['binding'],candidate_adapter_sha256=training['adapter']['sha256'],
            confirm_case_sha256=shared.CONFIRM_CASE_SHA,confirm_case_proof=case_proof)
    arms = ARMS if args.stage == 'full' else ARMS[:2]
    results = {}; replays = {}; started = time.time()
    for phase in (('ppl','mk') if args.stage == 'full' else ('ppl',)):
        for arm in arms:
            if phase == 'mk' and not results[arm].get('ppl_complete'): continue
            adapted = arm == 'w4_sq325_resurface'
            path = args.out_dir/(args.stage+'_'+arm+'.json')
            arm_common = dict(common,arm=arm,adapter_loaded=adapted,
                adapter_sha256=training['adapter']['sha256'] if adapted else None)
            try:
                need(v10.v6.no_adapter_hooks(model),'Adapter hooks leaked before arm')
                with native.install_fp16(model,adapter_path,expected_binding=training['binding']) if adapted else contextlib.nullcontext() as bank:
                    frozen.check()
                    before = adapter_storage(bank,training['adapter'] if adapted else {},arm)
                    if phase == 'ppl':
                        result = evaluate_ppl(model,table,windows,path,arm_common)
                        if pretrain_raw is not None and arm in pretrain_raw:
                            replays[arm] = exact_ppl(pretrain_raw[arm],result)
                    else:
                        result = evaluate_mk(model,tokenizer,table,cases,path,results[arm])
                    after = adapter_storage(bank,training['adapter'] if adapted else {},arm)
                    need(before == after,'Adapter tensor/storage changed within evaluation')
                    result['adapter_storage'] = after; result['adapter_unchanged'] = True
                result['controller_restoration'] = check_native_snapshot(original_snapshot)
                need(v10.v6.no_adapter_hooks(model),'Adapter hooks remain after arm')
                result['adapter_hooks_removed_after_arm'] = True
                result['frozen_source'] = frozen.check()
                result['backend_policy_check'] = check_replay_backend(policy)
                need(tensor_sha(table) == table_sha and code_hashes() == hashes,'Table or evaluation source changed')
                result['candidate_table_unchanged'] = True
            except v10.v6.CandidateInvalid as error:
                result = read_json(path)
                # Preserve known numerical quality failure; ownership/source/backend violations still raise.
                result.update(complete=False,error=str(error),error_type=type(error).__name__,
                    failure_kind='nonfinite_quality',invalid_candidate=True,fatal_failure=False)
                result['controller_restoration'] = check_native_snapshot(original_snapshot)
                result['frozen_source'] = frozen.check()
                result['backend_policy_check'] = check_replay_backend(policy)
                need(tensor_sha(table) == table_sha and code_hashes() == hashes,'Integrity failed after nonfinite outcome')
            except BaseException as error:
                failed = read_json(path) if path.exists() else arm_common
                failed.update(complete=False,error=repr(error),error_type=type(error).__name__,
                              failure_kind='integrity_or_runtime',fatal_failure=True)
                save_json(path,failed); raise
            save_json(path,result); results[arm] = result
    final_base = shared.assert_loaded_hashes(model,expected_base)
    need(code_hashes() == hashes,'Evaluation source changed before final report')
    complete = all(row.get('complete') is True for row in results.values())
    reports = {arm:dict(file=args.stage+'_'+arm+'.json',sha256=sha(args.out_dir/(args.stage+'_'+arm+'.json')),
        ppl=row.get('ppl',{}).get('ppl')) for arm,row in results.items()}
    if args.stage == 'pretrain':
        report = dict(format='W4_STATE_PRETRAIN_V1',complete=complete,stage='pretrain',
            protocol_sha256=PROTOCOL,w4_manifest_sha256=shared.W4_MANIFEST_SHA,
            calibration_sha256=sha(args.calibration),calibration_receipt_sha256=sha(args.calibration.with_suffix('.json')),
            binding=binding,code_sha256=hashes,backend_policy=policy,reports=reports,
            final_w4_content_check=final_base,heldout_used_for_selection=False,mk_used=False,
            independent_cpu_audit_required=True,elapsed_seconds=time.time()-started)
        if not complete: report['failure_kind'] = 'nonfinite_quality'
        save_json(args.out_dir/'pretrain_report.json',report)
    else:
        comparison = dict(format=COMPARE_FORMAT,complete=complete,stage='full',protocol_sha256=PROTOCOL,
            w4_manifest_sha256=shared.W4_MANIFEST_SHA,calibration_sha256=sha(args.calibration),binding=binding,
            code_sha256=hashes,backend_policy=policy,report_sha256={k:v['sha256'] for k,v in reports.items()},
            reports=reports,pretrain_report_sha256=sha(args.pretrain_report),pretrain_audit_sha256=sha(args.pretrain_audit),
            training_report_sha256=sha(args.training_report),training_audit_sha256=sha(args.training_audit),
            candidate_adapter_sha256=training['adapter']['sha256'],final_w4_content_check=final_base,
            pretrain_replay=replays,quality_gate_pass=False,independent_full_audit_required=True,
            quality_scope=common['quality_scope'],elapsed_seconds=time.time()-started)
        if complete:
            restorations = dict(w4_sq325=exact_removal(results['w4_sq325'],results['restored_w4_sq325']),
                                w4_s16=exact_removal(results['w4_s16'],results['restored_w4_s16']))
            repair,checks,separate = quality_checks(results,replays,restorations,training_audit)
            comparison.update(restoration=restorations,comparison=repair,quality_checks=checks,
                separate_reference_checks=separate,quality_gate_pass=all(checks.values()),
                fresh_w4_s16_comparison=compare_pair(results['w4_s16'],results['w4_sq325_resurface']),
                memory=dict(w4_file_bytes_including_manifest=4381415300,expanded_weight_payload_bytes=16473999360,
                    s16_cache_bytes=S16_CACHE_BYTES,sq325_cache_bytes=CACHE_BYTES,adapter_bytes=ADAPTER_BYTES,
                    adapter=results['w4_sq325_resurface']['adapter_storage'],batch_size=1,
                    scope='Weight files, loaded FP16 parameters, persistent caches, adapter payload and measured CUDA peaks are separate'))
        else:
            comparison.update(failure_kind='nonfinite_quality',invalid_arms=[k for k,v in results.items() if not v.get('complete')])
        save_json(args.out_dir/'full_comparison.json',comparison)
        report = comparison
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
