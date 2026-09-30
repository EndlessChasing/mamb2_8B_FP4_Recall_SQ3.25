#!/usr/bin/env python3
"""Full three-arm deployed PPL and recall validation of fresh V11 Resurface."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mamba2_recall import runtime,resurface_data as data,resurface_native as native
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from prepare_state_first_v5 import need,read_json,tensor_sha
from evaluate_quant_first import FrozenBase,compare_pair,check_restoration,VALIDATION_TOKENS_SHA
from evaluate_resurface_more import pin_replay_backend,check_replay_backend
from run_statequant import save_json
from state_ppl_codec_v10 import StatePPLQuantV10
import run_state_ppl_v10 as v10
import train_state_resurface_v11 as trainer

PROTOCOL='cebe06473806d74726ac006ff6d5dd316f4fdcfb779eb47f5af271ddb1fa04a6'
FORMAT='MAMBA2_STATE_RESURFACE_V11_EVAL_V1'
COMPARE='MAMBA2_STATE_RESURFACE_V11_COMPARISON_V1'
ARMS=('v10_no_adapter','v10_resurface','restored_v10_no_adapter')
LAYOUT='32_32_64'
CACHE_BYTES=28499968
ADAPTER_BYTES=2308208
PARENT_PPL=8.186186562837207
NUMERIC_PROTOCOL='24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb'
TRAIN_MANIFEST='451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0'


def code_hashes():
    result=trainer.code_hashes()
    for name in ('scripts/run_state_resurface_v11.py','scripts/audit_state_resurface_v11.py'):
        result[name]=data.sha_file(ROOT/name)
    return dict(sorted(result.items()))


def validate_training(path,parent_binding):
    """Check final training/export/checkpoint evidence before any model load."""
    report=read_json(path)
    need(report.get('format')=='MAMBA2_STATE_RESURFACE_V11_TRAIN_V1' and report.get('complete') is True
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
        teacher='separate unadapted source, S16 per-token carry',v2_trainer_sha256=trainer.V2_TRAINER_SHA,
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
    need(checkpoint['format']=='MAMBA2_STATE_RESURFACE_V11_CHECKPOINT_V1' and checkpoint['binding']==binding
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
    need(proof['format']=='MAMBA2_STATE_RESURFACE_V11_TRAIN_PROOF_V1' and proof['binding']==binding
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


def validate_training_audit(path,report_path,report):
    audit=read_json(path)
    need(audit.get('format')=='MAMBA2_STATE_RESURFACE_V11_AUDIT_V1' and audit.get('complete') is True
         and audit.get('passed') is True and audit.get('stage')=='training' and audit.get('cuda_initialized') is False
         and audit.get('protocol_sha256')==PROTOCOL
         and audit.get('source_sha256')==data.sha_file(ROOT/'scripts/audit_state_resurface_v11.py')
         and audit.get('training_report_sha256')==data.sha_file(report_path)
         and audit.get('checks_sha256')==report['binding']['kernel_checks_sha256']
         and audit.get('smoke_report_sha256')==report['discarded_smoke_check']['sha256'],
         'Passing independent V11 training/codec/smoke audit required')
    for name,digest in audit.get('auditor_dependency_sha256',{}).items():
        need(data.sha_file(ROOT/'scripts'/name)==digest,'Independent audit dependency changed: '+name)
    return audit


def classify_prediction(row):
    prediction=row['prediction']
    if prediction==row['answer']:
        return 'correct_target' if row['condition']=='normal' else 'removed_target_match'
    if prediction is None:return 'unparseable'
    if int(prediction) in {value for _,value in row['records']}:
        return 'wrong_present_value' if row['condition']=='normal' else 'other_present_value'
    return 'not_present_value'


def summary_rows(rows):
    correct=sum(row['correct'] for row in rows)
    categories={}
    for row in rows:
        kind=classify_prediction(row);categories[kind]=categories.get(kind,0)+1
    return dict(count=len(rows),correct=correct,accuracy=correct/len(rows) if rows else None,
        prediction_categories=dict(sorted(categories.items())))


def summarize_mk(rows):
    normal=[r for r in rows if r['condition']=='normal']
    need(len(rows)==768 and len(normal)==384 and len({r['id'] for r in rows})==768,'Full MK population differs')
    summary={condition:summary_rows([r for r in rows if r['condition']==condition]) for condition in ('normal','target_removed')}
    strata={}
    for label,key in (
        ('normal_by_N',lambda r:str(r['N'])),('normal_by_template',lambda r:str(r['template'])),
        ('normal_by_N_template',lambda r:f"N{r['N']}_t{r['template']}"),
        ('normal_by_N_query_position',lambda r:f"N{r['N']}_p{r['query_position']}"),
        ('normal_by_N_position_quartile',lambda r:f"N{r['N']}_q{4*r['query_position']//r['N']}")):
        groups={}
        for row in normal:groups.setdefault(key(row),[]).append(row)
        strata[label]={name:summary_rows(group) for name,group in sorted(groups.items())}
    return summary,strata


def adapter_storage(bank,export):
    if bank is None:
        return dict(loaded=False,tensors=0,parameters=0,fp16_payload_bytes=0,resident_storage_bytes=0,
            persistent_buffer_bytes=0,additional_recurrent_cache_bytes=0,ema_enabled=False,
            cache_plus_adapter_bytes=CACHE_BYTES)
    parameters=bank.masters
    need(len(parameters)==224 and all(v.dtype==torch.float16 and v.is_cuda and not v.requires_grad
         and v.grad is None for v in parameters.values()),'Installed FP16 adapter geometry/dtype differs')
    hashes={k:native.tensor_hash(v) for k,v in parameters.items()}
    need(hashes==export['tensor_sha256'],'Loaded adapter tensors differ')
    storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in parameters.values()}
    buffers=list(bank.adapters.named_buffers())
    need(not buffers and not bank._frames and not bank._requests,'Unexpected persistent adapter buffers or active frames')
    payload=sum(v.numel()*v.element_size() for v in parameters.values());resident=sum(storages.values())
    need(payload==resident==ADAPTER_BYTES,'Actual loaded adapter storage differs')
    return dict(loaded=True,tensors=len(parameters),parameters=sum(v.numel() for v in parameters.values()),
        fp16_payload_bytes=payload,resident_storage_bytes=resident,persistent_buffer_bytes=0,
        additional_recurrent_cache_bytes=0,ema_enabled=False,cache_plus_adapter_bytes=CACHE_BYTES+resident,
        serialized_file_bytes=export['bytes'],adapter_sha256=export['sha256'],tensor_sha256=hashes,
        tensors_unchanged=True,scope='Actual distinct CUDA parameter storage; source weights, temporaries and allocator reserve excluded')


@torch.inference_mode()
def evaluate_arm(model,tokenizer,table,windows,cases,path,common,archive=None):
    spec=v10.candidate_spec(LAYOUT,dict(candidate_order=list(v10.LAYOUTS)))
    result=v10.evaluate(model,table,spec,windows,path,common)
    v10.v6.check_ppl(result,v10.v6.window_identity(windows))
    result['ppl_cache']=copy.deepcopy(result['cache'])
    if archive is not None:
        result['archived_ppl_replay']=v10.archive_replay(archive,result)
    result.update(complete=False,mk=dict(rows=[]))
    save_json(path,result)
    started=time.time()
    with StatePPLQuantV10(model,table,layout=LAYOUT) as execution:
        for index,case in enumerate(cases):
            encoded=tokenizer.encode(case['prompt'])
            hidden=execution.backbone(torch.tensor(encoded,device='cuda',dtype=torch.long)[None],reset=True)[:,-1:]
            generated=[]
            for step in range(12):
                logits=model.lm_head(hidden)
                if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite MK logits')
                token=int(logits.argmax(-1).item());generated.append(token)
                if token==tokenizer.eos_token_id or step==11:break
                hidden=execution.backbone(torch.tensor([[token]],device='cuda'))
            v10.finite_cache(execution)
            output=tokenizer.decode(generated);match=re.search(r'(?<!\d)\d{6}(?!\d)',output)
            prediction=match.group() if match else None
            row=dict(case,prompt_tokens=len(encoded),prompt_token_sha256_int64le=runtime.token_digest(encoded),
                generated_ids=generated,output=output,prediction=prediction,correct=prediction==case['answer'])
            row['prediction_category']=classify_prediction(row)
            result['mk']['rows'].append(row)
            result['mk_cache']=v10.require_cache(execution,spec)
            if index==0 or (index+1)%8==0:
                save_json(path,result);print(f'[{common["arm"]} MK] {index+1}/{len(cases)}',flush=True)
        result['storage_descriptor_mk']=execution.storage_descriptor()
        v10.validate_storage_descriptor(result['storage_descriptor_mk'],LAYOUT)
        need(native.tensor_hash(execution.permutations)==tensor_sha(table),'MK runtime table changed')
    result['mk']['summary'],result['mk']['strata']=summarize_mk(result['mk']['rows'])
    result['mk']['elapsed_seconds']=time.time()-started
    result.update(complete=True,mk_persistent_float_finite_checks_passed=True,mk_runtime_table_unchanged=True,
        gpu_memory=runtime.gpu_memory_receipt())
    return result


def exact_removal(before,after):
    proof=check_restoration(before,after)
    for field in ('repeated_reset_probe','ppl_cache','cache','mk_cache','storage_descriptor_probe',
                  'storage_descriptor','storage_descriptor_mk'):
        need(before[field]==after[field],'Adapter-removal cache/probe/geometry differs: '+field)
    need(before['mk']['summary']==after['mk']['summary'] and before['mk']['strata']==after['mk']['strata'],
         'Restored MK summaries differ')
    return dict(proof,whole_reset_probe_exact=True,ppl_end_cache_exact=True,mk_end_cache_exact=True,
        all_storage_descriptors_exact=True)


def quality_checks(results,parent_replay,restoration,training_audit):
    repair=compare_pair(results[ARMS[0]],results[ARMS[1]])
    checks=dict(ppl_no_worse_than_v10=repair['candidate_ppl']<=repair['control_ppl'],
        ppl_strictly_below_8p25=repair['candidate_ppl']<8.25,normal_mk_increases=repair['normal_mk_correct_delta']>0,
        paired95_lower_positive=repair['normal_mk_paired_bootstrap_95ci'][0]>0,
        exact_archive_replay=parent_replay['complete'],exact_adapter_removal_replay=restoration['complete'],
        cache_same_budget=all(row[field]['total_bytes']==CACHE_BYTES for row in results.values() for field in ('ppl_cache','mk_cache')),
        adapter_storage_verified=results[ARMS[1]]['adapter_storage']['resident_storage_bytes']==ADAPTER_BYTES,
        training_audit_passed=training_audit['passed'])
    return repair,checks


def self_test():
    """Synthetic scoring/control/replay fixtures; no model construction."""
    import os
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU fixtures require CUDA_VISIBLE_DEVICES=')
    need(not torch.cuda.is_initialized(),'CUDA initialized before CPU fixtures')
    checks=[]
    def passed(name,condition):
        need(condition,name);checks.append(dict(name=name,passed=True))
    def rejects(name,fn):
        try:fn()
        except (ValueError,RuntimeError):passed(name,True)
        else:raise AssertionError('Expected rejection: '+name)
    base=dict(condition='normal',answer='123456',records=[[111111,123456],[222222,654321]])
    for prediction,expected in (('123456','correct_target'),('654321','wrong_present_value'),(None,'unparseable'),('999999','not_present_value')):
        passed('normal prediction '+expected,classify_prediction(dict(base,prediction=prediction))==expected)
    removed=dict(base,condition='target_removed',records=[[333333,777777],[222222,654321]])
    for prediction,expected in (('123456','removed_target_match'),('654321','other_present_value'),(None,'unparseable'),('999999','not_present_value')):
        passed('removed prediction '+expected,classify_prediction(dict(removed,prediction=prediction))==expected)
    cases=data.generate_cases('confirm');baseline=[];adapted=[]
    for case in cases:
        row=dict(case,prediction=None,correct=False,generated_ids=[1],output='no six-digit answer',
            prompt_token_sha256_int64le=data.sha_bytes(case['prompt'].encode()))
        baseline.append(row)
        other=copy.deepcopy(row)
        if case['condition']=='normal' and case['N']==16:
            other.update(prediction=case['answer'],correct=True,generated_ids=[2],output=case['answer'])
        adapted.append(other)
    summary,strata=summarize_mk(adapted)
    passed('exact768 cases and N16/N64 strata',summary['normal']['correct']==192 and summary['target_removed']['correct']==0
        and strata['normal_by_N']['16']['count']==192 and strata['normal_by_N']['64']['count']==192)
    passed('six template cells and all query positions',len(strata['normal_by_N_template'])==6
        and len(strata['normal_by_N_query_position'])==80 and len(strata['normal_by_N_position_quartile'])==8)
    rejects('duplicate case rejected',lambda:summarize_mk(adapted[:-1]+[adapted[0]]))
    win=dict(start=0,target_tokens=2,token_sha256_int64le='synthetic',nll=2.,ppl=math.e)
    def arm(name,rows,ppl):
        s,t=summarize_mk(rows)
        return dict(complete=True,arm=name,ppl=dict(windows=[win],nll=2.,target_tokens=2,ppl=ppl),
            mk=dict(rows=rows,summary=s,strata=t),cache=dict(total_bytes=CACHE_BYTES),
            ppl_cache=dict(total_bytes=CACHE_BYTES),mk_cache=dict(total_bytes=CACHE_BYTES),
            repeated_reset_probe=dict(fixture=True),storage_descriptor_probe=dict(fixture=True),
            storage_descriptor=dict(fixture=True),storage_descriptor_mk=dict(fixture=True),
            adapter_storage=dict(resident_storage_bytes=ADAPTER_BYTES if name==ARMS[1] else 0))
    results={ARMS[0]:arm(ARMS[0],baseline,PARENT_PPL),ARMS[1]:arm(ARMS[1],adapted,PARENT_PPL),
        ARMS[2]:arm(ARMS[2],copy.deepcopy(baseline),PARENT_PPL)}
    replay=exact_removal(results[ARMS[0]],results[ARMS[2]])
    passed('complete removal replay',replay['complete'] and replay['all_generated_ids_exact'])
    for field in ('ppl_cache','mk_cache','repeated_reset_probe','storage_descriptor_mk'):
        bad=copy.deepcopy(results[ARMS[2]]);bad[field]['extra']=1
        rejects('removal rejects '+field,lambda:exact_removal(results[ARMS[0]],bad))
    bad=copy.deepcopy(results[ARMS[2]]);bad['mk']['rows'][0]['generated_ids']=[99]
    rejects('removal rejects changed generated IDs',lambda:exact_removal(results[ARMS[0]],bad))
    delta,gates=quality_checks(results,dict(complete=True),replay,dict(passed=True))
    passed('positive paired MK gate and equality PPL',all(gates.values()) and delta['normal_mk_correct_delta']==192
        and delta['normal_mk_paired_bootstrap_95ci'][0]>0)
    results[ARMS[1]]['ppl']['ppl']=math.nextafter(PARENT_PPL,math.inf)
    _,gates=quality_checks(results,dict(complete=True),replay,dict(passed=True))
    passed('one ULP PPL degradation is rejected',not gates['ppl_no_worse_than_v10'] and gates['ppl_strictly_below_8p25'])
    results[ARMS[1]]['ppl']['ppl']=8.25
    _,gates=quality_checks(results,dict(complete=True),replay,dict(passed=True))
    passed('strict8p25 boundary',not gates['ppl_strictly_below_8p25'])
    results[ARMS[1]]['mk']=copy.deepcopy(results[ARMS[0]]['mk'])
    delta,gates=quality_checks(results,dict(complete=True),replay,dict(passed=True))
    passed('unchanged MK fails repair',not gates['normal_mk_increases'] and not gates['paired95_lower_positive']
        and delta['normal_mk_paired_bootstrap_95ci']==[0.,0.])
    passed('no loaded baseline adapter residency',adapter_storage(None,{})['cache_plus_adapter_bytes']==CACHE_BYTES)
    passed('CUDA uninitialized',not torch.cuda.is_initialized())
    return dict(format='MAMBA2_STATE_RESURFACE_V11_EVAL_CPU_FIXTURES_V1',complete=True,passed=True,
        checks=checks,count=len(checks),source_sha256=data.sha_file(__file__),protocol_sha256=PROTOCOL,
        cuda_initialized=False,quality_measured=False)


def main():
    if sys.argv[1:]==['--self-test']:
        print(json.dumps(self_test(),indent=2));return
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','calibration','v10-full-dir','v10-full-audit','training-report','training-audit','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--s16-report',type=Path)
    args=parser.parse_args()
    need(not args.out.exists(),'Fresh output directory required')
    need(data.sha_file(ROOT/'docs/STATE_RESURFACE_V11_PROTOCOL.md')==PROTOCOL,'Frozen protocol changed')
    selected,_,archive,parent_binding=trainer.load_v10_inputs(args)
    training,adapter_path=validate_training(args.training_report,parent_binding)
    training_audit=validate_training_audit(args.training_audit,args.training_report,training)
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==runtime.TOKENIZER_SHA256,'Tokenizer differs')
    ids,dataset=load_wikitext_tokens(tokenizer,'validation');windows=ppl_windows(ids,2048)
    need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764
         and dataset['token_stream_sha256_int64le']==VALIDATION_TOKENS_SHA,'Full validation population differs')
    cases=data.generate_cases('confirm');case_proof=data.validate_cases(cases,'confirm')
    need(len(cases)==768 and sum(c['condition']=='normal' for c in cases)==384,'Full CONFIRM population differs')
    s16=None
    if args.s16_report is not None:
        need(data.sha_file(args.s16_report)==v10.v6.S16_REPORT_SHA,'Original S16 archive differs')
        s16=read_json(args.s16_report)
    torch.set_num_threads(8);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');policy=pin_replay_backend()
    hashes=code_hashes();args.out.mkdir(parents=True)
    model=runtime.load_source_model(args.source_dir);frozen=FrozenBase(model)
    table=selected['permutations'];table_sha=tensor_sha(table)
    common=dict(format=FORMAT,stage='full',protocol_sha256=PROTOCOL,parent_binding=parent_binding,
        training_report_sha256=data.sha_file(args.training_report),training_audit_sha256=data.sha_file(args.training_audit),
        candidate_adapter_sha256=training['adapter']['sha256'],training_binding=training['binding'],
        dataset=dataset,confirm_case_sha256=data.sha_bytes(data.canonical_bytes(cases)),confirm_case_proof=case_proof,
        confirm_generation='Frozen resurface_data.generate_cases(confirm); full vocabulary greedy max12/EOS; first standalone six-digit integer',
        numeric_protocol_sha256=NUMERIC_PROTOCOL,environment=runtime.environment_receipt(),backend_policy=policy,code_hashes=hashes,
        candidate_table_sha256=table_sha,allocation_spec=v10.layout_spec(LAYOUT),heldout_used=True,
        heldout_used_for_selection=False,mk_used=True,s16_report_sha256=v10.v6.S16_REPORT_SHA if s16 else None,
        quality_scope='Historically exposed validation corpus and CONFIRM prompt families; no unseen-generalization claim')
    results={};started=time.time()
    for arm in ARMS:
        adapted=arm=='v10_resurface';path=args.out/('full_'+arm+'.json')
        need(v10.v6.no_adapter_hooks(model),'Adapter hooks leaked before arm')
        arm_common=dict(common,arm=arm,adapter_loaded=adapted,adapter_sha256=training['adapter']['sha256'] if adapted else None)
        try:
            with native.install_fp16(model,adapter_path,expected_binding=training['binding']) if adapted else contextlib.nullcontext() as bank:
                frozen.check();before=adapter_storage(bank,training['adapter'])
                result=evaluate_arm(model,tokenizer,table,windows,cases,path,arm_common,archive if not adapted else None)
                after=adapter_storage(bank,training['adapter']);need(before==after,'Adapter tensors/storage changed within arm')
                result['adapter_storage']=after;result['adapter_unchanged']=True
                result['frozen_source']=frozen.check();result['backend_policy_check']=check_replay_backend(policy)
                need(tensor_sha(table)==table_sha,'CPU coordinate table changed')
                result['candidate_table_unchanged']=True
            del bank
            need(v10.v6.no_adapter_hooks(model),'Adapter hooks remain after removal')
            result['adapter_hooks_removed_after_arm']=True
            save_json(path,result);results[arm]=result
        except BaseException as error:
            failed=read_json(path) if path.exists() else arm_common
            failed.update(complete=False,error=repr(error),error_type=type(error).__name__,fatal_failure=True)
            save_json(path,failed);raise
    parent_replay=results[ARMS[0]]['archived_ppl_replay'];save_json(args.out/'full_parent_replay.json',parent_replay)
    restoration=exact_removal(results[ARMS[0]],results[ARMS[2]]);save_json(args.out/'full_restoration.json',restoration)
    need(results[ARMS[0]]['ppl']['ppl']==PARENT_PPL,'Exact v10 parent PPL differs')
    repair,checks=quality_checks(results,parent_replay,restoration,training_audit)
    comparison=dict(format=COMPARE,complete=True,protocol_sha256=PROTOCOL,stage='full',parent_binding=parent_binding,
        training_report_sha256=common['training_report_sha256'],training_audit_sha256=common['training_audit_sha256'],
        candidate_adapter_sha256=training['adapter']['sha256'],backend_policy=policy,code_hashes=hashes,
        report_sha256={arm:data.sha_file(args.out/('full_'+arm+'.json')) for arm in ARMS},
        parent_replay=parent_replay,restoration=restoration,comparison=repair,quality_checks=checks,
        quality_gate_pass=all(checks.values()),independent_full_audit_required=True,
        selection_scope='Final1536 TRAIN export only; validation/CONFIRM never select an adapter or threshold',
        quality_scope=common['quality_scope'],
        memory=dict(persistent_cache_bytes=CACHE_BYTES,adapter=results[ARMS[1]]['adapter_storage'],
            batch_size=1,static_table_bytes=57344,
            scope='Batch-one recurrent/convolution cache plus one shared static table and one adapter; source weights, temporaries and allocator reserve excluded'),
        elapsed_seconds=time.time()-started)
    if s16 is not None:comparison['original_s16_comparison']=compare_pair(s16,results[ARMS[1]])
    save_json(args.out/'full_comparison.json',comparison)
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
