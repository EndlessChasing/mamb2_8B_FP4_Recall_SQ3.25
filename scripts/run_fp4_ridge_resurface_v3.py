#!/usr/bin/env python3
"""Explicit transferred/fresh Resurface with exact frozen group-ridge SQ3.25."""
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
import traceback
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamba2_recall import runtime, resurface_data as data, resurface_native as native
from mamba2_recall.fp4 import load_fp4_model
from evaluate_quant_first import FrozenBase, compare_pair
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from prepare_quant_first import load_train_tokens, TRAIN_SHA
from run_state_resurface_v11 import classify_prediction, summarize_mk
from run_w4_state_resurface_v1 import native_snapshot, check_native_snapshot, adapter_storage
from run_fp4_state_quality_v1 import assert_weight_content
import run_fp4_latent_state_quality_v1 as latent
import run_w4_state_repair_v2 as evaluator
from fp4_zero_predictor_codec_v1 import PredictorState
import ridge_resurface_binding_v3 as shared
import fp4_state_binding_v1 as fp4
import w4_state_binding_v1 as w4
need,sha,read,put=shared.need,shared.sha,shared.read,shared.put
FORMAT='FP4_RIDGE_RESURFACE_EVAL_V3'
ARMS=('ridge_parent','ridge_resurface')


def validate_adapter(args):
    train=read(args.training_report); audit=read(args.training_audit)
    need(train['complete'] is True and train['mode']=='formal' and
         audit['passed'] is True and audit['complete'] is True and
         audit['cuda_initialized'] is False and
         audit['training_report_sha256']==sha(args.training_report), 'Audited final training required')
    if args.adapter_origin=='transfer':
        need(train['format']=='FP4_G16_RESURFACE_TRAIN_V1' and
             audit['format']=='FP4_G16_RESURFACE_CPU_AUDIT_V1' and
             train['successful_updates']==1536 and train['fresh_initialization'] is True and
             train['parity_128_512_initial_and_export'] is True,
             'Frozen historical transfer training differs')
    else:
        need(train['format']=='FP4_RIDGE_RESURFACE_TRAIN_V3' and
             audit['format']=='FP4_RIDGE_RESURFACE_TRAIN_CPU_AUDIT_V3' and
             train['successful_updates']==1536 and train['fresh_initialization'] is True and
             train['parity_128_512_initial_and_export'] is True,
             'Fresh ridge training differs')
    for name,digest in train['code_sha256'].items():
        need(not Path(name).is_absolute() and (ROOT/name).resolve().is_relative_to(ROOT) and
             sha(ROOT/name)==digest, 'Training code changed: '+name)
    adapter=args.training_report.parent/train['adapter']['file']
    need(Path(train['adapter']['file']).name==train['adapter']['file'] and
         sha(adapter)==train['adapter']['sha256'] and
         adapter.stat().st_size==train['adapter']['bytes'] and
         train['adapter']['payload_bytes']==shared.ADAPTER_BYTES and
         train['adapter']['discarded'] is False, 'Frozen FP16 adapter differs')
    payload=native.read_fp16(adapter,expected_binding=train['binding'])
    need(sum(x.numel()*x.element_size() for x in payload['tensors'].values())==shared.ADAPTER_BYTES and
         {k:native.tensor_hash(v) for k,v in payload['tensors'].items()}==train['adapter']['tensor_sha256'],
         'Adapter tensor hashes differ')
    return train,adapter


@contextlib.contextmanager
def execution_for(model,parent):
    table,layouts,bases,scales,predictors=parent[:5]
    snapshot=native_snapshot(model)
    with PredictorState(model,table,layouts,bases,scales,predictors) as execution:
        yield execution
    check_native_snapshot(snapshot)


@torch.inference_mode()
def evaluate_ppl(model,parent,windows,path,common,frozen,policy):
    table,layouts,bases,scales,predictors=parent[:5]
    row=dict(common,complete=False,ppl=dict(windows=[]),
        static_basis_sha256=[native.tensor_hash(x) for x in bases],
        static_scale_sha256=[native.tensor_hash(x) for x in scales],
        static_predictor_sha256=[[native.tensor_hash(x) for x in p] for p in predictors],
        layer_layouts=list(layouts))
    put(path,row)
    with execution_for(model,parent) as execution:
        probe=windows[0][1][:128].cuda()[None]
        first=execution.backbone(probe,reset=True); identity=latent.cache_identity(execution)
        second=execution.backbone(probe,reset=True)
        need(torch.equal(first,second) and identity==latent.cache_identity(execution) and
             bool(torch.isfinite(first).all()), 'Repeated reset differs')
        row['repeated_reset_probe']=dict(tokens=128,hidden_sha256=native.tensor_hash(first),
            hidden_and_cache_exact=True,cache_tensor_sha256=identity,
            token_sha256_int64le=runtime.token_digest(probe.cpu().numpy()),
            cache=latent.check_cache(execution,128)[0])
        row['storage_descriptor_probe']=latent.check_cache(execution,128)[1]
        total=count=0
        for index,(start,window) in enumerate(windows):
            tokens=window.cuda(); hidden=execution.backbone(tokens[:-1][None],reset=True)
            need(bool(torch.isfinite(hidden).all()), 'Nonfinite PPL hidden')
            latent.finite_cache(execution); loss_sum=0.
            for pos in range(0,hidden.shape[1],64):
                end=min(pos+64,hidden.shape[1])
                logits=model.lm_head(hidden[:,pos:end]).float()
                loss=F.cross_entropy(logits.reshape(-1,256000),tokens[pos+1:end+1],reduction='sum')
                loss_sum+=float(loss); del logits,loss
            need(math.isfinite(loss_sum), 'Nonfinite PPL NLL')
            targets=len(window)-1; total+=loss_sum;count+=targets
            row['ppl']['windows'].append(dict(start=start,target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()),
                nll=loss_sum,ppl=math.exp(loss_sum/targets)))
            row['ppl'].update(nll=total,target_tokens=count,ppl=math.exp(total/count))
            row['cache'],row['storage_descriptor']=latent.check_cache(execution,targets)
            row['ppl_end_cache_tensor_sha256']=latent.cache_identity(execution)
            put(path,row)
            if index==0 or (index+1)%8==0 or index+1==len(windows):
                print(f'[{common["arm"]} PPL] {index+1}/{len(windows)} {row["ppl"]["ppl"]:.9f}',flush=True)
            del hidden,tokens
        row['ppl_cache']=copy.deepcopy(row['cache'])
    row.update(complete=True,ppl_complete=True,frozen_source=frozen.check(),
        backend_policy_check=check_replay_backend(policy),
        frozen_weight_check=assert_weight_content(model,parent[6]))
    put(path,row)
    return row


@torch.inference_mode()
def evaluate_mk(model,tokenizer,parent,cases,path,row):
    row.update(complete=False,mk=dict(rows=[]));put(path,row)
    with execution_for(model,parent) as execution:
        for index,case in enumerate(cases):
            encoded=tokenizer.encode(case['prompt'])
            hidden=execution.backbone(torch.tensor(encoded,device='cuda',dtype=torch.long)[None],reset=True)[:,-1:]
            generated=[]
            for step in range(12):
                logits=model.lm_head(hidden)
                need(bool(torch.isfinite(logits).all()), 'Nonfinite MK logits')
                token=int(logits.argmax(-1).item());generated.append(token)
                if token==tokenizer.eos_token_id or step==11:break
                hidden=execution.backbone(torch.tensor([[token]],device='cuda'))
            latent.finite_cache(execution)
            output=tokenizer.decode(generated);match=re.search(r'(?<!\d)\d{6}(?!\d)',output)
            prediction=match.group() if match else None
            item=dict(case,prompt_tokens=len(encoded),prompt_token_sha256_int64le=runtime.token_digest(encoded),
                generated_ids=generated,output=output,prediction=prediction,correct=prediction==case['answer'])
            item['prediction_category']=classify_prediction(item)
            row['mk']['rows'].append(item)
            row['mk_cache'],row['storage_descriptor_mk']=latent.check_cache(execution)
            if index==0 or (index+1)%8==0:
                put(path,row);print(f'[{row["arm"]} MK] {index+1}/{len(cases)}',flush=True)
        row['mk_end_cache_tensor_sha256']=latent.cache_identity(execution)
    row['mk']['summary'],row['mk']['strata']=summarize_mk(row['mk']['rows'])
    row.update(complete=True,mk_complete=True);put(path,row)
    return row


def exact_parent(row,saved):
    for key in ('ppl','repeated_reset_probe','cache','ppl_cache','ppl_end_cache_tensor_sha256'):
        need(row[key]==saved[key], 'Archived ridge parent differs: '+key)
    need(row['ppl']['ppl']==shared.PARENT_PPL, 'Archived ridge parent PPL differs')
    return dict(complete=True,windows=130,targets=264764,per_window_nll_exact=True,
                aggregate_ppl_exact=True,reset_hidden_and_cache_exact=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('screen','full'),required=True)
    ap.add_argument('--adapter-origin',choices=('transfer','fresh'),required=True)
    for name in ('source-dir','training-report','training-audit','out-dir'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--prose-tokens',type=Path)
    ap.add_argument('--screen-dir',type=Path)
    ap.add_argument('--mk-always',action='store_true',help='Measure full paired MK even when PPL>=8')
    shared.add_parent_arguments(ap);args=ap.parse_args()
    need(not args.out_dir.exists(), 'Fresh output directory required')
    parent=shared.load_parent(args);train,adapter=validate_adapter(args)
    if args.adapter_origin=='fresh':
        need(train['binding']['ridge_parent']==parent[7], 'Fresh training parent differs')
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==fp4.TOKENIZER_SHA, 'Tokenizer differs')
    if args.stage=='screen':
        need(args.prose_tokens is not None,'Pinned prose TRAIN required')
        train_tokens=load_train_tokens(args.prose_tokens)
        windows=[(i*2048,train_tokens[i].clone()) for i in range(432,448)]
        dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(432,448)),
                     windows=16,target_tokens=32752,historically_exposed_train=True)
    else:
        need(args.screen_dir is not None,'Audited positive screen required')
        screen=read(args.screen_dir/'comparison.json');audit=read(args.screen_dir/'audit.json')
        need(screen['complete'] is True and screen['advance_to_full'] is True and
             screen['adapter_sha256']==train['adapter']['sha256'] and
             audit['passed'] is True and audit['cuda_initialized'] is False and
             audit['input_report_sha256']==sha(args.screen_dir/'comparison.json'), 'Positive audited screen differs')
        ids,dataset=evaluator.load_wikitext_tokens(tokenizer,'validation')
        windows=evaluator.ppl_windows(ids,2048)
        need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and
             dataset['token_stream_sha256_int64le']==fp4.VALIDATION_TOKENS_SHA, 'Full validation differs')
    args.out_dir.mkdir(parents=True);start=time.time()
    report=dict(format=FORMAT,complete=False,stage=args.stage,adapter_origin=args.adapter_origin,
        protocol_sha256=sha(shared.PROTOCOL),code_sha256=shared.hashes(),parent=parent[7],
        dataset=dataset,training_report_sha256=sha(args.training_report),
        training_audit_sha256=sha(args.training_audit),adapter_sha256=train['adapter']['sha256'],
        reports={},historically_exposed_validation=True,quality_scope='historically exposed; no untouched test claim')
    try:
        torch.set_num_threads(8);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');policy=pin_replay_backend()
        need(policy==parent[5]['backend_policy'],'Archived replay backend differs')
        report['backend_policy']=policy
        model=load_fp4_model(args.source_dir,'fp4_g16_e4m3',device='cuda',chunk_rows=512,
            evidence_path=args.out_dir/'conversion_evidence.pt',
            progress=lambda row:print('[FP4 materialize] '+str(row),flush=True))
        report['initial_weight_check']=assert_weight_content(model,parent[6]);frozen=FrozenBase(model)
        snapshot=native_snapshot(model);rows={}
        for arm in ARMS:
            adapted=arm==ARMS[1];path=args.out_dir/(arm+'.json')
            common=dict(format=FORMAT,arm=arm,adapter_origin=args.adapter_origin if adapted else None,
                adapter_loaded=adapted,adapter_sha256=train['adapter']['sha256'] if adapted else None,
                table_sha256=parent[7]['table_sha256'],weight_format='fp4_g16_e4m3',
                source_checkpoint_sha256=fp4.SOURCE_SHA,dataset=dataset)
            with native.install_fp16(model,adapter,expected_binding=train['binding'],
                    expected_base_hashes=parent[6]) if adapted else contextlib.nullcontext() as bank:
                row=evaluate_ppl(model,parent,windows,path,common,frozen,policy)
                row['adapter_storage']=adapter_storage(bank,train['adapter'] if adapted else {},arm)
            row['native_restoration']=check_native_snapshot(snapshot)
            if not adapted and args.stage=='full':report['exact_parent_replay']=exact_parent(row,parent[5])
            put(path,row);rows[arm]=row
            report['reports'][arm]=dict(file=path.name,sha256=sha(path),ppl=row['ppl']['ppl'])
            put(args.out_dir/'comparison.json',report)
        with torch.inference_mode(),execution_for(model,parent) as restored:
            probe=windows[0][1][:128].cuda()[None];hidden=restored.backbone(probe,reset=True)
            need(native.tensor_hash(hidden)==rows[ARMS[0]]['repeated_reset_probe']['hidden_sha256'] and
                 latent.cache_identity(restored)==rows[ARMS[0]]['repeated_reset_probe']['cache_tensor_sha256'],
                 'Adapter removal reset/cache differs')
        report['adapter_removal_reset_and_cache_exact']=True
        full_ppl_pass=args.stage=='full' and rows[ARMS[1]]['ppl']['ppl']<8.0
        measure_mk=args.stage=='full' and (full_ppl_pass or args.mk_always)
        if measure_mk:
            cases=data.generate_cases('confirm')
            need(len(cases)==768 and data.sha_bytes(data.canonical_bytes(cases))==w4.CONFIRM_CASE_SHA,
                 'Full CONFIRM cases differ')
            for arm in ARMS:
                adapted=arm==ARMS[1];path=args.out_dir/(arm+'.json')
                with native.install_fp16(model,adapter,expected_binding=train['binding'],
                        expected_base_hashes=parent[6]) if adapted else contextlib.nullcontext() as bank:
                    rows[arm]=evaluate_mk(model,tokenizer,parent,cases,path,rows[arm])
                rows[arm]['native_restoration']=check_native_snapshot(snapshot);put(path,rows[arm])
                report['reports'][arm]=dict(file=path.name,sha256=sha(path),ppl=rows[arm]['ppl']['ppl'],
                    normal_mk_correct=rows[arm]['mk']['summary']['normal']['correct'])
                put(args.out_dir/'comparison.json',report)
            comparison=compare_pair(rows[ARMS[0]],rows[ARMS[1]])
            mk_pass=(comparison['normal_mk_correct_delta']>0 and
                     comparison['normal_mk_paired_bootstrap_95ci'][0]>0)
        else:comparison=None;mk_pass=False
        report.update(complete=True,ppl={k:rows[k]['ppl']['ppl'] for k in ARMS},
            nll={k:rows[k]['ppl']['nll'] for k in ARMS},comparison=comparison,mk_complete=measure_mk,
            advance_to_full=args.stage=='screen' and rows[ARMS[1]]['ppl']['nll']<rows[ARMS[0]]['ppl']['nll'],
            strict_ppl_below_8=full_ppl_pass,normal_mk_gate_pass=mk_pass,
            publication_gate_pass=full_ppl_pass and mk_pass,
            final_weight_check=assert_weight_content(model,parent[6]),
            backend_final_check=check_replay_backend(policy),elapsed_seconds=time.time()-start,
            memory=dict(logical_encoded_weight_payload_bytes=4638460360,
                expanded_fp16_weight_bytes=16473999360,state_conv_table_cache_bytes=28499968,
                static_basis_bytes=109952,static_latent_scale_bytes=1792,
                static_predictor_bytes=3673216,adapter_fp16_payload_bytes=2308208))
        need(report['code_sha256']==shared.hashes(),'Evaluation code changed')
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:put(args.out_dir/'comparison.json',report)
    print(json.dumps({k:report[k] for k in ('stage','adapter_origin','ppl','advance_to_full',
        'strict_ppl_below_8','mk_complete','publication_gate_pass')},indent=2),flush=True)


if __name__=='__main__':main()
