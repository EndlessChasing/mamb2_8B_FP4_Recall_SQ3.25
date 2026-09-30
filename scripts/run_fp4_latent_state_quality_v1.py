#!/usr/bin/env python3
"""Evaluate actual 52-byte FP8 latent recurrent carry against mixed top48."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime,resurface_native as native
from mamba2_recall.fp4 import load_fp4_model
from prepare_quant_first import load_train_tokens,TRAIN_SHA
from prepare_state_first_v5 import tensor_sha
from evaluate_quant_first import FrozenBase
from evaluate_resurface_more import pin_replay_backend,check_replay_backend
from run_statequant import save_json
import run_w4_state_repair_v2 as evaluator
import run_fp4_state_quality_v1 as fp4_quality
import run_fp4_g16_mixed_layout_v1 as mixed
import run_fp4_g16_historical_layer_v1 as prior
import fp4_latent_state_codec_v1 as latent
import fp4_state_binding_v1 as binding

need,sha,read_json=binding.need,binding.sha,binding.read_json
ARCHIVE=ROOT/'artifacts/fp4_weight_v1/fp4_g16_v1'
ARMS=('parent','latent_off','latent_on','restored_parent')


def code_hashes():
    names=('docs/FP4_G16_LATENT_STATE_QUALITY_V1_PROTOCOL.md',
           'scripts/run_fp4_latent_state_quality_v1.py',
           'scripts/fp4_latent_state_codec_v1.py',
           'scripts/check_fp4_latent_state_codec_v1.py',
           'scripts/run_fp4_g16_mixed_layout_v1.py',
           'scripts/run_fp4_g16_historical_layer_v1.py',
           'scripts/run_w4_state_repair_v2.py',
           'scripts/w4_state_repair_codec_v2.py',
           'mamba2_recall/fp4.py')
    return {name:sha(ROOT/name) for name in names}


def cache_identity(execution):
    out={}
    for index,entry in enumerate(execution._cache):
        for key,value in {**entry.state.tensors,'conv':entry.conv}.items():
            if value.dtype==torch.float8_e4m3fn:
                digest=hashlib.sha256()
                for start in range(0,value.shape[0],128):
                    raw=value[start:start+128].detach().cpu().contiguous().view(torch.uint8)
                    digest.update(raw.numpy().tobytes())
                out[f'{index}.{key}']=digest.hexdigest()
            else:
                out[f'{index}.{key}']=native.tensor_hash(value)
    return out


def finite_cache(execution):
    values=[entry.conv for entry in execution._cache]
    values.extend(t for entry in execution._cache for t in entry.state.tensors.values()
                  if t.is_floating_point())
    need(all(bool(torch.isfinite(t.float()).all()) for t in values),
         'Nonfinite persistent latent/scale/convolution cache')
    scales=[t for entry in execution._cache for name,t in entry.state.tensors.items()
            if name in ('s8','s4')]
    return sum(int((t==0).sum()) for t in scales)


def check_cache(execution,tokens=None):
    cache=execution.cache_breakdown()
    expected=dict(mode='sq3p25',scale_mode='stored_scale',int4_clip=1.,diagnostic=None,
        is_3p25_candidate=True,batch_size=1,allocated_layers=56,
        conv_fp16_bytes=4587520,ssm_payload_bytes=22020096,
        ssm_scale_bytes=1835008,ssm_total_bytes=23855104,
        permutation_bytes=57344,total_bytes=28499968,
        calibration_workspace_bytes=0,diagnostic_dense_fp32_bytes=0,row_bytes=52)
    if tokens is not None:expected['tokens_per_layer']=[tokens]*56
    need(all(cache.get(key)==value for key,value in expected.items()),
         'Actual latent request cache differs from fixed budget')
    desc=execution.storage_descriptor()
    need(desc['total_cache_bytes']==28499968 and desc['row_bytes']==52 and
         desc['payload_bytes']==48 and desc['scale_bytes']==4 and
         len(desc['layers'])==56 and
         desc['static_basis_bytes']==sum(b.untyped_storage().nbytes() for b in execution.bases) and
         desc['static_scale_bytes']==sum(s.untyped_storage().nbytes() for s in execution.latent_scales),
         'Latent physical descriptor or static metadata differs')
    for index,entry in enumerate(execution._cache):
        spec=latent.layout_descriptor(execution.layouts[index])
        latent.validate_state(entry.state,execution.device,execution.layouts[index])
        layer=desc['layers'][index]
        need(layer['state_bytes']==128*64*52 and layer['conv_storage_bytes']==81920 and
             layer['n8']==spec['n8'] and layer['n4']==spec['n4'] and
             layer['nzero']==spec['nzero'] and
             layer['tensors']['latent']['dtype']=='torch.float8_e4m3fn' and
             layer['tensors']['latent']['storage_bytes']==128*64*2,
             'Latent layer allocation differs')
    return cache,desc


@torch.inference_mode()
def evaluate_latent(model,name,table,layouts,bases,scales,windows,path,common,frozen,policy):
    result=dict(common,arm=name,candidate_table_sha256=tensor_sha(table),
        candidate_layout_tuple_sha256=mixed.tuple_hash(layouts),
        layer_layouts=list(layouts),complete=False,ppl=dict(windows=[]),
        latent_mode=name,static_basis_sha256=[tensor_sha(value) for value in bases],
        static_scale_sha256=[tensor_sha(value) for value in scales])
    save_json(path,result)
    started=time.time()
    snapshot=evaluator.native_snapshot(model)
    with latent.LatentState(model,table,layouts,bases,scales) as execution:
        execution.reset(1)
        result['allocated_cache'],result['storage_descriptor_initial']=check_cache(execution,0)
        save_json(path,result)
        probe=windows[0][1][:128].cuda()[None]
        first=execution.backbone(probe,reset=True)
        need(bool(torch.isfinite(first).all()),'Nonfinite latent probe hidden')
        zero_scales=finite_cache(execution)
        cache0=cache_identity(execution);hidden_sha=native.tensor_hash(first)
        second=execution.backbone(probe,reset=True)
        need(bool(torch.isfinite(second).all()) and
             torch.equal(first,second) and cache0==cache_identity(execution),
             'Latent repeated reset differed')
        finite_cache(execution)
        result['repeated_reset_probe']=dict(tokens=128,hidden_sha256=hidden_sha,
            hidden_and_cache_exact=True,cache_tensor_sha256=cache0,
            token_sha256_int64le=runtime.token_digest(probe.cpu().numpy()),
            cache=check_cache(execution,128)[0])
        result['storage_descriptor_probe']=check_cache(execution,128)[1]
        save_json(path,result)
        del first,second,probe,cache0
        total=0.;count=0
        for index,(start,window) in enumerate(windows):
            tokens=window.cuda()
            hidden=execution.backbone(tokens[:-1][None],reset=True)
            need(bool(torch.isfinite(hidden).all()),'Nonfinite latent PPL hidden')
            zero_scales+=finite_cache(execution)
            loss_sum=0.
            for pos in range(0,hidden.shape[1],64):
                end=min(pos+64,hidden.shape[1])
                logits=model.lm_head(hidden[:,pos:end]).float()
                loss=F.cross_entropy(logits.reshape(-1,256000),
                                     tokens[pos+1:end+1],reduction='sum')
                loss_sum+=float(loss)
                del logits,loss
            need(math.isfinite(loss_sum),'Nonfinite latent PPL NLL')
            targets=len(window)-1;total+=loss_sum;count+=targets
            result['ppl']['windows'].append(dict(start=start,target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()),
                nll=loss_sum,ppl=math.exp(loss_sum/targets)))
            result['ppl'].update(nll=total,target_tokens=count,ppl=math.exp(total/count))
            result['cache'],result['storage_descriptor']=check_cache(execution,targets)
            result['ppl_end_cache_tensor_sha256']=cache_identity(execution)
            save_json(path,result)
            if index==0 or (index+1)%8==0 or index+1==len(windows):
                print(f'[{name} PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.9f}',flush=True)
            del hidden,tokens
        result['ppl_cache']=dict(result['cache'])
    restoration=evaluator.check_native_snapshot(snapshot)
    result.update(complete=True,ppl_complete=True,runtime_table_unchanged=True,
        candidate_table_unchanged=tensor_sha(table)==common['table_sha256'],
        allocation_storage_validated=True,persistent_float_finite_checks_passed=True,
        controller_restoration=restoration,
        zero_scale_observations=zero_scales,
        frozen_source=frozen.check(),backend_policy_check=check_replay_backend(policy),
        adapter_hooks_absent=evaluator.no_adapter_hooks(model),
        gpu_memory=runtime.gpu_memory_receipt(),elapsed_seconds=time.time()-started)
    need(result['frozen_source']['identity_version_gradients_unchanged'] is True and
         result['adapter_hooks_absent'] is True and
         result['backend_policy_check']['singleton_config_unchanged'] is True,
         'Latent source/backend/adapter restoration failed')
    evaluator.check_ppl(result,windows)
    save_json(path,result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('screen','full'),required=True)
    ap.add_argument('--source-dir',type=Path,required=True)
    ap.add_argument('--prose-tokens',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--five-table-screen',type=Path,required=True)
    ap.add_argument('--rank-dir',type=Path,required=True)
    ap.add_argument('--combo-screen-dir',type=Path,required=True)
    ap.add_argument('--top4-full-dir',type=Path,required=True)
    ap.add_argument('--mixed-rank-dir',type=Path,required=True)
    ap.add_argument('--mixed-outcome-dir',type=Path,required=True)
    ap.add_argument('--transfer-screen-dir',type=Path,required=True)
    ap.add_argument('--historical-rank-dir',type=Path,required=True)
    ap.add_argument('--historical-screen-dir',type=Path,required=True)
    ap.add_argument('--probe-dir',type=Path,required=True)
    ap.add_argument('--latent-screen-dir',type=Path)
    args=ap.parse_args()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(),
         'Fresh latent quality output required')
    tables,layouts,upstream,sources=prior.frozen_inputs(args)
    historical=read_json(args.historical_screen_dir/'comparison.json')
    historical_audit=read_json(args.historical_screen_dir/'audit.json')
    need(historical['selection']['selected_id']=='parent' and
         historical_audit['passed'] is True and
         historical_audit['input_report_sha256']==sha(args.historical_screen_dir/'comparison.json'),
         'Audited negative historical screen required')
    upstream['historical_screen_sha256']=sha(args.historical_screen_dir/'comparison.json')
    probe=read_json(args.probe_dir/'report.json')
    probe_audit=read_json(args.probe_dir/'audit.json')
    need(probe['complete'] is True and probe['advance_to_codec'] is True and
         probe_audit['passed'] is True and probe_audit['cuda_initialized'] is False and
         probe_audit['input_report_sha256']==sha(args.probe_dir/'report.json') and
         probe_audit['input_payload_sha256']==sha(args.probe_dir/'probe.pt') and
         probe['table_sha256']==tensor_sha(tables['fp4_top4_parent']) and
         probe['mixed_layout_tuple_sha256']==mixed.tuple_hash(layouts),
         'Audited latent PCA probe differs')
    upstream['latent_probe_sha256']=sha(args.probe_dir/'report.json')
    payload=torch.load(args.probe_dir/'probe.pt',map_location='cpu',weights_only=True)
    bases=[item[:,:,-2:].contiguous().half() for item in payload['bases']]
    scales=[item.contiguous().half() for item in payload['scales']]
    need(len(bases)==len(scales)==56 and
         all(tuple(bases[i].shape)==(8,latent.layout_descriptor(layouts[i])['nzero'],2)
             and tuple(scales[i].shape)==(8,2) and bool((scales[i]>0).all())
             for i in range(56)),
         'Frozen FP8 latent basis/scale geometry differs')
    table=tables['fp4_top4_parent']
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==binding.TOKENIZER_SHA,'Tokenizer differs')
    torch.set_num_threads(8)
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    policy=pin_replay_backend()
    old=read_json(args.mixed_outcome_dir/'top48.json')
    need(old['backend_policy']==policy,'Backend differs from mixed top48')
    if args.stage=='screen':
        train=load_train_tokens(args.prose_tokens)
        windows=[(row*2048,train[row].clone()) for row in range(168,200)]
        dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(168,200)),
                     windows=32,target_tokens=65504)
    else:
        need(args.latent_screen_dir is not None,'Audited positive latent screen required')
        screen=read_json(args.latent_screen_dir/'comparison.json')
        audit=read_json(args.latent_screen_dir/'audit.json')
        need(screen['complete'] is True and screen['stage']=='screen' and
             screen['advance_to_full'] is True and
             audit['passed'] is True and audit['cuda_initialized'] is False and
             audit['input_report_sha256']==sha(args.latent_screen_dir/'comparison.json') and
             screen['upstream']==upstream,
             'Audited positive latent TRAIN gate required')
        ids,dataset=evaluator.load_wikitext_tokens(tokenizer,'validation')
        windows=evaluator.ppl_windows(ids,2048)
        need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and
             dataset['token_stream_sha256_int64le']==binding.VALIDATION_TOKENS_SHA,
             'Full validation population differs')
        upstream['latent_screen_sha256']=sha(args.latent_screen_dir/'comparison.json')
    args.out_dir.mkdir(parents=True)
    started=time.time()
    model=load_fp4_model(args.source_dir,'fp4_g16_e4m3',device='cuda',chunk_rows=512,
        evidence_path=args.out_dir/'conversion_evidence.pt',
        progress=lambda row:print('[FP4 materialize] '+str(row),flush=True))
    expected={name:item['decoded_sha256'] for name,item in
              read_json(ARCHIVE/'conversion_receipt.json')['tensors'].items()}
    initial=fp4_quality.assert_weight_content(model,expected)
    frozen=FrozenBase(model)
    need(evaluator.no_adapter_hooks(model),'Adapter hooks at entry')
    hashes=code_hashes()
    common=dict(format='FP4_G16_LATENT_STATE_QUALITY_V1',stage=args.stage,
        source_checkpoint_sha256=binding.SOURCE_SHA,tokenizer_sha256=tokenizer.sha256,
        dataset=dataset,backend_policy=policy,code_sha256=hashes,
        table_sha256=tensor_sha(table),adapter_loaded=False,adapter_sha256=None,
        adapter_used=False,mk_used=False,upstream=upstream,source_files=sources)
    reports={}
    for name in ARMS:
        print(f'[FP4 latent {args.stage}] {name}',flush=True)
        path=args.out_dir/f'{name}.json'
        if name in ('parent','restored_parent'):
            reports[name]=mixed.evaluate_arm(model,name,table,layouts,windows,
                                              args.out_dir,common,frozen,policy)
        else:
            selected=[torch.zeros_like(value) for value in bases] if name=='latent_off' else bases
            reports[name]=evaluate_latent(model,name,table,layouts,selected,scales,
                                           windows,path,common,frozen,policy)
    restoration=evaluator.exact_replay(reports['parent'],reports['restored_parent'])
    parent_nll=reports['parent']['ppl']['nll']
    off_nll=reports['latent_off']['ppl']['nll']
    on_nll=reports['latent_on']['ppl']['nll']
    advance=args.stage=='screen' and on_nll<parent_nll and on_nll<off_nll
    if args.stage=='full':
        replay=evaluator.exact_replay(old,reports['parent'],archive=True)
        need(replay['complete'] is True,'Archived mixed top48 parent failed replay')
    final=fp4_quality.assert_weight_content(model,expected)
    comp=dict(format='FP4_G16_LATENT_STATE_QUALITY_COMPARISON_V1',complete=True,
        stage=args.stage,upstream=upstream,candidate_order=list(ARMS[:-1]),
        report_sha256={name:sha(args.out_dir/f'{name}.json') for name in ARMS},
        ppl={name:reports[name]['ppl']['ppl'] for name in ARMS},
        nll={name:reports[name]['ppl']['nll'] for name in ARMS},
        advance_to_full=advance,selection_rule='latent_on < parent and latent_off on complete TRAIN',
        cache_bytes=28499968,static_basis_bytes=sum(x.numel()*2 for x in bases),
        static_scale_bytes=sum(x.numel()*2 for x in scales),
        table_sha256=tensor_sha(table),layout_tuple_sha256=mixed.tuple_hash(layouts),
        backend_policy=policy,code_sha256=hashes,
        source_content_initial=initial,source_content_final=final,
        restoration=restoration,archived_parent_replay=replay if args.stage=='full' else None,
        adapter_used=False,mk_used=False,
        target_pass=args.stage=='full' and reports['latent_on']['ppl']['ppl']<binding.TARGET,
        elapsed_seconds=time.time()-started)
    save_json(args.out_dir/'comparison.json',comp)
    print(json.dumps({k:comp[k] for k in ('stage','ppl','advance_to_full','target_pass','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
