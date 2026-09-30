#!/usr/bin/env python3
"""Train-select same-byte mixed state layouts for unadapted FP4 G16/top4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mamba2_recall import runtime
from mamba2_recall.fp4 import load_fp4_model
from prepare_quant_first import load_train_tokens,TRAIN_SHA
from prepare_state_first_v5 import tensor_sha
from evaluate_quant_first import FrozenBase
from evaluate_resurface_more import pin_replay_backend,check_replay_backend
from run_statequant import save_json
import run_w4_state_repair_v2 as evaluator
import run_fp4_state_quality_v1 as fp4_quality
import run_fp4_g16_layout_v1 as prior
import fp4_state_binding_v1 as binding

need,sha,read_json=binding.need,binding.sha,binding.read_json
PARENT='32_32_64'
ALTS=prior.ORDER[1:]
TOP=(1,2,4,8,16,32,48)
RESTORED='restored_parent'
ARCHIVE=ROOT/'artifacts/fp4_weight_v1/fp4_g16_v1'


def code_hashes():
    names=('docs/FP4_G16_MIXED_LAYOUT_V1_PROTOCOL.md',
           'scripts/run_fp4_g16_mixed_layout_v1.py',
           'scripts/run_fp4_g16_layout_v1.py','scripts/run_w4_state_repair_v2.py',
           'scripts/w4_state_repair_codec_v2.py','mamba2_recall/fp4.py')
    return {name:sha(ROOT/name) for name in names}


def tuple_hash(layouts):
    need(len(layouts)==56 and all(item in prior.ORDER for item in layouts),
         'Invalid 56-layer fixed-byte policy')
    return hashlib.sha256(json.dumps(list(layouts),separators=(',',':')).encode()).hexdigest()


def top4_table(args):
    table,upstream=prior.frozen_top4(args.five_table_screen,args.rank_dir,
                                    args.combo_screen_dir,args.top4_full_dir)
    outcome=read_json(args.global_layout_full_dir/'comparison.json')
    audit=read_json(args.global_layout_full_dir/'audit.json')
    need(outcome['complete'] is True and outcome['stage']=='full' and
         outcome['target_pass'] is False and
         outcome['ppl']['24_48_56']>outcome['ppl']['32_32_64'] and
         audit['passed'] is True and audit['cuda_initialized'] is False and
         audit['input_report_sha256']==sha(args.global_layout_full_dir/'comparison.json'),
         'Audited failed global-layout continuation required')
    upstream['global_layout_full_sha256']=sha(args.global_layout_full_dir/'comparison.json')
    return table,upstream


def isolated(layer=None,layout=None):
    result=[PARENT]*56
    if layer is not None:
        need(0<=layer<56 and layout in ALTS,'Invalid isolated layout arm')
        result[layer]=layout
    return tuple(result)


def combos_from_rank(ranked):
    need(len({row['layer'] for row in ranked})==len(ranked)<=56 and
         all(row['layout'] in ALTS and row['delta_nll']<0 for row in ranked),
         'Invalid ranked layout swaps')
    requests=[('parent',0)]+[(f'top{k}',min(k,len(ranked))) for k in TOP]
    requests.append(('allnegative',len(ranked)))
    out={};seen=set();specs={}
    for name,count in requests:
        value=[PARENT]*56
        for row in ranked[:count]:value[row['layer']]=row['layout']
        key=tuple(value)
        if key in seen:continue
        seen.add(key);out[name]=key
        specs[name]=dict(swap_count=count,swaps=ranked[:count],tuple_sha256=tuple_hash(key))
    return out,specs


def evaluate_arm(model,name,table,layouts,windows,out,common,frozen,policy):
    spec=dict(id=name,table_name='frozen_fp4_top4',table=table,layouts=layouts)
    path=out/f'{name}.json'
    try:
        row=evaluator.evaluate(model,spec,windows,path,dict(common,arm=name,
            candidate_layout_tuple_sha256=tuple_hash(layouts)))
        row.update(frozen_source=frozen.check(),backend_policy_check=check_replay_backend(policy),
                   candidate_table_unchanged=tensor_sha(table)==common['table_sha256'],
                   adapter_hooks_absent=evaluator.no_adapter_hooks(model))
        need(evaluator.valid_candidate(row),'Candidate failed finite/same-byte guards')
    except evaluator.CandidateInvalid as error:
        row=read_json(path)
        row.update(complete=False,error=str(error),error_type=type(error).__name__,
                   failure_kind='nonfinite_quality',invalid_candidate=True)
    save_json(path,row)
    return row


def rank_layers(rows):
    parent=rows['parent'];need(evaluator.valid_candidate(parent),'Valid rank parent required')
    nll0=parent['ppl']['nll'];ranked=[];details=[]
    for layer in range(56):
        options=[]
        for layout in ALTS:
            name=f'layer{layer:02d}_{layout}'
            item=rows[name];valid=evaluator.valid_candidate(item)
            options.append(dict(name=name,layout=layout,valid=valid,
                nll=item['ppl']['nll'] if valid else None,
                delta_nll=item['ppl']['nll']-nll0 if valid else None))
        valid=[row for row in options if row['valid']]
        best=min(valid,key=lambda row:(row['nll'],ALTS.index(row['layout']))) if valid else None
        details.append(dict(layer=layer,alternatives=options,best=best))
        if best and best['delta_nll']<0:
            ranked.append(dict(layer=layer,layout=best['layout'],arm=best['name'],
                               delta_nll=best['delta_nll']))
    ranked.sort(key=lambda row:(row['delta_nll'],row['layer'],ALTS.index(row['layout'])))
    return ranked,details


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('rank','screen','full'),required=True)
    ap.add_argument('--source-dir',type=Path,required=True)
    ap.add_argument('--prose-tokens',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--five-table-screen',type=Path,required=True)
    ap.add_argument('--rank-dir',type=Path,required=True)
    ap.add_argument('--combo-screen-dir',type=Path,required=True)
    ap.add_argument('--top4-full-dir',type=Path,required=True)
    ap.add_argument('--global-layout-full-dir',type=Path,required=True)
    ap.add_argument('--mixed-rank-dir',type=Path)
    ap.add_argument('--mixed-screen-dir',type=Path)
    args=ap.parse_args()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(),
         'Fresh output directory required')
    table,upstream=top4_table(args)
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==binding.TOKENIZER_SHA,'Tokenizer differs')
    train=load_train_tokens(args.prose_tokens) if args.stage!='full' else None
    torch.set_num_threads(8)
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    policy=pin_replay_backend()
    old=read_json(args.top4_full_dir/'top4.json')
    need(old['backend_policy']==policy,'Backend differs from top4 reference')
    hashes=code_hashes()
    if args.stage=='rank':
        windows=[(row*2048,train[row].clone()) for row in range(352,360)]
        dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(352,360)),
                     windows=8,target_tokens=16376)
        policies={'parent':isolated()}
        for layer in range(56):
            for layout in ALTS:
                policies[f'layer{layer:02d}_{layout}']=isolated(layer,layout)
        policies[RESTORED]=isolated()
    else:
        need(args.mixed_rank_dir is not None,'Frozen mixed-layout rank required')
        rank=read_json(args.mixed_rank_dir/'comparison.json')
        audit=read_json(args.mixed_rank_dir/'audit.json')
        need(rank['complete'] is True and rank['stage']=='rank' and
             rank['upstream']==upstream and audit['passed'] is True and
             audit['cuda_initialized'] is False and
             audit['input_report_sha256']==sha(args.mixed_rank_dir/'comparison.json'),
             'Audited mixed-layout ranking differs')
        policies,_=combos_from_rank(rank['ranked_improving_swaps'])
        need({name:tuple_hash(value) for name,value in policies.items()}==rank['combo_tuple_sha256'],
             'Frozen layout tuples differ')
        upstream['mixed_rank_sha256']=sha(args.mixed_rank_dir/'comparison.json')
        if args.stage=='screen':
            windows=[(row*2048,train[row].clone()) for row in range(360,376)]
            dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(360,376)),
                         windows=16,target_tokens=32752)
            policies[RESTORED]=isolated()
        else:
            need(args.mixed_screen_dir is not None,'Frozen mixed screen required')
            screen=read_json(args.mixed_screen_dir/'comparison.json')
            screen_audit=read_json(args.mixed_screen_dir/'audit.json')
            selected=screen['selection']['selected_id']
            need(screen['complete'] is True and screen['stage']=='screen' and
                 selected!='parent' and selected in policies and
                 screen_audit['passed'] is True and screen_audit['cuda_initialized'] is False and
                 screen_audit['input_report_sha256']==sha(args.mixed_screen_dir/'comparison.json') and
                 screen['upstream']==upstream,
                 'Audited nonparent mixed-layout TRAIN winner required')
            ids,dataset=evaluator.load_wikitext_tokens(tokenizer,'validation')
            windows=evaluator.ppl_windows(ids,2048)
            need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and
                 dataset['token_stream_sha256_int64le']==binding.VALIDATION_TOKENS_SHA,
                 'Validation population differs')
            policies={'parent':isolated(),selected:policies[selected],RESTORED:isolated()}
            upstream['mixed_screen_sha256']=sha(args.mixed_screen_dir/'comparison.json')
    args.out_dir.mkdir(parents=True)
    started=time.time()
    model=load_fp4_model(args.source_dir,'fp4_g16_e4m3',device='cuda',chunk_rows=512,
        evidence_path=args.out_dir/'conversion_evidence.pt',
        progress=lambda row:print('[FP4 materialize] '+str(row),flush=True))
    expected={name:item['decoded_sha256'] for name,item in
              read_json(ARCHIVE/'conversion_receipt.json')['tensors'].items()}
    initial=fp4_quality.assert_weight_content(model,expected)
    frozen=FrozenBase(model)
    need(evaluator.no_adapter_hooks(model),'Unexpected adapter hooks at entry')
    common=dict(format='FP4_G16_UNADAPTED_MIXED_LAYOUT_V1',stage=args.stage,
        source_checkpoint_sha256=binding.SOURCE_SHA,tokenizer_sha256=tokenizer.sha256,
        dataset=dataset,backend_policy=policy,code_sha256=hashes,
        table_sha256=tensor_sha(table),adapter_loaded=False,adapter_sha256=None,
        adapter_used=False,mk_used=False,upstream=upstream)
    results={}
    for index,(name,layouts) in enumerate(policies.items()):
        print(f'[FP4 mixed {args.stage}] {index+1}/{len(policies)} {name}',flush=True)
        results[name]=evaluate_arm(model,name,table,layouts,windows,args.out_dir,common,frozen,policy)
    restoration=evaluator.exact_replay(results['parent'],results[RESTORED])
    if args.stage=='rank':
        ranked,details=rank_layers(results)
        combos,specs=combos_from_rank(ranked)
        selected='parent'
        valid=[name for name in list(policies)[:-1] if evaluator.valid_candidate(results[name])]
    elif args.stage=='screen':
        order=list(policies)[:-1]
        valid=[name for name in order if evaluator.valid_candidate(results[name])]
        need(valid and valid[0]=='parent','Finite parent required')
        selected=min(valid,key=lambda name:(results[name]['ppl']['nll'],order.index(name)))
    else:
        valid=['parent',selected]
        replay=evaluator.exact_replay(old,results['parent'],archive=True)
        need(replay['complete'] is True,'Archived top4 parent did not replay')
    final=fp4_quality.assert_weight_content(model,expected)
    comp=dict(format='FP4_G16_UNADAPTED_MIXED_LAYOUT_V1_COMPARISON',complete=True,
        stage=args.stage,upstream=upstream,candidate_order=list(policies)[:-1],
        policy_tuple_sha256={name:tuple_hash(value) for name,value in policies.items()},
        selection=dict(selected_id=selected,valid=valid,
          rule='Minimum finite TRAIN NLL, parent then frozen order' if args.stage=='screen'
          else 'Frozen TRAIN-selected mixed layout' if args.stage=='full'
          else 'Single-layer delta rank only'),
        report_sha256={name:sha(args.out_dir/f'{name}.json') for name in policies},
        ppl={name:results[name]['ppl'].get('ppl') for name in policies},
        cache_bytes=binding.CACHE_BYTES,table_sha256=tensor_sha(table),
        backend_policy=policy,code_sha256=hashes,
        source_content_initial=initial,source_content_final=final,
        restoration=restoration,
        archived_parent_replay=replay if args.stage=='full' else None,
        adapter_used=False,mk_used=False,
        target_pass=args.stage=='full' and results[selected]['ppl']['ppl']<binding.TARGET,
        elapsed_seconds=time.time()-started)
    if args.stage=='rank':
        comp.update(ranked_improving_swaps=ranked,layer_details=details,
                    combo_specs=specs,
                    combo_tuple_sha256={name:tuple_hash(value) for name,value in combos.items()})
    save_json(args.out_dir/'comparison.json',comp)
    print(json.dumps({k:comp[k] for k in ('stage','selection','target_pass','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
