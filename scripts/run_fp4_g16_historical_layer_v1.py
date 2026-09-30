#!/usr/bin/env python3
"""TRAIN-select FP16-derived coordinate layers on frozen FP4 mixed SQ3.25."""
from __future__ import annotations

import argparse
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
from prepare_state_first_v5 import tensor_sha,check_table
from evaluate_quant_first import FrozenBase
from evaluate_resurface_more import pin_replay_backend,check_replay_backend
from run_statequant import save_json
import run_w4_state_repair_v2 as evaluator
import run_fp4_state_quality_v1 as fp4_quality
import run_fp4_g16_mixed_layout_v1 as mixed
import run_fp4_g16_fp16_table_transfer_v1 as transfer
import fp4_state_binding_v1 as binding

need,sha,read_json=binding.need,binding.sha,binding.read_json
PARENT='parent'
ALTS=('fp16_v8_top8','fp16_v9_top2')
TOP=(1,2,4,8,16,32,48)
RESTORED='restored_parent'
ARCHIVE=ROOT/'artifacts/fp4_weight_v1/fp4_g16_v1'


def code_hashes():
    names=('docs/FP4_G16_HISTORICAL_LAYER_V1_PROTOCOL.md',
           'scripts/run_fp4_g16_historical_layer_v1.py',
           'scripts/run_fp4_g16_mixed_layout_v1.py',
           'scripts/run_fp4_g16_fp16_table_transfer_v1.py',
           'scripts/run_fp4_g16_layout_v1.py',
           'scripts/run_fp4_state_quality_v1.py',
           'scripts/run_w4_state_repair_v2.py',
           'scripts/w4_state_repair_codec_v2.py','mamba2_recall/fp4.py')
    return {name:sha(ROOT/name) for name in names}


def frozen_inputs(args):
    tables,upstream,sources=transfer.load_tables(args)
    screen=read_json(args.transfer_screen_dir/'comparison.json')
    audit=read_json(args.transfer_screen_dir/'audit.json')
    need(screen['complete'] is True and screen['stage']=='screen' and
         screen['selection']['selected_id']=='fp4_top4_parent' and
         audit['passed'] is True and audit['cuda_initialized'] is False and
         audit['input_report_sha256']==sha(args.transfer_screen_dir/'comparison.json'),
         'Audited parent-winning whole-table transfer screen required')
    upstream['transfer_screen_sha256']=sha(args.transfer_screen_dir/'comparison.json')
    rank=read_json(args.mixed_rank_dir/'comparison.json')
    rank_audit=read_json(args.mixed_rank_dir/'audit.json')
    full=read_json(args.mixed_outcome_dir/'comparison.json')
    full_audit=read_json(args.mixed_outcome_dir/'audit.json')
    need(rank['complete'] is True and rank['stage']=='rank' and
         rank_audit['passed'] is True and rank_audit['cuda_initialized'] is False and
         rank_audit['input_report_sha256']==sha(args.mixed_rank_dir/'comparison.json') and
         full['complete'] is True and full['stage']=='full' and
         full['selection']['selected_id']=='top48' and full['target_pass'] is False and
         full['ppl']['top48']==8.512022460539045 and
         full_audit['passed'] is True and full_audit['cuda_initialized'] is False and
         full_audit['input_report_sha256']==sha(args.mixed_outcome_dir/'comparison.json'),
         'Audited failed mixed top48 reference required')
    policies,_=mixed.combos_from_rank(rank['ranked_improving_swaps'])
    layouts=policies['top48']
    need(mixed.tuple_hash(layouts)==full['policy_tuple_sha256']['top48'] and
         tensor_sha(tables['fp4_top4_parent'])==full['table_sha256'],
         'Frozen mixed layout/table binding differs')
    upstream['mixed_rank_sha256']=sha(args.mixed_rank_dir/'comparison.json')
    upstream['mixed_full_sha256']=sha(args.mixed_outcome_dir/'comparison.json')
    return tables,layouts,upstream,sources


def layer_table(tables,layer=None,kind=None):
    table=tables['fp4_top4_parent'].clone().contiguous()
    if layer is not None:
        need(0<=layer<56 and kind in ALTS,'Invalid historical layer swap')
        table[layer].copy_(tables[kind][layer])
    check_table(table)
    return table


def eligible_arms(tables):
    seen={tensor_sha(layer_table(tables))}
    out=[]
    for layer in range(56):
        for kind in ALTS:
            table=layer_table(tables,layer,kind)
            digest=tensor_sha(table)
            if digest in seen:continue
            seen.add(digest)
            out.append(dict(layer=layer,kind=kind,arm=f'layer{layer:02d}_{kind}',
                            table_sha256=digest))
    return out


def combos_from_rank(tables,ranked):
    need(len({row['layer'] for row in ranked})==len(ranked)<=56 and
         all(row['kind'] in ALTS and row['delta_nll']<0 for row in ranked),
         'Invalid ranked historical swaps')
    requests=[(PARENT,0)]+[(f'top{k}',min(k,len(ranked))) for k in TOP]
    requests.append(('allnegative',len(ranked)))
    out={};specs={};seen=set()
    for name,count in requests:
        table=layer_table(tables)
        for row in ranked[:count]:table[row['layer']].copy_(tables[row['kind']][row['layer']])
        digest=tensor_sha(table)
        if digest in seen:continue
        seen.add(digest);out[name]=table
        specs[name]=dict(swap_count=count,swaps=ranked[:count],table_sha256=digest)
    return out,specs


def rank_layers(rows,eligible):
    parent=rows[PARENT];need(evaluator.valid_candidate(parent),'Valid rank parent required')
    nll0=parent['ppl']['nll'];ranked=[];details=[]
    for layer in range(56):
        options=[]
        for proposal in eligible:
            if proposal['layer']!=layer:continue
            kind=proposal['kind'];name=proposal['arm']
            item=rows[name];valid=evaluator.valid_candidate(item)
            options.append(dict(name=name,kind=kind,valid=valid,
                nll=item['ppl']['nll'] if valid else None,
                delta_nll=item['ppl']['nll']-nll0 if valid else None))
        valid=[row for row in options if row['valid']]
        best=min(valid,key=lambda row:(row['nll'],ALTS.index(row['kind']))) if valid else None
        details.append(dict(layer=layer,alternatives=options,best=best))
        if best and best['delta_nll']<0:
            ranked.append(dict(layer=layer,kind=best['kind'],arm=best['name'],
                               delta_nll=best['delta_nll']))
    ranked.sort(key=lambda row:(row['delta_nll'],row['layer'],ALTS.index(row['kind'])))
    return ranked,details


def evaluate_arm(model,name,table,layouts,windows,out,common,frozen,policy):
    spec=dict(id=name,table_name=name,table=table,layouts=layouts)
    path=out/f'{name}.json'
    try:
        row=evaluator.evaluate(model,spec,windows,path,dict(common,arm=name,
            candidate_layout_tuple_sha256=mixed.tuple_hash(layouts)))
        row.update(frozen_source=frozen.check(),backend_policy_check=check_replay_backend(policy),
                   candidate_table_unchanged=tensor_sha(table)==common['table_sha256'][name],
                   adapter_hooks_absent=evaluator.no_adapter_hooks(model))
        need(evaluator.valid_candidate(row),'Candidate failed finite/same-byte guards')
    except evaluator.CandidateInvalid as error:
        row=read_json(path)
        row.update(complete=False,error=str(error),error_type=type(error).__name__,
                   failure_kind='nonfinite_quality',invalid_candidate=True)
    save_json(path,row)
    return row


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
    ap.add_argument('--mixed-rank-dir',type=Path,required=True)
    ap.add_argument('--mixed-outcome-dir',type=Path,required=True)
    ap.add_argument('--transfer-screen-dir',type=Path,required=True)
    ap.add_argument('--historical-rank-dir',type=Path)
    ap.add_argument('--historical-screen-dir',type=Path)
    args=ap.parse_args()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(),
         'Fresh output directory required')
    tables,layouts,upstream,sources=frozen_inputs(args)
    eligible=eligible_arms(tables)
    need(len(eligible)==18 and all(row['kind']=='fp16_v8_top8' for row in eligible),
         'Frozen distinct historical layer inventory differs')
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==binding.TOKENIZER_SHA,'Tokenizer differs')
    train=load_train_tokens(args.prose_tokens) if args.stage!='full' else None
    torch.set_num_threads(8)
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    policy=pin_replay_backend()
    old=read_json(args.mixed_outcome_dir/'top48.json')
    need(old['backend_policy']==policy,'Backend differs from frozen mixed top48')
    hashes=code_hashes()
    if args.stage=='rank':
        windows=[(row*2048,train[row].clone()) for row in range(128,136)]
        dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(128,136)),
                     windows=8,target_tokens=16376)
        candidates={PARENT:layer_table(tables)}
        candidates.update({row['arm']:layer_table(tables,row['layer'],row['kind'])
                           for row in eligible})
        candidates[RESTORED]=layer_table(tables)
    else:
        need(args.historical_rank_dir is not None,'Audited historical rank required')
        rank=read_json(args.historical_rank_dir/'comparison.json')
        audit=read_json(args.historical_rank_dir/'audit.json')
        need(rank['complete'] is True and rank['stage']=='rank' and
             rank['upstream']==upstream and audit['passed'] is True and
             audit['cuda_initialized'] is False and
             audit['input_report_sha256']==sha(args.historical_rank_dir/'comparison.json'),
             'Audited historical rank differs')
        candidates,_=combos_from_rank(tables,rank['ranked_improving_swaps'])
        need({name:tensor_sha(value) for name,value in candidates.items()}==rank['combo_table_sha256'],
             'Frozen historical combo tables differ')
        upstream['historical_rank_sha256']=sha(args.historical_rank_dir/'comparison.json')
        if args.stage=='screen':
            windows=[(row*2048,train[row].clone()) for row in range(136,168)]
            dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(136,168)),
                         windows=32,target_tokens=65504)
            candidates[RESTORED]=layer_table(tables)
        else:
            need(args.historical_screen_dir is not None,'Audited historical screen required')
            screen=read_json(args.historical_screen_dir/'comparison.json')
            audit=read_json(args.historical_screen_dir/'audit.json')
            selected=screen['selection']['selected_id']
            need(screen['complete'] is True and screen['stage']=='screen' and
                 selected in candidates and selected!=PARENT and
                 screen['upstream']==upstream and audit['passed'] is True and
                 audit['cuda_initialized'] is False and
                 audit['input_report_sha256']==sha(args.historical_screen_dir/'comparison.json'),
                 'Audited nonparent historical TRAIN winner required')
            ids,dataset=evaluator.load_wikitext_tokens(tokenizer,'validation')
            windows=evaluator.ppl_windows(ids,2048)
            need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and
                 dataset['token_stream_sha256_int64le']==binding.VALIDATION_TOKENS_SHA,
                 'Validation population differs')
            candidates={PARENT:layer_table(tables),selected:candidates[selected],
                        RESTORED:layer_table(tables)}
            upstream['historical_screen_sha256']=sha(args.historical_screen_dir/'comparison.json')
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
    common=dict(format='FP4_G16_UNADAPTED_HISTORICAL_LAYER_V1',stage=args.stage,
        source_checkpoint_sha256=binding.SOURCE_SHA,tokenizer_sha256=tokenizer.sha256,
        dataset=dataset,backend_policy=policy,code_sha256=hashes,
        table_sha256={name:tensor_sha(value) for name,value in candidates.items()},
        adapter_loaded=False,adapter_sha256=None,adapter_used=False,mk_used=False,
        upstream=upstream,source_files=sources)
    reports={}
    for index,(name,table) in enumerate(candidates.items()):
        print(f'[FP4 historical {args.stage}] {index+1}/{len(candidates)} {name}',flush=True)
        reports[name]=evaluate_arm(model,name,table,layouts,windows,args.out_dir,common,frozen,policy)
    restoration=evaluator.exact_replay(reports[PARENT],reports[RESTORED])
    if args.stage=='rank':
        ranked,details=rank_layers(reports,eligible)
        combos,specs=combos_from_rank(tables,ranked)
        selected=PARENT
        valid=[name for name in list(candidates)[:-1] if evaluator.valid_candidate(reports[name])]
    elif args.stage=='screen':
        order=list(candidates)[:-1]
        valid=[name for name in order if evaluator.valid_candidate(reports[name])]
        need(valid and valid[0]==PARENT,'Finite screen parent required')
        selected=min(valid,key=lambda name:(reports[name]['ppl']['nll'],order.index(name)))
    else:
        valid=[PARENT,selected]
        replay=evaluator.exact_replay(old,reports[PARENT],archive=True)
        need(replay['complete'] is True,'Archived mixed top48 parent failed replay')
    final=fp4_quality.assert_weight_content(model,expected)
    comp=dict(format='FP4_G16_UNADAPTED_HISTORICAL_LAYER_V1_COMPARISON',complete=True,
        stage=args.stage,upstream=upstream,source_files=sources,
        eligible_layer_arms=eligible,
        candidate_order=list(candidates)[:-1],
        table_sha256=common['table_sha256'],
        layout_tuple_sha256=mixed.tuple_hash(layouts),
        report_sha256={name:sha(args.out_dir/f'{name}.json') for name in candidates},
        selection=dict(selected_id=selected,valid=valid,
          rule='Minimum finite TRAIN NLL, parent then frozen order' if args.stage=='screen'
          else 'Frozen TRAIN winner' if args.stage=='full'
          else 'Single-layer delta rank only'),
        ppl={name:reports[name]['ppl'].get('ppl') for name in candidates},
        cache_bytes=binding.CACHE_BYTES,backend_policy=policy,code_sha256=hashes,
        source_content_initial=initial,source_content_final=final,
        restoration=restoration,
        archived_parent_replay=replay if args.stage=='full' else None,
        adapter_used=False,mk_used=False,
        target_pass=args.stage=='full' and reports[selected]['ppl']['ppl']<binding.TARGET,
        elapsed_seconds=time.time()-started)
    if args.stage=='rank':
        comp.update(ranked_improving_swaps=ranked,layer_details=details,
                    combo_specs=specs,
                    combo_table_sha256={name:tensor_sha(value) for name,value in combos.items()})
    save_json(args.out_dir/'comparison.json',comp)
    print(json.dumps({k:comp[k] for k in ('stage','selection','target_pass','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
