#!/usr/bin/env python3
"""FP4 G16-only unadapted per-layer Q3.25 table ranking and confirmation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba2_recall import runtime
from mamba2_recall.fp4 import load_fp4_model
from prepare_quant_first import load_train_tokens, TRAIN_SHA
from prepare_state_first_v5 import check_table, tensor_sha, write_payload
from evaluate_quant_first import FrozenBase
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from run_statequant import save_json
import run_w4_state_repair_v2 as evaluator
import fp4_state_binding_v1 as binding
import run_fp4_state_quality_v1 as fp4_quality

need, sha, read_json = binding.need, binding.sha, binding.read_json
ALTERNATIVES = ('fp4_magnitude','fp4_readout','fp4_preserve32','fp4_preserve64')
TOP_K = (1,2,4,8,16,32)
ARCHIVE = ROOT / 'artifacts/fp4_weight_v1/fp4_g16_v1'


def code_hashes():
    names = ('docs/FP4_G16_STATE_LAYER_V1_PROTOCOL.md',
             'scripts/run_fp4_g16_state_layer_v1.py',
             'scripts/run_w4_state_repair_v2.py',
             'scripts/w4_state_repair_codec_v2.py',
             'scripts/run_fp4_state_quality_v1.py',
             'mamba2_recall/fp4.py')
    return {name: sha(ROOT/name) for name in names}


def screen_tables(screen_dir):
    comparison = read_json(screen_dir/'comparison.json')
    audit = read_json(screen_dir/'audit.json')
    need(comparison['complete'] is True and comparison['stage'] == 'screen' and
         comparison['selection']['selected_id'] == 'parent',
         'Audited five-table parent win required')
    need(audit.get('complete') is True and audit.get('passed') is True and
         audit.get('cuda_initialized') is False and audit.get('stage') == 'screen' and
         audit.get('selected_id') == 'parent' and
         audit.get('input_report_sha256') == sha(screen_dir/'comparison.json'),
         'Independent CPU audit of first-stage screen required')
    path = screen_dir/'raw_stats.pt'
    need(sha(path) == comparison['raw_stats']['sha256'], 'Frozen FP4 stats changed')
    tables = torch.load(path,map_location='cpu',weights_only=True)['tables']
    need(tuple(tables) == ('parent',*ALTERNATIVES) and
         all(tensor_sha(table) == comparison['table_sha256'][name] for name,table in tables.items()) and
         tensor_sha(tables['parent']) == binding.TABLE_SHA, 'FP4 tables differ')
    return tables, comparison


def arm_table(tables, layer=None, kind=None):
    table = tables['parent'].clone().contiguous()
    if layer is not None:
        need(0 <= layer < 56 and kind in ALTERNATIVES, 'Invalid isolated layer substitution')
        table[layer].copy_(tables[kind][layer])
    check_table(table)
    return table


def make_combos(tables, ranked):
    requests = [('parent',0)] + [(f'top{k}',min(k,len(ranked))) for k in TOP_K]
    requests += [('allnegative',len(ranked))]
    out={}; specs={}; seen={}
    for name,count in requests:
        table=arm_table(tables)
        for row in ranked[:count]:
            table[row['layer']].copy_(tables[row['kind']][row['layer']])
        digest=tensor_sha(table)
        if digest in seen:
            continue
        seen[digest]=name; out[name]=table
        specs[name]=dict(swap_count=count, swaps=ranked[:count], table_sha256=digest)
    return out,specs


def evaluate_arm(model,name,table,windows,out,common,frozen,policy):
    spec=dict(id=name,table_name=name,table=table,layouts=binding.LAYOUTS)
    path=out/f'{name}.json'
    try:
        row=evaluator.evaluate(model,spec,windows,path,dict(common,arm=name))
        row.update(frozen_source=frozen.check(),backend_policy_check=check_replay_backend(policy),
                   candidate_table_unchanged=tensor_sha(table)==common['table_sha256'][name],
                   adapter_hooks_absent=evaluator.no_adapter_hooks(model))
        need(evaluator.valid_candidate(row), 'Completed arm failed finite/same-byte gates')
    except evaluator.CandidateInvalid as error:
        row=read_json(path)
        row.update(complete=False,error=str(error),error_type=type(error).__name__,
                   failure_kind='nonfinite_quality',invalid_candidate=True)
    save_json(path,row)
    return row


def rank_layers(rows):
    parent=rows['parent']; need(evaluator.valid_candidate(parent),'Valid rank parent required')
    nll0=parent['ppl']['nll']; ranked=[]; details=[]
    for layer in range(56):
        options=[]
        for kind in ALTERNATIVES:
            name=f'layer{layer:02d}_{kind}'
            item=rows[name]; valid=evaluator.valid_candidate(item)
            options.append(dict(name=name,kind=kind,valid=valid,
                nll=item['ppl']['nll'] if valid else None,
                delta_nll=item['ppl']['nll']-nll0 if valid else None))
        valid=[row for row in options if row['valid']]
        best=min(valid,key=lambda row:(row['nll'],ALTERNATIVES.index(row['kind']))) if valid else None
        details.append(dict(layer=layer,alternatives=options,best=best))
        if best and best['delta_nll'] < 0:
            ranked.append(dict(layer=layer,kind=best['kind'],arm=best['name'],
                               delta_nll=best['delta_nll']))
    ranked.sort(key=lambda row:(row['delta_nll'],row['layer'],ALTERNATIVES.index(row['kind'])))
    return ranked,details


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('rank','screen','full'),required=True)
    ap.add_argument('--source-dir',type=Path,required=True)
    ap.add_argument('--prose-tokens',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--five-table-screen',type=Path,required=True)
    ap.add_argument('--rank-dir',type=Path)
    ap.add_argument('--combo-screen-dir',type=Path)
    args=ap.parse_args()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(),'Fresh output directory required')
    tables,first=screen_tables(args.five_table_screen)
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==binding.TOKENIZER_SHA,'Tokenizer differs')
    train=load_train_tokens(args.prose_tokens) if args.stage!='full' else None
    torch.set_num_threads(8)
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    policy=pin_replay_backend()
    archive=read_json(ARCHIVE/'full_sq325.json')
    need(archive['backend_policy']==policy,'Backend differs from archived FP4')
    hashes=code_hashes()
    args.out_dir.mkdir(parents=True)
    started=time.time()
    model=load_fp4_model(args.source_dir,'fp4_g16_e4m3',device='cuda',chunk_rows=512,
        evidence_path=args.out_dir/'conversion_evidence.pt',
        progress=lambda row: print('[FP4 materialize] '+str(row),flush=True))
    expected={name:item['decoded_sha256'] for name,item in
              read_json(ARCHIVE/'conversion_receipt.json')['tensors'].items()}
    initial=fp4_quality.assert_weight_content(model,expected)
    frozen=FrozenBase(model)
    need(evaluator.no_adapter_hooks(model),'Adapter hooks at entry')
    common=dict(format='FP4_G16_UNADAPTED_LAYER_V1',stage=args.stage,
        source_checkpoint_sha256=binding.SOURCE_SHA,tokenizer_sha256=tokenizer.sha256,
        five_table_screen_sha256=sha(args.five_table_screen/'comparison.json'),
        backend_policy=policy,code_sha256=hashes,adapter_loaded=False,
        adapter_sha256=None,adapter_used=False,mk_used=False)
    if args.stage=='rank':
        windows=[(row*2048,train[row].clone()) for row in range(440,448)]
        arms=['parent']+[f'layer{layer:02d}_{kind}' for layer in range(56)
                         for kind in ALTERNATIVES]+['restored_parent']
        arm_tables={'parent':tables['parent'],'restored_parent':tables['parent']}
        arm_tables.update({f'layer{layer:02d}_{kind}':arm_table(tables,layer,kind)
                           for layer in range(56) for kind in ALTERNATIVES})
        common['dataset']=dict(split='train',file_sha256=TRAIN_SHA,
                               rows=list(range(440,448)),windows=8,target_tokens=16376)
    else:
        need(args.rank_dir is not None,'Frozen rank required')
        ranking=read_json(args.rank_dir/'comparison.json')
        need(ranking['complete'] is True and ranking['stage']=='rank' and
             ranking['five_table_screen_sha256']==common['five_table_screen_sha256'],
             'Rank binding differs')
        ranked=ranking['ranked_improving_swaps']
        arm_tables,specs=make_combos(tables,ranked)
        need({name:tensor_sha(table) for name,table in arm_tables.items()}==ranking['combo_table_sha256'],
             'Frozen combination tables differ')
        if args.stage=='screen':
            windows=[(row*2048,train[row].clone()) for row in range(408,440)]
            common['dataset']=dict(split='train',file_sha256=TRAIN_SHA,
                                   rows=list(range(408,440)),windows=32,target_tokens=65504)
            arms=[*arm_tables,'restored_parent']
        else:
            need(args.combo_screen_dir is not None,'Frozen combination screen required')
            screen=read_json(args.combo_screen_dir/'comparison.json')
            selected=screen['selection']['selected_id']
            need(screen['complete'] is True and screen['stage']=='screen' and
                 selected!='parent' and selected in arm_tables and
                 screen['rank_comparison_sha256']==sha(args.rank_dir/'comparison.json'),
                 'Nonparent frozen TRAIN combination winner required')
            ids,dataset=evaluator.load_wikitext_tokens(tokenizer,'validation')
            windows=evaluator.ppl_windows(ids,2048)
            need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and
                 dataset['token_stream_sha256_int64le']==binding.VALIDATION_TOKENS_SHA,
                 'Full validation population differs')
            common['dataset']=dataset
            common['combo_screen_sha256']=sha(args.combo_screen_dir/'comparison.json')
            arms=['parent',selected,'restored_parent']
        arm_tables['restored_parent']=tables['parent']
        common['rank_comparison_sha256']=sha(args.rank_dir/'comparison.json')
    common['table_sha256']={name:tensor_sha(arm_tables[name]) for name in arms}
    reports={}
    for index,name in enumerate(arms):
        print(f'[FP4 layer {args.stage}] {index+1}/{len(arms)} {name}',flush=True)
        reports[name]=evaluate_arm(model,name,arm_tables[name],windows,args.out_dir,common,frozen,policy)
    restoration=evaluator.exact_replay(reports['parent'],reports['restored_parent'])
    if args.stage=='rank':
        ranked,details=rank_layers(reports)
        combos,specs=make_combos(tables,ranked)
        selected='parent';valid=[name for name in arms[:-1] if evaluator.valid_candidate(reports[name])]
    elif args.stage=='screen':
        valid=[name for name in arms[:-1] if evaluator.valid_candidate(reports[name])]
        need(valid and valid[0]=='parent','Finite parent required')
        selected=min(valid,key=lambda name:(reports[name]['ppl']['nll'],arms.index(name)))
    else:
        valid=['parent',selected]
        replay=evaluator.exact_replay(archive,reports['parent'],archive=True)
        need(replay['complete'] is True,'Archived FP4 baseline did not replay')
    final=fp4_quality.assert_weight_content(model,expected)
    comp=dict(format='FP4_G16_UNADAPTED_LAYER_V1_COMPARISON',complete=True,stage=args.stage,
        five_table_screen_sha256=common['five_table_screen_sha256'],
        rank_comparison_sha256=common.get('rank_comparison_sha256'),
        combo_screen_sha256=common.get('combo_screen_sha256'),
        source_content_initial=initial,source_content_final=final,
        backend_policy=policy,code_sha256=hashes,cache_bytes=binding.CACHE_BYTES,
        candidate_order=arms[:-1],table_sha256=common['table_sha256'],
        report_sha256={name:sha(args.out_dir/f'{name}.json') for name in arms},
        restoration=restoration,selection=dict(selected_id=selected,valid=valid,
            rule='Minimum finite TRAIN NLL, parent then frozen order' if args.stage=='screen'
            else 'Frozen TRAIN winner' if args.stage=='full' else 'Single-layer delta rank only'),
        ppl={name:reports[name]['ppl'].get('ppl') for name in arms},
        adapter_used=False,mk_used=False,
        target_pass=args.stage=='full' and reports[selected]['ppl']['ppl']<binding.TARGET,
        archived_parent_replay=replay if args.stage=='full' else None,
        elapsed_seconds=time.time()-started)
    if args.stage=='rank':
        comp.update(ranked_improving_swaps=ranked,layer_details=details,
                    combo_specs=specs,combo_table_sha256={name:tensor_sha(value)
                    for name,value in combos.items()})
        write_payload(args.out_dir/'combos.pt',dict(format='FP4_G16_LAYER_COMBOS_V1',
            tables=combos,ranked=ranked,code_sha256=hashes))
        comp['combos_sha256']=sha(args.out_dir/'combos.pt')
    save_json(args.out_dir/'comparison.json',comp)
    print(json.dumps({k:comp[k] for k in ('stage','selection','target_pass','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':
    main()
