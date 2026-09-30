#!/usr/bin/env python3
"""Screen seven equal-byte state-tier layouts on frozen FP4 G16/top4."""
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
import run_fp4_g16_state_layer_v1 as layer
import fp4_state_binding_v1 as binding

need,sha,read_json=binding.need,binding.sha,binding.read_json
ORDER=('32_32_64','16_64_48','8_80_40','24_48_56',
       '36_24_68','40_16_72','44_8_76')
RESTORED='restored_parent'
ARCHIVE=ROOT/'artifacts/fp4_weight_v1/fp4_g16_v1'


def code_hashes():
    names=('docs/FP4_G16_LAYOUT_V1_PROTOCOL.md','scripts/run_fp4_g16_layout_v1.py',
           'scripts/run_fp4_g16_state_layer_v1.py','scripts/run_w4_state_repair_v2.py',
           'scripts/w4_state_repair_codec_v2.py','mamba2_recall/fp4.py')
    return {name:sha(ROOT/name) for name in names}


def frozen_top4(five_dir,rank_dir,combo_dir,full_dir):
    tables,first=layer.screen_tables(five_dir)
    rank=read_json(rank_dir/'comparison.json')
    rank_audit=read_json(rank_dir/'audit.json')
    combo=read_json(combo_dir/'comparison.json')
    combo_audit=read_json(combo_dir/'audit.json')
    full=read_json(full_dir/'comparison.json')
    full_audit=read_json(full_dir/'audit.json')
    for report,audit,path,stage in ((rank,rank_audit,rank_dir,'rank'),
                                    (combo,combo_audit,combo_dir,'screen'),
                                    (full,full_audit,full_dir,'full')):
        need(report['complete'] is True and report['stage']==stage and
             audit['passed'] is True and audit['cuda_initialized'] is False and
             audit['input_report_sha256']==sha(path/'comparison.json'),
             'Audited FP4 per-layer prerequisite differs: '+stage)
    need(combo['selection']['selected_id']=='top4' and
         full['selection']['selected_id']=='top4' and full['target_pass'] is False and
         full['ppl']['top4']>=binding.TARGET,
         'Strict full top4 target miss required')
    combos,_=layer.make_combos(tables,rank['ranked_improving_swaps'])
    top4=combos['top4']
    need(tensor_sha(top4)==rank['combo_table_sha256']['top4'] and
         tensor_sha(top4)==combo['table_sha256']['top4'] and
         tensor_sha(top4)==full['table_sha256']['top4'],
         'Frozen top4 table bytes differ')
    check_table(top4)
    return top4,dict(five_screen_sha256=sha(five_dir/'comparison.json'),
                     rank_sha256=sha(rank_dir/'comparison.json'),
                     combo_screen_sha256=sha(combo_dir/'comparison.json'),
                     top4_full_sha256=sha(full_dir/'comparison.json'),
                     top4_table_sha256=tensor_sha(top4))


def evaluate_arm(model,name,table,windows,out,common,frozen,policy):
    layout=ORDER[0] if name==RESTORED else name
    spec=dict(id=name,table_name='frozen_fp4_top4',table=table,layouts=(layout,)*56)
    path=out/f'{name}.json'
    try:
        row=evaluator.evaluate(model,spec,windows,path,dict(common,arm=name))
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
    ap.add_argument('--layout-screen-dir',type=Path)
    args=ap.parse_args()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(),
         'Fresh output directory required')
    table,upstream=frozen_top4(args.five_table_screen,args.rank_dir,
                               args.combo_screen_dir,args.top4_full_dir)
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==binding.TOKENIZER_SHA,'Tokenizer differs')
    torch.set_num_threads(8)
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    policy=pin_replay_backend()
    old=read_json(args.top4_full_dir/'top4.json')
    need(old['backend_policy']==policy,'Backend differs from frozen top4')
    hashes=code_hashes()
    if args.stage=='screen':
        train=load_train_tokens(args.prose_tokens)
        windows=[(row*2048,train[row].clone()) for row in range(376,400)]
        dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(376,400)),
                     windows=24,target_tokens=49128)
        arms=(*ORDER,RESTORED)
    else:
        need(args.layout_screen_dir is not None,'Frozen layout screen required')
        screen=read_json(args.layout_screen_dir/'comparison.json')
        audit=read_json(args.layout_screen_dir/'audit.json')
        selected=screen['selection']['selected_id']
        need(screen['complete'] is True and screen['stage']=='screen' and
             selected in ORDER[1:] and audit['passed'] is True and
             audit['cuda_initialized'] is False and
             audit['input_report_sha256']==sha(args.layout_screen_dir/'comparison.json'),
             'Audited nonparent TRAIN layout winner required')
        ids,dataset=evaluator.load_wikitext_tokens(tokenizer,'validation')
        windows=evaluator.ppl_windows(ids,2048)
        need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and
             dataset['token_stream_sha256_int64le']==binding.VALIDATION_TOKENS_SHA,
             'Validation population differs')
        arms=(ORDER[0],selected,RESTORED)
        upstream['layout_screen_sha256']=sha(args.layout_screen_dir/'comparison.json')
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
    common=dict(format='FP4_G16_UNADAPTED_LAYOUT_V1',stage=args.stage,
        source_checkpoint_sha256=binding.SOURCE_SHA,tokenizer_sha256=tokenizer.sha256,
        dataset=dataset,backend_policy=policy,code_sha256=hashes,
        table_sha256=tensor_sha(table),adapter_loaded=False,adapter_sha256=None,
        adapter_used=False,mk_used=False,upstream=upstream)
    reports={}
    for index,name in enumerate(arms):
        print(f'[FP4 layout {args.stage}] {index+1}/{len(arms)} {name}',flush=True)
        reports[name]=evaluate_arm(model,name,table,windows,args.out_dir,common,frozen,policy)
    restoration=evaluator.exact_replay(reports[ORDER[0]],reports[RESTORED])
    if args.stage=='screen':
        valid=[name for name in ORDER if evaluator.valid_candidate(reports[name])]
        need(valid and valid[0]==ORDER[0],'Finite layout parent required')
        selected=min(valid,key=lambda name:(reports[name]['ppl']['nll'],ORDER.index(name)))
        replay=None
    else:
        valid=[ORDER[0],selected]
        replay=evaluator.exact_replay(old,reports[ORDER[0]],archive=True)
        need(replay['complete'] is True,'Archived top4 full parent did not replay')
    final=fp4_quality.assert_weight_content(model,expected)
    result=dict(format='FP4_G16_UNADAPTED_LAYOUT_V1_COMPARISON',complete=True,
        stage=args.stage,upstream=upstream,candidate_order=list(arms[:-1]),
        selection=dict(selected_id=selected,valid=valid,
          rule='Minimum finite TRAIN NLL, parent then frozen order' if args.stage=='screen'
          else 'Frozen TRAIN-selected layout'),
        report_sha256={name:sha(args.out_dir/f'{name}.json') for name in arms},
        ppl={name:reports[name]['ppl']['ppl'] for name in arms},
        cache_bytes=binding.CACHE_BYTES,table_sha256=tensor_sha(table),
        backend_policy=policy,code_sha256=hashes,
        source_content_initial=initial,source_content_final=final,
        restoration=restoration,archived_parent_replay=replay,
        adapter_used=False,mk_used=False,
        target_pass=args.stage=='full' and reports[selected]['ppl']['ppl']<binding.TARGET,
        elapsed_seconds=time.time()-started)
    save_json(args.out_dir/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('stage','selection','ppl','target_pass','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
