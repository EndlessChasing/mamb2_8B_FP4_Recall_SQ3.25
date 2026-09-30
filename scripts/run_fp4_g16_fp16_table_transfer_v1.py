#!/usr/bin/env python3
"""Screen historical FP16-base state tables on unadapted FP4 G16 SQ3.25."""
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
import run_fp4_g16_layout_v1 as prior
import fp4_state_binding_v1 as binding

need,sha,read_json=binding.need,binding.sha,binding.read_json
ORDER=('fp4_top4_parent','fp16_v8_top8','fp16_v9_top2')
RESTORED='restored_parent'
TABLES={
    'fp16_v8_top8':('v8','top8',
      '9b7c04814085abbb67d7a10e0b70d1a6e9b34e97299fc0dc26e4373ea8f8ea39',
      '9f9b6764856274222e227683adf27d155a2b4727faa82b3c0b43b558f7fca596',
      '281f7c9fbfdabd6bfa04964ed44b19761708a08447989436b978fae37dc27298'),
    'fp16_v9_top2':('v9','top2',
      '3467897358f33de22b1b629819cb2035f4cb8912914f0183959aa581fafeab50',
      '4e4bb614ff1ba0dfbbd1ba6f3116474cd959555e8fb22406590190e65e9d9a05',
      'b1865e81ff3fbed028027a883872aef9bb91e614e7cf08c72211496a78aeb597')}
ARCHIVE=ROOT/'artifacts/fp4_weight_v1/fp4_g16_v1'


def code_hashes():
    names=('docs/FP4_G16_FP16_TABLE_TRANSFER_V1_PROTOCOL.md',
           'scripts/run_fp4_g16_fp16_table_transfer_v1.py',
           'scripts/run_fp4_g16_layout_v1.py','scripts/run_w4_state_repair_v2.py',
           'scripts/w4_state_repair_codec_v2.py','mamba2_recall/fp4.py')
    return {name:sha(ROOT/name) for name in names}


def load_tables(args):
    parent,upstream=prior.frozen_top4(args.five_table_screen,args.rank_dir,
                                      args.combo_screen_dir,args.top4_full_dir)
    outcome=read_json(args.mixed_outcome_dir/'comparison.json')
    audit=read_json(args.mixed_outcome_dir/'audit.json')
    need(outcome['complete'] is True and outcome['stage'] in ('screen','full') and
         outcome['target_pass'] is False and audit['passed'] is True and
         audit['cuda_initialized'] is False and
         audit['input_report_sha256']==sha(args.mixed_outcome_dir/'comparison.json'),
         'Audited mixed-layout nonpass required')
    upstream['mixed_outcome_sha256']=sha(args.mixed_outcome_dir/'comparison.json')
    tables={ORDER[0]:parent}
    source_files={}
    for name,(version,selected,pt_sha,json_sha,table_sha) in TABLES.items():
        root=ROOT/'reference/fp16_state_tables'/version
        pt=root/'selected_calibration.pt';receipt=root/'selected_calibration.json'
        need(sha(pt)==pt_sha and sha(receipt)==json_sha,
             'Frozen historical selected file changed: '+name)
        payload=torch.load(pt,map_location='cpu',weights_only=True)
        metadata=read_json(receipt)
        table=payload['permutations'].contiguous()
        check_table(table)
        need(payload['selected_id']==selected and payload['table_sha256']==table_sha and
             tensor_sha(table)==table_sha and payload['cache_bytes']==binding.CACHE_BYTES and
             payload['adapter_used'] is False and payload['heldout_used'] is False and
             payload['mk_used'] is False and metadata['selected_id']==selected and
             metadata['table_sha256']==table_sha and metadata['sha256']==pt_sha,
             'Historical selected table provenance differs: '+name)
        tables[name]=table
        source_files[name]=dict(pt_sha256=pt_sha,receipt_sha256=json_sha,
                                table_sha256=table_sha)
    return tables,upstream,source_files


def evaluate_arm(model,name,table,windows,out,common,frozen,policy):
    spec=dict(id=name,table_name=name,table=table,layouts=(binding.LAYOUT,)*56)
    path=out/f'{name}.json'
    try:
        row=evaluator.evaluate(model,spec,windows,path,dict(common,arm=name,
            candidate_table_sha256=tensor_sha(table)))
        row.update(frozen_source=frozen.check(),backend_policy_check=check_replay_backend(policy),
                   candidate_table_unchanged=tensor_sha(table)==common['table_sha256'][
                       ORDER[0] if name==RESTORED else name],
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
    ap.add_argument('--mixed-outcome-dir',type=Path,required=True)
    ap.add_argument('--transfer-screen-dir',type=Path)
    args=ap.parse_args()
    need(not args.out_dir.exists() and not args.out_dir.is_symlink(),
         'Fresh output directory required')
    tables,upstream,sources=load_tables(args)
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
        windows=[(row*2048,train[row].clone()) for row in range(320,352)]
        dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(320,352)),
                     windows=32,target_tokens=65504)
        arms=(*ORDER,RESTORED)
    else:
        need(args.transfer_screen_dir is not None,'Frozen transfer screen required')
        screen=read_json(args.transfer_screen_dir/'comparison.json')
        audit=read_json(args.transfer_screen_dir/'audit.json')
        selected=screen['selection']['selected_id']
        need(screen['complete'] is True and screen['stage']=='screen' and
             selected in ORDER[1:] and audit['passed'] is True and
             audit['cuda_initialized'] is False and
             audit['input_report_sha256']==sha(args.transfer_screen_dir/'comparison.json') and
             screen['table_sha256']=={name:tensor_sha(value) for name,value in tables.items()},
             'Audited frozen transfer selection differs')
        ids,dataset=evaluator.load_wikitext_tokens(tokenizer,'validation')
        windows=evaluator.ppl_windows(ids,2048)
        need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and
             dataset['token_stream_sha256_int64le']==binding.VALIDATION_TOKENS_SHA,
             'Validation population differs')
        arms=(ORDER[0],selected,RESTORED)
        upstream['transfer_screen_sha256']=sha(args.transfer_screen_dir/'comparison.json')
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
    common=dict(format='FP4_G16_UNADAPTED_TABLE_TRANSFER_V1',stage=args.stage,
        source_checkpoint_sha256=binding.SOURCE_SHA,tokenizer_sha256=tokenizer.sha256,
        dataset=dataset,backend_policy=policy,code_sha256=hashes,
        table_sha256={name:tensor_sha(value) for name,value in tables.items()},
        adapter_loaded=False,adapter_sha256=None,adapter_used=False,mk_used=False,
        upstream=upstream,source_files=sources)
    reports={}
    for index,name in enumerate(arms):
        table=tables[ORDER[0] if name==RESTORED else name]
        print(f'[FP4 transfer {args.stage}] {index+1}/{len(arms)} {name}',flush=True)
        reports[name]=evaluate_arm(model,name,table,windows,args.out_dir,common,frozen,policy)
    restoration=evaluator.exact_replay(reports[ORDER[0]],reports[RESTORED])
    if args.stage=='screen':
        valid=[name for name in ORDER if evaluator.valid_candidate(reports[name])]
        need(valid and valid[0]==ORDER[0],'Finite transfer parent required')
        selected=min(valid,key=lambda name:(reports[name]['ppl']['nll'],ORDER.index(name)))
        replay=None
    else:
        valid=[ORDER[0],selected]
        replay=evaluator.exact_replay(old,reports[ORDER[0]],archive=True)
        need(replay['complete'] is True,'Archived top4 parent did not replay')
    final=fp4_quality.assert_weight_content(model,expected)
    comp=dict(format='FP4_G16_UNADAPTED_TABLE_TRANSFER_V1_COMPARISON',complete=True,
        stage=args.stage,upstream=upstream,source_files=sources,
        candidate_order=list(arms[:-1]),
        selection=dict(selected_id=selected,valid=valid,
          rule='Minimum finite TRAIN NLL, parent then fixed order' if args.stage=='screen'
          else 'Frozen TRAIN-selected transferred table'),
        report_sha256={name:sha(args.out_dir/f'{name}.json') for name in arms},
        ppl={name:reports[name]['ppl']['ppl'] for name in arms},
        cache_bytes=binding.CACHE_BYTES,
        table_sha256={name:tensor_sha(value) for name,value in tables.items()},
        backend_policy=policy,code_sha256=hashes,
        source_content_initial=initial,source_content_final=final,
        restoration=restoration,archived_parent_replay=replay,
        adapter_used=False,mk_used=False,
        target_pass=args.stage=='full' and reports[selected]['ppl']['ppl']<binding.TARGET,
        elapsed_seconds=time.time()-started)
    save_json(args.out_dir/'comparison.json',comp)
    print(json.dumps({k:comp[k] for k in ('stage','selection','ppl','target_pass','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
