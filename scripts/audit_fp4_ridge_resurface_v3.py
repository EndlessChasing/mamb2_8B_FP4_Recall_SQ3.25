#!/usr/bin/env python3
"""CPU-only audit of ridge-state Resurface transfer/fresh quality and gate."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamba2_recall import resurface_native as native, runtime, resurface_data as data
from prepare_quant_first import load_train_tokens,TRAIN_SHA
from evaluate_quant_first import compare_pair
from audit_fp4_g16_resurface_v1 import check_mk
import ridge_resurface_binding_v3 as shared
need,sha,read=shared.need,shared.sha,shared.read
ARMS=('ridge_parent','ridge_resurface')


def verify_ppl(row,identities,count):
    need(row['complete'] is True and row['ppl_complete'] is True and
         len(row['ppl']['windows'])==len(identities), 'PPL window population differs')
    total=0.
    for old,now in zip(identities,row['ppl']['windows']):
        need({k:now[k] for k in ('start','target_tokens','token_sha256_int64le')}==old and
             math.isfinite(now['nll']) and now['nll']>=0 and
             now['ppl']==math.exp(now['nll']/now['target_tokens']), 'Window identity/arithmetic differs')
        total+=now['nll']
    need(sum(v['target_tokens'] for v in row['ppl']['windows'])==count and
         row['ppl']['target_tokens']==count and row['ppl']['nll']==total and
         row['ppl']['ppl']==math.exp(total/count), 'Aggregate NLL/PPL differs')
    for field in ('cache','ppl_cache'):
        need(row[field]['total_bytes']==shared.CACHE_BYTES and row[field]['row_bytes']==52,
             'Physical persistent state bytes differ')
    for field in ('storage_descriptor_probe','storage_descriptor'):
        descriptor=row[field]
        need(descriptor['row_bytes']==52 and descriptor['total_cache_bytes']==shared.CACHE_BYTES and
             descriptor['static_basis_bytes']==109952 and descriptor['static_scale_bytes']==1792 and
             descriptor['static_predictor_bytes']==3673216 and len(descriptor['layers'])==56,
             'Static or physical storage descriptor differs')
        for layer in descriptor['layers']:
            need(layer['state_bytes']==128*64*52 and layer['conv_storage_bytes']==81920 and
                 sum(v['storage_bytes'] for v in layer['tensors'].values())==128*64*52,
                 'Physical state tensors differ')
    return row['ppl']['ppl']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('comparison','training-report','training-audit','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--prose-tokens',type=Path)
    ap.add_argument('--screen-dir',type=Path)
    shared.add_parent_arguments(ap);args=ap.parse_args()
    need(not args.out.exists(),'Fresh audit output required')
    parent=shared.load_parent(args);comp=read(args.comparison)
    need(comp['format']=='FP4_RIDGE_RESURFACE_EVAL_V3' and comp['complete'] is True and
         comp['stage'] in ('screen','full') and comp['adapter_origin'] in ('transfer','fresh') and
         comp['parent']==parent[7] and comp['code_sha256']==shared.hashes() and
         comp['protocol_sha256']==sha(shared.PROTOCOL) and
         comp['training_report_sha256']==sha(args.training_report) and
         comp['training_audit_sha256']==sha(args.training_audit), 'Complete bound comparison differs')
    train=read(args.training_report);ta=read(args.training_audit)
    need(train['complete'] is True and train['mode']=='formal' and ta['passed'] is True and
         ta['cuda_initialized'] is False and ta['training_report_sha256']==sha(args.training_report),
         'Training provenance/audit differs')
    if comp['adapter_origin']=='transfer':
        need(train['format']=='FP4_G16_RESURFACE_TRAIN_V1' and
             ta['format']=='FP4_G16_RESURFACE_CPU_AUDIT_V1' and
             train['successful_updates']==1536 and train['parity_128_512_initial_and_export'] is True and
             ta['input_report_sha256']==sha(args.training_audit.parent/'comparison.json'),
             'Historical adapter transfer differs')
    else:
        need(train['format']=='FP4_RIDGE_RESURFACE_TRAIN_V3' and
             train['binding']['ridge_parent']==parent[7] and
             ta['format']=='FP4_RIDGE_RESURFACE_TRAIN_CPU_AUDIT_V3','Fresh ridge training differs')
    for name,digest in train['code_sha256'].items():
        need(not Path(name).is_absolute() and (ROOT/name).resolve().is_relative_to(ROOT) and
             sha(ROOT/name)==digest,'Training code source differs')
    path=args.training_report.parent/train['adapter']['file']
    need(path.parent==args.training_report.parent and sha(path)==comp['adapter_sha256']==
         train['adapter']['sha256'] and path.stat().st_size==train['adapter']['bytes'] and
         train['adapter']['payload_bytes']==2308208,'FP16 adapter file differs')
    tensors=native.read_fp16(path,expected_binding=train['binding'])['tensors']
    need(len(tensors)==224 and sum(x.numel()*x.element_size() for x in tensors.values())==2308208 and
         {k:native.tensor_hash(v) for k,v in tensors.items()}==train['adapter']['tensor_sha256'],
         'FP16 adapter tensors differ')
    if comp['stage']=='screen':
        need(args.prose_tokens is not None,'Pinned TRAIN population required')
        tokens=load_train_tokens(args.prose_tokens)
        identities=[dict(start=i*2048,target_tokens=2047,
            token_sha256_int64le=runtime.token_digest(tokens[i].numpy())) for i in range(432,448)]
        count=32752
        need(comp['dataset']['split']=='train' and comp['dataset']['rows']==list(range(432,448)) and
             comp['dataset']['file_sha256']==TRAIN_SHA,'Screen TRAIN population differs')
    else:
        identities=[{k:v[k] for k in ('start','target_tokens','token_sha256_int64le')}
                    for v in parent[5]['ppl']['windows']];count=264764
        need(args.screen_dir is not None,'Audited positive screen required')
        screen=read(args.screen_dir/'comparison.json');sa=read(args.screen_dir/'audit.json')
        need(screen['complete'] is True and screen['advance_to_full'] is True and
             sa['passed'] is True and sa['cuda_initialized'] is False and
             sa['input_report_sha256']==sha(args.screen_dir/'comparison.json') and
             screen['adapter_sha256']==comp['adapter_sha256'] and
             comp['dataset']['split']=='validation', 'Full frozen screen/validation differs')
    rows={}
    for arm in ARMS:
        item=comp['reports'][arm];path=args.comparison.parent/item['file']
        need(path.parent==args.comparison.parent and sha(path)==item['sha256'], 'Arm file binding differs')
        row=read(path);rows[arm]=row
        need(row['arm']==arm and row['table_sha256']==parent[7]['table_sha256'] and
             row['layer_layouts']==parent[7]['layouts'] and
             row['static_basis_sha256']==parent[7]['static_tensor_sha256']['bases'] and
             row['static_scale_sha256']==parent[7]['static_tensor_sha256']['scales'] and
             row['static_predictor_sha256']==parent[7]['static_tensor_sha256']['predictors'] and
             row['adapter_loaded']==(arm==ARMS[1]) and
             row['frozen_weight_check']['decoded_tensor_sha256']==parent[6] and
             row['native_restoration']['complete'] is True,
             'Arm frozen static/source/restoration differs')
        need(verify_ppl(row,identities,count)==item['ppl']==comp['ppl'][arm] and
             row['ppl']['nll']==comp['nll'][arm], 'Arm comparison PPL differs')
        if arm==ARMS[1]:
            need(row['adapter_storage']['resident_storage_bytes']==2308208 and
                 row['adapter_storage']['additional_recurrent_cache_bytes']==0 and
                 row['adapter_origin']==comp['adapter_origin'], 'Adapted storage/origin differs')
        else:need(row['adapter_storage']['resident_storage_bytes']==0,'Control has an adapter')
    control,adapted=(rows[k] for k in ARMS)
    if comp['stage']=='full':
        for field in ('ppl','repeated_reset_probe','cache','ppl_cache','ppl_end_cache_tensor_sha256'):
            need(control[field]==parent[5][field],'Archived8.40828 parent does not exactly replay')
        need(comp['exact_parent_replay']['complete'] is True,'Exact parent replay missing')
    need(comp['adapter_removal_reset_and_cache_exact'] is True,'Adapter removal not exact')
    ppl_pass=comp['stage']=='full' and adapted['ppl']['ppl']<8.
    if comp['mk_complete']:
        cases=data.generate_cases('confirm');byid={x['id']:x for x in cases}
        for row in rows.values():
            need(row['mk_complete'] is True and row['mk_cache']['total_bytes']==shared.CACHE_BYTES,
                 'MK physical state differs')
            check_mk(row)
            need(len(row['mk']['rows'])==len(cases)==768,'MK population differs')
            for case in row['mk']['rows']:
                need(all(case[key]==value for key,value in byid[case['id']].items()),
                     'MK prompt/answer population changed')
        pair=compare_pair(control,adapted)
        need(pair==comp['comparison'],'Paired MK/PPL bootstrap differs')
        mk_pass=pair['normal_mk_correct_delta']>0 and pair['normal_mk_paired_bootstrap_95ci'][0]>0
    else:
        need(comp['comparison'] is None,'MK result fabricated');mk_pass=False
    need(comp['advance_to_full']==(comp['stage']=='screen' and
         adapted['ppl']['nll']<control['ppl']['nll']) and comp['strict_ppl_below_8']==ppl_pass and
         comp['normal_mk_gate_pass']==mk_pass and comp['publication_gate_pass']==(ppl_pass and mk_pass),
         'Strict selection/publication gate differs')
    for field in ('initial_weight_check','final_weight_check'):
        need(comp[field]['decoded_tensor_sha256']==parent[6] and
             comp[field]['actual_content_checked'] is True,'507 frozen FP4 tensors differ')
    need(comp['memory']==dict(logical_encoded_weight_payload_bytes=4638460360,
        expanded_fp16_weight_bytes=16473999360,state_conv_table_cache_bytes=28499968,
        static_basis_bytes=109952,static_latent_scale_bytes=1792,static_predictor_bytes=3673216,
        adapter_fp16_payload_bytes=2308208),'Memory accounting differs')
    result=dict(format='FP4_RIDGE_RESURFACE_CPU_AUDIT_V3',complete=True,passed=True,
        cuda_initialized=torch.cuda.is_initialized(),input_report_sha256=sha(args.comparison),
        training_report_sha256=sha(args.training_report),source_sha256=sha(__file__),
        stage=comp['stage'],adapter_origin=comp['adapter_origin'],ppl=comp['ppl'],target_tokens=count,
        mk_complete=comp['mk_complete'],advance_to_full=comp['advance_to_full'],
        strict_ppl_below_8=ppl_pass,publication_gate_pass=ppl_pass and mk_pass)
    need(result['cuda_initialized'] is False,'Audit must be CPU only')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
