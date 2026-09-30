#!/usr/bin/env python3
"""Independent CPU-only final fresh ridge Resurface training evidence audit."""
from pathlib import Path
import argparse
import json
import math
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamba2_recall import resurface_native as native
import ridge_resurface_binding_v3 as shared
import train_fp4_ridge_resurface_v3 as trainer
from train_quant_first import STEPS,PROSE_TOKENS_SHA,PROSE_MANIFEST_SHA,schedule
need,sha,read=shared.need,shared.sha,shared.read


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('training-report','smoke-report','prose-tokens','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    shared.add_parent_arguments(ap);args=ap.parse_args()
    need(not args.out.exists(),'Fresh audit output required')
    parent=shared.load_parent(args);report=read(args.training_report);smoke=read(args.smoke_report)
    need(report['format']=='FP4_RIDGE_RESURFACE_TRAIN_V3' and report['complete'] is True and
         report['mode']=='formal' and report['successful_updates']==STEPS==1536 and
         report['attempts']==len(report['history']) and report['overflows']<=8 and
         report['attempts']-report['overflows']==1536 and
         report['fresh_initialization'] is True and report['parity_128_512_initial_and_export'] is True and
         report['code_sha256']==trainer.hashes() and
         report['binding']['ridge_parent']==parent[7], 'Formal fresh training completion/binding differs')
    binding=report['binding']
    need(binding['prose_tokens_sha256']==PROSE_TOKENS_SHA==sha(args.prose_tokens) and
         binding['prose_manifest_sha256']==PROSE_MANIFEST_SHA and
         binding['forward']=='exact frozen deployed ridge state' and
         binding['backward']=='legacy32/32/64 live-mask STE; ignores ridge latent/predictor derivatives' and
         binding['planned_successful_updates']==1536 and
         binding['initial_adapter']=='fresh V=0,g=1,w=0,b=-4' and
         binding['objective']==dict(numeric_ce=.25,prose_ce=1.,prose_kl=.5,prose_closure=.1,
             closure_budget=.006,closure_budget_coefficient=10.,temperature=1.) and
         binding['recipe_selected_before_training'] is True and
         binding['recipe_selection_used_validation'] is False, 'Training data/gradient recipe differs')
    need(smoke['format']==report['format'] and smoke['complete'] is True and smoke['mode']=='smoke' and
         smoke['successful_updates']==1 and smoke['binding']==binding and
         smoke['adapter']['discarded'] is True and smoke['parity_128_512_initial_and_export'] is True and
         report['smoke_report_sha256']==sha(args.smoke_report) and
         args.smoke_report.parent.resolve()!=args.training_report.parent.resolve(), 'Distinct discarded smoke differs')
    ordered=schedule();success=overflow=0
    prose_manifest=read(ROOT/'docs/prose_train_manifest.json')
    need(sha(ROOT/'docs/prose_train_manifest.json')==PROSE_MANIFEST_SHA,'Pinned prose schedule differs')
    prose_order=prose_manifest['schedule']
    for i,row in enumerate(report['history'],1):
        need(row['attempt']==i and type(row['overflow']) is bool and success<1536 and
             row['update_index']==success and row['schedule_entry']==ordered[success] and
             row['prose_window']==prose_order[success%448] and
             row['prose_start']==512*((success//448)%4),
             'Successful update history differs')
        success+=int(not row['overflow']);overflow+=int(row['overflow'])
        need(row['successful_updates']==success and 0<=row['prose_window']<448 and
             all(math.isfinite(row[k]) for k in ('mk_ce','prose_ce','prose_kl','prose_closure','loss_scale')),
             'Training history scalar/step differs')
        need(row['objective']==binding['objective'] and
             row['weighted_numeric_ce']==.25*row['mk_ce'] and
             row['weighted_prose_ce']==row['prose_ce'] and
             row['weighted_prose_kl']==.5*row['prose_kl'] and
             row['weighted_prose_closure']==.1*row['prose_closure'],
             'PPL-priority objective history differs')
    need(success==1536 and overflow==report['overflows'],'Final training success counter differs')
    for field in ('initial_student_base_check','initial_teacher_base_check',
                  'final_student_base_check','final_teacher_base_check'):
        proof=report[field]
        need(proof['actual_content_checked'] is True and proof['tensors']==507 and
             proof['decoded_tensor_sha256']==parent[6], 'Frozen FP4 weight ledger differs: '+field)
    need(report['frozen_static_ridge_sha256']==parent[7]['static_tensor_sha256'],
         'Frozen ridge static state values differ')
    need([x['successful_updates'] for x in report['checkpoints']]==[384,768,1152,1536],
         'Frozen checkpoint sequence differs')
    for item in report['checkpoints']:
        path=args.training_report.parent/item['file']
        need(path.parent==args.training_report.parent and sha(path)==item['sha256'] and
             path.stat().st_size==item['bytes'], 'Checkpoint receipt differs')
    final=args.training_report.parent/report['checkpoints'][-1]['file']
    cp=torch.load(final,map_location='cpu',weights_only=True)
    need(cp['format']=='FP4_RIDGE_RESURFACE_CHECKPOINT_V3' and cp['binding']==binding and
         cp['successful_updates']==1536 and cp['attempts']==report['attempts'],
         'Final checkpoint metadata differs')
    adapter=args.training_report.parent/report['adapter']['file']
    need(adapter.parent==args.training_report.parent and sha(adapter)==report['adapter']['sha256'] and
         adapter.stat().st_size==report['adapter']['bytes'] and
         report['adapter']['payload_bytes']==2308208 and report['adapter']['discarded'] is False,
         'Adapter FP16 file differs')
    tensors=native.read_fp16(adapter,expected_binding=binding)['tensors']
    masters=cp['masters']
    need(len(tensors)==len(masters)==224 and set(tensors)==set(masters) and
         sum(x.numel()*2 for x in tensors.values())==2308208 and
         all(x.dtype==torch.float32 and torch.isfinite(x).all() and
             torch.equal(x.half(),tensors[k]) for k,x in masters.items()) and
         {k:native.tensor_hash(x) for k,x in tensors.items()}==report['adapter']['tensor_sha256'],
         'Final master/FP16 export differs')
    need(len(cp['optimizer']['state'])==224 and
         all(int(x['step'])==1536 for x in cp['optimizer']['state'].values()),
         'Final optimizer parameter count/steps differ')
    evidence_path=args.training_report.parent/report['parity_evidence']['file']
    need(sha(evidence_path)==report['parity_evidence']['sha256'] and
         evidence_path.stat().st_size==report['parity_evidence']['bytes'], 'Raw parity evidence differs')
    evidence=torch.load(evidence_path,map_location='cpu',weights_only=True)
    need(evidence['format']=='FP4_RIDGE_RESURFACE_PARITY_V3' and evidence['binding']==binding,
         'Raw parity metadata differs')
    train_tokens=torch.load(args.prose_tokens,map_location='cpu',weights_only=True)
    for phase in ('initial','export'):
        need(set(evidence[phase])=={'128','512'}, '128/512 probe population differs')
        for length in (128,512):
            entry=evidence[phase][str(length)]
            need(torch.equal(entry['ids'],train_tokens[0,:length][None]) and
                 entry['packed_hidden'].dtype==entry['training_hidden'].dtype==torch.float16 and
                 tuple(entry['packed_hidden'].shape)==(1,length,4096) and
                 torch.equal(entry['packed_hidden'],entry['training_hidden']) and
                 torch.isfinite(entry['packed_hidden']).all(), 'Raw bitwise deployment/training parity differs')
    result=dict(format='FP4_RIDGE_RESURFACE_TRAIN_CPU_AUDIT_V3',complete=True,passed=True,
        cuda_initialized=torch.cuda.is_initialized(),training_report_sha256=sha(args.training_report),
        input_report_sha256=sha(args.training_report),source_sha256=sha(__file__),
        smoke_report_sha256=sha(args.smoke_report),successful_updates=1536,
        frozen_weights=507,exact_128_512_training_deployment_forward=True,
        parent_ppl=parent[7]['parent_ppl'],surrogate_backward=True)
    need(result['cuda_initialized'] is False,'Independent audit must be CPU only')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
