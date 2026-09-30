#!/usr/bin/env python3
"""Exact ridge forward and compact/strided surrogate-gradient V3 fixture."""
from pathlib import Path
import argparse
import json
import torch
from fp4_zero_predictor_codec_v1 import allocate_state,scan,layout_descriptor
from w4_state_repair_codec_v2 import LAYOUT_ORDER
from state_ppl_training_scan_v11 import scan_training as legacy
from state_ridge_training_v2 import scan_training as previous
from state_ridge_training_v3 import scan_training
from ridge_resurface_binding_v3 import sha
ROOT=Path(__file__).resolve().parents[1]
FILES=('scripts/state_ridge_training_v3.py','scripts/check_state_ridge_training_v3.py',
       'scripts/state_ridge_training_v2.py','scripts/state_ppl_training_scan_v11.py',
       'scripts/fp4_zero_predictor_codec_v1.py')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.manual_seed(2026093007);torch.cuda.manual_seed_all(2026093007)
    rows=[];old_mismatch=False
    for layout in LAYOUT_ORDER:
        spec=layout_descriptor(layout);nz=spec['nzero'];features=spec['n8']+spec['n4']+2
        for length in (1,17,65):
            for input_layout in ('compact','projection_strided'):
                b,h,p,g=1,8,16,2
                projected=torch.randn(b,length,h*p+2*g*128+h,device='cuda').half()
                xv,bv,cv,dv=torch.split(projected,(h*p,g*128,g*128,h),dim=-1)
                xv.mul_(.05);bv.mul_(.1);cv.mul_(.1);dv.mul_(.03)
                x=xv.reshape(b,length,h,p);B=bv.reshape(b,length,g,128)
                C=cv.reshape(b,length,g,128);dt=dv
                if input_layout=='compact':x,dt,B,C=(t.contiguous() for t in (x,dt,B,C))
                if input_layout=='projection_strided' and length>1 and all(t.is_contiguous() for t in (x,dt,B,C)):
                    raise RuntimeError('Fixture failed to make projection-strided inputs')
                A=-torch.ones(h,device='cuda');D=torch.ones(h,device='cuda').half()
                bias=torch.full((h,),-3.,device='cuda').half()
                table=torch.stack([torch.randperm(128,device='cuda') for _ in range(g)]).to(torch.uint8)
                basis=(torch.randn(g,nz,2,device='cuda')*.03).half()
                scale=torch.full((g,2),.001,device='cuda').half()
                predictor=(torch.randn(g,features,nz,device='cuda')*.01).half()
                xx,dd,bb,cc=(t.detach().requires_grad_() for t in (x,dt,B,C))
                actual=scan_training(xx,dd,A,bb,cc,D,bias,table,basis,scale,predictor,layout=layout)
                state=allocate_state(b,h,p,128,'cuda',layout=layout)
                reference,_=scan(x,dt,A,B,C,D,bias,state,table,basis,scale,predictor,layout=layout)
                if not torch.equal(actual,reference):raise RuntimeError('Exact ridge forward failed')
                dy=(torch.randn_like(actual)*.01).half()
                grads=torch.autograd.grad(actual,(xx,dd,bb,cc),dy)
                old_inputs=tuple(t.detach().requires_grad_() for t in (x,dt,B,C))
                ox,od,ob,oc=old_inputs
                old=legacy(ox,od,A,ob,oc,D,bias,table)
                old_grads=torch.autograd.grad(old,old_inputs,dy)
                errors=[]
                for now,prior in zip(grads,old_grads):
                    if not torch.isfinite(now).all() or not torch.allclose(now,prior,rtol=.005,atol=.0001):
                        raise RuntimeError('V3 surrogate backward differs on '+input_layout)
                    errors.append(float((now.float()-prior.float()).abs().max()))
                bad_errors=None;bad_matches=None
                if input_layout=='projection_strided' and length>1:
                    old_views=tuple(t.detach().requires_grad_() for t in (x,dt,B,C))
                    px,pd,pb,pc=old_views
                    bad=previous(px,pd,A,pb,pc,D,bias,table,basis,scale,predictor,layout=layout)
                    bad_grads=torch.autograd.grad(bad,old_views,dy)
                    bad_matches=all(torch.allclose(a,z,rtol=.005,atol=.0001) for a,z in zip(bad_grads,old_grads))
                    bad_errors=[float((a.float()-z.float()).abs().max()) for a,z in zip(bad_grads,old_grads)]
                    old_mismatch|=not bad_matches
                rows.append(dict(layout=layout,tokens=length,input_layout=input_layout,
                    input_strides=[list(t.stride()) for t in (x,dt,B,C)],
                    forward_exact=True,gradient_finite=True,legacy_surrogate_allclose=True,
                    max_abs_gradient_error=errors,v2_legacy_surrogate_allclose=bad_matches,
                    v2_max_abs_gradient_error=bad_errors))
    if not old_mismatch:raise RuntimeError('Fixture did not reproduce V2 strided-gradient bug')
    result=dict(format='RIDGE_STATE_TRAINING_CHECK_V3',passed=True,rows=rows,
        source_sha256={name:sha(ROOT/name) for name in FILES},
        v2_strided_gradient_mismatch_seen=old_mismatch,
        backward='legacy32/32/64 live-mask STE; ignores ridge latent/predictor derivatives')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
