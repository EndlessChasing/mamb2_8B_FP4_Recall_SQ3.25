#!/usr/bin/env python3
"""Same-52-byte latent carry with static zero-tier residual prediction.

The current zero-tier input is exact. Previous carry uses the FP8 latent plus
a prediction from already-resident INT8/INT4/latent state. The prediction
matrix is static, and the per-token readout correction is transient.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import triton
import triton.language as tl
from mamba2_recall import state_codec as legacy_codec
from mamba2_recall.state_quant import _LayerCache
from state_ppl_codec_v6 import _stored_quantize
import w4_state_repair_codec_v2 as parent

RP=16
CACHE_BYTES=28499968
TABLE_BYTES=57344


def layout_descriptor(layout):
    old=parent.layout_descriptor(layout)
    n8,n4,nzero=old['n8'],old['n4']-4,old['nzero']+4
    if n4<4 or n8+n4+nzero!=128 or n8+n4//2+2+4!=52:
        raise ValueError('Latent tier layout violates same-byte geometry')
    return dict(layout=layout,n8=n8,n4=n4,nzero=nzero,
        payload_bytes=48,scale_bytes=4,row_bytes=52,
        latent_coefficients=2,latent_dtype='torch.float8_e4m3fn',
        tensor_widths=dict(lo=n8//2,hi=n8//2,q4=n4//2,latent=2))


def allocate_state(batch,heads,dim,dstate,device,*,layout):
    spec=layout_descriptor(layout)
    if dstate!=128 or any(type(v) is not int or v<=0 for v in (batch,heads,dim)):
        raise ValueError('Positive state geometry and dstate=128 required')
    base=(batch,heads,dim)
    tensors={name:torch.zeros((*base,width),device=device,
               dtype=torch.float8_e4m3fn if name=='latent' else torch.uint8)
             for name,width in spec['tensor_widths'].items()}
    tensors.update({name:torch.zeros(base,device=device,dtype=torch.float16)
                    for name in ('s8','s4')})
    return legacy_codec.PackedState('latent_fp8_'+layout,tensors,(*base,128))


def validate_state(state,device,layout):
    spec=layout_descriptor(layout)
    if (state.mode!='latent_fp8_'+layout or len(state.shape)!=4 or
            state.shape[-1]!=128 or
            set(state.tensors)!={'lo','hi','q4','latent','s8','s4'}):
        raise ValueError('Latent packed state mode/inventory differs')
    base=tuple(state.shape[:3])
    if min(base)<=0:raise ValueError('Invalid latent state geometry')
    storages=set()
    for name,value in state.tensors.items():
        shape=base+((spec['tensor_widths'][name],) if name in spec['tensor_widths'] else ())
        dtype=(torch.float16 if name in ('s8','s4') else
               torch.float8_e4m3fn if name=='latent' else torch.uint8)
        if (tuple(value.shape)!=shape or value.dtype!=dtype or
                value.device!=torch.device(device) or not value.is_contiguous() or
                value.storage_offset()!=0 or
                value.untyped_storage().nbytes()!=value.numel()*value.element_size()):
            raise ValueError('Latent packed state backing allocation differs: '+name)
        storages.add(value.untyped_storage().data_ptr())
    if len(storages)!=6 or state.nbytes!=base[0]*base[1]*base[2]*52:
        raise ValueError('Latent state aliases or exceeds 52-byte row')
    return True


@triton.jit
def _latent_scan(X,DT,A,B,C,CPred,D,DB,Perm,Lo,Hi,Q4,Latent,S8,S4,Basis,LScale,Y,
                 L:tl.constexpr,H:tl.constexpr,P:tl.constexpr,G:tl.constexpr,
                 N8:tl.constexpr,N4:tl.constexpr,NZ:tl.constexpr,
                 R8:tl.constexpr,R4:tl.constexpr,RZ:tl.constexpr,TILE:tl.constexpr):
    b,h,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    p=tile*TILE+tl.arange(0,TILE)
    mp=p<P
    row=(b*H+h)*P+p
    n8=tl.arange(0,R8);n4=tl.arange(0,R4);nd=tl.arange(0,RZ)
    lo=(tl.load(Lo+row[:,None]*(N8//2)+n8[None,:]//2,
                mask=mp[:,None]&(n8[None,:]<N8),other=0).to(tl.int32)
        >>((n8[None,:]%2)*4))&15
    hi=(tl.load(Hi+row[:,None]*(N8//2)+n8[None,:]//2,
                mask=mp[:,None]&(n8[None,:]<N8),other=0).to(tl.int32)
        >>((n8[None,:]%2)*4))&15
    ub=(hi<<4)|lo
    q8=ub-256*(ub>127).to(tl.int32)
    nib=(tl.load(Q4+row[:,None]*(N4//2)+n4[None,:]//2,
                 mask=mp[:,None]&(n4[None,:]<N4),other=0).to(tl.int32)
         >>((n4[None,:]%2)*4))&15
    q4=nib-16*(nib>7).to(tl.int32)
    s8=tl.load(S8+row,mask=mp,other=0).to(tl.float32)
    s4=tl.load(S4+row,mask=mp,other=0).to(tl.float32)
    state8=q8.to(tl.float32)*s8[:,None]
    state4=q4.to(tl.float32)*s4[:,None]
    av=tl.load(A+h).to(tl.float32)
    dv=tl.load(D+h).to(tl.float32)
    bias=tl.load(DB+h).to(tl.float32)
    group=h//(H//G)
    pos8=tl.load(Perm+group*128+n8,mask=n8<N8,other=0).to(tl.int32)
    pos4=tl.load(Perm+group*128+N8+n4,mask=n4<N4,other=0).to(tl.int32)
    posd=tl.load(Perm+group*128+N8+N4+nd,mask=nd<NZ,other=0).to(tl.int32)
    basis0=tl.load(Basis+(group*NZ+nd)*2,mask=nd<NZ,other=0).to(tl.float32)
    basis1=tl.load(Basis+(group*NZ+nd)*2+1,mask=nd<NZ,other=0).to(tl.float32)
    ls0=tl.load(LScale+group*2).to(tl.float32)
    ls1=tl.load(LScale+group*2+1).to(tl.float32)
    z0=tl.load(Latent+row*2).to(tl.float32)*ls0
    z1=tl.load(Latent+row*2+1).to(tl.float32)*ls1
    for t in range(L):
        xoff=((b*L+t)*H+h)*P+p
        pred_base=((b*L+t)*G+group)*(N8+N4+2)
        pc8=tl.load(CPred+pred_base+n8,mask=n8<N8,other=0).to(tl.float32)
        pc4=tl.load(CPred+pred_base+N8+n4,mask=n4<N4,other=0).to(tl.float32)
        pcz0=tl.load(CPred+pred_base+N8+N4).to(tl.float32)
        pcz1=tl.load(CPred+pred_base+N8+N4+1).to(tl.float32)
        x=tl.load(X+xoff,mask=mp,other=0).to(tl.float32)
        dt=tl.load(DT+(b*L+t)*H+h).to(tl.float32)+bias
        dt=tl.where(dt<=20.,tl.math.log(tl.math.exp(dt)+1.),dt)
        decay=tl.exp(av*dt)
        bbase=((b*L+t)*G+group)*128
        bv8=tl.load(B+bbase+pos8,mask=n8<N8,other=0).to(tl.float32)
        cv8=tl.load(C+bbase+pos8,mask=n8<N8,other=0).to(tl.float32)
        bv4=tl.load(B+bbase+pos4,mask=n4<N4,other=0).to(tl.float32)
        cv4=tl.load(C+bbase+pos4,mask=n4<N4,other=0).to(tl.float32)
        bvd=tl.load(B+bbase+posd,mask=nd<NZ,other=0).to(tl.float32)
        cvd=tl.load(C+bbase+posd,mask=nd<NZ,other=0).to(tl.float32)
        prior8=state8
        prior4=state4
        state8=prior8*decay+(bv8[None,:]*dt)*x[:,None]
        state4=prior4*decay+(bv4[None,:]*dt)*x[:,None]
        state8=tl.where(n8[None,:]<N8,state8,0.)
        state4=tl.where(n4[None,:]<N4,state4,0.)
        dot_bc=tl.sum(bvd*cvd,0)
        dot_c0=tl.sum(basis0*cvd,0)
        dot_c1=tl.sum(basis1*cvd,0)
        dot_b0=tl.sum(basis0*bvd,0)
        dot_b1=tl.sum(basis1*bvd,0)
        out=tl.sum(state8*cv8[None,:],1)+tl.sum(state4*cv4[None,:],1)
        out+=decay*(z0*dot_c0+z1*dot_c1)+(x*dt)*dot_bc+x*dv
        out+=decay*(tl.sum(prior8*pc8[None,:],1)+
                    tl.sum(prior4*pc4[None,:],1)+z0*pcz0+z1*pcz1)
        tl.store(Y+xoff,out,mask=mp)
        z0=z0*decay+(x*dt)*dot_b0
        z1=z1*decay+(x*dt)*dot_b1
        qz0=tl.minimum(tl.maximum(z0/ls0,-448.),448.).to(tl.float8e4nv)
        qz1=tl.minimum(tl.maximum(z1/ls1,-448.),448.).to(tl.float8e4nv)
        z0=qz0.to(tl.float32)*ls0
        z1=qz1.to(tl.float32)*ls1
        den8=tl.maximum(tl.div_rn(tl.max(tl.abs(state8),1),127.),1e-8)
        den4=tl.maximum(tl.div_rn(tl.max(tl.abs(state4),1),7.),1e-8)
        s8=den8.to(tl.float16).to(tl.float32)
        s4=den4.to(tl.float16).to(tl.float32)
        q8=_stored_quantize(state8,s8[:,None],127)
        q4=_stored_quantize(state4,s4[:,None],7)
        state8=q8.to(tl.float32)*s8[:,None]
        state4=q4.to(tl.float32)*s4[:,None]
    qlo=(q8&15).to(tl.uint8);qhi=((q8>>4)&15).to(tl.uint8)
    lo0,lo1=tl.split(tl.reshape(qlo,(TILE,R8//2,2)))
    hi0,hi1=tl.split(tl.reshape(qhi,(TILE,R8//2,2)))
    pairs8=tl.arange(0,R8//2)
    tl.store(Lo+row[:,None]*(N8//2)+pairs8[None,:],lo0|(lo1<<4),
             mask=mp[:,None]&(pairs8[None,:]<N8//2))
    tl.store(Hi+row[:,None]*(N8//2)+pairs8[None,:],hi0|(hi1<<4),
             mask=mp[:,None]&(pairs8[None,:]<N8//2))
    qn=(q4&15).to(tl.uint8)
    f0,f1=tl.split(tl.reshape(qn,(TILE,R4//2,2)))
    pairs4=tl.arange(0,R4//2)
    tl.store(Q4+row[:,None]*(N4//2)+pairs4[None,:],f0|(f1<<4),
             mask=mp[:,None]&(pairs4[None,:]<N4//2))
    qz0=tl.minimum(tl.maximum(z0/ls0,-448.),448.).to(tl.float8e4nv)
    qz1=tl.minimum(tl.maximum(z1/ls1,-448.),448.).to(tl.float8e4nv)
    tl.store(Latent+row*2,qz0)
    tl.store(Latent+row*2+1,qz1)
    tl.store(S8+row,s8,mask=mp)
    tl.store(S4+row,s4,mask=mp)


@torch.no_grad()
def scan(x,dt,A,B,C,D,dt_bias,state,permutation,basis,latent_scale,predictor,*,layout):
    spec=layout_descriptor(layout)
    if x.ndim!=4 or x.dtype!=torch.float16 or not x.is_cuda:
        raise ValueError('x must be CUDA FP16 [batch,tokens,heads,dim]')
    batch,length,heads,dim=x.shape
    if length<1 or dim%RP or state.shape!=(batch,heads,dim,128):
        raise ValueError('Positive sequence and matching latent state required')
    validate_state(state,x.device,layout)
    groups=B.shape[2]
    expected=((dt,(batch,length,heads)),(A,(heads,)),(B,(batch,length,groups,128)),
              (C,(batch,length,groups,128)),(D,(heads,)),(dt_bias,(heads,)))
    for value,shape in expected:
        if tuple(value.shape)!=shape or value.device!=x.device:
            raise ValueError('Latent scan input geometry/device differs')
    if (heads%groups or A.dtype!=torch.float32 or
            any(v.dtype!=torch.float16 for v in (dt,B,C)) or
            any(v.dtype not in (torch.float16,torch.float32) for v in (D,dt_bias))):
        raise ValueError('Latent scan dtype/grouping differs')
    if (permutation.dtype!=torch.uint8 or tuple(permutation.shape)!=(groups,128) or
            permutation.device!=x.device or
            basis.dtype!=torch.float16 or tuple(basis.shape)!=(groups,spec['nzero'],2) or
            basis.device!=x.device or
            latent_scale.dtype!=torch.float16 or tuple(latent_scale.shape)!=(groups,2) or
            latent_scale.device!=x.device or not bool(torch.isfinite(basis).all()) or
            not bool(torch.isfinite(latent_scale).all()) or
            not bool((latent_scale>0).all())):
        raise ValueError('Latent static basis/scales differ')
    features=spec['n8']+spec['n4']+2
    if (predictor.dtype!=torch.float16 or
            tuple(predictor.shape)!=(groups,features,spec['nzero']) or
            predictor.device!=x.device or not predictor.is_contiguous() or
            not bool(torch.isfinite(predictor).all())):
        raise ValueError('Static zero-tier predictor differs')
    x,dt,A,B,C,D,dt_bias,permutation,basis,latent_scale=(
        value.contiguous() for value in
        (x,dt,A,B,C,D,dt_bias,permutation,basis,latent_scale))
    dead_ix=permutation[:,spec['n8']+spec['n4']:].long()
    dead_c=C.float().gather(3,dead_ix[None,None,:,:].expand(batch,length,-1,-1))
    correction=torch.einsum('blgd,gfd->blgf',dead_c,predictor.float()).contiguous()
    if not bool(torch.isfinite(correction).all()):
        raise FloatingPointError('Nonfinite transient predictor readout')
    output=torch.empty_like(x)
    n8,n4,nzero=spec['n8'],spec['n4'],spec['nzero']
    with torch.cuda.device(x.device):
        _latent_scan[(batch,heads,triton.cdiv(dim,RP))](
            x,dt,A,B,C,correction,D,dt_bias,permutation,
            *(state.tensors[k] for k in ('lo','hi','q4','latent','s8','s4')),
            basis,latent_scale,output,
            length,heads,dim,groups,n8,n4,nzero,
            triton.next_power_of_2(n8),triton.next_power_of_2(n4),
            triton.next_power_of_2(nzero),RP,num_warps=4)
    return output,None


class PredictorState(parent.StateRepairV2):
    """Own native Mamba mixer forwards with predicted zero-tier history."""

    def __init__(self,model,table,layouts,bases,scales,predictors):
        super().__init__(model,table,layouts)
        if len(bases)!=56 or len(scales)!=56 or len(predictors)!=56:
            raise ValueError('Exactly 56 static bases/scales/predictors required')
        self.bases=[];self.latent_scales=[];self.predictors=[]
        for layer,layout in enumerate(self.layouts):
            spec=layout_descriptor(layout)
            basis=bases[layer].to(device=self.device,dtype=torch.float16).contiguous()
            scale=scales[layer].to(device=self.device,dtype=torch.float16).contiguous()
            predictor=predictors[layer].to(device=self.device,dtype=torch.float16).contiguous()
            if (tuple(basis.shape)!=(8,spec['nzero'],2) or
                    tuple(scale.shape)!=(8,2) or
                    tuple(predictor.shape)!=(8,spec['n8']+spec['n4']+2,spec['nzero']) or
                    not bool(torch.isfinite(basis).all()) or
                    not bool(torch.isfinite(scale).all()) or
                    not bool(torch.isfinite(predictor).all()) or
                    not bool((scale>0).all()) or
                    basis.untyped_storage().nbytes()!=basis.numel()*2 or
                    scale.untyped_storage().nbytes()!=scale.numel()*2 or
                    predictor.untyped_storage().nbytes()!=predictor.numel()*2):
                raise ValueError('Static predictor/basis/scale allocation differs')
            self.bases.append(basis);self.latent_scales.append(scale)
            self.predictors.append(predictor)

    @torch.no_grad()
    def reset(self,batch_size=1):
        if not self._installed or type(batch_size) is not int or batch_size<=0:
            raise ValueError('Installed controller and positive batch required')
        self.clear()
        for layout in self.layouts:
            conv=torch.zeros(batch_size,4,10240,device=self.device,dtype=torch.float16).transpose(1,2)
            state=allocate_state(batch_size,128,64,128,self.device,layout=layout)
            validate_state(state,self.device,layout)
            self._cache.append(_LayerCache(conv=conv,state=state))
        self._batch_size=batch_size
        expected=batch_size*(CACHE_BYTES-TABLE_BYTES)+TABLE_BYTES
        if self.cache_breakdown()['total_bytes']!=expected:
            raise RuntimeError('Latent request cache violates fixed physical budget')
        return self

    def storage_descriptor(self):
        layers=[]
        for layer,(entry,layout,basis,scale,predictor) in enumerate(zip(
                self._cache,self.layouts,self.bases,self.latent_scales,self.predictors)):
            validate_state(entry.state,self.device,layout)
            layers.append(dict(**layout_descriptor(layout),layer=layer,
                state_shape=list(entry.state.shape),state_bytes=entry.state.nbytes,
                tensors={key:dict(shape=list(value.shape),dtype=str(value.dtype),
                    storage_bytes=value.untyped_storage().nbytes())
                    for key,value in entry.state.tensors.items()},
                conv_shape=list(entry.conv.shape),
            conv_storage_bytes=entry.conv.untyped_storage().nbytes(),
            static_basis_bytes=basis.untyped_storage().nbytes(),
            static_scale_bytes=scale.untyped_storage().nbytes(),
            static_predictor_bytes=predictor.untyped_storage().nbytes()))
        return dict(format='FP4_LATENT_STATE_STORAGE_V1',layouts=list(self.layouts),
            row_bytes=52,payload_bytes=48,scale_bytes=4,
            permutation_storage_bytes=self.permutations.untyped_storage().nbytes(),
            total_cache_bytes=self.cache_breakdown()['total_bytes'],
            static_basis_bytes=sum(x.untyped_storage().nbytes() for x in self.bases),
            static_scale_bytes=sum(x.untyped_storage().nbytes() for x in self.latent_scales),
            static_predictor_bytes=sum(x.untyped_storage().nbytes() for x in self.predictors),
            layers=layers)

    def assert_finite_cache(self):
        values=[entry.conv for entry in self._cache]
        values.extend(t for entry in self._cache for t in entry.state.tensors.values()
                      if t.is_floating_point())
        if values and not all(bool(torch.isfinite(value.float()).all()) for value in values):
            raise FloatingPointError('Nonfinite persistent latent/scale/convolution cache')
        return True

    @torch.no_grad()
    def _forward(self,index,mx,u,seqlen,seq_idx,cu_seqlens,inference_params):
        if not self._installed or self._failed:
            raise RuntimeError('Latent controller is inactive or failed')
        if any(value is not None for value in (seqlen,seq_idx,cu_seqlens,inference_params)):
            raise ValueError('External inference cache/varlen unsupported')
        if (not isinstance(u,torch.Tensor) or u.ndim!=3 or u.dtype!=torch.float16 or
                u.device!=self.device or u.shape[-1]!=4096 or u.shape[1]==0):
            raise ValueError('Mixer input must be CUDA FP16 [batch,tokens,4096]')
        if not self._cache:
            if index!=0:raise RuntimeError('A request must start at layer zero')
            self.reset(u.shape[0])
        if u.shape[0]!=self._batch_size or index!=self._expected_layer:
            raise RuntimeError('Batch shape or layer order changed')
        length=u.shape[1]
        if index==0:
            if len({entry.tokens for entry in self._cache})!=1:
                raise RuntimeError('Layer cache positions differ')
            self._call_length=length
        elif length!=self._call_length:
            raise RuntimeError('Layer token lengths differ')
        entry=self._cache[index]
        try:
            is_step=entry.tokens>0 and length==1
            projected=mx.in_proj(u.squeeze(1) if is_step else u)
            zxbcdt=projected.unsqueeze(1) if is_step else projected
            z,xbc,dt=torch.split(zxbcdt,(8192,10240,128),dim=-1)
            xbc=self._convolution(mx,xbc,entry)
            x,bm,cm=torch.split(xbc,(8192,1024,1024),dim=-1)
            batch=u.shape[0]
            x=x.reshape(batch,length,128,64)
            bm=bm.reshape(batch,length,8,128)
            cm=cm.reshape(batch,length,8,128)
            aa=-torch.exp(mx.A_log.float())
            y,_=scan(x,dt,aa,bm,cm,mx.D,mx.dt_bias,entry.state,
                     self.permutations[index],self.bases[index],self.latent_scales[index],
                     self.predictors[index],
                     layout=self.layouts[index])
            if y.dtype!=torch.float16 or tuple(y.shape)!=(batch,length,128,64):
                raise RuntimeError('Latent scan returned wrong dtype/shape')
            y=y.reshape(batch,length,8192)
            if is_step:
                y=mx.norm(y[:,0],z[:,0])
                out=mx.out_proj(y).unsqueeze(1)
            else:
                y=mx.norm(y,z)
                out=mx.out_proj(y)
            entry.tokens+=length
            self._expected_layer=(index+1)%len(self._mixers)
            if self._expected_layer==0:self._call_length=None
            return out
        except Exception:
            self._failed=True
            raise
