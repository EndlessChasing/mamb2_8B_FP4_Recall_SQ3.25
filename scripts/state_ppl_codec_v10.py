#!/usr/bin/env python3
"""Same-52-byte variable StateQuant tiers; immutable v6 baseline delegation.

Derived from the frozen Apache-2.0 state_codec and v6 kernels. Only tier
lengths and masked register/payload geometry change. No dense carry, padded
backing storage, resident codebook or resident layout tensor is allocated.
See reference/statequant/LICENSE and docs/THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import state_ppl_codec_v6 as v6
from state_ppl_codec_v6 import _stored_quantize
from mamba2_recall import state_codec as legacy_codec
from mamba2_recall.state_quant import _LayerCache

PROTOCOL_SHA = '565707cf401b2ab472b9b78368a7192afeebe73056efd811b7acba610e3d9567'
LAYOUTS = ('16_64_48', '8_80_40', '24_48_56', '32_32_64')
COUNTS = {'16_64_48': (16,64,48), '8_80_40': (8,80,40),
          '24_48_56': (24,48,56), '32_32_64': (32,32,64)}
BASELINE = LAYOUTS[0]
RP = 16


def layout_descriptor(layout=BASELINE):
    if layout not in COUNTS:
        raise ValueError('Unknown frozen v10 tier layout')
    n8,n4,nzero = COUNTS[layout]
    return dict(layout=layout,n8=n8,n4=n4,nzero=nzero,payload_bytes=48,
        scale_bytes=4,row_bytes=52,resident_layout_metadata_bytes=0,
        tensor_widths=dict(lo=n8//2,hi=n8//2,q4=n4//2))


def allocate_state(batch,heads,dim,dstate,device,*,layout=BASELINE):
    descriptor = layout_descriptor(layout)
    if dstate != 128 or any(type(v) is not int or v <= 0 for v in (batch,heads,dim)):
        raise ValueError('Positive integer geometry and128 state coordinates required')
    if layout == BASELINE:
        return v6.allocate_state(batch,heads,dim,dstate,device)
    shape = (batch,heads,dim,128)
    tensors = {key: torch.zeros((batch,heads,dim,width),dtype=torch.uint8,device=device)
               for key,width in descriptor['tensor_widths'].items()}
    tensors.update({key: torch.zeros((batch,heads,dim),dtype=torch.float16,device=device)
                    for key in ('s8','s4')})
    return legacy_codec.PackedState('v10_'+layout,tensors,shape)


def validate_state(state,device,layout):
    descriptor = layout_descriptor(layout)
    batch,heads,dim,dstate = state.shape
    if dstate != 128 or min(batch,heads,dim) <= 0:
        raise ValueError('Invalid packed state geometry')
    mode = 'sq3p25' if layout == BASELINE else 'v10_'+layout
    if state.mode != mode or set(state.tensors) != {'lo','hi','q4','s8','s4'}:
        raise ValueError('Packed state layout or tensor inventory differs')
    storages = set()
    for key,value in state.tensors.items():
        shape = (batch,heads,dim)
        dtype = torch.float16 if key in ('s8','s4') else torch.uint8
        if key in descriptor['tensor_widths']:
            shape += (descriptor['tensor_widths'][key],)
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device != torch.device(device)
                or not value.is_contiguous() or value.storage_offset() != 0
                or value.untyped_storage().nbytes() != value.numel()*value.element_size()):
            raise ValueError('State must use exact unpadded physical tensor allocations')
        storages.add(value.untyped_storage().data_ptr())
    if len(storages) != 5 or state.nbytes != batch*heads*dim*52:
        raise ValueError('State tensors alias or exceed the52-byte budget')
    return True


@triton.jit
def _tier_scan(X,DT,A,B,C,D,DB,Perm,Lo,Hi,Q4,S8,S4,Y,
               L:tl.constexpr,H:tl.constexpr,P:tl.constexpr,G:tl.constexpr,
               N8:tl.constexpr,N4:tl.constexpr,NZ:tl.constexpr,
               R8:tl.constexpr,R4:tl.constexpr,RZ:tl.constexpr,TILE:tl.constexpr):
    b,h,tile = tl.program_id(0),tl.program_id(1),tl.program_id(2)
    p = tile*TILE+tl.arange(0,TILE)
    mp = p < P
    row = (b*H+h)*P+p
    n8 = tl.arange(0,R8)
    n4 = tl.arange(0,R4)
    nd = tl.arange(0,RZ)
    lo = (tl.load(Lo+row[:,None]*(N8//2)+n8[None,:]//2,
                  mask=mp[:,None] & (n8[None,:]<N8),other=0).to(tl.int32) >> ((n8[None,:]%2)*4)) & 15
    hi = (tl.load(Hi+row[:,None]*(N8//2)+n8[None,:]//2,
                  mask=mp[:,None] & (n8[None,:]<N8),other=0).to(tl.int32) >> ((n8[None,:]%2)*4)) & 15
    ub = (hi << 4) | lo
    q8 = ub-256*(ub>127).to(tl.int32)
    nib = (tl.load(Q4+row[:,None]*(N4//2)+n4[None,:]//2,
                   mask=mp[:,None] & (n4[None,:]<N4),other=0).to(tl.int32) >> ((n4[None,:]%2)*4)) & 15
    q4 = nib-16*(nib>7).to(tl.int32)
    scale8 = tl.load(S8+row,mask=mp,other=0).to(tl.float32)
    scale4 = tl.load(S4+row,mask=mp,other=0).to(tl.float32)
    state8 = q8.to(tl.float32)*scale8[:,None]
    state4 = q4.to(tl.float32)*scale4[:,None]
    av = tl.load(A+h).to(tl.float32)
    dv = tl.load(D+h).to(tl.float32)
    bias = tl.load(DB+h).to(tl.float32)
    group = h//(H//G)
    pos8 = tl.load(Perm+group*128+n8,mask=n8<N8,other=0).to(tl.int32)
    pos4 = tl.load(Perm+group*128+N8+n4,mask=n4<N4,other=0).to(tl.int32)
    posd = tl.load(Perm+group*128+N8+N4+nd,mask=nd<NZ,other=0).to(tl.int32)
    for t in range(L):
        xoff = ((b*L+t)*H+h)*P+p
        x = tl.load(X+xoff,mask=mp,other=0).to(tl.float32)
        dt = tl.load(DT+(b*L+t)*H+h).to(tl.float32)+bias
        dt = tl.where(dt<=20.,tl.math.log(tl.math.exp(dt)+1.),dt)
        decay = tl.exp(av*dt)
        bbase = ((b*L+t)*G+group)*128
        bv8 = tl.load(B+bbase+pos8,mask=n8<N8,other=0).to(tl.float32)
        cv8 = tl.load(C+bbase+pos8,mask=n8<N8,other=0).to(tl.float32)
        bv4 = tl.load(B+bbase+pos4,mask=n4<N4,other=0).to(tl.float32)
        cv4 = tl.load(C+bbase+pos4,mask=n4<N4,other=0).to(tl.float32)
        bvd = tl.load(B+bbase+posd,mask=nd<NZ,other=0).to(tl.float32)
        cvd = tl.load(C+bbase+posd,mask=nd<NZ,other=0).to(tl.float32)
        state8 = state8*decay+(bv8[None,:]*dt)*x[:,None]
        state4 = state4*decay+(bv4[None,:]*dt)*x[:,None]
        stated = (bvd[None,:]*dt)*x[:,None]
        # Make padded register lanes explicit zeros before either reduction.
        state8 = tl.where(n8[None,:]<N8,state8,0.)
        state4 = tl.where(n4[None,:]<N4,state4,0.)
        stated = tl.where(nd[None,:]<NZ,stated,0.)
        out = tl.sum(state8*cv8[None,:],1)
        out += tl.sum(state4*cv4[None,:],1)
        out += tl.sum(stated*cvd[None,:],1)+x*dv
        tl.store(Y+xoff,out,mask=mp)
        den8 = tl.maximum(tl.div_rn(tl.max(tl.abs(state8),1),127.),1e-8)
        den4 = tl.maximum(tl.div_rn(tl.max(tl.abs(state4),1),7.),1e-8)
        scale8 = den8.to(tl.float16).to(tl.float32)
        scale4 = den4.to(tl.float16).to(tl.float32)
        q8 = _stored_quantize(state8,scale8[:,None],127)
        q4 = _stored_quantize(state4,scale4[:,None],7)
        state8 = q8.to(tl.float32)*scale8[:,None]
        state4 = q4.to(tl.float32)*scale4[:,None]
    qlo = (q8 & 15).to(tl.uint8)
    qhi = ((q8 >> 4) & 15).to(tl.uint8)
    lo0,lo1 = tl.split(tl.reshape(qlo,(TILE,R8//2,2)))
    hi0,hi1 = tl.split(tl.reshape(qhi,(TILE,R8//2,2)))
    pairs8 = tl.arange(0,R8//2)
    tl.store(Lo+row[:,None]*(N8//2)+pairs8[None,:],lo0|(lo1<<4),mask=mp[:,None] & (pairs8[None,:]<N8//2))
    tl.store(Hi+row[:,None]*(N8//2)+pairs8[None,:],hi0|(hi1<<4),mask=mp[:,None] & (pairs8[None,:]<N8//2))
    qn = (q4 & 15).to(tl.uint8)
    f0,f1 = tl.split(tl.reshape(qn,(TILE,R4//2,2)))
    pairs4 = tl.arange(0,R4//2)
    tl.store(Q4+row[:,None]*(N4//2)+pairs4[None,:],f0|(f1<<4),mask=mp[:,None] & (pairs4[None,:]<N4//2))
    tl.store(S8+row,scale8,mask=mp)
    tl.store(S4+row,scale4,mask=mp)


@torch.no_grad()
def scan(x,dt,A,B,C,D,dt_bias,state,permutation,*,layout=BASELINE):
    descriptor = layout_descriptor(layout)
    if layout == BASELINE:
        # This is the original measured v6 path, not an equivalent new kernel.
        return v6.scan(x,dt,A,B,C,D,dt_bias,state,permutation,scale_mode='stored_scale',int4_clip=1.)
    if x.ndim != 4 or x.dtype != torch.float16 or not x.is_cuda:
        raise ValueError('x must be CUDA FP16[batch,tokens,heads,dim]')
    batch,length,heads,dim = x.shape
    if length < 1 or state.shape != (batch,heads,dim,128):
        raise ValueError('Positive sequence and matching packed geometry required')
    validate_state(state,x.device,layout)
    if B.ndim != 4:raise ValueError('B must have four axes')
    groups = B.shape[2]
    if groups < 1 or heads%groups:raise ValueError('Heads must be divisible by groups')
    shapes = ((dt,(batch,length,heads)),(A,(heads,)),(B,(batch,length,groups,128)),
              (C,(batch,length,groups,128)),(D,(heads,)),(dt_bias,(heads,)))
    for tensor,shape in shapes:
        if tuple(tensor.shape) != shape or tensor.device != x.device:
            raise ValueError('Input device/geometry differs')
    if A.dtype != torch.float32 or any(v.dtype != torch.float16 for v in (dt,B,C)):
        raise ValueError('A must be FP32; dt/B/C must be FP16')
    if any(v.dtype not in (torch.float16,torch.float32) for v in (D,dt_bias)):
        raise ValueError('D/dt_bias must be FP16 or FP32')
    if (permutation.dtype != torch.uint8 or tuple(permutation.shape) != (groups,128)
            or permutation.device != x.device):
        raise ValueError('Permutation must be CUDA uint8[groups,128]')
    x,dt,A,B,C,D,dt_bias,permutation = (v.contiguous() for v in (x,dt,A,B,C,D,dt_bias,permutation))
    output = torch.empty_like(x)
    n8,n4,nzero = descriptor['n8'],descriptor['n4'],descriptor['nzero']
    with torch.cuda.device(x.device):
        _tier_scan[(batch,heads,triton.cdiv(dim,RP))](x,dt,A,B,C,D,dt_bias,permutation,
            *(state.tensors[k] for k in ('lo','hi','q4','s8','s4')),output,
            length,heads,dim,groups,n8,n4,nzero,triton.next_power_of_2(n8),
            triton.next_power_of_2(n4),triton.next_power_of_2(nzero),RP,num_warps=4)
    return output,None


@torch.no_grad()
def decode_state(state,permutation=None,*,layout=BASELINE):
    if layout == BASELINE:return v6.decode_state(state,permutation)
    descriptor = layout_descriptor(layout)
    validate_state(state,state.tensors['lo'].device,layout)
    n8,n4 = descriptor['n8'],descriptor['n4']
    shape = state.shape
    values = torch.zeros(shape,dtype=torch.float32,device=state.tensors['lo'].device)
    def unpack(tensor):
        return torch.stack((tensor.int() & 15,tensor.int() >> 4),dim=-1).flatten(-2)
    q8 = (unpack(state.tensors['hi']) << 4) | unpack(state.tensors['lo'])
    q8 = torch.where(q8>127,q8-256,q8)
    q4 = unpack(state.tensors['q4'])
    q4 = torch.where(q4>7,q4-16,q4)
    values[...,:n8] = q8.float()*state.tensors['s8'].float()[...,None]
    values[...,n8:n8+n4] = q4.float()*state.tensors['s4'].float()[...,None]
    if permutation is None:return values
    groups = permutation.shape[0]
    legacy_codec.validate_permutation(permutation,groups)
    batch,heads,dim,_ = shape
    if heads%groups:raise ValueError('Invalid permutation grouping')
    indices = permutation.long().repeat_interleave(heads//groups,0)[None,:,None,:].expand(shape)
    return torch.zeros_like(values).scatter_(-1,indices,values)


validate_finite_result = v6.validate_finite_result


class StatePPLQuantV10(v6.StatePPLQuant):
    """Same native FP16 orchestration and one unchanged static permutation."""
    def __init__(self,model,permutations,*,layout=BASELINE):
        layout_descriptor(layout)
        self.layout = layout
        super().__init__(model,permutations,scale_mode='stored_scale',int4_clip=1.,diagnostic=None)

    @torch.no_grad()
    def reset(self,batch_size=1):
        if self.layout == BASELINE:return super().reset(batch_size)
        if not self._installed:raise RuntimeError('Install controller before resetting')
        if type(batch_size) is not int or batch_size<=0:raise ValueError('Positive integer batch required')
        self.clear()
        for _ in self._mixers:
            conv = torch.zeros(batch_size,4,10240,device=self.device,dtype=torch.float16).transpose(1,2)
            state = allocate_state(batch_size,128,64,128,self.device,layout=self.layout)
            self._cache.append(_LayerCache(conv=conv,state=state))
        self._batch_size = batch_size
        return self

    def storage_descriptor(self):
        # Evidence is computed on demand; no persistent tensor metadata added.
        return dict(**layout_descriptor(self.layout),layers=[dict(
            state_shape=list(entry.state.shape),state_bytes=entry.state.nbytes,
            tensors={key:dict(shape=list(value.shape),dtype=str(value.dtype),
                storage_bytes=value.untyped_storage().nbytes()) for key,value in entry.state.tensors.items()},
            conv_shape=list(entry.conv.shape),conv_storage_bytes=entry.conv.untyped_storage().nbytes())
            for entry in self._cache])

    # cache_breakdown is inherited unchanged, including the exact baseline dict.

    @torch.no_grad()
    def _forward(self, index, mx, u, seqlen, seq_idx, cu_seqlens, inference_params):
        if self.layout == BASELINE:
            return super()._forward(index,mx,u,seqlen,seq_idx,cu_seqlens,inference_params)
        if not self._installed or self._failed:
            raise RuntimeError("Controller is inactive or failed; reset before reusing it")
        if any(value is not None for value in (seqlen, seq_idx, cu_seqlens, inference_params)):
            raise ValueError("Use controller-owned caches; native InferenceParams/varlen are unsupported")
        if (not isinstance(u, torch.Tensor) or u.ndim != 3 or u.dtype != torch.float16
                or u.device != self.device or u.shape[-1] != 4096 or u.shape[1] == 0):
            raise ValueError("Mixer input must be CUDA FP16 [batch,tokens,4096]")
        if not self._cache:
            if index != 0:
                raise RuntimeError("A new request must start with backbone layer zero")
            self.reset(u.shape[0])
        if u.shape[0] != self._batch_size or index != self._expected_layer:
            raise RuntimeError("Batch shape or backbone layer order changed without a reset")
        length = u.shape[1]
        if index == 0:
            if len({entry.tokens for entry in self._cache}) != 1:
                raise RuntimeError("Per-layer cache positions disagree; reset the request")
            self._call_length = length
        elif length != self._call_length:
            raise RuntimeError("Backbone layers received different token counts")
        entry = self._cache[index]
        try:
            is_step = entry.tokens > 0 and length == 1
            # Native step uses rank-two linear/norm inputs; retain those GEMM
            # and adapter-hook shapes as well as its convolution arithmetic.
            projected = mx.in_proj(u.squeeze(1) if is_step else u)
            zxbcdt = projected.unsqueeze(1) if is_step else projected
            z, xbc, dt = torch.split(zxbcdt, (8192, 10240, 128), dim=-1)
            xbc = self._convolution(mx, xbc, entry)
            x, bm, cm = torch.split(xbc, (8192, 1024, 1024), dim=-1)
            batch = u.shape[0]
            x = x.reshape(batch, length, 128, 64)
            bm = bm.reshape(batch, length, 8, 128)
            cm = cm.reshape(batch, length, 8, 128)
            aa = -torch.exp(mx.A_log.float())
            permutation = None if self.permutations is None else self.permutations[index]
            y, stats = scan(x, dt, aa, bm, cm, mx.D, mx.dt_bias, entry.state,
                            permutation=permutation, layout=self.layout)
            if y.dtype != torch.float16 or tuple(y.shape) != (batch, length, 128, 64):
                raise RuntimeError("State codec returned unexpected readout dtype/shape")
            # Keep this module call intact: the external Resurface norm prehook
            # adds its post-D correction here, using the matching mixer input.
            y = y.reshape(batch, length, 8192)
            if is_step:
                y = mx.norm(y[:, 0], z[:, 0])
                out = mx.out_proj(y).unsqueeze(1)
            else:
                y = mx.norm(y, z)
                out = mx.out_proj(y)
            entry.tokens += length
            self._expected_layer = (index + 1) % len(self._mixers)
            if self._expected_layer == 0:
                self._call_length = None
            return out
        except Exception:
            self._failed = True
            raise
