#!/usr/bin/env python3
"""Same-byte W4 state repair layouts with frozen v10 arithmetic.

The four existing layouts delegate directly to v10. The three new layouts
invoke v10's unchanged generic Triton kernel with new compile-time tier counts.
Per-layer layout IDs are a host tuple; no layout tensor, padded carry, lookup
buffer or shadow state is resident on the GPU. Original weights stay frozen.
Derived from Apache-2.0 upstream sources; see docs/THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import triton
import state_ppl_codec_v10 as v10
from mamba2_recall import state_codec as legacy_codec
from mamba2_recall.state_quant import _LayerCache

BASELINE = '32_32_64'
LEGACY_LAYOUTS = tuple(v10.LAYOUTS)
NEW_LAYOUTS = ('36_24_68', '40_16_72', '44_8_76')
LAYOUT_ORDER = (BASELINE, '16_64_48', '8_80_40', '24_48_56', *NEW_LAYOUTS)
LAYOUTS = LAYOUT_ORDER
COUNTS = {**v10.COUNTS, '36_24_68': (36,24,68),
          '40_16_72': (40,16,72), '44_8_76': (44,8,76)}
RP = v10.RP
CACHE_BYTES = 28499968
TABLE_BYTES = 57344


def layout_descriptor(layout=BASELINE):
    if layout not in COUNTS:
        raise ValueError('Unknown fixed W4 state repair tier layout')
    if layout in LEGACY_LAYOUTS:
        return v10.layout_descriptor(layout)
    n8,n4,nzero = COUNTS[layout]
    if n8+n4+nzero != 128 or n8+n4//2 != 48 or n8%2 or n4%2:
        raise RuntimeError('Internal repair layout violates the 52-byte row')
    return dict(layout=layout,n8=n8,n4=n4,nzero=nzero,payload_bytes=48,
        scale_bytes=4,row_bytes=52,resident_layout_metadata_bytes=0,
        tensor_widths=dict(lo=n8//2,hi=n8//2,q4=n4//2))


def validate_layouts(layouts):
    if not isinstance(layouts, (tuple,list)) or len(layouts) != 56:
        raise ValueError('Exactly 56 per-layer layout names are required')
    result = tuple(layouts)
    for layout in result:
        if not isinstance(layout,str):
            raise ValueError('Layout IDs must be strings on the CPU')
        layout_descriptor(layout)
    return result


def uniform_layouts(layout=BASELINE):
    layout_descriptor(layout)
    return (layout,)*56


def allocate_state(batch,heads,dim,dstate,device,*,layout=BASELINE):
    descriptor = layout_descriptor(layout)
    if layout in LEGACY_LAYOUTS:
        return v10.allocate_state(batch,heads,dim,dstate,device,layout=layout)
    if dstate != 128 or any(type(v) is not int or v <= 0 for v in (batch,heads,dim)):
        raise ValueError('Positive integer geometry and 128 state coordinates required')
    shape = (batch,heads,dim,128)
    tensors = {key: torch.zeros((batch,heads,dim,width),dtype=torch.uint8,device=device)
               for key,width in descriptor['tensor_widths'].items()}
    tensors.update({key: torch.zeros((batch,heads,dim),dtype=torch.float16,device=device)
                    for key in ('s8','s4')})
    return legacy_codec.PackedState('w4_repair_v2_'+layout,tensors,shape)


def validate_state(state,device,layout):
    if layout in LEGACY_LAYOUTS:
        return v10.validate_state(state,device,layout)
    descriptor = layout_descriptor(layout)
    batch,heads,dim,dstate = state.shape
    if dstate != 128 or min(batch,heads,dim) <= 0:
        raise ValueError('Invalid packed state geometry')
    mode = 'w4_repair_v2_'+layout
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


@torch.no_grad()
def scan(x,dt,A,B,C,D,dt_bias,state,permutation,*,layout=BASELINE):
    descriptor = layout_descriptor(layout)
    if layout in LEGACY_LAYOUTS:
        return v10.scan(x,dt,A,B,C,D,dt_bias,state,permutation,layout=layout)
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
        v10._tier_scan[(batch,heads,triton.cdiv(dim,RP))](x,dt,A,B,C,D,dt_bias,permutation,
            *(state.tensors[k] for k in ('lo','hi','q4','s8','s4')),output,
            length,heads,dim,groups,n8,n4,nzero,triton.next_power_of_2(n8),
            triton.next_power_of_2(n4),triton.next_power_of_2(nzero),RP,num_warps=4)
    return output,None


@torch.no_grad()
def decode_state(state,permutation=None,*,layout=BASELINE):
    if layout in LEGACY_LAYOUTS:return v10.decode_state(state,permutation,layout=layout)
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


validate_finite_result = v10.validate_finite_result


class StateRepairV2(v10.v6.StatePPLQuant):
    """Own native mixer overrides, one table and a fixed host-only layout tuple.

    Installation, ownership, convolution, request ordering and cleanup inherit
    the frozen StateQuant controller. Norm remains an actual module invocation.
    """
    def __init__(self,model,table,layouts):
        self._layouts = validate_layouts(layouts)
        super().__init__(model,table,scale_mode='stored_scale',int4_clip=1.,diagnostic=None)
        if (self.permutations.numel() != TABLE_BYTES
                or self.permutations.untyped_storage().nbytes() != TABLE_BYTES):
            raise ValueError('The one coordinate table must occupy exactly 57,344 bytes')

    @property
    def layouts(self):
        return self._layouts

    @torch.no_grad()
    def reset(self,batch_size=1):
        if not self._installed:
            raise RuntimeError('Install controller before resetting')
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Positive integer batch required')
        self.clear()
        for layout in self.layouts:
            conv = torch.zeros(batch_size,4,10240,device=self.device,dtype=torch.float16).transpose(1,2)
            state = allocate_state(batch_size,128,64,128,self.device,layout=layout)
            validate_state(state,self.device,layout)
            self._cache.append(_LayerCache(conv=conv,state=state))
        self._batch_size = batch_size
        expected = batch_size*(CACHE_BYTES-TABLE_BYTES)+TABLE_BYTES
        if self.cache_breakdown()['total_bytes'] != expected:
            raise RuntimeError('Per-layer repair cache exceeds the fixed physical budget')
        return self

    def storage_descriptor(self):
        layers = []
        for index,entry in enumerate(self._cache):
            layout = self.layouts[index]
            validate_state(entry.state,self.device,layout)
            layers.append(dict(**layout_descriptor(layout),
                state_shape=list(entry.state.shape),state_bytes=entry.state.nbytes,
                tensors={key:dict(shape=list(value.shape),dtype=str(value.dtype),
                    storage_bytes=value.untyped_storage().nbytes())
                    for key,value in entry.state.tensors.items()},
                conv_shape=list(entry.conv.shape),
                conv_storage_bytes=entry.conv.untyped_storage().nbytes()))
        return dict(format='W4_STATE_REPAIR_STORAGE_V2',layouts=list(self.layouts),
            row_bytes=52,payload_bytes=48,scale_bytes=4,resident_layout_metadata_bytes=0,
            cpu_layout_metadata=dict(kind='tuple[str,...]',entries=56,
                scope='Host orchestration only; Python heap is not GPU recurrent storage'),
            permutation_shape=list(self.permutations.shape),
            permutation_dtype=str(self.permutations.dtype),
            permutation_storage_bytes=self.permutations.untyped_storage().nbytes(),
            total_cache_bytes=self.cache_breakdown()['total_bytes'],layers=layers)

    # Actual tensor accounting, install/close ownership and convolution are
    # inherited unchanged. The following orchestration is v10's nonbaseline
    # _forward, changing only scan selection to the layout for this layer.
    @torch.no_grad()
    def _forward(self, index, mx, u, seqlen, seq_idx, cu_seqlens, inference_params):
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
                            permutation=permutation, layout=self.layouts[index])
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
