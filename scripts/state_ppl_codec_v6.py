#!/usr/bin/env python3
"""Isolated v6 SQ3.25 scale/INT4 clipping variants and causal diagnostics.

The packed kernel is derived from the frozen Apache-2.0 state_codec.py kernel.
It retains that kernel's16/64/48 coordinate reductions and per-token ordering.
Legacy+clip1 delegates it directly. Diagnostics intentionally persist FP32 dense
carry to avoid rounding q*FP16scale products; they are NOT3.25-bit candidates.
See reference/statequant/LICENSE and docs/THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import state_codec as legacy_codec
from mamba2_recall.state_quant import StateQuant, _LayerCache

SCALE_MODES = ('legacy', 'stored_scale')
CLIP_FACTORS = (1.0, .95, .90, .80)
DIAGNOSTICS = ('prune_only', 'quant_only', 'legacy_emulation')
RP = 16
PROTOCOL_SHA = '86d4e8dc85d79c2c867ce15a846c5938893c86d27e67ad174472975e654e9bff'


def validate_options(scale_mode='legacy', int4_clip=1.0, diagnostic=None):
    if scale_mode not in SCALE_MODES or type(int4_clip) not in (float, int) or int4_clip not in CLIP_FACTORS:
        raise ValueError('Unknown frozen scale mode or INT4 clipping factor')
    if diagnostic is not None and diagnostic not in DIAGNOSTICS:
        raise ValueError('Unknown diagnostic')
    if scale_mode == 'legacy' and int4_clip != 1.:
        raise ValueError('Legacy means exact unchanged codec; clipping requires stored_scale')
    if diagnostic is not None and (scale_mode != 'legacy' or int4_clip != 1.):
        raise ValueError('Diagnostics isolate the unchanged legacy factor1 baseline')


def allocate_state(batch, heads, dim, dstate, device, *, diagnostic=None):
    validate_options(diagnostic=diagnostic)
    if diagnostic is None:
        return legacy_codec.allocate_state(batch, heads, dim, dstate, 'sq3p25', device)
    if dstate != 128 or min(batch, heads, dim) <= 0:
        raise ValueError('Positive dimensions and128 state coordinates required')
    shape = (batch, heads, dim, dstate)
    return legacy_codec.PackedState('diagnostic_'+diagnostic,
        {'dense': torch.zeros(shape, dtype=torch.float32, device=device)}, shape)


@triton.jit
def _stored_quantize(value, scale, LIMIT: tl.constexpr):
    # Approximate reciprocal division can move an EXACT half tie (e.g.3.5)
    # below the boundary. Correctly rounded FP32 division implements the
    # declared nearest/half-away rule using the actual stored FP16 scale.
    safe = tl.where(scale > 0., scale, 1.)
    quotient = tl.div_rn(value, safe)
    rounded = libdevice.round(quotient)
    clipped = tl.minimum(tl.maximum(rounded, -LIMIT), LIMIT)
    return tl.where(scale > 0., clipped, 0.).to(tl.int32)


@triton.jit
def _stored_probe(X, S, Q, COUNT: tl.constexpr, LIMIT: tl.constexpr):
    i = tl.program_id(0)*128+tl.arange(0, 128)
    value = tl.load(X+i, mask=i < COUNT, other=0.).to(tl.float32)
    scale = tl.load(S+i, mask=i < COUNT, other=0.).to(tl.float32)
    tl.store(Q+i, _stored_quantize(value, scale, LIMIT), mask=i < COUNT)


def quantize_stored_probe(values, scales, limit):
    """Diagnostic only: exercise the exact production rounding primitive."""
    if (not values.is_cuda or values.dtype != torch.float32 or scales.dtype != torch.float16
            or scales.device != values.device or values.shape != scales.shape or limit not in (7, 127)):
        raise ValueError('Probe expects CUDA FP32 values and same-shape FP16 scales')
    values, scales = values.contiguous(), scales.contiguous()
    result = torch.empty_like(values, dtype=torch.int32)
    _stored_probe[(triton.cdiv(values.numel(), 128),)](values, scales, result, values.numel(), limit)
    return result


@triton.jit
def _variant_scan(X, DT, A, B, C, D, DB, Perm, Lo, Hi, Q4, S8, S4, Dense, Y,
             L: tl.constexpr, H: tl.constexpr, P: tl.constexpr, G: tl.constexpr,
             TILE: tl.constexpr, STORED_SCALE: tl.constexpr, CLIP4: tl.constexpr, DIAG: tl.constexpr):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tile * TILE + tl.arange(0, TILE)
    mp = p < P
    row = (b * H + h) * P + p
    n8 = tl.arange(0, 16)
    n4 = tl.arange(0, 64)
    nd = tl.arange(0, 64)
    if DIAG == 0:
        lo = (tl.load(Lo + row[:, None] * 8 + n8[None, :] // 2,
                      mask=mp[:, None], other=0).to(tl.int32) >> ((n8[None, :] % 2) * 4)) & 15
        hi = (tl.load(Hi + row[:, None] * 8 + n8[None, :] // 2,
                      mask=mp[:, None], other=0).to(tl.int32) >> ((n8[None, :] % 2) * 4)) & 15
        ub = (hi << 4) | lo
        q8 = ub - 256 * (ub > 127).to(tl.int32)
        nib = (tl.load(Q4 + row[:, None] * 32 + n4[None, :] // 2,
                       mask=mp[:, None], other=0).to(tl.int32) >> ((n4[None, :] % 2) * 4)) & 15
        q4 = nib - 16 * (nib > 7).to(tl.int32)
        scale8 = tl.load(S8 + row, mask=mp, other=0).to(tl.float32)
        scale4 = tl.load(S4 + row, mask=mp, other=0).to(tl.float32)
        state8 = q8.to(tl.float32) * scale8[:, None]
        state4 = q4.to(tl.float32) * scale4[:, None]
    else:
        state8 = tl.load(Dense + row[:, None] * 128 + n8[None, :], mask=mp[:, None], other=0)
        state4 = tl.load(Dense + row[:, None] * 128 + 16 + n4[None, :], mask=mp[:, None], other=0)
        q8 = tl.full((TILE, 16), 0, tl.int32)
        q4 = tl.full((TILE, 64), 0, tl.int32)
        scale8 = tl.full((TILE,), 0., tl.float32)
        scale4 = tl.full((TILE,), 0., tl.float32)
    if DIAG == 2:
        stated = tl.load(Dense + row[:, None] * 128 + 80 + nd[None, :],
                        mask=mp[:, None] & (nd[None, :] < 48), other=0)
    av = tl.load(A + h).to(tl.float32)
    dv = tl.load(D + h).to(tl.float32)
    bias = tl.load(DB + h).to(tl.float32)
    group = h // (H // G)
    pos8 = tl.load(Perm + group * 128 + n8).to(tl.int32)
    pos4 = tl.load(Perm + group * 128 + 16 + n4).to(tl.int32)
    posd = tl.load(Perm + group * 128 + 80 + nd, mask=nd < 48, other=0).to(tl.int32)
    for t in range(L):
        xoff = ((b * L + t) * H + h) * P + p
        x = tl.load(X + xoff, mask=mp, other=0).to(tl.float32)
        dt = tl.load(DT + (b * L + t) * H + h).to(tl.float32) + bias
        dt = tl.where(dt <= 20., tl.math.log(tl.math.exp(dt) + 1.), dt)
        decay = tl.exp(av * dt)
        bbase = ((b * L + t) * G + group) * 128
        bv8 = tl.load(B + bbase + pos8).to(tl.float32)
        cv8 = tl.load(C + bbase + pos8).to(tl.float32)
        bv4 = tl.load(B + bbase + pos4).to(tl.float32)
        cv4 = tl.load(C + bbase + pos4).to(tl.float32)
        bvd = tl.load(B + bbase + posd, mask=nd < 48, other=0).to(tl.float32)
        cvd = tl.load(C + bbase + posd, mask=nd < 48, other=0).to(tl.float32)
        state8 = state8 * decay + (bv8[None, :] * dt) * x[:, None]
        state4 = state4 * decay + (bv4[None, :] * dt) * x[:, None]
        if DIAG == 2:
            stated = stated * decay + (bvd[None, :] * dt) * x[:, None]
        else:
            stated = (bvd[None, :] * dt) * x[:, None]
        out = tl.sum(state8 * cv8[None, :], 1)
        out += tl.sum(state4 * cv4[None, :], 1)
        out += tl.sum(stated * cvd[None, :], 1) + x * dv
        tl.store(Y + xoff, out, mask=mp)
        if DIAG == 1:
            state8 = state8.to(tl.float16).to(tl.float32)
            state4 = state4.to(tl.float16).to(tl.float32)
        else:
            if STORED_SCALE:
                # Preserve two FP32 operations (clip multiplication, division)
                # before the FP16 cast, including exact FP16 halfway cases.
                den8 = tl.maximum(tl.div_rn(tl.max(tl.abs(state8), 1), 127.), 1e-8)
                den4 = tl.maximum(tl.div_rn(tl.max(tl.abs(state4), 1) * CLIP4, 7.), 1e-8)
                scale8 = den8.to(tl.float16).to(tl.float32)
                scale4 = den4.to(tl.float16).to(tl.float32)
                q8 = _stored_quantize(state8, scale8[:, None], 127)
                q4 = _stored_quantize(state4, scale4[:, None], 7)
            else:
                # Preserve the original legacy expression and statement order.
                den8 = tl.maximum(tl.max(tl.abs(state8), 1) / 127., 1e-8)
                den4 = tl.maximum(tl.max(tl.abs(state4), 1) / 7., 1e-8)
                q8 = tl.minimum(tl.maximum(libdevice.round(state8 / den8[:, None]), -127.), 127.).to(tl.int32)
                q4 = tl.minimum(tl.maximum(libdevice.round(state4 / den4[:, None]), -7.), 7.).to(tl.int32)
                scale8 = den8.to(tl.float16).to(tl.float32)
                scale4 = den4.to(tl.float16).to(tl.float32)
            state8 = q8.to(tl.float32) * scale8[:, None]
            state4 = q4.to(tl.float32) * scale4[:, None]
        if DIAG == 2:
            stated = stated.to(tl.float16).to(tl.float32)
    if DIAG == 0:
        qlo = (q8 & 15).to(tl.uint8)
        qhi = ((q8 >> 4) & 15).to(tl.uint8)
        lo0, lo1 = tl.split(tl.reshape(qlo, (TILE, 8, 2)))
        hi0, hi1 = tl.split(tl.reshape(qhi, (TILE, 8, 2)))
        pairs8 = tl.arange(0, 8)
        tl.store(Lo + row[:, None] * 8 + pairs8[None, :], lo0 | (lo1 << 4), mask=mp[:, None])
        tl.store(Hi + row[:, None] * 8 + pairs8[None, :], hi0 | (hi1 << 4), mask=mp[:, None])
        qn = (q4 & 15).to(tl.uint8)
        f0, f1 = tl.split(tl.reshape(qn, (TILE, 32, 2)))
        pairs4 = tl.arange(0, 32)
        tl.store(Q4 + row[:, None] * 32 + pairs4[None, :], f0 | (f1 << 4), mask=mp[:, None])
        tl.store(S8 + row, scale8, mask=mp)
        tl.store(S4 + row, scale4, mask=mp)
    else:
        tl.store(Dense + row[:, None] * 128 + n8[None, :], state8, mask=mp[:, None])
        tl.store(Dense + row[:, None] * 128 + 16 + n4[None, :], state4, mask=mp[:, None])
        if DIAG == 2:
            final_dead = stated
        else:
            final_dead = tl.full((TILE, 64), 0., tl.float32)
        tl.store(Dense + row[:, None] * 128 + 80 + nd[None, :], final_dead,
                 mask=mp[:, None] & (nd[None, :] < 48))


@torch.no_grad()
def scan(x, dt, A, B, C, D, dt_bias, state, permutation, *, scale_mode='legacy',
         int4_clip=1.0, diagnostic=None):
    """Return(FP16 readout,None), mutating carried state after EVERY token.

    Current readout always uses the FP32 update before its carry is quantized.
    Clipping changes only the INT4 denominator, never INT8 or current readout.
    Stored-scale zero underflow deterministically writes integer0/carry0.
    """
    validate_options(scale_mode, int4_clip, diagnostic)
    if diagnostic is None and scale_mode == 'legacy' and int4_clip == 1.:
        return legacy_codec.scan(x, dt, A, B, C, D, dt_bias, state, permutation)
    if x.ndim != 4 or x.dtype != torch.float16 or not x.is_cuda:
        raise ValueError('x must be CUDA FP16[batch,tokens,heads,dim]')
    batch, length, heads, dim = x.shape
    if length < 1 or state.shape != (batch, heads, dim, 128):
        raise ValueError('Positive sequence and matching state geometry required')
    if diagnostic is None:
        legacy_codec._validate_state(state, x.device)
        if state.mode != 'sq3p25': raise ValueError('Packed SQ3.25 state required')
    else:
        dense = state.tensors.get('dense')
        if (state.mode != 'diagnostic_'+diagnostic or set(state.tensors) != {'dense'}
                or dense.dtype != torch.float32 or tuple(dense.shape) != state.shape
                or dense.device != x.device or not dense.is_contiguous()):
            raise ValueError('Diagnostic must use the declared dense FP32 backing storage')
    if B.ndim != 4: raise ValueError('B must have4 axes')
    groups = B.shape[2]
    if groups < 1 or heads % groups: raise ValueError('Heads must be divisible by groups')
    shapes = ((dt, (batch, length, heads)), (A, (heads,)),
        (B, (batch, length, groups, 128)), (C, (batch, length, groups, 128)),
        (D, (heads,)), (dt_bias, (heads,)))
    for tensor, shape in shapes:
        if tuple(tensor.shape) != shape or tensor.device != x.device:
            raise ValueError('Input device/geometry differs')
    if A.dtype != torch.float32 or any(v.dtype != torch.float16 for v in (dt, B, C)):
        raise ValueError('A must be FP32; dt/B/C must be FP16')
    if any(v.dtype not in (torch.float16, torch.float32) for v in (D, dt_bias)):
        raise ValueError('D/dt_bias must be FP16 or FP32')
    if (permutation.dtype != torch.uint8 or tuple(permutation.shape) != (groups, 128)
            or permutation.device != x.device):
        raise ValueError('Permutation must be CUDA uint8[groups,128]')
    x, dt, A, B, C, D, dt_bias, permutation = (
        v.contiguous() for v in (x, dt, A, B, C, D, dt_bias, permutation))
    output = torch.empty_like(x)
    if diagnostic is None:
        buffers = [state.tensors[k] for k in ('lo', 'hi', 'q4', 's8', 's4')]
        dense = output  # Compile-time-unused pointer, no additional allocation.
    else:
        dense = state.tensors['dense']
        buffers = [dense]*5  # Compile-time-unused packed pointers.
    diag = 0 if diagnostic is None else DIAGNOSTICS.index(diagnostic)+1
    with torch.cuda.device(x.device):
        _variant_scan[(batch, heads, triton.cdiv(dim, RP))](x, dt, A, B, C, D,
            dt_bias, permutation, *buffers, dense, output, length, heads, dim,
            groups, RP, scale_mode == 'stored_scale', float(int4_clip), diag, num_warps=4)
    return output, None


@torch.no_grad()
def decode_state(state, permutation=None):
    if state.mode == 'sq3p25':
        return legacy_codec.decode_state(state, permutation)
    if state.mode not in tuple('diagnostic_'+v for v in DIAGNOSTICS):
        raise ValueError('Unknown state mode')
    values = state.tensors['dense'].clone()
    if permutation is None: return values
    groups = permutation.shape[0]
    legacy_codec.validate_permutation(permutation, groups)
    batch, heads, dim, _ = state.shape
    if heads % groups: raise ValueError('Invalid permutation grouping')
    indices = permutation.long().repeat_interleave(heads//groups, 0)[None, :, None, :].expand(batch, heads, dim, 128)
    return torch.zeros_like(values).scatter_(-1, indices, values)


def validate_finite_result(state, output=None):
    values = [v for v in state.tensors.values() if v.is_floating_point()]
    if output is not None: values.append(output)
    if values and not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
        raise FloatingPointError('Nonfinite state scale, diagnostic carry or readout')
    return True


class StatePPLQuant(StateQuant):
    """Native orchestration plus one unchanged57,344-byte permutation table."""
    def __init__(self, model, permutations, *, scale_mode='legacy', int4_clip=1.0, diagnostic=None):
        validate_options(scale_mode, int4_clip, diagnostic)
        self.scale_mode, self.int4_clip, self.diagnostic = scale_mode, float(int4_clip), diagnostic
        super().__init__(model, mode='sq3p25', permutations=permutations, collect_stats=False)
        if diagnostic is not None: self.mode = 'diagnostic_'+diagnostic

    @torch.no_grad()
    def reset(self, batch_size=1):
        if not self._installed: raise RuntimeError('Install controller before resetting')
        if type(batch_size) is not int or batch_size <= 0: raise ValueError('Positive integer batch required')
        self.clear()
        for _ in self._mixers:
            conv = torch.zeros(batch_size, 4, 10240, device=self.device, dtype=torch.float16).transpose(1, 2)
            state = allocate_state(batch_size, 128, 64, 128, self.device, diagnostic=self.diagnostic)
            self._cache.append(_LayerCache(conv=conv, state=state))
        self._batch_size = batch_size
        return self

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
                            permutation=permutation, scale_mode=self.scale_mode,
                            int4_clip=self.int4_clip, diagnostic=self.diagnostic)
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

    def assert_finite_cache(self):
        values = [v.conv for v in self._cache]
        values += [t for v in self._cache for t in v.state.tensors.values() if t.is_floating_point()]
        if values and not bool(torch.stack([torch.isfinite(t).all() for t in values]).all()):
            raise FloatingPointError('Nonfinite persistent scale, convolution or diagnostic carry')
        return True

    def cache_breakdown(self):
        conv = sum(v.conv.untyped_storage().nbytes() for v in self._cache)
        state = sum(v.state.nbytes for v in self._cache)
        scales = 0 if self.diagnostic else sum(
            v.state.tensors[k].untyped_storage().nbytes() for v in self._cache for k in ('s8', 's4'))
        table = self.permutations.untyped_storage().nbytes()
        return {'mode': self.mode, 'scale_mode': self.scale_mode, 'int4_clip': self.int4_clip,
            'diagnostic': self.diagnostic, 'is_3p25_candidate': self.diagnostic is None,
            'batch_size': self._batch_size, 'allocated_layers': len(self._cache),
            'conv_fp16_bytes': conv, 'ssm_payload_bytes': state-scales,
            'ssm_scale_bytes': scales, 'ssm_total_bytes': state, 'permutation_bytes': table,
            'total_bytes': conv+state+table, 'calibration_workspace_bytes': 0,
            'diagnostic_dense_fp32_bytes': state if self.diagnostic else 0,
            'row_bytes': 512 if self.diagnostic else 52,
            'tokens_per_layer': [v.tokens for v in self._cache],
            'scope': 'Actual persistent request cache and permutation table; diagnostics use dense FP32 backing storage and are not3.25-bit. Weights, activations, registers and allocator reserve excluded.'}
