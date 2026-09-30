"""Packed recurrent StateQuant cache for native grouped Mamba-2.

Derived from StateQuant selective_state_update_pairnib.py, archived commit
22156f74edf428fd192948d75a0dabf3cf152d48, Copyright 2026 Kun Yue, Apache-2.0.
Modifications: sequence scan, native scalar-head geometry, explicit packed state
object, statistics, input validation and diagnostic codec helpers. The s16 scan
follows state-spaces/mamba selective_state_update.py (Tri Dao, Albert Gu, 2024).
See reference/statequant/LICENSE and docs/THIRD_PARTY_NOTICES.md.

Only s16 and sq3p25 exist. Every token reads out its FP32 update BEFORE carrying
FP16 or quantized state. Sequence kernels retain one tile in registers, never a
dense persistent SQ cache. Statistics are sums of abs(FP16 rounded carry) over
tokens and valid P rows, shape [batch,head,ceil(P/16),128].
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

MODES = ('s16', 'sq3p25')
RP = 16
S16_TILE = 4  # Match installed native selective_state_update reduction layout.
N = 128


@dataclass
class PackedState:
    mode: str
    tensors: dict[str, torch.Tensor]
    shape: tuple[int, int, int, int]

    @property
    def nbytes(self):
        # Count actual backing storages, including a view's complete allocation.
        storages = {(t.device, t.untyped_storage().data_ptr()):
                    t.untyped_storage().nbytes() for t in self.tensors.values()}
        return sum(storages.values())

    def clone(self):
        return PackedState(self.mode, {k: v.clone() for k, v in self.tensors.items()}, self.shape)

    def zero_(self):
        for value in self.tensors.values():
            value.zero_()
        return self


def allocate_state(batch, heads, dim, dstate, mode, device):
    if mode not in MODES or dstate != N or min(batch, heads, dim) <= 0:
        raise ValueError('Require s16/sq3p25, dstate=128, and positive geometry')
    shape = (batch, heads, dim, dstate)
    if mode == 's16':
        tensors = {'state': torch.zeros(shape, dtype=torch.float16, device=device)}
    else:
        tensors = {key: torch.zeros((batch, heads, dim, size), dtype=torch.uint8, device=device)
                   for key, size in [('lo', 8), ('hi', 8), ('q4', 32)]}
        tensors.update({key: torch.zeros((batch, heads, dim), dtype=torch.float16, device=device)
                        for key in ('s8', 's4')})
    return PackedState(mode, tensors, shape)


@triton.jit
def _s16_scan(X, DT, A, B, C, D, DB, S, Y, Stats,
              L: tl.constexpr, H: tl.constexpr, P: tl.constexpr, G: tl.constexpr,
              COLLECT: tl.constexpr, TILE: tl.constexpr):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tile * TILE + tl.arange(0, TILE)
    n = tl.arange(0, 128)
    mask = p < P
    offsets = ((b * H + h) * P + p[:, None]) * 128 + n[None, :]
    state = tl.load(S + offsets, mask=mask[:, None], other=0).to(tl.float32)
    av = tl.load(A + h).to(tl.float32)
    dv = tl.load(D + h).to(tl.float32)
    bias = tl.load(DB + h).to(tl.float32)
    group = h // (H // G)
    if COLLECT:
        acc = tl.full((TILE, 128), 0., tl.float32)
    for t in range(L):
        xoff = ((b * L + t) * H + h) * P + p
        x = tl.load(X + xoff, mask=mask, other=0).to(tl.float32)
        dt = tl.load(DT + (b * L + t) * H + h).to(tl.float32) + bias
        dt = tl.where(dt <= 20., tl.math.log(tl.math.exp(dt) + 1.), dt)
        decay = tl.exp(av * dt)
        boff = ((b * L + t) * G + group) * 128 + n
        bv = tl.load(B + boff).to(tl.float32)
        cv = tl.load(C + boff).to(tl.float32)
        dB = bv * dt
        state = state * decay + dB[None, :] * x[:, None]
        out = tl.sum(state * cv[None, :], 1)
        out += x * dv
        tl.store(Y + xoff, out, mask=mask)
        state = state.to(tl.float16).to(tl.float32)
        if COLLECT:
            acc += tl.where(mask[:, None], tl.abs(state), 0.)
    tl.store(S + offsets, state, mask=mask[:, None])
    if COLLECT:
        ntile = tl.cdiv(P, TILE)
        off = ((b * H + h) * ntile + tile) * 128 + n
        tl.store(Stats + off, tl.sum(acc, 0))


@triton.jit
def _sq_scan(X, DT, A, B, C, D, DB, Perm, Lo, Hi, Q4, S8, S4, Y,
             L: tl.constexpr, H: tl.constexpr, P: tl.constexpr, G: tl.constexpr,
             TILE: tl.constexpr):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tile * TILE + tl.arange(0, TILE)
    mp = p < P
    row = (b * H + h) * P + p
    n8 = tl.arange(0, 16)
    n4 = tl.arange(0, 64)
    nd = tl.arange(0, 64)
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
        stated = (bvd[None, :] * dt) * x[:, None]
        out = tl.sum(state8 * cv8[None, :], 1)
        out += tl.sum(state4 * cv4[None, :], 1)
        out += tl.sum(stated * cvd[None, :], 1) + x * dv
        tl.store(Y + xoff, out, mask=mp)
        den8 = tl.maximum(tl.max(tl.abs(state8), 1) / 127., 1e-8)
        den4 = tl.maximum(tl.max(tl.abs(state4), 1) / 7., 1e-8)
        q8 = tl.minimum(tl.maximum(libdevice.round(state8 / den8[:, None]), -127.), 127.).to(tl.int32)
        q4 = tl.minimum(tl.maximum(libdevice.round(state4 / den4[:, None]), -7.), 7.).to(tl.int32)
        scale8 = den8.to(tl.float16).to(tl.float32)
        scale4 = den4.to(tl.float16).to(tl.float32)
        state8 = q8.to(tl.float32) * scale8[:, None]
        state4 = q4.to(tl.float32) * scale4[:, None]
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


def validate_permutation(permutation, groups):
    if permutation is None or permutation.dtype != torch.uint8 or tuple(permutation.shape) != (groups, N):
        raise ValueError('Permutation must be uint8[groups,128]')
    expected = torch.arange(N, device=permutation.device).expand(groups, N)
    if not torch.equal(permutation.long().sort(-1).values, expected):
        raise ValueError('Each permutation row must contain 0..127 exactly once')


def _validate_state(state, device):
    if not isinstance(state, PackedState) or state.mode not in MODES:
        raise ValueError('Invalid packed state')
    b, h, p, n = state.shape
    spec = ({'state': (state.shape, torch.float16)} if state.mode == 's16' else {
        'lo': ((b, h, p, 8), torch.uint8), 'hi': ((b, h, p, 8), torch.uint8),
        'q4': ((b, h, p, 32), torch.uint8), 's8': ((b, h, p), torch.float16),
        's4': ((b, h, p), torch.float16)})
    if n != N or set(state.tensors) != set(spec):
        raise ValueError('State keys or geometry differ')
    for key, (shape, dtype) in spec.items():
        t = state.tensors[key]
        if tuple(t.shape) != tuple(shape) or t.dtype != dtype or t.device != device or not t.is_contiguous():
            raise ValueError(f'Invalid state buffer {key}')


@torch.no_grad()
def scan(x, dt, A, B, C, D, dt_bias, state, permutation=None, collect_stats=False):
    """Mutate compact cache for every token; return (FP16 readouts, statistics).

    Inputs: x[B,L,H,P], dt[B,L,H], B/C[B,L,G,128] in FP16;
    A[H] FP32 negative, D[H] and dt_bias[H] FP16/FP32. Inputs may be views.
    Permutation validation is explicit via validate_permutation at controller
    installation, avoiding a CPU/GPU synchronization on every layer invocation.
    """
    if x.ndim != 4 or x.dtype != torch.float16 or not x.is_cuda:
        raise ValueError('x must be CUDA FP16[batch,tokens,heads,dim]')
    b, length, h, p = x.shape
    if length < 1 or state.shape != (b, h, p, N):
        raise ValueError('Positive sequence length and matching state required')
    _validate_state(state, x.device)
    if B.ndim != 4:
        raise ValueError('B must have four axes')
    g = B.shape[2]
    if g < 1 or h % g:
        raise ValueError('heads must be divisible by B/C groups')
    shapes = ((dt, (b, length, h)), (A, (h,)), (B, (b, length, g, N)),
              (C, (b, length, g, N)), (D, (h,)), (dt_bias, (h,)))
    for t, shape in shapes:
        if tuple(t.shape) != shape or t.device != x.device:
            raise ValueError('Input device/geometry mismatch')
    if A.dtype != torch.float32 or any(t.dtype != torch.float16 for t in (dt, B, C)):
        raise ValueError('A must be FP32; dt/B/C must be FP16')
    if any(t.dtype not in (torch.float16, torch.float32) for t in (D, dt_bias)):
        raise ValueError('D and dt_bias must be FP16 or FP32')
    if collect_stats and state.mode != 's16':
        raise ValueError('Calibration statistics require s16')
    if state.mode == 'sq3p25':
        if permutation is None or permutation.dtype != torch.uint8 or tuple(permutation.shape) != (g, N) or permutation.device != x.device:
            raise ValueError('SQ requires CUDA uint8[groups,128] permutation')
        permutation = permutation.contiguous()
    elif permutation is not None:
        raise ValueError('s16 does not accept a permutation')
    x, dt, A, B, C, D, dt_bias = (v.contiguous() for v in (x, dt, A, B, C, D, dt_bias))
    out = torch.empty_like(x)
    stats = (torch.empty((b, h, triton.cdiv(p, S16_TILE), N), device=x.device, dtype=torch.float32)
             if collect_stats else None)
    grid = (b, h, triton.cdiv(p, RP))
    with torch.cuda.device(x.device):
        if state.mode == 's16':
            _s16_scan[(b, h, triton.cdiv(p, S16_TILE))](x, dt, A, B, C, D, dt_bias, state.tensors['state'], out,
                            stats if stats is not None else out,
                            length, h, p, g, collect_stats, S16_TILE, num_warps=4)
        else:
            _sq_scan[grid](x, dt, A, B, C, D, dt_bias, permutation,
                          *(state.tensors[k] for k in ('lo', 'hi', 'q4', 's8', 's4')), out,
                          length, h, p, g, RP, num_warps=4)
    if stats is not None:
        # Public statistics tiles remain P=16; native arithmetic tiles are P=4.
        target_tiles = triton.cdiv(p, RP)
        if stats.shape[2] != target_tiles * (RP // S16_TILE):
            pad = torch.zeros((b, h, target_tiles * (RP // S16_TILE) - stats.shape[2], N),
                              device=stats.device, dtype=stats.dtype)
            stats = torch.cat((stats, pad), 2)
        stats = stats.reshape(b, h, target_tiles, RP // S16_TILE, N).sum(3)
    return out, stats


def _unpack_nibbles(value):
    return torch.stack((value & 15, (value >> 4) & 15), -1).flatten(-2).to(torch.int32)


@torch.no_grad()
def decode_state(state, permutation=None):
    """Diagnostic full FP32 decode only; do not retain in the inference runtime.

    SQ returns permuted coordinates unless a [G,N] permutation is supplied, in
    which case it returns original coordinates. s16 always original coordinates.
    """
    if state.mode == 's16':
        return state.tensors['state'].float()
    if state.mode != 'sq3p25':
        raise ValueError('Unsupported mode')
    t = state.tensors
    ub = (_unpack_nibbles(t['hi']) << 4) | _unpack_nibbles(t['lo'])
    q8 = ub - (ub > 127).to(torch.int32) * 256
    q4 = _unpack_nibbles(t['q4'])
    q4 = q4 - (q4 > 7).to(torch.int32) * 16
    values = torch.cat((q8.float() * t['s8'].float()[..., None],
                        q4.float() * t['s4'].float()[..., None],
                        torch.zeros((*state.shape[:-1], 48), device=q8.device)), -1)
    if permutation is None:
        return values
    g = permutation.shape[0]
    validate_permutation(permutation, g)
    b, h, p, _ = state.shape
    if h % g:
        raise ValueError('Invalid grouping')
    indices = permutation.long().repeat_interleave(h // g, 0)[None, :, None, :].expand(b, h, p, N)
    return torch.zeros_like(values).scatter_(-1, indices, values)


@torch.no_grad()
def encode_state_(state, dense_permuted):
    """Diagnostic initialization, using half-away and FP32 quant denominators.

    Input coordinates are already permuted for sq3p25. Not used by runtime,
    which allocates zero buffers and updates directly through scan.
    """
    if tuple(dense_permuted.shape) != state.shape or not bool(torch.isfinite(dense_permuted).all()):
        raise ValueError('Diagnostic input must be finite and match state shape')
    if state.mode == 's16':
        state.tensors['state'].copy_(dense_permuted)
        return state
    if state.mode != 'sq3p25':
        raise ValueError('Unsupported mode')
    v = dense_permuted.float()
    v8, v4 = v[..., :16], v[..., 16:80]
    s8 = (v8.abs().amax(-1) / 127.).clamp_min(1e-8)
    s4 = (v4.abs().amax(-1) / 7.).clamp_min(1e-8)
    if not bool(torch.isfinite(s8.half()).all() & torch.isfinite(s4.half()).all()):
        raise ValueError('Scale exceeds FP16 range')
    def quant(x, scale, limit):
        z = x / scale[..., None]
        return (z.sign() * torch.floor(z.abs() + .5)).clamp(-limit, limit).to(torch.int32)
    q8, q4 = quant(v8, s8, 127), quant(v4, s4, 7)
    def pack(q):
        nib = (q & 15).to(torch.uint8)
        return nib[..., 0::2] | (nib[..., 1::2] << 4)
    state.tensors['lo'].copy_(pack(q8))
    state.tensors['hi'].copy_(pack(q8 >> 4))
    state.tensors['q4'].copy_(pack(q4))
    state.tensors['s8'].copy_(s8)
    state.tensors['s4'].copy_(s4)
    return state
