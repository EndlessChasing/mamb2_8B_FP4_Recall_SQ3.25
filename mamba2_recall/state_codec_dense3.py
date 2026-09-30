"""Experimental dense Q3.25 carry: all128 coordinates, 48B codes +4B scales.

This is a new codec; the frozen {0,4,8} implementation is not changed. Three
uint8 bitplanes encode signed three-bit values (codes -3..3; -4 unused). Two
64-coordinate blocks have independent FP16 scales. Quantization occurs after
every token; the current readout uses the pre-quantized FP32 update.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

N = 128
TILE = 16
TABLE_KINDS = ('permutation', 'equalizer')
MIN_EXPONENT, MAX_EXPONENT = -8, 8


@dataclass
class Dense3State:
    tensors: dict[str, torch.Tensor]
    shape: tuple[int, int, int, int]
    mode: str = 'dense3'

    @property
    def nbytes(self):
        storages = {(v.device, v.untyped_storage().data_ptr()):
                    v.untyped_storage().nbytes() for v in self.tensors.values()}
        return sum(storages.values())

    def clone(self):
        return Dense3State({k: v.clone() for k, v in self.tensors.items()}, self.shape)

    def zero_(self):
        for value in self.tensors.values():
            value.zero_()
        return self


def allocate_state(batch, heads, dim, dstate, device):
    if dstate != N or min(batch, heads, dim) <= 0:
        raise ValueError('Dense Q3 requires positive geometry and dstate128')
    return Dense3State({'codes': torch.zeros((batch, heads, dim, 3, 16),
                                            dtype=torch.uint8, device=device),
                        'scales': torch.zeros((batch, heads, dim, 2),
                                             dtype=torch.float16, device=device)},
                       (batch, heads, dim, dstate))


def validate_table(table, groups, kind):
    if kind not in TABLE_KINDS or not isinstance(table, torch.Tensor):
        raise ValueError('Dense Q3 table kind must be permutation or equalizer')
    dtype = torch.uint8 if kind == 'permutation' else torch.int8
    if table.dtype != dtype or tuple(table.shape) != (groups, N):
        raise ValueError('Dense Q3 table dtype or shape differs')
    if kind == 'permutation':
        expected = torch.arange(N, device=table.device).expand(groups, N)
        if not torch.equal(table.long().sort(-1).values, expected):
            raise ValueError('Each permutation must contain0..127 exactly')
    elif not bool(((table >= MIN_EXPONENT) & (table <= MAX_EXPONENT)).all()):
        raise ValueError('Power2 exponents must lie in[-8,8]')


def validate_state(state, device):
    if not isinstance(state, Dense3State) or state.mode != 'dense3':
        raise ValueError('Expected a dense Q3 packed state')
    b, h, p, n = state.shape
    spec = {'codes': ((b, h, p, 3, 16), torch.uint8),
            'scales': ((b, h, p, 2), torch.float16)}
    if n != N or set(state.tensors) != set(spec):
        raise ValueError('Dense Q3 state geometry/keys differ')
    for key, (shape, dtype) in spec.items():
        value = state.tensors[key]
        if (tuple(value.shape) != shape or value.dtype != dtype
                or value.device != device or not value.is_contiguous()):
            raise ValueError('Invalid packed state tensor: '+key)


@triton.jit
def _scan(X, DT, A, B, C, D, DB, Table, Codes, Scales, Y,
          L: tl.constexpr, H: tl.constexpr, P: tl.constexpr, G: tl.constexpr,
          POWER2: tl.constexpr, RP: tl.constexpr):
    batch, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tile * RP + tl.arange(0, RP)
    mask = p < P
    n = tl.arange(0, 128)
    row = (batch * H + head) * P + p
    bit = n % 8
    code_base = row[:, None] * 48 + n[None, :] // 8
    u0 = (tl.load(Codes + code_base, mask=mask[:, None], other=0).to(tl.int32)
          >> bit[None, :]) & 1
    u1 = (tl.load(Codes + code_base + 16, mask=mask[:, None], other=0).to(tl.int32)
          >> bit[None, :]) & 1
    u2 = (tl.load(Codes + code_base + 32, mask=mask[:, None], other=0).to(tl.int32)
          >> bit[None, :]) & 1
    unsigned = u0 | (u1 << 1) | (u2 << 2)
    q = unsigned - 8 * (unsigned >= 4).to(tl.int32)
    scales2 = tl.arange(0, 2)
    stored_scale = tl.load(Scales + row[:, None] * 2 + scales2[None, :],
                           mask=mask[:, None], other=0).to(tl.float32)
    scale = tl.reshape(tl.broadcast_to(stored_scale[:, :, None], (RP, 2, 64)), (RP, 128))
    state = q.to(tl.float32) * scale
    group = head // (H // G)
    table = tl.load(Table + group * 128 + n).to(tl.int32)
    if POWER2:
        position = n
        write_factor = tl.exp2(-table.to(tl.float32))
        read_factor = tl.exp2(table.to(tl.float32))
    else:
        position = table
    av = tl.load(A + head).to(tl.float32)
    dv = tl.load(D + head).to(tl.float32)
    bias = tl.load(DB + head).to(tl.float32)
    for t in range(L):
        xoff = ((batch * L + t) * H + head) * P + p
        x = tl.load(X + xoff, mask=mask, other=0).to(tl.float32)
        dt = tl.load(DT + (batch * L + t) * H + head).to(tl.float32) + bias
        dt = tl.where(dt <= 20., tl.math.log(tl.math.exp(dt) + 1.), dt)
        decay = tl.exp(av * dt)
        offset = ((batch * L + t) * G + group) * 128 + position
        bv = tl.load(B + offset).to(tl.float32)
        cv = tl.load(C + offset).to(tl.float32)
        if POWER2:
            bv *= write_factor
            cv *= read_factor
        state = state * decay + (bv[None, :] * dt) * x[:, None]
        out = tl.sum(state * cv[None, :], 1) + x * dv
        tl.store(Y + xoff, out, mask=mask)
        blocks = tl.reshape(state, (RP, 2, 64))
        denominator = tl.maximum(tl.max(tl.abs(blocks), 2) / 3., 1e-8)
        expanded = tl.reshape(tl.broadcast_to(denominator[:, :, None], (RP, 2, 64)), (RP, 128))
        q = tl.minimum(tl.maximum(libdevice.round(state / expanded), -3.), 3.).to(tl.int32)
        stored_scale = denominator.to(tl.float16).to(tl.float32)
        scale = tl.reshape(tl.broadcast_to(stored_scale[:, :, None], (RP, 2, 64)), (RP, 128))
        state = q.to(tl.float32) * scale
    unsigned = q & 7
    byte = tl.arange(0, 16)
    packed0 = tl.sum(tl.reshape(((unsigned >> 0) & 1) << bit[None, :], (RP, 16, 8)), 2)
    packed1 = tl.sum(tl.reshape(((unsigned >> 1) & 1) << bit[None, :], (RP, 16, 8)), 2)
    packed2 = tl.sum(tl.reshape(((unsigned >> 2) & 1) << bit[None, :], (RP, 16, 8)), 2)
    tl.store(Codes + row[:, None] * 48 + byte[None, :], packed0.to(tl.uint8), mask=mask[:, None])
    tl.store(Codes + row[:, None] * 48 + 16 + byte[None, :], packed1.to(tl.uint8), mask=mask[:, None])
    tl.store(Codes + row[:, None] * 48 + 32 + byte[None, :], packed2.to(tl.uint8), mask=mask[:, None])
    scales2 = tl.arange(0, 2)
    tl.store(Scales + row[:, None] * 2 + scales2[None, :], stored_scale, mask=mask[:, None])


@torch.no_grad()
def scan(x, dt, A, B, C, D, dt_bias, state, table, table_kind='permutation'):
    """Mutate true packed carry; table validation is done once by the controller.

    x[B,L,H,P], dt[B,L,H], B/C[B,L,G,128] are FP16; A[H] is FP32.
    Power2 mode stores original-coordinate state divided by2**k, and applies
    B*=2**(-k), C*=2**k after convolution. No FP16 transformed B/C is retained.
    """
    if x.ndim != 4 or x.dtype != torch.float16 or not x.is_cuda:
        raise ValueError('x must be CUDA FP16[batch,tokens,heads,dim]')
    b, length, h, p = x.shape
    if length < 1 or state.shape != (b, h, p, N):
        raise ValueError('Positive sequence length and matching state required')
    validate_state(state, x.device)
    if B.ndim != 4 or B.shape[2] < 1 or h % B.shape[2]:
        raise ValueError('Invalid B/C head grouping')
    g = B.shape[2]
    for tensor, shape in ((dt, (b, length, h)), (A, (h,)), (B, (b, length, g, N)),
                          (C, (b, length, g, N)), (D, (h,)), (dt_bias, (h,))):
        if tensor.device != x.device or tuple(tensor.shape) != shape:
            raise ValueError('Input shape/device mismatch')
    if A.dtype != torch.float32 or any(v.dtype != torch.float16 for v in (dt, B, C)):
        raise ValueError('A must be FP32; dt/B/C must be FP16')
    if any(v.dtype not in (torch.float16, torch.float32) for v in (D, dt_bias)):
        raise ValueError('D and dt_bias must be FP16/FP32')
    expected_dtype = torch.uint8 if table_kind == 'permutation' else torch.int8
    if (table_kind not in TABLE_KINDS or table.dtype != expected_dtype
            or tuple(table.shape) != (g, N) or table.device != x.device):
        raise ValueError('Dense Q3 table differs')
    x, dt, A, B, C, D, dt_bias, table = (v.contiguous() for v in (x, dt, A, B, C, D, dt_bias, table))
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _scan[(b, h, triton.cdiv(p, TILE))](x, dt, A, B, C, D, dt_bias, table,
            state.tensors['codes'], state.tensors['scales'], out,
            length, h, p, g, table_kind == 'equalizer', TILE, num_warps=4)
    return out


@torch.no_grad()
def decode_state(state, table=None, table_kind='permutation'):
    """Diagnostic FP32 decode; full dense state is never persistent in inference."""
    validate_state(state, state.tensors['codes'].device)
    codes = state.tensors['codes']
    lanes = torch.arange(128, device=codes.device)
    unsigned = torch.zeros(state.shape, dtype=torch.int32, device=codes.device)
    for plane in range(3):
        bits = (codes[..., plane, lanes//8].to(torch.int32) >> (lanes % 8)) & 1
        unsigned |= bits << plane
    q = unsigned - (unsigned >= 4).to(torch.int32) * 8
    value = q.float() * state.tensors['scales'].float().repeat_interleave(64, -1)
    if table is None:
        return value
    g = table.shape[0]
    validate_table(table, g, table_kind)
    b, h, p, _ = state.shape
    if h % g or table.device != value.device:
        raise ValueError('Table grouping/device differs')
    expanded = table.repeat_interleave(h//g, 0)[None, :, None, :]
    if table_kind == 'equalizer':
        return torch.ldexp(value, expanded.to(torch.int32))
    indices = expanded.long().expand(b, h, p, N)
    return torch.zeros_like(value).scatter_(-1, indices, value)


@torch.no_grad()
def validate_finite_result(state, output=None):
    """Explicit boundary guard; arbitrary finite inputs can overflow FP16 scales.

    Zero scales from FP16 underflow are permitted and represent a zero carry.
    The codec avoids per-layer host synchronization; evaluation calls this at
    documented boundaries and treats any nonfinite scale/readout as invalid.
    """
    validate_state(state, state.tensors['codes'].device)
    if not bool(torch.isfinite(state.tensors['scales']).all()):
        raise FloatingPointError('Dense Q3 FP16 scale overflow/nonfinite carry')
    if output is not None and not bool(torch.isfinite(output).all()):
        raise FloatingPointError('Dense Q3 nonfinite readout')
    return True
