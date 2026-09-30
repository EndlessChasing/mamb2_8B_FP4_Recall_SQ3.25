"""Differentiable training companion to the exact packed SQ3.25 scan.

The forward output is the deployment codec's output, bit for bit. Backward uses
an explicitly approximate straight-through derivative: identity on the 80 live
state coordinates and zero through the 48 pruned carry coordinates. Scales and
rounding are detached. Current-token readout gradients include all 128 lanes.

Backward recomputes the actual dequantized carries, checkpoints live coordinates
every ``chunk_size`` tokens, and uses one chunk of temporary state history. This
is training workspace, never part of the deployed recurrent cache. The frozen
A, D and dt_bias parameters deliberately receive no parameter gradients.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .state_codec import allocate_state, scan, validate_permutation

TILE = 16


@triton.jit
def _carry_checkpoints(X, DT, A, B, DB, Perm, CP,
                       L: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
                       G: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
                       TILE: tl.constexpr):
    batch, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tile * TILE + tl.arange(0, TILE)
    mask = p < P
    n8, n4 = tl.arange(0, 16), tl.arange(0, 64)
    group = h // (H // G)
    pos8 = tl.load(Perm + group * 128 + n8).to(tl.int32)
    pos4 = tl.load(Perm + group * 128 + 16 + n4).to(tl.int32)
    a, bias = tl.load(A + h), tl.load(DB + h).to(tl.float32)
    s8 = tl.full((TILE, 16), 0., tl.float32)
    s4 = tl.full((TILE, 64), 0., tl.float32)
    for chunk in range(NC):
        base = (((batch * NC + chunk) * H + h) * P + p[:, None]) * 80
        tl.store(CP + base + n8[None, :], s8, mask=mask[:, None])
        tl.store(CP + base + 16 + n4[None, :], s4, mask=mask[:, None])
        for j in range(K):
            t = chunk * K + j
            if t < L:
                x = tl.load(X + ((batch * L + t) * H + h) * P + p,
                            mask=mask, other=0).to(tl.float32)
                dt = tl.load(DT + (batch * L + t) * H + h).to(tl.float32) + bias
                dt = tl.where(dt <= 20., tl.math.log(tl.math.exp(dt) + 1.), dt)
                decay = tl.exp(a * dt)
                bbase = ((batch * L + t) * G + group) * 128
                b8 = tl.load(B + bbase + pos8).to(tl.float32)
                b4 = tl.load(B + bbase + pos4).to(tl.float32)
                s8 = s8 * decay + (b8[None, :] * dt) * x[:, None]
                s4 = s4 * decay + (b4[None, :] * dt) * x[:, None]
                den8 = tl.maximum(tl.max(tl.abs(s8), 1) / 127., 1e-8)
                den4 = tl.maximum(tl.max(tl.abs(s4), 1) / 7., 1e-8)
                q8 = tl.minimum(tl.maximum(libdevice.round(s8 / den8[:, None]), -127.), 127.)
                q4 = tl.minimum(tl.maximum(libdevice.round(s4 / den4[:, None]), -7.), 7.)
                s8 = q8 * den8.to(tl.float16).to(tl.float32)[:, None]
                s4 = q4 * den4.to(tl.float16).to(tl.float32)[:, None]


@triton.jit
def _chunk_history(X, DT, A, B, DB, Perm, CP, Hist,
                   L: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
                   G: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
                   CHUNK, TILE: tl.constexpr):
    batch, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tile * TILE + tl.arange(0, TILE)
    mask = p < P
    n8, n4 = tl.arange(0, 16), tl.arange(0, 64)
    group = h // (H // G)
    pos8 = tl.load(Perm + group * 128 + n8).to(tl.int32)
    pos4 = tl.load(Perm + group * 128 + 16 + n4).to(tl.int32)
    a, bias = tl.load(A + h), tl.load(DB + h).to(tl.float32)
    base = (((batch * NC + CHUNK) * H + h) * P + p[:, None]) * 80
    s8 = tl.load(CP + base + n8[None, :], mask=mask[:, None], other=0)
    s4 = tl.load(CP + base + 16 + n4[None, :], mask=mask[:, None], other=0)
    for j in range(K):
        t = CHUNK * K + j
        if t < L:
            off = (((batch * K + j) * H + h) * P + p[:, None]) * 80
            tl.store(Hist + off + n8[None, :], s8, mask=mask[:, None])
            tl.store(Hist + off + 16 + n4[None, :], s4, mask=mask[:, None])
            x = tl.load(X + ((batch * L + t) * H + h) * P + p,
                        mask=mask, other=0).to(tl.float32)
            dt = tl.load(DT + (batch * L + t) * H + h).to(tl.float32) + bias
            dt = tl.where(dt <= 20., tl.math.log(tl.math.exp(dt) + 1.), dt)
            decay = tl.exp(a * dt)
            bbase = ((batch * L + t) * G + group) * 128
            b8 = tl.load(B + bbase + pos8).to(tl.float32)
            b4 = tl.load(B + bbase + pos4).to(tl.float32)
            s8 = s8 * decay + (b8[None, :] * dt) * x[:, None]
            s4 = s4 * decay + (b4[None, :] * dt) * x[:, None]
            den8 = tl.maximum(tl.max(tl.abs(s8), 1) / 127., 1e-8)
            den4 = tl.maximum(tl.max(tl.abs(s4), 1) / 7., 1e-8)
            q8 = tl.minimum(tl.maximum(libdevice.round(s8 / den8[:, None]), -127.), 127.)
            q4 = tl.minimum(tl.maximum(libdevice.round(s4 / den4[:, None]), -7.), 7.)
            s8 = q8 * den8.to(tl.float16).to(tl.float32)[:, None]
            s4 = q4 * den4.to(tl.float16).to(tl.float32)[:, None]


@triton.jit
def _chunk_backward(X, DT, A, B, C, D, DB, Perm, Hist, Adj,
                    DY, DX, DDT, DBV, DCV,
                    L: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
                    G: tl.constexpr, K: tl.constexpr, CHUNK,
                    TILE: tl.constexpr):
    batch, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tile * TILE + tl.arange(0, TILE)
    mask = p < P
    n8, n4, nd = tl.arange(0, 16), tl.arange(0, 64), tl.arange(0, 64)
    group = h // (H // G)
    pos8 = tl.load(Perm + group * 128 + n8).to(tl.int32)
    pos4 = tl.load(Perm + group * 128 + 16 + n4).to(tl.int32)
    posd = tl.load(Perm + group * 128 + 80 + nd, mask=nd < 48, other=0).to(tl.int32)
    a = tl.load(A + h)
    bias, dv = tl.load(DB + h).to(tl.float32), tl.load(D + h).to(tl.float32)
    adjbase = ((batch * H + h) * P + p[:, None]) * 80
    adj8 = tl.load(Adj + adjbase + n8[None, :], mask=mask[:, None], other=0)
    adj4 = tl.load(Adj + adjbase + 16 + n4[None, :], mask=mask[:, None], other=0)
    for rev in range(K):
        j = K - 1 - rev
        t = CHUNK * K + j
        if t < L:
            xoff = ((batch * L + t) * H + h) * P + p
            x = tl.load(X + xoff, mask=mask, other=0).to(tl.float32)
            gy = tl.load(DY + xoff, mask=mask, other=0).to(tl.float32)
            rawdt = tl.load(DT + (batch * L + t) * H + h).to(tl.float32) + bias
            dt = tl.where(rawdt <= 20., tl.math.log(tl.math.exp(rawdt) + 1.), rawdt)
            sigmoid = tl.where(rawdt <= 20., 1. / (1. + tl.exp(-rawdt)), 1.)
            decay = tl.exp(a * dt)
            off = (((batch * K + j) * H + h) * P + p[:, None]) * 80
            prev8 = tl.load(Hist + off + n8[None, :], mask=mask[:, None], other=0)
            prev4 = tl.load(Hist + off + 16 + n4[None, :], mask=mask[:, None], other=0)
            bbase = ((batch * L + t) * G + group) * 128
            b8 = tl.load(B + bbase + pos8).to(tl.float32)
            b4 = tl.load(B + bbase + pos4).to(tl.float32)
            bd = tl.load(B + bbase + posd, mask=nd < 48, other=0).to(tl.float32)
            c8 = tl.load(C + bbase + pos8).to(tl.float32)
            c4 = tl.load(C + bbase + pos4).to(tl.float32)
            cd = tl.load(C + bbase + posd, mask=nd < 48, other=0).to(tl.float32)
            # STE future derivative is identity on live lanes, zero on dead lanes.
            v8 = gy[:, None] * c8[None, :] + adj8
            v4 = gy[:, None] * c4[None, :] + adj4
            vd = gy[:, None] * cd[None, :]
            h8 = prev8 * decay + (b8[None, :] * dt) * x[:, None]
            h4 = prev4 * decay + (b4[None, :] * dt) * x[:, None]
            hd = (bd[None, :] * dt) * x[:, None]
            dx = tl.sum(v8 * (b8[None, :] * dt), 1)
            dx += tl.sum(v4 * (b4[None, :] * dt), 1)
            dx += tl.sum(vd * (bd[None, :] * dt), 1) + gy * dv
            tl.store(DX + xoff, dx, mask=mask)
            ddelta = tl.sum(v8 * (prev8 * (a * decay) + b8[None, :] * x[:, None]), 1)
            ddelta += tl.sum(v4 * (prev4 * (a * decay) + b4[None, :] * x[:, None]), 1)
            ddelta += tl.sum(vd * (bd[None, :] * x[:, None]), 1)
            ddelta = tl.sum(tl.where(mask, ddelta, 0.), 0) * sigmoid
            tl.atomic_add(DDT + (batch * L + t) * H + h, ddelta)
            factor = tl.where(mask, dt * x, 0.)
            db8, db4 = tl.sum(v8 * factor[:, None], 0), tl.sum(v4 * factor[:, None], 0)
            dbd = tl.sum(vd * factor[:, None], 0)
            dc8, dc4 = tl.sum(h8 * gy[:, None], 0), tl.sum(h4 * gy[:, None], 0)
            dcd = tl.sum(hd * gy[:, None], 0)
            tl.atomic_add(DBV + bbase + pos8, db8)
            tl.atomic_add(DBV + bbase + pos4, db4)
            tl.atomic_add(DBV + bbase + posd, dbd, mask=nd < 48)
            tl.atomic_add(DCV + bbase + pos8, dc8)
            tl.atomic_add(DCV + bbase + pos4, dc4)
            tl.atomic_add(DCV + bbase + posd, dcd, mask=nd < 48)
            adj8, adj4 = v8 * decay, v4 * decay
    tl.store(Adj + adjbase + n8[None, :], adj8, mask=mask[:, None])
    tl.store(Adj + adjbase + 16 + n4[None, :], adj4, mask=mask[:, None])


class _StateQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D, dt_bias, permutation, chunk_size):
        tensors = tuple(v.contiguous() for v in (x, dt, A, B, C, D, dt_bias, permutation))
        x, dt, A, B, C, D, dt_bias, permutation = tensors
        batch, length, heads, dim = x.shape
        state = allocate_state(batch, heads, dim, 128, 'sq3p25', x.device)
        y, _ = scan(x, dt, A, B, C, D, dt_bias, state, permutation=permutation)
        ctx.save_for_backward(*tensors)
        ctx.chunk_size = chunk_size
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, dt, A, B, C, D, dt_bias, permutation = ctx.saved_tensors
        batch, length, heads, dim = x.shape
        groups, k = B.shape[2], ctx.chunk_size
        nc = triton.cdiv(length, k)
        cp = torch.empty((batch, nc, heads, dim, 80), device=x.device, dtype=torch.float32)
        hist = torch.empty((batch, k, heads, dim, 80), device=x.device, dtype=torch.float32)
        adj = torch.zeros((batch, heads, dim, 80), device=x.device, dtype=torch.float32)
        dx = torch.empty_like(x, dtype=torch.float32)
        ddt = torch.zeros_like(dt, dtype=torch.float32)
        db = torch.zeros_like(B, dtype=torch.float32)
        dc = torch.zeros_like(C, dtype=torch.float32)
        dy = grad_y.contiguous()
        grid = (batch, heads, triton.cdiv(dim, TILE))
        with torch.cuda.device(x.device):
            _carry_checkpoints[grid](x, dt, A, B, dt_bias, permutation, cp,
                                     length, heads, dim, groups, k, nc, TILE, num_warps=4)
            for chunk in reversed(range(nc)):
                _chunk_history[grid](x, dt, A, B, dt_bias, permutation, cp, hist,
                                     length, heads, dim, groups, k, nc, chunk, TILE, num_warps=4)
                _chunk_backward[grid](x, dt, A, B, C, D, dt_bias, permutation, hist, adj,
                                      dy, dx, ddt, db, dc, length, heads, dim, groups,
                                      k, chunk, TILE, num_warps=4)
        return dx.to(x.dtype), ddt.to(dt.dtype), None, db.to(B.dtype), dc.to(C.dtype), None, None, None, None


def training_workspace_bytes(batch, length, heads=128, dim=64, groups=8, chunk_size=32):
    """Explicit scratch allocation count, excluding inputs/output and CUDA allocator."""
    nc = triton.cdiv(length, chunk_size)
    carries = batch * (nc + chunk_size + 1) * heads * dim * 80 * 4
    gradients = batch * length * (heads * dim + heads + 2 * groups * 128) * 4
    return {'carry_scratch_bytes': carries, 'gradient_scratch_bytes': gradients,
            'total_scratch_bytes': carries + gradients}


def scan_training(x, dt, A, B, C, D, dt_bias, permutation,
                  mode='sq3p25', chunk_size=32):
    """Exact SQ3.25 forward with the documented approximate STE backward.

    Whole sequence with fresh zero state only. Frozen A/D/dt_bias are required;
    this routine intentionally does not train the base SSM parameters. Validation
    of each permutation's contents is the controller's responsibility.
    """
    if mode != 'sq3p25':
        raise ValueError('Training scan supports sq3p25 only; Q8 is excluded')
    if not isinstance(chunk_size, int) or chunk_size < 1 or chunk_size > 256:
        raise ValueError('chunk_size must be an integer in [1,256]')
    if any(v.requires_grad for v in (A, D, dt_bias)):
        raise ValueError('A, D, and dt_bias must be frozen')
    return _StateQuantSTE.apply(x, dt, A, B, C, D, dt_bias, permutation, chunk_size)
