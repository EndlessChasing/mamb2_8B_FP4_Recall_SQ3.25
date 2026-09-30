#!/usr/bin/env python3
"""PAIRED-NIBBLE {0,4,8}: store the {0,4,8} quantization as a UNIFORM 4-bit (nibble) layout by splitting each
INT8 outlier into two nibbles -- low nibble in its own slot, high nibble in a paired slot -- and SKIPPING the
dead positions from storage (they are always 0 at storage, so they cost 0 bytes; they are still SCANNED as a
constant-0 state that accumulates one step of dB*x and contributes to the readout -- exact {0,4,8} semantics).
A static per-GROUP d_state permutation orders positions [outliers | int4 | dead] so every load is coalesced
(no gather). Dequant is bit-identical to {0,4,8}, so accuracy == {0,4,8} (no new approximation); the novelty
is the storage/kernel. Config dead=48/int4=64/int8=16: c_lo_out 8B + c_hi_out 8B + c_int4 32B = 48B = 3.0 b/elem
payload + fp16 s4,s8 (0.25) = 3.25 b/elem -- below PERM-PREFIX (5.25b) at full {0,4,8} accuracy, single uniform
nibble representation. Correctness-gated vs the identical PyTorch dequant->scan->requant."""
import os, torch
import triton
import triton.language as tl

D_STATE = 128
K8 = 16          # INT8 outliers (stored as 2 nibbles each)
K4 = 64          # INT4
KD = D_STATE - K8 - K4   # 48 dead (skipped from storage, scanned as 0)


@triton.jit
def _pairnib_kernel(
    clo_ptr, chi_ptr, c4_ptr, s8_ptr, s4_ptr,            # outlier-lo nibbles, outlier-hi nibbles, int4 nibbles, scales
    x_ptr, dt_ptr, dtb_ptr, A_ptr, B_ptr, C_ptr, D_ptr, out_ptr,
    batch, nheads, dim, dstate, k8, k4, kd, nhg,
    stride_clo_b, stride_clo_h, stride_clo_d, stride_clo_n,
    stride_chi_b, stride_chi_h, stride_chi_d, stride_chi_n,
    stride_c4_b, stride_c4_h, stride_c4_d, stride_c4_n,
    stride_sc_b, stride_sc_h, stride_sc_d,
    stride_x_b, stride_x_h, stride_x_d,
    stride_dt_b, stride_dt_h, stride_dt_d,
    stride_dtb_h, stride_dtb_d,
    stride_A_h, stride_A_d, stride_A_n,
    stride_B_b, stride_B_g, stride_B_n,
    stride_C_b, stride_C_g, stride_C_n,
    stride_D_h, stride_D_d,
    stride_out_b, stride_out_h, stride_out_d,
    DT_SOFTPLUS: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr, BLOCK_4: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_b = tl.program_id(1); pid_h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); mask_m = offs_m < dim
    clo_ptr += pid_b * stride_clo_b + pid_h * stride_clo_h
    chi_ptr += pid_b * stride_chi_b + pid_h * stride_chi_h
    c4_ptr += pid_b * stride_c4_b + pid_h * stride_c4_h
    s8_ptr += pid_b * stride_sc_b + pid_h * stride_sc_h
    s4_ptr += pid_b * stride_sc_b + pid_h * stride_sc_h
    x_ptr += pid_b * stride_x_b + pid_h * stride_x_h
    dt_ptr += pid_b * stride_dt_b + pid_h * stride_dt_h
    dtb_ptr += pid_h * stride_dtb_h
    A_ptr += pid_h * stride_A_h
    B_ptr += pid_b * stride_B_b + (pid_h // nhg) * stride_B_g
    C_ptr += pid_b * stride_C_b + (pid_h // nhg) * stride_C_g
    D_ptr += pid_h * stride_D_h
    out_ptr += pid_b * stride_out_b + pid_h * stride_out_h

    s8 = tl.load(s8_ptr + offs_m * stride_sc_d, mask=mask_m, other=0.0).to(tl.float32)
    s4 = tl.load(s4_ptr + offs_m * stride_sc_d, mask=mask_m, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs_m * stride_x_d, mask=mask_m, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptr + offs_m * stride_dt_d, mask=mask_m, other=0.0).to(tl.float32)
    dt += tl.load(dtb_ptr + offs_m * stride_dtb_d, mask=mask_m, other=0.0).to(tl.float32)
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
    D = tl.load(D_ptr + offs_m * stride_D_d, mask=mask_m, other=0.0).to(tl.float32)

    # ---- OUTLIERS [0:k8): INT8 from two nibbles (lo own slot + hi paired slot), coalesced ----
    o = tl.arange(0, BLOCK_O); mo = o < k8; mmo = mask_m[:, None] & mo[None, :]
    bo = o // 2; sho = (o % 2) * 4
    lo = (tl.load(clo_ptr + (offs_m[:, None] * stride_clo_d + bo[None, :] * stride_clo_n), mask=mmo, other=0) >> sho[None, :]) & 0xF
    hi = (tl.load(chi_ptr + (offs_m[:, None] * stride_chi_d + bo[None, :] * stride_chi_n), mask=mmo, other=0) >> sho[None, :]) & 0xF
    ub = (hi.to(tl.int32) << 4) | lo.to(tl.int32)                    # 0..255
    v8 = ub - 256 * (ub > 127).to(tl.int32)                         # signed int8
    Ao = tl.load(A_ptr + (offs_m[:, None] * stride_A_d + o[None, :] * stride_A_n), mask=mmo, other=0.0).to(tl.float32)
    Bo = tl.load(B_ptr + o * stride_B_n, mask=mo, other=0.0).to(tl.float32)
    Co = tl.load(C_ptr + o * stride_C_n, mask=mo, other=0.0).to(tl.float32)
    sto = v8.to(tl.float32) * s8[:, None] * tl.exp(Ao * dt[:, None]) + (Bo[None, :] * dt[:, None]) * x[:, None]
    out = tl.sum(tl.where(mmo, sto * Co[None, :], 0.0), axis=1)

    # ---- INT4 [k8:k8+k4): one nibble, coalesced ----
    f = tl.arange(0, BLOCK_4); mf = f < k4; mmf = mask_m[:, None] & mf[None, :]
    p4 = k8 + f; b4 = f // 2; sh4 = (f % 2) * 4
    nib = (tl.load(c4_ptr + (offs_m[:, None] * stride_c4_d + b4[None, :] * stride_c4_n), mask=mmf, other=0) >> sh4[None, :]) & 0xF
    v4 = nib.to(tl.int32) - 16 * (nib > 7).to(tl.int32)
    A4 = tl.load(A_ptr + (offs_m[:, None] * stride_A_d + p4[None, :] * stride_A_n), mask=mmf, other=0.0).to(tl.float32)
    B4 = tl.load(B_ptr + p4 * stride_B_n, mask=mf, other=0.0).to(tl.float32)
    C4 = tl.load(C_ptr + p4 * stride_C_n, mask=mf, other=0.0).to(tl.float32)
    st4 = v4.to(tl.float32) * s4[:, None] * tl.exp(A4 * dt[:, None]) + (B4[None, :] * dt[:, None]) * x[:, None]
    out += tl.sum(tl.where(mmf, st4 * C4[None, :], 0.0), axis=1)

    # ---- DEAD [k8+k4:dstate): state==0 (not stored); scanned -> st = dB*x; contributes dB*x*C ----
    d = tl.arange(0, BLOCK_D); md = d < kd; mmd = mask_m[:, None] & md[None, :]
    pd = k8 + k4 + d
    Bd = tl.load(B_ptr + pd * stride_B_n, mask=md, other=0.0).to(tl.float32)
    Cd = tl.load(C_ptr + pd * stride_C_n, mask=md, other=0.0).to(tl.float32)
    std = (Bd[None, :] * dt[:, None]) * x[:, None]
    out += tl.sum(tl.where(mmd, std * Cd[None, :], 0.0), axis=1) + x * D
    tl.store(out_ptr + offs_m * stride_out_d, out, mask=mask_m)

    # ---- REQUANT + STORE (outliers -> 2 nibble bufs; int4 -> nibble buf; dead -> skip) ----
    ns8 = tl.maximum(tl.max(tl.where(mmo, tl.abs(sto), 0.0), axis=1) / 127.0, 1e-8)
    q8 = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(sto / ns8[:, None]), -127.0), 127.0).to(tl.int32)
    qlo = tl.where(mo[None, :], q8 & 0xF, 0).to(tl.uint8)
    qhi = tl.where(mo[None, :], (q8 >> 4) & 0xF, 0).to(tl.uint8)
    lo0, lo1 = tl.split(tl.reshape(qlo, (BLOCK_M, BLOCK_O // 2, 2))); plo = lo0 | (lo1 << 4)
    hi0, hi1 = tl.split(tl.reshape(qhi, (BLOCK_M, BLOCK_O // 2, 2))); phi = hi0 | (hi1 << 4)
    op = tl.arange(0, BLOCK_O // 2); mop = op < (k8 // 2)
    tl.store(clo_ptr + (offs_m[:, None] * stride_clo_d + op[None, :] * stride_clo_n), plo, mask=mask_m[:, None] & mop[None, :])
    tl.store(chi_ptr + (offs_m[:, None] * stride_chi_d + op[None, :] * stride_chi_n), phi, mask=mask_m[:, None] & mop[None, :])
    ns4 = tl.maximum(tl.max(tl.where(mmf, tl.abs(st4), 0.0), axis=1) / 7.0, 1e-8)
    q4 = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(st4 / ns4[:, None]), -7.0), 7.0)
    qn = tl.where(mf[None, :], q4.to(tl.int32) & 0xF, 0).to(tl.uint8)
    f0, f1 = tl.split(tl.reshape(qn, (BLOCK_M, BLOCK_4 // 2, 2))); pf = f0 | (f1 << 4)
    fp = tl.arange(0, BLOCK_4 // 2); mfp = fp < (k4 // 2)
    tl.store(c4_ptr + (offs_m[:, None] * stride_c4_d + fp[None, :] * stride_c4_n), pf, mask=mask_m[:, None] & mfp[None, :])
    tl.store(s8_ptr + offs_m * stride_sc_d, ns8.to(tl.float16), mask=mask_m)
    tl.store(s4_ptr + offs_m * stride_sc_d, ns4.to(tl.float16), mask=mask_m)


def pairnib_scan(clo, chi, c4, s8, s4, x, dt, A, B, C, D, dt_bias, dstate=D_STATE, dt_softplus=True):
    Bb, H, dim, _ = clo.shape; G = B.shape[1]
    out = torch.empty(Bb, H, dim, device=clo.device, dtype=torch.float32)
    BLOCK_M = 16
    grid = (triton.cdiv(dim, BLOCK_M), Bb, H)
    _pairnib_kernel[grid](
        clo, chi, c4, s8, s4, x, dt, dt_bias, A, B, C, D, out,
        Bb, H, dim, dstate, K8, K4, KD, H // G,
        clo.stride(0), clo.stride(1), clo.stride(2), clo.stride(3),
        chi.stride(0), chi.stride(1), chi.stride(2), chi.stride(3),
        c4.stride(0), c4.stride(1), c4.stride(2), c4.stride(3),
        s8.stride(0), s8.stride(1), s8.stride(2),
        x.stride(0), x.stride(1), x.stride(2), dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0), dt_bias.stride(1),
        A.stride(0), A.stride(1), A.stride(2), B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2), D.stride(0), D.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        DT_SOFTPLUS=dt_softplus, BLOCK_M=BLOCK_M,
        BLOCK_O=triton.next_power_of_2(K8), BLOCK_4=triton.next_power_of_2(K4), BLOCK_D=triton.next_power_of_2(KD))
    return out


def pack_pairnib(state_perm):
    """state_perm:[B,H,dim,N] in permuted order [outliers 0:K8 | int4 K8:K8+K4 | dead K8+K4:N] -> nibble bufs."""
    pre = state_perm[..., :K8]; mid = state_perm[..., K8:K8 + K4]
    s8 = (pre.abs().amax(-1) / 127.0).clamp(min=1e-8)
    s4 = (mid.abs().amax(-1) / 7.0).clamp(min=1e-8)
    q8 = (pre / s8[..., None]).round().clamp(-127, 127).to(torch.int32)
    qlo = (q8 & 0xF).to(torch.uint8); qhi = ((q8 >> 4) & 0xF).to(torch.uint8)
    clo = (qlo[..., 0::2] | (qlo[..., 1::2] << 4)).contiguous()
    chi = (qhi[..., 0::2] | (qhi[..., 1::2] << 4)).contiguous()
    q4 = ((mid / s4[..., None]).round().clamp(-7, 7).to(torch.int32) & 0xF).to(torch.uint8)
    c4 = (q4[..., 0::2] | (q4[..., 1::2] << 4)).contiguous()
    return clo, chi, c4, s8.half().contiguous(), s4.half().contiguous()


def dequant_pairnib(clo, chi, c4, s8, s4):
    B, H, dim, _ = clo.shape
    lo = torch.stack([(clo & 0xF), (clo >> 4) & 0xF], -1).reshape(B, H, dim, K8).to(torch.int32)
    hi = torch.stack([(chi & 0xF), (chi >> 4) & 0xF], -1).reshape(B, H, dim, K8).to(torch.int32)
    ub = (hi << 4) | lo; v8 = ub - 256 * (ub > 127).to(torch.int32)
    pre = v8.float() * s8[..., None].float()
    n4 = torch.stack([(c4 & 0xF), (c4 >> 4) & 0xF], -1).reshape(B, H, dim, K4).to(torch.int32)
    v4 = n4 - 16 * (n4 > 7).to(torch.int32)
    mid = v4.float() * s4[..., None].float()
    dead = torch.zeros(B, H, dim, KD, device=clo.device)
    return torch.cat([pre, mid, dead], -1)


if __name__ == "__main__":
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    torch.manual_seed(0); dev = "cuda"
    Bb, H, dim, N, G = 2, 128, 80, 128, 8
    state = (torch.randn(Bb, H, dim, N, device=dev) * torch.rand(Bb, H, dim, 1, device=dev) * 3).float()
    state[..., K8 + K4:] = 0.0                                       # dead region starts at 0 (the format)
    clo, chi, c4, s8, s4 = pack_pairnib(state)
    A = -torch.exp(torch.randn(H, dim, N, device=dev)); D = torch.randn(H, dim, device=dev)
    x = torch.randn(Bb, H, dim, device=dev); dt = torch.rand(Bb, H, dim, device=dev) * 0.1
    dtb = torch.randn(H, dim, device=dev) * 0.1
    Bg = torch.randn(Bb, G, N, device=dev); Cg = torch.randn(Bb, G, N, device=dev)
    Bm = Bg.repeat_interleave(H // G, 1); Cm = Cg.repeat_interleave(H // G, 1)

    clo_k, chi_k, c4_k, s8_k, s4_k = clo.clone(), chi.clone(), c4.clone(), s8.clone(), s4.clone()
    out_k = pairnib_scan(clo_k, chi_k, c4_k, s8_k, s4_k, x, dt, A, Bg, Cg, D, dtb)
    deq = dequant_pairnib(clo, chi, c4, s8, s4)
    dtf = dt.float() + dtb.float(); dtf = torch.where(dtf <= 20, torch.log1p(torch.exp(dtf)), dtf)
    st = deq * torch.exp(A.float() * dtf[..., None]) + (Bm.float()[:, :, None, :] * dtf[..., None]) * x.float()[..., None]
    out_r = (st * Cm.float()[:, :, None, :]).sum(-1) + x.float() * D.float()
    rel = (out_k - out_r).abs().max().item() / (out_r.abs().max().item() + 1e-9)

    rha = lambda z: torch.sign(z) * torch.floor(torch.abs(z) + 0.5)
    pre, mid = st[..., :K8], st[..., K8:K8 + K4]
    ns8 = (pre.abs().amax(-1) / 127.0).clamp(min=1e-8).half().float()
    ns4 = (mid.abs().amax(-1) / 7.0).clamp(min=1e-8).half().float()
    q8n = rha(pre / ns8[..., None]).clamp(-127, 127); q4n = rha(mid / ns4[..., None]).clamp(-7, 7)
    lo = torch.stack([(clo_k & 0xF), (clo_k >> 4) & 0xF], -1).reshape(Bb, H, dim, K8).to(torch.int32)
    hi = torch.stack([(chi_k & 0xF), (chi_k >> 4) & 0xF], -1).reshape(Bb, H, dim, K8).to(torch.int32)
    ub = (hi << 4) | lo; q8k = (ub - 256 * (ub > 127).to(torch.int32)).float()
    n4 = torch.stack([(c4_k & 0xF), (c4_k >> 4) & 0xF], -1).reshape(Bb, H, dim, K4).to(torch.int32)
    q4k = (n4 - 16 * (n4 > 7).to(torch.int32)).float()
    m8 = (q8k == q8n).float().mean().item(); md8 = (q8k - q8n).abs().max().item()
    m4 = (q4k == q4n).float().mean().item(); md4 = (q4k - q4n).abs().max().item()

    bpe = (K8 * 4 + K8 * 4 + K4 * 4) / N + 2 * 16.0 / N             # lo+hi outliers + int4 nibbles + scales
    ok = rel < 5e-3 and m8 > 0.99 and m4 > 0.99 and max(md8, md4) <= 1.0
    print(f"[gate-pairnib] out rel={rel:.2e}  int8-match={m8:.4f}(maxD{md8:.0f})  int4-match={m4:.4f}(maxD{md4:.0f})  "
          f"({bpe:.2f} b/elem, dead={KD} skipped) -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("PAIRNIB_GATE_DONE", flush=True)
