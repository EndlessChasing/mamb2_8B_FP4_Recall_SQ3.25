# FP4 weight comparison v1

User direction: confirm current W4 is INT4 and try FP4. This is a new weight
format experiment after the completed F4 state-table experiment. Preserve
all historical sources and results. No Resurface or weight training occurs.

## Fixed comparisons

Run full paired WikiText-2 validation for each of three weight formats with
both FP16 state and the frozen F4 top16 Q3.25 state table:

1. Existing affine INT4 G128, FP16 scale and offset: control.
2. FP4 E2M1 G64, one FP16 scale, no offset: equal matrix payload budget.
3. FP4 E2M1 G16, FP8 E4M3 block scale and one FP32 global scale per tensor:
   NVFP4 scaling hierarchy, with weight-only MSE range selection.

The first FP4 variant uses 4.25 bits per quantized weight including scales,
exactly matching the existing INT4 matrix payload. The second uses 4.5 bits
plus 32 bits per matrix; it is not an equal-size comparison. Neither recipe
claims native FP4 GEMM execution, full NVIDIA recipe equivalence, or lower
resident VRAM. Actual inference weights and activations remain FP16.

All 114 original quantized matrices are covered (8,233,418,752 weights),
including embedding and lm_head. The remaining 393 tensors retain their
original FP16 bytes (7,161,856 bytes). Original source checkpoint SHA-256:
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
Quantize directly from this source cast to FP16, never from decoded INT4.
The INT4 control uses manifest
`3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05`.

## Normative FP4 representation

Nibble bits 0..2 index magnitudes `[0, .5, 1, 1.5, 2, 3, 4, 6]`; bit 3 is
the sign. Pack the first input-axis element into the low nibble. Quantize
normalized values with round-to-nearest, ties to even code LSB, saturation
at magnitude 6. Canonicalize encoded zero to positive zero; decoder must
support both zero codes. Zero groups use zero scale and zero codes.

Use these fixed scale multipliers in this order:
`[1, .99, .98, .97, .96, .95, .94, .92, .90, 1.25, 1.5]`.
For each block, choose minimum SSE against actual decoded FP16 weights;
exact ties retain the first multiplier. This is source-weight calibration,
without language text or validation-driven scale tuning. The last two
choices allow a wider represented range for the nonuniform FP4 codebook.

G64: scale = FP16(FP32(block_absmax * multiplier / 6)). Use the stored scale
for code selection. Normative decode is FP32(code_value * stored_scale),
then FP16 rounding. Constant nonzero groups are handled by this same rule.

G16: global_scale = FP32(tensor_absmax / (448 * 6)); for a zero tensor use
global_scale = 1. For each multiplier, block_scale =
E4M3FN(clamp(FP32(block_absmax * multiplier / (6 * global_scale)), 0, 448)).
For code selection compute effective_scale = FP32(stored_block_scale *
stored_global_scale), then normalized = FP32(source / effective_scale).
Use these actual stored scales, not unrounded candidates. Normative decode
performs FP32(code_value * block_scale), then FP32(result * global_scale),
then FP16 rounding. Scale underflow uses zero codes. Reject nonfinite inputs,
scales or decoded weights; do not silently substitute another recipe.

Both tensor shapes in the real model are divisible by their block sizes.
Bounded fixture support may pad a final group, excluding padding from maxima,
SSE and parameter counts and recording physical padding in payload bytes.

## Materialization and audit

GPU disk is nearly full. Do not create two additional full checkpoints.
Actually pack and unpack every quantized chunk in memory before assigning
the decoded FP16 weights to the model. Record total physical packed bytes,
per-matrix codes/scales/global-scale hashes, decoded FP16 tensor hashes,
group counts, selected-multiplier counts and SSE. These are measured payload
bytes, not an exported checkpoint-file size. Preserve a complete deterministic
source/recipe receipt so the package can be materialized later if justified.
No high-precision residual is permitted. Expanded FP16 weights total
16,473,999,360 bytes in all arms and are a separate residency scope.

Before model evaluation, bounded independent checks must cover all E2M1
codes, both zeros, nearest-even midpoint and neighboring cases, scale
rounding/underflow, packing order, padding, exact byte accounting and GPU
quantize/pack/unpack/decode parity against an independent CPU decoder.
Reuse the unchanged admitted state kernels. Do not expand state-kernel tests.
Freeze new source identities before GPU quality evaluation.

## Full quality protocol and controls

Tokenizer SHA
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
Use the same 130 reset windows / 264,764 targets, length 2048, full 256K
vocabulary, head chunks 64, TF32 off and pinned native 16-warp RMSNorm.
Validation token stream SHA (int64 little endian):
`5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027`.
Report this as historically exposed validation, not untouched test quality.

Q3.25 keeps the frozen top16 table, SHA
`214b47a4dfdc20fce4aa552f954e3f0af84b49edbfc14cbcc1569946a4777ef8`,
from `artifacts/repair_v5/screen_v1/selected_calibration.pt`, payload SHA
`7ac1c824936adbcf372570f2f49a22b02a9d42f007e7ae506528e84392560fb2`.
No per-FP4 table optimization in this comparison. State storage is exactly
28,499,968 bytes, including convolution and one table. S16 cache is
122,028,032 bytes. Do not introduce Q8.

Run INT4 S16 and INT4 Q3.25 controls first, requiring exact archived full PPL
and per-window NLL: 8.014129751718814 and 9.012334876908318 respectively.
Then run G64 S16/Q3.25 and G16 S16/Q3.25, in that fixed order. Check repeated
reset outputs/cache, finite losses, actual physical cache, absent adapters,
unchanged native functions/backend and all 507 actual parameter hashes
before/after each format's paired evaluation. For each FP4 format, repeat
the S16 probe after Q3.25 removal to prove restoration without a redundant
second full corpus. Any integrity failure is fatal; numerical nonfinite
quality may be preserved as a failed arm with clear scope.

Independent CPU audit recomputes full NLL/PPL, source/recipe/storage identities,
controls and the strict unadapted Q3.25 PPL <8.4 gate. A lower SSE alone cannot
establish quality. A target miss does not permit Resurface. A successful
FP4/Q3.25 configuration needs a separately bound fresh Resurface stage;
do not reuse old adapters or publish from this comparison.

## Primary format references

- [CUDA E2M1 type](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html)
- [NVIDIA NVFP4 hierarchy](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
- [TensorRT quantized types and FP4 rounding](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/work-quantized-types.html#quantization-schemes)
