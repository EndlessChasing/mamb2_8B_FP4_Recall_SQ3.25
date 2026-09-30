---
license: other
license_name: Apache-2.0 base weights and StateQuant; GPL-3.0 project runtime and Resurface adapter
license_link: https://github.com/EndlessChasing/mamb2_8B_FP4_Recall_SQ3.25
language:
  - en
pipeline_tag: text-generation
base_model: nvidia/mamba2-8b-3t-4k
tags:
  - mamba-2
  - ssm
  - fp4
  - state-quantization
  - resurface
  - custom-runtime
---

# Mamb2_8B_FP4_Recall_SQ3.25

A member of the **Mamb2_8B_Recall** family, combining a quantized pure
Mamba-2 8B base, a packed 3.25-bit recurrent-state row, a frozen group-ridge
state predictor, and a Resurface adapter trained for this exact configuration.

The complete download contains the packed base weights, tokenizer, state
configuration, adapter, custom inference code, and evaluation evidence.
Generation loads the packed files and decodes the weights into resident FP16.
The published implementation requires the recorded CUDA/Mamba/Triton stack;
it does not provide a packed-resident FP4 matrix-multiplication kernel or a
Transformers `AutoModel.from_pretrained` entry point.

## Measured quality

| Configuration | WikiText-2 validation PPL | Normal MK, correct / 384 | Target-removed matches / 384 |
|---|---:|---:|---:|
| FP4 G16 + ridge SQ3.25, without Resurface | 8.40828258 | 37 | 0 |
| Same frozen base and state + released Resurface | **7.58730687** | **255** | 0 |

The paired normal-MK gain is **56.77 percentage points**;
its paired-bootstrap 95% interval is **51.30 to 61.98
percentage points** (10,000 draws). The release passes the fixed
condition **full PPL strictly below 8.0** and a positive normal-MK gain with a
strictly positive paired-bootstrap lower bound. The independent CPU audit
binds the measured reports, training export, source code, and weight hashes.
Removing the adapter exactly restores the frozen parent reset output and cache.

**Evaluation scope:** PPL uses the pinned `wikitext-2-raw-v1` validation split,
130 windows with up to 2,048 input/target tokens per window, and **264,764
next-token targets**. Recurrent state resets at each window; probabilities are
computed with quantized state updated at every token. The final partial window
is included. This validation set has been used during development; the result
is not an untouched-test estimate. The pinned normal/target-removed MK protocol
contains 768 paired cases, split into 384 ordinary recall prompts and 384 prompts
with the queried binding removed. Each case uses greedy generation of at most
12 tokens. A target-removed match is an output matching the removed value; it
does not measure successful retrieval from the prompt.

The exact arm reports are `evidence/ridge_parent.json` and
`evidence/ridge_resurface.json`; the paired result is
`evidence/comparison.json`, with its independent `evidence/audit.json`.
Hashes in `manifest.json` and `SHA256SUMS` bind these files to the release.

## Storage and runtime memory

| Scope | Exact bytes | Meaning |
|---|---:|---|
| Packed weight tensor payload | 4,638,460,360 | FP4 codes, scale values, and retained FP16 tensors |
| 119 weight safetensors files | 4,638,539,864 | Actual weight containers, about 4.639 GB / 4.320 GiB |
| Decoded resident FP16 weight tensors | 16,473,999,360 | Weights held by this inference implementation |
| Recurrent state, convolution caches, and permutation table | 28,499,968 | One request, batch size 1 |
| Static latent bases | 109,952 | Frozen FP16 basis tensors |
| Static latent scales | 1,792 | Frozen FP16 latent scale tensors |
| Static group-ridge predictors | 3,673,216 | Frozen FP16 predictor tensors |
| Complete persistent state-side tensors | **32,284,928** | All preceding cache/table/basis/scale/predictor tensors |
| Resurface FP16 adapter tensor payload | 2,308,208 | 1,154,104 adapter parameters |
| Complete state-side tensors + adapter | **34,593,136** | About 32.991 MiB, batch size 1 |
| Serialized adapter file | 2,426,943 | File size including serialization metadata |

The weight payload averages about **4.505 bits per original model parameter**,
including scales and retained FP16 tensors. Four-bit codes alone are not the
complete weight size. The table excludes tokenizer, code, evidence, manifest,
and other outer files; `manifest.json` lists their actual sizes.

These values count persistent tensor payloads, not total GPU memory. Activations,
temporary readout corrections, FP32 intermediates, logits, loading buffers,
CPU copies, and allocator reserve need additional memory. Decoding weights in
chunks bounds conversion scratch, but all decoded FP16 weights remain resident
during generation. Packed file size therefore does not establish 4.6 GB VRAM
inference. The complete state-side total includes the predictor and latent
objects in addition to the recurrent row; the label SQ3.25 applies to that row.

## Representation

**Weights.** The base contains 8,236,999,680 parameters. The 114 large matrices
use E2M1 FP4 codes, with signed magnitudes `0, 0.5, 1, 1.5, 2, 3, 4, 6` and
16-element groups along the input axis. Each group has an E4M3FN FP8 block scale;
each matrix has an FP32 global scale. Scale selection minimizes the actual
FP16-decoded weight reconstruction error over the fixed range-search candidates.
This uses an NVFP4-style scaling hierarchy with a custom weight-only range search.
The remaining 393 small tensors remain FP16. No language data or adapter is used
in this weight conversion.

**State.** Each 128-coordinate recurrent row occupies 52 physical bytes:
46 bytes of packed INT8/INT4 codes, two one-byte E4M3FN latent coefficients,
and four bytes for two FP16 dynamic scales. Thus `52 × 8 / 128 = 3.25` physical bits per original
state coordinate. The precision allocation and permutation are fixed per layer
and group. Two latent bytes replace four directly stored INT4 coordinates.
Omitted coordinates contribute through the current exact input term and a
frozen predictor of earlier carry; this is a lossy reconstruction, not lossless
storage or simple permanent zeroing. Predictors, bases, and scales are included
in the memory accounting above. A recurrent row is requantized each token.

**Resurface.** A gated post-D readout correction mixes information across heads
and is trained with the base weights and state representation frozen. The
released adapter is bound to this precise weight conversion, table, basis,
latent scale, predictor, and execution policy. Adapters from other family
members are not interchangeable.

The released adapter was trained from fresh initialization on the exact frozen group-ridge state parent. Training used the pinned numeric TRAIN examples and WikiText TRAIN windows, with 1,536 successful updates; the final export was used without selecting a checkpoint on validation. The frozen FP4 G16 model with FP16 state supplied the prose teacher. See `evidence/training_report.json` and `evidence/training_audit.json` for the exact recipe, data hashes, export and frozen-tensor checks. The state forward matches the deployed ridge codec. Training uses a surrogate backward with a fixed live-mask straight-through estimator and omits latent/predictor derivative terms; it does not implement an exact adjoint of the codec.

The frozen loss weights were `{"closure_budget": 0.006, "closure_budget_coefficient": 10.0, "numeric_ce": 0.25, "prose_ce": 1.0, "prose_closure": 0.1, "prose_kl": 0.5, "temperature": 1.0}`.

## Download and run

### Hugging Face

Download the complete snapshot; the complete GitHub bundle is preserved under
`release/` in the Hub repository:

```bash
hf download EndlessChasing/Mamb2_8B_FP4_Recall_SQ3.25 \
  --revision v0.1.0-fp4g16-sq325-resurface \
  --local-dir Mamb2_8B_FP4_Recall_SQ3.25
cd Mamb2_8B_FP4_Recall_SQ3.25/release
python scripts/infer_fp4_ridge_resurface_v1.py --bundle . --verify-only
```

Verification uses Python's standard library and does not require CUDA.
For generation, install the compatible environment below, then run:

```bash
python scripts/infer_fp4_ridge_resurface_v1.py --bundle . \
  --prompt 'The capital of France is' --max-new-tokens 64
python scripts/infer_fp4_ridge_resurface_v1.py --bundle . --smoke-check
```

`--smoke-check` verifies exact generation for one published ordinary MK case.
Use `--without-adapter` with a prompt to run the frozen unadapted ridge parent.
The complete bundle is sufficient; loading does not require downloading the
original NVIDIA checkpoint.

### GitHub Release

The same payload is distributed as 119 weight assets plus a runtime/configuration
archive. Download and assemble it as follows:

```bash
gh release download v0.1.0-fp4g16-sq325-resurface \
  --repo EndlessChasing/mamb2_8B_FP4_Recall_SQ3.25 \
  --pattern '*-runtime.tar.gz'
tar -xzf mamba2-8b-fp4g16-sq325-resurface-v1-runtime.tar.gz
gh release download v0.1.0-fp4g16-sq325-resurface \
  --repo EndlessChasing/mamb2_8B_FP4_Recall_SQ3.25 \
  --pattern '*.safetensors' \
  --dir mamba2-8b-fp4g16-sq325-resurface-v1/weights
cd mamba2-8b-fp4g16-sq325-resurface-v1
python scripts/infer_fp4_ridge_resurface_v1.py --bundle . --verify-only
```

All 119 safetensors assets and the runtime archive are needed. The verifier
checks the full inventory and SHA-256 hashes, not just filenames.

### Compatible reference environment

The measured CUDA stack is Python **3.10.12**, PyTorch **2.11.0+cu128**
(CUDA 12.8), Triton **3.6.0**, and `mamba-ssm`
**2.3.2.post1**, installed from
[`state-spaces/mamba` commit
`e9594ce1c732d97440f0332fdc43170a2294dbfa`](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa).
The experiments ran on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU.

On Linux with Python 3.10 and a matching CUDA build toolchain, the setup is:
Create the environment beside the bundle so its files do not alter the verified
bundle inventory. The inference script loads the included source directly.

```bash
python -m venv ../mamba2-fp4-env
source ../mamba2-fp4-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'triton==3.6.0' 'numpy==1.26.4' \
  'sentencepiece==0.2.1' 'einops==0.8.2' 'safetensors==0.8.0' \
  'datasets==4.8.5' 'huggingface-hub==0.36.2' 'packaging==26.3' 'ninja==1.13.2'
python -m pip install --no-build-isolation \
  'mamba-ssm @ git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa'
```

The Mamba install may compile an extension for the installed PyTorch/CUDA ABI.
The recorded environment has no separate `causal-conv1d` package. The runtime
also checks the external kernel source hashes, version receipt, precision flags,
and pinned RMSNorm configuration from `manifest.json`; a version string alone
is insufficient for exact replay. TF32 is disabled. Batch-size-1 generation
and the measured 2,048-token quality protocol are the verified scope; upstream
4K training context is not a new long-context validation claim. See
`docs/RESURFACE_MORE_BACKEND_REPLAY.md` for the policy.

## Provenance and license

Source: the pure Mamba-2 checkpoint
[`nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302).
The original model's Apache-2.0 license applies to the redistributed modified
base weights. The StateQuant reference is Apache-2.0. The inherited Recall
runtime, project code, and newly trained Resurface adapter are GPL-3.0.
The full scope and attributions are in `THIRD_PARTY_NOTICES.md` in the complete
bundle and `LICENSES.md` at the Hugging Face snapshot root. Root `LICENSE` is
the GPL-3.0 text; `reference/w4/WEIGHTS_LICENSE.txt` and
`reference/statequant/LICENSE` contain Apache-2.0.

The quantized weights are independently converted from NVIDIA's source. No
Quamba2 checkpoint or research-only weight package is used. This release is an
independent modification and does not imply endorsement by NVIDIA, the Mamba
authors, or the original Resurface authors.

Key identities:

- Original checkpoint SHA-256: `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
- Packed weight manifest SHA-256: `4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07`.
- Tokenizer SHA-256: `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
- Released adapter SHA-256: `46cec16c6b6619808f5bc9af65da06a8889232d2842d552c9d0424ba62349a7b`.
- Full comparison SHA-256: `bfb94f7035c12cc41be1969ee7c4c9407db05aee1103cc56cc7fb7a3128bfd16`.

The weights reload without the original checkpoint, with all 507 decoded
tensor hashes exactly matching the weight ledger used for quality evaluation.
`evidence/packed_reload.json` records that check. These identities establish
which artifact was measured; they do not establish unmeasured hardware speed,
native FP4 inference support, or performance on other benchmarks.

## References

- Waleffe et al., [*An Empirical Study of Mamba-based Language Models*
  (2024)](https://arxiv.org/abs/2406.07887), source model.
- Dao and Gu, [*Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality*
  (2024)](https://arxiv.org/abs/2405.21060), Mamba-2.
- [StateQuant](https://github.com/Oso1106/StateQuant), paired-nibble state
  representation; the pinned reference and Apache-2.0 license are included.
- [Resurface: Multi-Binding Recall Is Latent in Mamba's
  State](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State),
  readout-recovery inspiration; this project's post-D implementation and
  training binding are described above.
- Merity et al., [*Pointer Sentinel Mixture Models*
  (2016)](https://arxiv.org/abs/1609.07843), WikiText.
