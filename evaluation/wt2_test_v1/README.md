# Complete WT2 test evaluation of the fixed published FP4 model

This directory adds evaluation evidence for the unchanged
`v0.1.0-fp4g16-sq325-resurface` model. It does not replace any file inside the
original model bundle. The fixed-checkpoint protocol was frozen before running
both complete arms; no training, recalibration, checkpoint selection, or
test-dependent stopping was performed.

| Arm | Pooled-token PPL | Total NLL | Targets | Windows |
|---|---:|---:|---:|---:|
| Without Resurface | 8.327234815511874 | 637900.5426273346 | 300963 | 147 |
| Released Resurface | 7.508139013005539 | 606737.686252594 | 300963 | 147 |

All 147 matched windows have lower NLL with the adapter. The relative PPL change
is −9.836348087369085%. The last window contains 1,955 next-token targets.
This evaluation does not include a new MK run; recall remains the original
published synthetic CONFIRM result in the sealed release.

## Included evidence and its scope

- `comparison.json`: exact complete paired GPU report and model/backend bindings.
- `without_resurface.json`, `resurface.json`: exact per-arm reports, all per-window
  scores and token hashes, recorded cache/weight identities, and storage receipts.
- `cpu_audit_v1.json`: independent CPU audit of token/window coverage and score
  arithmetic, plus sealed-ledger checks of the recorded tensor identities.
- `inventory.json`: hashes of these reports and the reproduction sources.

The CPU audit does not recompute GPU logits or reread the large packed shards.
Dataset text hashes/fingerprints in that audit are recorded metadata. Its input
token stream was verified in full; raw token IDs are deliberately excluded from
this public update and can be regenerated from the pinned public dataset below.
Original reports have not been shortened or rewritten.

The adapter used separate official TRAIN data. This is the official WT2 `test`
split scored with the fixed published 8B configuration, after publication and
without test selection. Earlier 2.7B experiments in this project used the WT2
test text; it is therefore not an untouched test for the whole research project.
Base-model pretraining contamination and cross-split text duplication have not
been independently audited. See the exact
[protocol](../../docs/FP4_G16_SQ325_RESURFACE_WT2_TEST_V1_PROTOCOL.md).

## Reproduce the GPU evaluation

Use this repository's current documentation/evaluation source separately from
the immutable model bundle. The measured runner imports the runtime from the
bundle and never installs or edits source inside it. Obtain the current source
from GitHub, or download the HF root `scripts/`, `docs/`, and `evaluation/`
overlay. The measured runner and protocol hashes are pinned in the reports and
inventory, independently of the mutable `main` branch name.

From the evaluation source root:

```bash
SQ325_EVAL_SOURCE="$PWD"
hf download EndlessChasing/Mamb2_8B_FP4_Recall_SQ3.25 \
  --revision v0.1.0-fp4g16-sq325-resurface \
  --local-dir ../fp4-sq325-frozen-model
SQ325_RELEASE_BUNDLE="$(cd ../fp4-sq325-frozen-model/release && pwd)"
python "$SQ325_RELEASE_BUNDLE/scripts/infer_fp4_ridge_resurface_v1.py" \
  --bundle "$SQ325_RELEASE_BUNDLE" --verify-only
```

Install the CUDA/Mamba/Triton environment documented in the root model card,
in a virtual environment outside the sealed bundle. The exact GPU backend
policy is checked before inference. The measured runtime expands the packed
weights into 16,473,999,360 bytes of resident FP16 weights; packed file size is
not its VRAM requirement.

Both output locations below must be fresh. The runner regenerates the full
token stream locally and executes both arms without a score threshold:

```bash
python "$SQ325_EVAL_SOURCE/scripts/run_fp4_sq325_wt2_test_v1.py" \
  --bundle "$SQ325_RELEASE_BUNDLE" \
  --out-dir "$SQ325_EVAL_SOURCE/local_wt2_test_replay"
python "$SQ325_EVAL_SOURCE/scripts/audit_fp4_sq325_wt2_test_v1.py" \
  --comparison "$SQ325_EVAL_SOURCE/local_wt2_test_replay/comparison.json" \
  --bundle "$SQ325_RELEASE_BUNDLE" \
  --protocol "$SQ325_EVAL_SOURCE/docs/FP4_G16_SQ325_RESURFACE_WT2_TEST_V1_PROTOCOL.md" \
  --runner "$SQ325_EVAL_SOURCE/scripts/run_fp4_sq325_wt2_test_v1.py" \
  --out "$SQ325_EVAL_SOURCE/local_wt2_test_replay/cpu_audit_replay.json"
```

The runner prints a result only after all targets in both arms are complete.
The auditor checks all window hashes, the final partial window, pooled NLL/PPL,
recorded model/adapter/state bindings, and exact adapter-removal cache/reset
restoration. A partial or failed run is not a complete test result.

## Re-audit the published reports without GPU inference

This route reconstructs the public test tokens and audits the supplied reports;
it does not reproduce logits. It needs the original sealed bundle for its small
metadata, adapter, and state files, plus `datasets==4.8.5` and
`sentencepiece==0.2.1` to recreate tokens. The auditor itself uses only Python's
standard library. Use the same source and bundle variables as above:

```bash
mkdir ../fp4-wt2-audit-input
cp evaluation/wt2_test_v1/comparison.json \
  evaluation/wt2_test_v1/without_resurface.json \
  evaluation/wt2_test_v1/resurface.json ../fp4-wt2-audit-input/
python scripts/reconstruct_fp4_wt2_test_tokens_v1.py \
  --bundle "$SQ325_RELEASE_BUNDLE" \
  --out ../fp4-wt2-audit-input/tokens.int64le
python scripts/audit_fp4_sq325_wt2_test_v1.py \
  --comparison ../fp4-wt2-audit-input/comparison.json \
  --bundle "$SQ325_RELEASE_BUNDLE" \
  --protocol docs/FP4_G16_SQ325_RESURFACE_WT2_TEST_V1_PROTOCOL.md \
  --runner scripts/run_fp4_sq325_wt2_test_v1.py \
  --out ../fp4-wt2-audit-input/cpu_audit_replay.json
```

Do not put generated token IDs, outputs, virtual environments, or extra scripts
inside the sealed model bundle. Its strict manifest rejects added files.
The original release tag and HF `release/` payload remain byte-identical; the
new reports live only in this separate evaluation overlay.
