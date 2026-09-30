# RMSNorm replay clarification for the fixed v3 continuation

The first full parent replay failed before the continued candidate was evaluated.
The unchanged v2 parent produced PPL 8.39286420757536 instead of archived
8.388905741798535. Preserve that failed run in
`artifacts/resurface_more_v3_eval`; it is not a completed v3 comparison.

A bounded first-window diagnostic identified the numerical cause. Mamba's
backbone RMSNorm forward kernel, `_layer_norm_fwd_1pass_kernel`, benchmarks
Triton configurations with 1, 2, 4, 8, 16 or 32 warps. Changing the reduction
configuration changes FP16 outputs. Its process-local autotune key omits the
number of token rows. Package versions and source hashes alone therefore do
not fix this numerical choice.

| RMSNorm configuration | Original S16 first-window NLL | Parent SQ3.25 first-window NLL |
|---|---:|---:|
| Archived v2 | 3292.7154846191406 | 3528.2185134887695 |
| Forced 8 warps | 3292.566764831543 | 3532.2641410827637 |
| Forced 16 warps | 3292.7154846191406 | 3528.2185134887695 |

Eight warps also reproduced the failed run's hidden-state hashes. Sixteen warps
was the only tested configuration that exactly reproduced both archived NLLs.
The first-window evidence does not establish full replay by itself. No continued
candidate metric was used to choose this configuration.

## Evaluation execution clarification

The evaluator selects the existing 16-warp, 3-stage, 1-CTA configuration before
loading the model or executing any forward. Setting the autotuner's configuration
list to this single entry bypasses benchmarking and cached configuration choices.
No model weights, adapter tensors, codec, training source, calibration, data,
training schedule, frozen protocol or quality gate changes.

Every new evaluation arm records the original configuration inventory, selected
configuration, precision flags, relevant environment settings, package versions,
cuDNN version, and SHA-256 hashes of the installed Mamba RMSNorm source,
Mamba determinism helper and Triton autotuner source. Checks before completion
require the singleton configuration and actual selected configuration to remain
unchanged. The independent CPU auditor verifies those receipts against the
installed sources and this clarification's hash.

Run the fixed final adapter in a fresh output directory, such as
`artifacts/resurface_more_v3_eval_pinned16`. The original strict requirements
remain: all 130 parent window NLLs and all 768 generated token sequences must
exactly reproduce the archive before the continued candidate is evaluated;
the final restored parent must exactly repeat the fresh parent. The improvement
gate remains PPL no more than 1% worse than the parent, observed MK improvement,
and a strictly positive paired-bootstrap 95% lower bound.

The already-completed training receipt and its original 128-token export parity
check remain historical evidence from that training process. They did not record
the process-local RMSNorm autotune choice. This clarification pins the evaluation
backend and does not retroactively claim a pinned configuration during training.
