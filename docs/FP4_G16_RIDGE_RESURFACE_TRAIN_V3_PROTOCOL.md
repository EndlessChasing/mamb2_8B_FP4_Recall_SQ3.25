# Fresh training on frozen FP4 G16 group-ridge SQ3.25

This supplements the frozen Ridge Resurface V3 protocol. The full parent,
state table, latent bases/scales and group-ridge predictor remain unchanged.
The new stateless training forward delegates to the identical deployment scan.
The surrogate backward reuses V11 fixed32/32/64 live-mask STE recomputation.
It omits latent/predictor derivative terms. No exact-adjoint claim is made.
V2 proxy omitted contiguous copies of saved inputs; the legacy Triton adjoint
requires compact storage, so real split/reshape views would receive wrong
gradients. V2 was never used for full-model training. Original V1 prepared
contiguous saved inputs and is unaffected. V3 copies all scan inputs before
saving them, exactly as V11 does. Fixture42 layout/length/input-layout combinations must match deployment forward bitwise;
surrogate gradients must be finite and agree with V11 within absolute1e-4,
relative.005 to accommodate FP32 atomic reduction ordering.

Run and discard a one-update full-model smoke, with exact128/512-token parity
before and after FP16 export and a complete frozen507-tensor weight ledger.
Formal training starts entirely new224 FP32 masters and empty AdamW state,
GradScaler1024. Seed2026092803. Use existing pinned1536 numeric TRAIN examples
and448 WikiText TRAIN windows; never use validation or CONFIRM in training.
Separate frozen FP4 G16/S16 teacher. Student exact ridge-state forward.
Only external Resurface parameters may change. Before any smoke/formal
training, select one PPL-priority recipe: numeric answer CE.25, prose CE1.,
teacher-to-student KL.5, prose gate closure.1. Closure internal budget.006 and
penalty coefficient10 remain unchanged; temperature1. This increases prose CE
relative to numeric CE eightfold versus the old1/.5 recipe and reduces closure
30fold. Numeric loss remains to encourage recall; the S16 teacher anchors language
recovery. Recipe selection uses code/objective analysis and TRAIN evidence, not
validation. Fixed1536 successful updates, max8 overflow retries,
checkpoints384/768/1152/1536 and final FP16
export2,308,208 bytes. Freeze recipe before smoke and do not select checkpoints
on validation. Report inherited gradient-estimator limitation.

A CPU-only training audit binds source, data, code, raw128/512 hidden parity,
checkpoint master/export tensors, optimizer steps, scalar history and frozen
weight/static state provenance. Evaluation requires this audit, then follows
unchanged Ridge Resurface V3 TRAIN screen and full validation/MK gate.

The V3 fixture includes compact inputs and strided views constructed from one
projection-shaped allocation. It must demonstrate at least one V2 gradient
mismatch on strided inputs, then verify V3 matches independent V11 gradients
and exact deployed ridge forward on all42 cases. No frozen V2 file is edited.

Use the final1536 checkpoint only, without validation checkpoint selection.
If comparing future recipes, predeclare all recipes and evaluate their final
exports on the pinned64 reserved WikiText TRAIN-heldout windows
(heldout_tokens.pt SHA256a3fc0803052890d973b74f91581a6bfc1d2f93ed28db5725f23c16eb8ec20546),
which have zero overlap with448 training windows. Choose minimum finite NLL,
with parent tie priority. Additional historical use of this selector must be
disclosed; no untouched-test claim. Only then evaluate the selected candidate
on the fixed validation and full paired MK; never alter the strict PPL<8 gate.
