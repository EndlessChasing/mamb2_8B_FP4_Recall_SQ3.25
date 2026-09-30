# FP4 G16 ridge SQ3.25 Resurface V3

The current user authorizes applying Resurface to the independently audited
unadapted group-ridge parent PPL 8.408282583578627. The publication prerequisite
is full adapted PPL strictly less than 8.0. This supersedes the earlier no-adapter
<8.4 prerequisite for this experiment; it does not make the parent pass that gate.

Freeze all 507 decoded FP4 G16 weight tensors, top4 state permutations, per-layer
mixed layouts, latent bases, latent scales and group-ridge replacement predictor.
Each recurrent state row remains 52 bytes. The state + convolution + table cache
is 28,499,968 bytes; separately count static bases 109,952 bytes, latent scales
1,792 bytes and predictors 3,673,216 bytes. FP16 adapter payload is 2,308,208 bytes.
The logical encoded weight payload is 4,638,460,360 bytes and the reference runtime
expands weights to 16,473,999,360 FP16 bytes. No packed-resident weight claim.

First evaluate the existing fresh-trained FP4 G16 adapter as an explicitly labeled
transfer experiment. It was trained on an earlier fixed32/32/64 state; it is not a
fresh adapter for the ridge parent. Its original training report and successful
independent audit must be pinned. A 16-window WikiText TRAIN screen chooses only
whether this frozen transfer has improved NLL. Use the pinned prose TRAIN windows
432..447. They are historically exposed TRAIN data, including adapter training,
so screen improvement is not a generalization result. Full quality alone gates
publication, and no candidate is selected from validation results.

Full quality uses the unchanged130 WikiText-2 validation reset windows and264,764
targets with64-token full-vocabulary NLL chunks. Replay adapter-off output, cache
and each window's NLL exactly against the archived8.408282583578627 parent. Measure
both arms on all768 CONFIRM recall cases,384 normal and384 target-removed,12-token
greedy generation. Require normal MK increase and a positive paired bootstrap95%
lower bound, report target-removed controls. Adapter removal must restore reset
hidden state/cache, native forwards/hooks and frozen weight checks. Validation and
CONFIRM are historically exposed; make no untouched-test claim.

Fresh ridge training, if performed, uses an exact frozen deployed ridge-state
forward. Its explicit gradient estimator delegates backward to the earlier V11
fixed32/32/64 live-mask STE; it ignores latent/predictor derivative contributions.
It is a surrogate backward, not an exact derivative of the ridge codec. Verify
forward bitwise equality across layouts and lengths, verify finite surrogate
backward against V11, then run and discard a one-update full-model smoke. Formal
training starts fresh V=0,g=1,w=0,b=-4 and a fresh optimizer/scaler; only Resurface
masters update. Require128/512-token training/deployment parity before and after
FP16 export. Use numeric TRAIN and448 pinned WikiText TRAIN windows, no validation
or CONFIRM in training/selection. Teacher is separate frozen FP4 G16 with S16 state.
The first formal recipe follows1536 successful updates with numeric CE.25,
prose CE1., KL.5 and prose closure.1, selected before training to prioritize PPL.
Closure internal budget.006 and coefficient10 remain unchanged. Seed2026092803;
max8 overflow retries. See the V3 fresh-training protocol for the strided-gradient
fix and exact source-bound objective audit.

Before publishing, perform CPU-only arithmetic/provenance/storage audit. Full
adapted PPL <8.0, positive normal MK effect, exact parent replay/removal and all
integrity audits must pass. Publish GitHub first, then Hugging Face; no publication
is authorized when the strict PPL gate fails.
