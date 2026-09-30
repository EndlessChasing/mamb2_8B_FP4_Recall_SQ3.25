# Frozen published model: WikiText-2 test PPL

Measure the already released FP4 G16 / group-ridge SQ3.25 model, without and
with its final V3 Resurface adapter. No training, calibration, scale changes,
checkpoint selection, candidate selection, or test-dependent stopping is allowed.
Run both complete arms regardless of the first arm's score.

The model is the byte-identical complete GitHub/HF v0.1.0 bundle, source commit
c15ca4f8f7178f7956076e3d124fe19abbb77cd3 and HF commit
6ddb462279d2e9bc9fd92c9bb1e14400cfb96176. Verify its manifest/files and bind every
507 decoded weight tensor, state configuration, all 224 adapter tensors, and
the exact recorded execution backend. Load real packed weights without the
original source checkpoint. Do not modify files inside the sealed bundle.

Use Salesforce/wikitext, wikitext-2-raw-v1, official test split, revision
b08601e04326c79dfdd32d625aee71d232d685c3, joined with two newlines and tokenized
with the pinned NVIDIA SentencePiece tokenizer without automatic BOS/EOS.
Freeze and save text/token hashes before inference. Use all tokens, predicting
every token except the first, with windows of at most 2,048 next-token targets.
Windows reset state and overlap by one boundary token; include the final
partial window. Requantize state every token with the unchanged PredictorState
codec. Reuse the released V3 PPL loop: FP16 compute, FP32 logits, sum cross
entropy in chunks of 64, and aggregate exp(total NLL / total target count).
This is not the generic FP16-cache tokenwise evaluator.

Compare published Resurface against its identical adapter-free parent on the
same window/token population. Record complete per-window scores, actual state
storage, exact initial/final weight hashes, backend policy, adapter removal
and reset/cache restoration. Save raw little-endian int64 token IDs outside
the bundle so an independent CPU audit can reproduce all window hashes/counts
and NLL/PPL arithmetic. A failed or partial run cannot be called a test result.

This is a fixed-checkpoint compression evaluation on the official test split.
Do not claim the NVIDIA base pretraining never contained these public texts,
or that documents across official splits have been independently deduplicated.
The bounded history check found earlier 2.7B experiments on the official WT2
test text, including a complete test stream, using a different tokenizer and
model configuration. No test run for this published 8B configuration was found
in the checked family scripts and main reports. Thus this cannot be called an
untouched test for the whole research project. It is the published fixed 8B
configuration evaluated on the official test split, without test selection.
