# FP8/FP4 state diagnosis and practical stopping decision

**Retain the verified INT8/INT4 group-ridge state parent, whose full validation
PPL is 8.408282583578627, for the next Resurface stage.** The bounded FP-state
investigation found no encoding defect and no recurrent PPL winner, including
a selective FP4 control chosen from favorable endpoint measurements. This
supports stopping the tested recipes. It does not prove that all possible
floating formats, transforms, scale optimizers, or retrained predictors fail.

All comparisons freeze FP4 G16 weights and use the same physical **52-byte
recurrent row**. None of the experiments below uses Resurface. Screening PPL
comes from TRAIN; it is not the 130-window full validation metric.

## 1. The original floating tiers lose consistently

On v12's sixteen WikiText-2 TRAIN windows / 32,752 targets:

| Recurrent tier formats | PPL |
| --- | ---: |
| INT8 + INT4 parent | **8.696807873536711** |
| E4M3FN FP8 + INT4 | 9.147942597630397 |
| INT8 + E2M1 FP4 | 9.037462918815429 |
| E4M3FN FP8 + E2M1 FP4 | 9.638968884457402 |

Every FP arm has higher NLL on all sixteen matched windows. The independent
[v12 CPU audit](evidence/fp_state_diagnostics/v12/audit.json)
passed physical-byte, frozen-weight, arithmetic, and parent-restoration checks.
The parent won; no FP candidate advanced to full validation.

## 2. Independent encoding and scale checks exclude the suspected defects

The old first-token output fixture precedes quantization, so it did not alone
verify the new stored codes. The v13 oracle addresses that gap by injecting
constructed one-token states into the actual frozen v12 production kernel and
comparing `lo`, `hi`, `q4`, `s8`, and `s4` against an independent CPU reference.
It covers finite FP8 levels, ties and representable neighbors, signs, three
scale ranges, four modes, and seven layouts. **All 28 combinations passed
with zero mismatched bytes.** The reference implements E4M3FN exponent/sign
decoding and table-based nearest-even selection independently of the kernel.
FP4 tie handling, FP8 saturation and negative zero, stored FP16 scales, and
nibble packing match the declared recipe within these fixtures.

## 3. More dynamic range does not guarantee a better recurrent readout

TRAIN S16 teacher snapshots from prose rows 446/447 at 128, 512, and 2047
tokens sample all 56 layers. Each tier is quantized separately; its endpoint
readout proxy includes current C and the frozen zero-tier ridge sensitivity.

| Candidate | MSE / same INT tier | Readout proxy SSE / INT | Layers with better proxy |
| --- | ---: | ---: | ---: |
| FP8 max/448 | 14.7516 | 14.7809 | 0/56 |
| FP8 max/384 | 16.1768 | 15.9556 | 0/56 |
| FP8 upward power-of-two scale | 25.8043 | 14.7086 | 0/56 |
| FP4 max/6 | 0.57944 | 1.51090 | 52/56 |
| FP4 row-MSE scale search, max divided by 2/4/6/8 | 0.56104 | 1.56895 | 52/56 |

FP8 spends exponent bits on range already supplied by the row scale, leaving
coarser spacing for medium and large coordinates. It loses to INT8 throughout
the sampled layers, even after the tested scale changes.

FP4 preserves small values and improves reconstruction MSE, but its errors
project more strongly onto sensitive readout directions in layers 0, 1, 2,
and 15. Layer 1 contributes most of the aggregate harm. This is measured
directional error, not evidence distinguishing its possible causes, such as
coordinate sensitivity versus correlated rounding errors. Endpoint MSE and
the current-C proxy cannot determine recurrent PPL.

The [v13 CPU audit](evidence/fp_state_diagnostics/v13/endpoints_audit.json)
passed code/static binding, reported 507 decoded-weight hashes, finite values,
sample counts, layer/bin sums, and normalization arithmetic. No PPL is
inferred from these endpoint measurements.

## 4. Selective FP4 also fails the bounded recurrent control

v14 retains INT8 everywhere and applies max/6 FP4 only to selected layers.
It uses sixteen further WikiText-2 TRAIN windows / 32,752 targets, with starts
disjoint from the declared WikiText screening history through v12. Text
overlap with the separate prose TRAIN corpus has not been audited. Its PPL
numbers must be compared within this table, not directly against v12's
different window population.

| Static layer policy | PPL | Matched windows worse than parent |
| --- | ---: | ---: |
| Original INT8/INT4 parent | **8.406065636256484** | — |
| FP4 only on proxy-best layers 4/9/5/46/6/50 | 8.433642089861635 | 10/16 |
| FP4 everywhere except harmful layers 0/1/2/15 | 8.623837446969981 | 16/16 |
| All-INT through the new selective wrapper | **8.406065636256484** | exact replay |

The static mode tuple adds no GPU tensor or per-row state storage. The
fourteen mode/layout CUDA fixtures passed byte-exact segmentation and rejected
incorrect declared modes. The all-INT wrapper exactly reproduces the original
parent's per-window NLL, reset output, physical allocations, and cache hashes;
this excludes the wrapper itself as the source of the candidate regressions.
The [v14 CPU audit](evidence/fp_state_diagnostics/v14/audit.json)
passed and independently selected the parent. Neither FP4 policy advances to
full validation.

## Selected baseline and next gate

The [existing full-validation parent audit](evidence/fp_state_diagnostics/parent_full_audit.json)
supports **8.408282583578627 on 130 windows / 264,764 target tokens**. This
remains the measured unadapted baseline. The new sixteen-window score
8.406065636256484 does not replace it.

Proceed with Resurface on that exact parent. Publication requires a separately
audited full-validation **PPL <8**, along with the applicable recall, integrity,
and storage evidence. Publish GitHub first, then Hugging Face, after those
conditions are actually met. This diagnostic records no adapter quality or
publication result.
