# Supplemental FP8/FP4 state evidence

These original JSON summaries, CPU audits and constructed codec fixtures document the bounded state-format investigation. They do not add a publication quality gate and do not claim that every possible floating-format recipe is ineffective.

- `v12/`: matched TRAIN screening comparison, CPU audit and 28 codec/layout fixtures.
- `v13/`: endpoint metric report/audit, independent production-storage oracle and CPU reference checks. The endpoint `snapshots` field contains only row, endpoint and sample-count metadata; no state tensors are included.
- `v14/`: disjoint-window TRAIN screening comparison, CPU audit, explicit static layer selection policy and selective-codec fixtures.
- `parent_full_audit.json`: historical full unadapted group-ridge parent audit.

Per-arm source reports are referenced by their original summary/audit hashes; raw training token streams and state snapshots are omitted. `inventory.json` records original paths, bytes and SHA-256 values. All copied JSON bytes are unchanged; only the packaged diagnosis document's links are adjusted to point to this included evidence.
