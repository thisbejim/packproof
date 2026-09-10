# Opportunity scorecard

| Criterion | Assessment | Reason |
| --- | --- | --- |
| Pain frequency | High | Packing is a standard efficiency switch in modern fine-tuning and pretraining. |
| Failure cost | High | Boundary or mask mistakes can contaminate training while still producing valid tensors. |
| Public evidence | High | Framework docs plus an implementation issue and published packing research. |
| Existing alternatives | Partial | Framework-local tests exist; a small provider-neutral artifact gate is missing. |
| Standalone value | High | Works on synthetic or user-exported JSONL with no model, GPU, or account. |
| Adoption friction | Low | One CLI command or library call; SARIF fits ordinary CI. |
| Technical depth | Medium-high | Requires careful boundary, mask, and position semantics without assuming one trainer. |
| Privacy fit | High | Reports omit token arrays and never upload data. |
| Differentiation | High | It is low-level packed-record integrity, not dataset deduplication, model-bundle inspection, or runtime tracing. |

## Alternatives considered

- **A generic token-count profiler:** useful but crowded, and unable to prove
  the packed attention contract.
- **A pack builder:** would compete with TRL/torchtune and would make users
  restructure their pipeline. The unmet job is validation after custom packing.
- **A GPU benchmark:** expensive and cannot catch metadata mistakes before the
  kernel runs.
- **A full training framework:** violates the focused-tool requirement and
  obscures the portable artifact boundary.
