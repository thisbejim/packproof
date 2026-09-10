# Skeptical review

## Could this be a thin wrapper?

No. The implementation has its own bounded JSONL reader, segment derivation
for lengths/offsets/IDs, causal and block-diagonal matrix checks, position
policies, optional label alignment, and payload-free report schema. It does not
call TRL, torchtune, Transformers, or a model server.

## Could the defaults reject valid data?

The defaults are intentionally conservative: labels are opaque, positions may
be either reset or global, and missing optional metadata is not an error. The
stronger assumptions are explicit flags. A custom convention can therefore
start with structural checks and opt into its known contract.

## What remains outside the proof?

The tool cannot infer a tokenizer's vocabulary, validate a binary tensor shard,
or prove that the trainer consumes the fields as intended. It also cannot
measure model quality. It proves only the serialized invariants it can see;
users should pair it with a small end-to-end loss-equivalence test for a new
collator or kernel.

## Abuse and privacy

Large or hostile JSONL could consume resources, so line, record, and attention
cell limits are explicit. Reports contain no token arrays, prompts, or model
outputs. The tool is offline and does not execute values from input records.

## Why ship anyway?

The remaining uncertainty is exactly why this is a preflight gate rather than a
trainer. A cheap, deterministic check that catches one boundary or mask bug
before a long run has a favorable cost-to-value ratio, and the portable JSONL
surface makes it easy to integrate without adopting a framework.
