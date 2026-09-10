# packproof

Offline, dependency-free integrity checks for packed language-model training
records.

Sequence packing makes a training batch denser by concatenating several
examples into one token row. That speedup is only correct when the pack's
boundaries, positions, labels, loss mask, and attention behavior agree. A
single off-by-one can let one example attend to another or train on padding
without producing an obvious crash.

`packproof` is the small preflight gate for that boundary. It reads JSONL,
reports stable line-level findings, and exits non-zero on errors. It never
loads a tokenizer or model, allocates a tensor, imports user code, calls a
service, or copies token payloads into reports.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "git+https://github.com/thisbejim/packproof.git@v0.1.1"
```

The same command can install a downloaded wheel from the
[`v0.1.1` release](https://github.com/thisbejim/packproof/releases/tag/v0.1.1).

For a checkout with the test tools:

```bash
uv sync --dev
uv run pytest
```

## Quick start

The repository includes a clean packed record and intentionally broken
records:

```bash
packproof examples/clean.jsonl --format text
# packproof: PASS
# records=1 lines=1 errors=0 warnings=0 infos=0

packproof examples/broken.jsonl --format text
# packproof: FAIL
```

The input can also be piped through stdin:

```bash
cat examples/clean.jsonl | packproof - --format json > report.json
```

Use `--format sarif` for GitHub code scanning or another SARIF consumer. Use
`--format markdown` for a review artifact. All formats contain line, field,
code, severity, and a concise explanation, never raw token arrays.

## Record shape

Each JSONL object needs a non-empty integer `input_ids` array. Optional fields
are deliberately close to the conventions emitted by Hugging Face TRL,
torchtune, Megatron-style preprocessing scripts, and custom collators:

| Field | Meaning |
| --- | --- |
| `seq_lengths` / `sequence_lengths` | Positive segment lengths summing to `input_ids` length. |
| `cu_seqlens` | Strictly increasing prefix offsets starting at 0 and ending at the token count. |
| `segment_ids` / `document_ids` | One integer per token; contiguous runs define segments. |
| `position_ids` | Per-token positions. `--position-mode reset` expects 0, 1, … in every segment; `global` expects token indexes. `auto` accepts either. |
| `attention_mask` | A 1-D 0/1 active-token mask, or a 2-D causal/block mask. Use `attention_mask_2d` when a 1-D active mask is also needed. |
| `labels` | Optional token IDs or the configured `--ignore-index` (default `-100`). `--label-alignment copy` or `next` enables explicit alignment checks. |
| `loss_mask` | Optional finite weights in `[0, 1]`; inactive/padding positions must be zero. |
| `id` / `record_id` | Optional stable identifier used to detect duplicate records. |

The checker treats missing optional metadata conservatively. For example, a
packed record with only a 1-D attention mask receives a warning because
cross-segment isolation cannot be proved from that mask alone. Turn useful
assumptions into hard gates with `--require-segments`, `--require-positions`,
`--require-eos`, `--pad-id`, `--eos-id`, `--vocab-size`, and the explicit
alignment modes.

## What is checked

- JSONL validity, UTF-8, bounded line size, record count, and object shape.
- Integer token IDs, optional vocabulary bounds, non-empty sequences, and
  maximum sequence length.
- Segment-length and cumulative-offset arithmetic, contiguous segment IDs, and
  repeated non-contiguous IDs.
- 1-D active masks, padding placement, and optional 2-D causal/block masks:
  no future edges, no cross-segment edges, no padding edges, and no empty
  active query rows.
- Position reset/global-index policies, label length and ignore-index rules,
  optional copy/next-token alignment, and loss-mask bounds.
- Optional EOS-at-boundary requirements and duplicate record identifiers.

The default is intentionally structural rather than model-specific. A custom
collator can use different label semantics without being rejected; opt into
the stronger `--label-alignment` policy when that contract is known.

## CI example

```yaml
- name: Validate packed training data
  run: >-
    packproof data/packed.jsonl
    --max-length 8192
    --pad-id 0
    --eos-id 2
    --require-segments
    --require-positions
    --require-eos
    --format sarif
    --output packproof.sarif
- uses: github/codeql-action/upload-sarif@v3
  if: always()
  with:
    sarif_file: packproof.sarif
```

## Finding codes

Codes are stable enough to gate or suppress in automation. The prefixes are
`INPUT` (JSON and bounds), `SEG` (boundaries), `MASK`/`ATTN` (attention),
`PAD`, `POS` (positions), `LABEL`, `LOSS`, `EOS`, and `META`. See
[`docs/report-schema.md`](docs/report-schema.md) for the machine-readable
report contract.

## Why this exists

Hugging Face TRL exposes packing as a dataset transformation with sequence
length metadata, while Meta's torchtune documents that packed training needs
document masking and relative position IDs to prevent irrelevant samples from
cross-attending. A TRL issue describes `seq_lengths` being lost before a
collator generated whole-sequence positions; public discussions and the
packing literature repeatedly call out cross-contamination as a correctness
failure, not just a performance detail. The evidence and scope decision are
recorded in [`docs/research.md`](docs/research.md).

## Non-goals

`packproof` does not tokenize text, build packs, read framework-specific binary
datasets, run a model, calculate loss, benchmark kernels, or certify that a
training recipe is statistically correct. It checks the serialized contract
that a pack-producing step hands to a trainer.

## License

MIT. See [`LICENSE`](LICENSE).
