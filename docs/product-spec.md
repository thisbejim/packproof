# Product specification

## User problem

Training teams often serialize pretokenized or packed examples and hand them
to a high-throughput trainer. The serialized row is small and easy to inspect,
but correctness depends on several parallel arrays and a hidden boundary
contract. A malformed row can train for hours before its effect appears as a
quality regression.

## User and job

The primary user is an ML or data engineer who owns dataset preparation,
collators, or a training CI pipeline. Their job is: “before a costly run,
prove that these packed rows have consistent boundaries and masks, and show me
the exact line that violates the contract.”

## Product promise

Given a local JSONL capture, produce deterministic, payload-free diagnostics in
one command, with a useful zero-dependency library API and a SARIF exit path.

## Input contract

One JSON object per line with `input_ids` and optional metadata described in the
README. The tool accepts common boundary representations rather than binding
to one trainer. It is intentionally not a loader for Arrow, Parquet, NumPy,
or framework-specific binary shards.

## Correctness policies

The default policy checks facts that can be proved from the row itself. Stronger
assumptions are opt-in: maximum length, padding/EOS IDs, vocabulary size,
required segment and position metadata, label alignment, and attention-matrix
cell checks. This avoids silently treating one framework's convention as
universal.

## Output contract

Every finding has a stable code, severity, line, optional field/record ID, and
content-free message. Text is for humans; JSON is for scripts; Markdown is for
reviews; SARIF is for CI annotations. Exit 0 means no errors, 1 means at least
one error, and 2 means the command could not be run (bad arguments or I/O).

## Safety and bounds

The parser never evaluates record values or imports model code. It bounds line
size, record count, and quadratic attention checks. It emits no input arrays.
Users inspecting untrusted data should lower these limits as appropriate.

## Success criteria

- A developer can install and run a clean fixture in under a minute.
- A synthetic cross-segment edge, bad boundary sum, padding leak, and position
  mismatch each produce a distinct line-level code.
- The core remains dependency-free and works with Python 3.10+.
- CI can consume SARIF without a service account or data upload.
