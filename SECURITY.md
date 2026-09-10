# Security policy

## Scope

`packproof` reads local JSONL and emits diagnostics. It does not import model
code, execute values from records, open network connections, or copy token
payloads into reports.

The parser is bounded by configurable line, record, and attention-matrix cell
limits. Use conservative limits when inspecting untrusted files.

## Reporting a vulnerability

Please report security issues privately through GitHub's security advisory
workflow for `thisbejim/packproof`. Do not include private training data in a
public issue. Include the version, command line, a minimal synthetic fixture,
and the observed impact.
