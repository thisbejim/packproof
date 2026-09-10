"""Command-line interface for packproof."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from .checks import audit_jsonl
from .model import CheckConfig
from .report import render


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="packproof",
        description="Offline integrity checks for packed language-model training JSONL.",
        epilog="Input is JSONL; use - for stdin. Reports never include token payloads.",
    )
    parser.add_argument(
        "input", nargs="?", default="-", help="JSONL path, or - for stdin (default)"
    )
    parser.add_argument("--format", choices=("text", "json", "markdown", "sarif"), default="text")
    parser.add_argument("--output", default="-", help="output path, or - for stdout")
    parser.add_argument("--max-length", type=_positive)
    parser.add_argument("--max-records", type=_positive, default=100_000)
    parser.add_argument("--max-line-bytes", type=_positive, default=4_000_000)
    parser.add_argument("--max-attention-cells", type=_positive, default=2_000_000)
    parser.add_argument("--pad-id", type=_nonnegative)
    parser.add_argument("--eos-id", type=_nonnegative)
    parser.add_argument("--vocab-size", type=_positive)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument(
        "--segment-field",
        choices=(
            "auto",
            "seq_lengths",
            "sequence_lengths",
            "cu_seqlens",
            "segment_ids",
            "document_ids",
            "segments",
        ),
        default="auto",
    )
    parser.add_argument(
        "--position-mode", choices=("auto", "reset", "global", "ignore"), default="auto"
    )
    parser.add_argument("--label-alignment", choices=("opaque", "copy", "next"), default="opaque")
    parser.add_argument(
        "--require-segments", action="store_true", help="fail records without explicit boundaries"
    )
    parser.add_argument(
        "--require-positions", action="store_true", help="fail packed records without position_ids"
    )
    parser.add_argument(
        "--require-eos", action="store_true", help="require eos_id at every segment boundary"
    )
    return parser


def _config(args: argparse.Namespace) -> CheckConfig:
    return CheckConfig(
        max_length=args.max_length,
        max_records=args.max_records,
        max_line_bytes=args.max_line_bytes,
        max_attention_cells=args.max_attention_cells,
        pad_id=args.pad_id,
        eos_id=args.eos_id,
        vocab_size=args.vocab_size,
        ignore_index=args.ignore_index,
        segment_field=args.segment_field,
        position_mode=args.position_mode,
        label_alignment=args.label_alignment,
        require_segments=args.require_segments,
        require_positions=args.require_positions,
        require_eos=args.require_eos,
    )


def _open_input(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdin, False
    return Path(path).open("r", encoding="utf-8"), True


def _write_output(path: str, content: str) -> None:
    if path == "-":
        sys.stdout.write(content)
        return
    Path(path).write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config(args)
        stream, should_close = _open_input(args.input)
        try:
            report = audit_jsonl(stream, config)
        finally:
            if should_close:
                stream.close()
        _write_output(args.output, render(report, args.format))
    except (OSError, ValueError) as exc:
        print(f"packproof: {exc}", file=sys.stderr)
        return 2
    return 1 if report.errors else 0
