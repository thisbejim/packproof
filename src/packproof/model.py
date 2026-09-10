"""Public data structures used by packproof."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warning", "info"]
SegmentField = Literal[
    "auto",
    "seq_lengths",
    "sequence_lengths",
    "cu_seqlens",
    "segment_ids",
    "document_ids",
    "segments",
]
PositionMode = Literal["auto", "reset", "global", "ignore"]
LabelAlignment = Literal["opaque", "copy", "next"]


@dataclass(frozen=True)
class CheckConfig:
    """Bounded, explicit policy for one audit run."""

    max_length: int | None = None
    max_records: int = 100_000
    max_line_bytes: int = 4_000_000
    max_attention_cells: int = 2_000_000
    pad_id: int | None = None
    eos_id: int | None = None
    vocab_size: int | None = None
    ignore_index: int = -100
    segment_field: SegmentField = "auto"
    position_mode: PositionMode = "auto"
    label_alignment: LabelAlignment = "opaque"
    require_segments: bool = False
    require_positions: bool = False
    require_eos: bool = False

    def __post_init__(self) -> None:
        positive = {
            "max_length": self.max_length,
            "max_records": self.max_records,
            "max_line_bytes": self.max_line_bytes,
            "max_attention_cells": self.max_attention_cells,
            "vocab_size": self.vocab_size,
        }
        for name, value in positive.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Finding:
    """A stable, payload-free diagnostic."""

    code: str
    severity: Severity
    message: str
    line: int | None = None
    field: str | None = None
    record_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.field is not None:
            result["field"] = self.field
        if self.record_id is not None:
            result["record_id"] = self.record_id
        return result


@dataclass(frozen=True)
class AuditReport:
    """The complete result of a JSONL audit."""

    findings: tuple[Finding, ...] = field(default_factory=tuple)
    records_checked: int = 0
    lines_read: int = 0
    truncated: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> int:
        return sum(f.severity == "error" for f in self.findings)

    @property
    def warnings(self) -> int:
        return sum(f.severity == "warning" for f in self.findings)

    @property
    def infos(self) -> int:
        return sum(f.severity == "info" for f in self.findings)

    @property
    def ok(self) -> bool:
        return self.errors == 0

    def summary(self) -> dict[str, Any]:
        return {
            "records_checked": self.records_checked,
            "lines_read": self.lines_read,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "truncated": self.truncated,
            "status": "PASS" if self.ok else "FAIL",
        }
