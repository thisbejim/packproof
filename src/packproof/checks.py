"""Core JSONL checks.  This module deliberately has no model or tensor dependency."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO, TextIO, cast

from .model import AuditReport, CheckConfig, Finding

_SEGMENT_KEYS = (
    "seq_lengths",
    "sequence_lengths",
    "cu_seqlens",
    "segment_ids",
    "document_ids",
    "segments",
)


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    label: int | str | None = None


def _finding(
    findings: list[Finding],
    code: str,
    severity: str,
    message: str,
    *,
    line: int | None = None,
    field: str | None = None,
    record_id: str | None = None,
) -> None:
    if severity not in {"error", "warning", "info"}:
        raise ValueError(f"invalid severity: {severity}")
    findings.append(
        Finding(
            code=code,
            severity=severity,  # type: ignore[arg-type]
            message=message,
            line=line,
            field=field,
            record_id=record_id,
        )
    )


def _display_id(record: Mapping[str, Any], line: int) -> str | None:
    for key in ("id", "record_id", "pack_id"):
        value = record.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return str(value)
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_list(
    value: Any,
    *,
    name: str,
    line: int,
    findings: list[Finding],
    record_id: str | None,
    allow_negative: bool = False,
) -> list[int] | None:
    if not isinstance(value, list):
        _finding(
            findings,
            "INPUT004",
            "error",
            f"{name} must be an array",
            line=line,
            field=name,
            record_id=record_id,
        )
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        if not _is_int(item) or (not allow_negative and item < 0):
            suffix = " or the configured ignore index" if allow_negative else ""
            _finding(
                findings,
                "INPUT005",
                "error",
                f"{name}[{index}] must be a non-negative integer{suffix}",
                line=line,
                field=name,
                record_id=record_id,
            )
            continue
        result.append(int(item))
    return result


def _parse_segments(
    record: Mapping[str, Any],
    n: int,
    config: CheckConfig,
    *,
    line: int,
    findings: list[Finding],
    record_id: str | None,
) -> tuple[list[_Segment], bool]:
    chosen: str | None = None
    if config.segment_field == "auto":
        for key in _SEGMENT_KEYS:
            if key in record:
                chosen = key
                break
    elif config.segment_field in record:
        chosen = config.segment_field
    else:
        _finding(
            findings,
            "SEG001",
            "error",
            f"configured segment field {config.segment_field!r} is missing",
            line=line,
            field=config.segment_field,
            record_id=record_id,
        )
        return [], False

    if chosen is None:
        if config.require_segments:
            _finding(
                findings,
                "SEG006",
                "error",
                "packed record has no segment boundaries; provide seq_lengths, "
                "cu_seqlens, or segment_ids",
                line=line,
                field="segments",
                record_id=record_id,
            )
        return [_Segment(0, n)], False

    raw = record[chosen]
    spans: list[_Segment] = []
    if chosen in {"seq_lengths", "sequence_lengths", "segments"}:
        lengths = _int_list(raw, name=chosen, line=line, findings=findings, record_id=record_id)
        if lengths is None:
            return [], False
        cursor = 0
        for index, length in enumerate(lengths):
            if length <= 0:
                _finding(
                    findings,
                    "SEG002",
                    "error",
                    f"{chosen}[{index}] must be greater than zero",
                    line=line,
                    field=chosen,
                    record_id=record_id,
                )
                continue
            end = cursor + length
            spans.append(_Segment(cursor, end, index))
            cursor = end
        if cursor != n:
            _finding(
                findings,
                "SEG003",
                "error",
                f"segment lengths sum to {cursor}, but input_ids has length {n}",
                line=line,
                field=chosen,
                record_id=record_id,
            )
            return [], False
    elif chosen == "cu_seqlens":
        offsets = _int_list(raw, name=chosen, line=line, findings=findings, record_id=record_id)
        if offsets is None:
            return [], False
        if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n:
            _finding(
                findings,
                "SEG003",
                "error",
                f"cu_seqlens must start at 0 and end at {n}",
                line=line,
                field=chosen,
                record_id=record_id,
            )
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            if end <= start:
                _finding(
                    findings,
                    "SEG002",
                    "error",
                    f"cu_seqlens must be strictly increasing (segment {index})",
                    line=line,
                    field=chosen,
                    record_id=record_id,
                )
            spans.append(_Segment(start, end, index))
    else:
        ids = _int_list(raw, name=chosen, line=line, findings=findings, record_id=record_id)
        if ids is None:
            return [], False
        if len(ids) != n:
            _finding(
                findings,
                "SEG004",
                "error",
                f"{chosen} has length {len(ids)}, but input_ids has length {n}",
                line=line,
                field=chosen,
                record_id=record_id,
            )
            return [], False
        start = 0
        seen: set[int] = set()
        previous: int | None = None
        for index in range(n + 1):
            value: int | None = ids[index] if index < n else None
            if index == n or value != previous:
                if previous is not None:
                    if previous in seen:
                        _finding(
                            findings,
                            "SEG005",
                            "error",
                            f"{chosen} value {previous} appears in multiple non-contiguous spans",
                            line=line,
                            field=chosen,
                            record_id=record_id,
                        )
                    seen.add(previous)
                    spans.append(_Segment(start, index, previous))
                start = index
                previous = value
    valid = bool(spans) and all(0 <= span.start < span.end <= n for span in spans)
    if not valid:
        return [], False
    return spans, len(spans) > 1


def _active_mask(
    record: Mapping[str, Any],
    n: int,
    config: CheckConfig,
    *,
    line: int,
    findings: list[Finding],
    record_id: str | None,
) -> list[bool]:
    raw = record.get("attention_mask")
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = record.get("token_attention_mask")
        if raw is None:
            return [
                not (config.pad_id is not None and token == config.pad_id)
                for token in record["input_ids"]
            ]
    if raw is None:
        return [
            not (config.pad_id is not None and token == config.pad_id)
            for token in record["input_ids"]
        ]
    if not isinstance(raw, list):
        _finding(
            findings,
            "MASK001",
            "error",
            "attention_mask must be an array",
            line=line,
            field="attention_mask",
            record_id=record_id,
        )
        return [True] * n
    values: list[bool] = []
    for index, item in enumerate(raw):
        if not _is_int(item) or item not in (0, 1):
            _finding(
                findings,
                "MASK003",
                "error",
                f"attention_mask[{index}] must be 0 or 1",
                line=line,
                field="attention_mask",
                record_id=record_id,
            )
            values.append(bool(item))
        else:
            values.append(item == 1)
    if len(values) != n:
        _finding(
            findings,
            "MASK002",
            "error",
            f"attention_mask has length {len(values)}, but input_ids has length {n}",
            line=line,
            field="attention_mask",
            record_id=record_id,
        )
        return (values + [True] * n)[:n]
    seen_inactive = False
    for index, active in enumerate(values):
        if not active:
            seen_inactive = True
        elif seen_inactive:
            _finding(
                findings,
                "MASK004",
                "error",
                f"attention_mask reactivates token at position {index} after "
                "padding/inactive tokens",
                line=line,
                field="attention_mask",
                record_id=record_id,
            )
            break
    if config.pad_id is not None:
        for index, (token, active) in enumerate(zip(record["input_ids"], values, strict=True)):
            if token == config.pad_id and active:
                _finding(
                    findings,
                    "PAD001",
                    "error",
                    f"pad token is marked active at position {index}",
                    line=line,
                    field="input_ids",
                    record_id=record_id,
                )
    return values


def _check_positions(
    record: Mapping[str, Any],
    n: int,
    spans: list[_Segment],
    active: list[bool],
    config: CheckConfig,
    *,
    line: int,
    findings: list[Finding],
    record_id: str | None,
) -> None:
    raw = record.get("position_ids")
    if raw is None:
        if config.require_positions and len(spans) > 1:
            _finding(
                findings,
                "POS005",
                "error",
                "packed record has no position_ids",
                line=line,
                field="position_ids",
                record_id=record_id,
            )
        return
    positions = _int_list(
        raw, name="position_ids", line=line, findings=findings, record_id=record_id
    )
    if positions is None:
        return
    if len(positions) != n:
        _finding(
            findings,
            "POS001",
            "error",
            f"position_ids has length {len(positions)}, but input_ids has length {n}",
            line=line,
            field="position_ids",
            record_id=record_id,
        )
        return
    active_positions = [index for index, flag in enumerate(active) if flag]
    mode = config.position_mode

    def is_reset() -> bool:
        return all(
            positions[index] == offset
            for span in spans
            for offset, index in enumerate(range(span.start, span.end))
            if active[index]
        )

    def is_global() -> bool:
        return all(positions[index] == index for index in active_positions)

    valid = is_reset() if mode == "reset" else is_global() if mode == "global" else True
    if mode == "auto":
        valid = is_reset() or is_global()
    if mode != "ignore" and not valid:
        expected = (
            "reset at each segment boundary"
            if mode == "reset"
            else "global token indexes"
            if mode == "global"
            else "either reset-per-segment or global token indexes"
        )
        _finding(
            findings,
            "POS003" if mode == "reset" else "POS004" if mode == "global" else "POS006",
            "error",
            f"position_ids do not follow {expected}",
            line=line,
            field="position_ids",
            record_id=record_id,
        )


def _check_labels(
    record: Mapping[str, Any],
    n: int,
    spans: list[_Segment],
    active: list[bool],
    input_ids: list[int],
    config: CheckConfig,
    *,
    line: int,
    findings: list[Finding],
    record_id: str | None,
) -> None:
    if "labels" not in record:
        return
    raw = record["labels"]
    if not isinstance(raw, list):
        _finding(
            findings,
            "LABEL001",
            "error",
            "labels must be an array",
            line=line,
            field="labels",
            record_id=record_id,
        )
        return
    if len(raw) != n:
        _finding(
            findings,
            "LABEL002",
            "error",
            f"labels has length {len(raw)}, but input_ids has length {n}",
            line=line,
            field="labels",
            record_id=record_id,
        )
        return
    labels: list[int] = []
    for index, value in enumerate(raw):
        if not _is_int(value) or (value < 0 and value != config.ignore_index):
            _finding(
                findings,
                "LABEL003",
                "error",
                f"labels[{index}] must be a token id or ignore_index {config.ignore_index}",
                line=line,
                field="labels",
                record_id=record_id,
            )
            labels.append(config.ignore_index)
        else:
            labels.append(int(value))
    for index, (value, is_active) in enumerate(zip(labels, active, strict=True)):
        if not is_active and value != config.ignore_index:
            _finding(
                findings,
                "LABEL004",
                "error",
                f"padding position {index} has a non-ignored label",
                line=line,
                field="labels",
                record_id=record_id,
            )
            break
    if config.label_alignment == "opaque":
        return
    mismatches: list[int] = []
    if config.label_alignment == "copy":
        mismatches = [
            index
            for index, (label, token, flag) in enumerate(
                zip(labels, input_ids, active, strict=True)
            )
            if flag and label != config.ignore_index and label != token
        ]
    elif config.label_alignment == "next":
        for span in spans:
            for index in range(span.start, span.end - 1):
                if (
                    active[index]
                    and labels[index] != config.ignore_index
                    and labels[index] != input_ids[index + 1]
                ):
                    mismatches.append(index)
            last = span.end - 1
            if active[last] and labels[last] != config.ignore_index:
                mismatches.append(last)
    if mismatches:
        _finding(
            findings,
            "LABEL005" if config.label_alignment == "copy" else "LABEL006",
            "error",
            f"{len(mismatches)} labels do not match the requested "
            f"{config.label_alignment} alignment",
            line=line,
            field="labels",
            record_id=record_id,
        )


def _check_loss_mask(
    record: Mapping[str, Any],
    n: int,
    active: list[bool],
    *,
    line: int,
    findings: list[Finding],
    record_id: str | None,
) -> None:
    if "loss_mask" not in record:
        return
    raw = record["loss_mask"]
    if not isinstance(raw, list):
        _finding(
            findings,
            "LOSS001",
            "error",
            "loss_mask must be an array",
            line=line,
            field="loss_mask",
            record_id=record_id,
        )
        return
    if len(raw) != n:
        _finding(
            findings,
            "LOSS002",
            "error",
            f"loss_mask has length {len(raw)}, but input_ids has length {n}",
            line=line,
            field="loss_mask",
            record_id=record_id,
        )
        return
    for index, value in enumerate(raw):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            _finding(
                findings,
                "LOSS003",
                "error",
                f"loss_mask[{index}] must be a finite number in [0, 1]",
                line=line,
                field="loss_mask",
                record_id=record_id,
            )
            break
        if not active[index] and float(value) != 0:
            _finding(
                findings,
                "LOSS004",
                "error",
                f"padding position {index} has non-zero loss weight",
                line=line,
                field="loss_mask",
                record_id=record_id,
            )
            break


def _matrix(record: Mapping[str, Any]) -> tuple[str, list[list[Any]]] | None:
    for key in ("attention_mask", "attention_mask_2d", "block_attention_mask"):
        raw = record.get(key)
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return key, raw
    return None


def _check_attention_matrix(
    record: Mapping[str, Any],
    n: int,
    spans: list[_Segment],
    active: list[bool],
    config: CheckConfig,
    *,
    line: int,
    findings: list[Finding],
    record_id: str | None,
) -> None:
    matrix_info = _matrix(record)
    if matrix_info is None:
        return
    field, matrix = matrix_info
    if n * n > config.max_attention_cells:
        _finding(
            findings,
            "ATTN006",
            "warning",
            f"skipped {field} cell checks because {n}x{n} exceeds "
            f"max_attention_cells={config.max_attention_cells}",
            line=line,
            field=field,
            record_id=record_id,
        )
        return
    if len(matrix) != n or any(not isinstance(row, list) or len(row) != n for row in matrix):
        _finding(
            findings,
            "ATTN001",
            "error",
            f"{field} must be a {n}x{n} matrix",
            line=line,
            field=field,
            record_id=record_id,
        )
        return
    segment_for = [0] * n
    for segment_index, span in enumerate(spans):
        for index in range(span.start, span.end):
            segment_for[index] = segment_index
    bad_binary = False
    reported_future = False
    reported_cross = False
    reported_inactive = False
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if not _is_int(value) or value not in (0, 1):
                bad_binary = True
                break
            if value == 1 and col_index > row_index and not reported_future:
                _finding(
                    findings,
                    "ATTN002",
                    "error",
                    f"{field} allows future attention from row {row_index} to column {col_index}",
                    line=line,
                    field=field,
                    record_id=record_id,
                )
                reported_future = True
            if (
                value == 1
                and segment_for[row_index] != segment_for[col_index]
                and not reported_cross
            ):
                _finding(
                    findings,
                    "ATTN003",
                    "error",
                    f"{field} allows cross-segment attention at ({row_index}, {col_index})",
                    line=line,
                    field=field,
                    record_id=record_id,
                )
                reported_cross = True
            if (
                value == 1
                and (not active[row_index] or not active[col_index])
                and not reported_inactive
            ):
                _finding(
                    findings,
                    "ATTN004",
                    "error",
                    f"{field} attends to an inactive/padding position at "
                    f"({row_index}, {col_index})",
                    line=line,
                    field=field,
                    record_id=record_id,
                )
                reported_inactive = True
        if bad_binary:
            break
    if bad_binary:
        _finding(
            findings,
            "ATTN005",
            "error",
            f"{field} values must be 0 or 1",
            line=line,
            field=field,
            record_id=record_id,
        )
    reported_empty = False
    for row_index, row in enumerate(matrix):
        if active[row_index] and not any(value == 1 for value in row) and not reported_empty:
            _finding(
                findings,
                "ATTN007",
                "error",
                f"active attention row {row_index} has no allowed keys",
                line=line,
                field=field,
                record_id=record_id,
            )
            reported_empty = True


def _check_record(
    record: Mapping[str, Any],
    *,
    line: int,
    config: CheckConfig,
    findings: list[Finding],
    seen_ids: set[str],
) -> None:
    record_id = _display_id(record, line)
    if record_id is not None:
        if record_id in seen_ids:
            _finding(
                findings,
                "META001",
                "error",
                f"duplicate record id {record_id!r}",
                line=line,
                field="id",
                record_id=record_id,
            )
        seen_ids.add(record_id)
    if "input_ids" not in record:
        _finding(
            findings,
            "INPUT003",
            "error",
            "record is missing input_ids",
            line=line,
            field="input_ids",
            record_id=record_id,
        )
        return
    input_ids = _int_list(
        record["input_ids"], name="input_ids", line=line, findings=findings, record_id=record_id
    )
    if input_ids is None:
        return
    n = len(input_ids)
    if n == 0:
        _finding(
            findings,
            "INPUT006",
            "error",
            "input_ids must not be empty",
            line=line,
            field="input_ids",
            record_id=record_id,
        )
        return
    if config.max_length is not None and n > config.max_length:
        _finding(
            findings,
            "INPUT007",
            "error",
            f"input_ids length {n} exceeds max_length={config.max_length}",
            line=line,
            field="input_ids",
            record_id=record_id,
        )
    if config.vocab_size is not None:
        for index, token in enumerate(input_ids):
            if token >= config.vocab_size:
                _finding(
                    findings,
                    "INPUT010",
                    "error",
                    f"input_ids[{index}]={token} is outside vocab_size={config.vocab_size}",
                    line=line,
                    field="input_ids",
                    record_id=record_id,
                )
                break
    if config.pad_id is not None:
        first_pad = next(
            (index for index, token in enumerate(input_ids) if token == config.pad_id), None
        )
        if first_pad is not None and any(token != config.pad_id for token in input_ids[first_pad:]):
            _finding(
                findings,
                "PAD002",
                "error",
                f"non-pad token follows pad token at position {first_pad}",
                line=line,
                field="input_ids",
                record_id=record_id,
            )
    spans, packed = _parse_segments(
        record, n, config, line=line, findings=findings, record_id=record_id
    )
    if not spans:
        spans = [_Segment(0, n)]
    active = _active_mask(record, n, config, line=line, findings=findings, record_id=record_id)
    _check_positions(
        record, n, spans, active, config, line=line, findings=findings, record_id=record_id
    )
    _check_labels(
        record,
        n,
        spans,
        active,
        input_ids,
        config,
        line=line,
        findings=findings,
        record_id=record_id,
    )
    _check_loss_mask(record, n, active, line=line, findings=findings, record_id=record_id)
    _check_attention_matrix(
        record, n, spans, active, config, line=line, findings=findings, record_id=record_id
    )
    if config.require_eos and config.eos_id is None:
        _finding(
            findings,
            "EOS002",
            "error",
            "require_eos is enabled but eos_id was not configured",
            line=line,
            field="eos_id",
            record_id=record_id,
        )
    elif config.require_eos and config.eos_id is not None:
        for span in spans:
            last_active = next(
                (index for index in range(span.end - 1, span.start - 1, -1) if active[index]), None
            )
            if last_active is not None and input_ids[last_active] != config.eos_id:
                _finding(
                    findings,
                    "EOS001",
                    "error",
                    f"segment ending at {span.end} does not end with eos_id={config.eos_id}",
                    line=line,
                    field="input_ids",
                    record_id=record_id,
                )
    if packed and "attention_mask" in record and not _matrix(record):
        _finding(
            findings,
            "ATTN008",
            "warning",
            "packed record has only a 1-D attention_mask; cross-segment "
            "isolation cannot be verified",
            line=line,
            field="attention_mask",
            record_id=record_id,
        )


def audit_records(
    records: Iterable[tuple[int, Mapping[str, Any]]], config: CheckConfig | None = None
) -> AuditReport:
    """Audit already-decoded records, preserving caller-provided line numbers."""
    cfg = config or CheckConfig()
    findings: list[Finding] = []
    seen_ids: set[str] = set()
    checked = 0
    lines = 0
    truncated = False
    for line, record in records:
        lines = max(lines, line)
        if checked >= cfg.max_records:
            truncated = True
            _finding(
                findings,
                "INPUT009",
                "error",
                f"record limit max_records={cfg.max_records} reached",
                line=line,
            )
            break
        if not isinstance(record, Mapping):
            _finding(findings, "INPUT002", "error", "JSON value must be an object", line=line)
            continue
        checked += 1
        _check_record(record, line=line, config=cfg, findings=findings, seen_ids=seen_ids)
    return AuditReport(
        findings=tuple(findings),
        records_checked=checked,
        lines_read=lines,
        truncated=truncated,
        config=_config_dict(cfg),
    )


def _config_dict(config: CheckConfig) -> dict[str, Any]:
    return {
        "max_length": config.max_length,
        "max_records": config.max_records,
        "max_line_bytes": config.max_line_bytes,
        "max_attention_cells": config.max_attention_cells,
        "pad_id": config.pad_id,
        "eos_id": config.eos_id,
        "vocab_size": config.vocab_size,
        "ignore_index": config.ignore_index,
        "segment_field": config.segment_field,
        "position_mode": config.position_mode,
        "label_alignment": config.label_alignment,
        "require_segments": config.require_segments,
        "require_positions": config.require_positions,
        "require_eos": config.require_eos,
    }


def _iter_jsonl(
    stream: TextIO | BinaryIO,
    config: CheckConfig,
    findings: list[Finding],
    line_counter: list[int],
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    for line_number, raw_value in enumerate(stream, start=1):
        line_counter[0] = line_number
        raw = cast(str | bytes, raw_value)
        if isinstance(raw, bytes):
            size = len(raw)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                _finding(
                    findings, "INPUT001", "error", "line is not valid UTF-8 JSON", line=line_number
                )
                continue
        else:
            text = raw
            size = len(raw.encode("utf-8"))
        if size > config.max_line_bytes:
            _finding(
                findings,
                "INPUT008",
                "error",
                f"line exceeds max_line_bytes={config.max_line_bytes}",
                line=line_number,
            )
            continue
        if not text.strip():
            _finding(findings, "INPUT011", "warning", "blank line ignored", line=line_number)
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            _finding(findings, "INPUT001", "error", f"invalid JSON ({exc.msg})", line=line_number)
            continue
        if isinstance(value, Mapping):
            yield line_number, value
        else:
            _finding(
                findings, "INPUT002", "error", "JSON value must be an object", line=line_number
            )


def audit_jsonl(stream: TextIO | BinaryIO, config: CheckConfig | None = None) -> AuditReport:
    """Audit a UTF-8 JSONL stream with bounded line and record handling."""
    cfg = config or CheckConfig()
    parse_findings: list[Finding] = []
    line_counter = [0]
    report = audit_records(_iter_jsonl(stream, cfg, parse_findings, line_counter), cfg)
    findings = tuple(parse_findings) + report.findings
    return AuditReport(
        findings=findings,
        records_checked=report.records_checked,
        lines_read=max(report.lines_read, line_counter[0]),
        truncated=report.truncated,
        config=report.config,
    )
