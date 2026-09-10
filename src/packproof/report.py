"""Content-free report renderers."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .model import AuditReport, Severity


def _payload(report: AuditReport) -> dict[str, Any]:
    by_code = Counter(f.code for f in report.findings)
    return {
        "schema_version": "1",
        "tool": {"name": "packproof", "version": "0.1.0"},
        "summary": report.summary(),
        "config": dict(report.config),
        "finding_counts": dict(sorted(by_code.items())),
        "findings": [finding.as_dict() for finding in report.findings],
    }


def render_json(report: AuditReport) -> str:
    return json.dumps(_payload(report), indent=2, sort_keys=True) + "\n"


def render_text(report: AuditReport) -> str:
    summary = report.summary()
    lines = [
        f"packproof: {summary['status']}",
        f"records={summary['records_checked']} lines={summary['lines_read']} "
        f"errors={summary['errors']} warnings={summary['warnings']} infos={summary['infos']}",
    ]
    if summary["truncated"]:
        lines.append("input was truncated at the configured record limit")
    for finding in report.findings:
        location = f"line {finding.line}" if finding.line is not None else "input"
        field = f" [{finding.field}]" if finding.field else ""
        lines.append(
            f"{finding.severity.upper()} {finding.code} {location}{field}: {finding.message}"
        )
    return "\n".join(lines) + "\n"


def render_markdown(report: AuditReport) -> str:
    summary = report.summary()
    lines = [
        f"# packproof report: {summary['status']}",
        "",
        f"Checked **{summary['records_checked']}** records across "
        f"**{summary['lines_read']}** lines. Errors: **{summary['errors']}** · "
        f"warnings: **{summary['warnings']}** · infos: **{summary['infos']}**.",
        "",
        "| Severity | Code | Line | Field | Finding |",
        "| --- | --- | ---: | --- | --- |",
    ]
    if report.findings:
        for finding in report.findings:
            lines.append(
                f"| {finding.severity} | `{finding.code}` | {finding.line or ''} | "
                f"{finding.field or ''} | {finding.message} |"
            )
    else:
        lines.append("| — | — | — | — | No findings. |")
    lines.extend(["", "The report intentionally omits token payloads and model data.", ""])
    return "\n".join(lines)


def render_sarif(report: AuditReport) -> str:
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    level_map: dict[Severity, str] = {"error": "error", "warning": "warning", "info": "note"}
    for finding in report.findings:
        rules.setdefault(
            finding.code, {"id": finding.code, "shortDescription": {"text": finding.message}}
        )
        result: dict[str, Any] = {
            "ruleId": finding.code,
            "level": level_map[finding.severity],
            "message": {"text": finding.message},
        }
        if finding.line is not None:
            result["locations"] = [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": "input.jsonl"},
                        "region": {"startLine": finding.line},
                    }
                }
            ]
        results.append(result)
    payload = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "packproof",
                        "version": "0.1.0",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render(report: AuditReport, format_name: str) -> str:
    """Render using one of text, json, markdown, or sarif."""
    if format_name == "text":
        return render_text(report)
    if format_name == "json":
        return render_json(report)
    if format_name == "markdown":
        return render_markdown(report)
    if format_name == "sarif":
        return render_sarif(report)
    raise ValueError(f"unknown format: {format_name}")
