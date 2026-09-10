from __future__ import annotations

import json

from packproof import AuditReport, Finding
from packproof.report import render_json, render_markdown, render_sarif, render_text


def sample() -> AuditReport:
    return AuditReport(
        findings=(
            Finding("POS003", "error", "position_ids do not reset", line=4, field="position_ids"),
        ),
        records_checked=2,
        lines_read=4,
        config={"position_mode": "reset"},
    )


def test_json_is_stable_and_payload_free() -> None:
    payload = json.loads(render_json(sample()))
    assert payload["summary"]["status"] == "FAIL"
    assert payload["findings"][0]["code"] == "POS003"
    assert "token" not in render_json(sample()).lower()


def test_text_and_markdown_include_summary() -> None:
    assert "packproof: FAIL" in render_text(sample())
    markdown = render_markdown(sample())
    assert "# packproof report: FAIL" in markdown
    assert "`POS003`" in markdown


def test_sarif_has_line_location() -> None:
    payload = json.loads(render_sarif(sample()))
    result = payload["runs"][0]["results"][0]
    assert result["ruleId"] == "POS003"
    assert result["locations"][0]["physicalLocation"]["region"]["startLine"] == 4
