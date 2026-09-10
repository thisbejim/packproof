from __future__ import annotations

import json
from pathlib import Path

from packproof.cli import main


def test_cli_clean_fixture(capsys) -> None:  # type: ignore[no-untyped-def]
    fixture = Path(__file__).parents[1] / "examples" / "clean.jsonl"
    assert main([str(fixture), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["status"] == "PASS"


def test_cli_broken_fixture_returns_one(capsys) -> None:  # type: ignore[no-untyped-def]
    fixture = Path(__file__).parents[1] / "examples" / "broken.jsonl"
    assert main([str(fixture), "--position-mode", "reset", "--format", "text"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_stdin_and_output_file(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "report.json"
    import sys
    from io import StringIO

    original = sys.stdin
    try:
        sys.stdin = StringIO('{"input_ids":[1]}\n')
        assert main(["-", "--format", "json", "--output", str(output)]) == 0
    finally:
        sys.stdin = original
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["status"] == "PASS"
    assert capsys.readouterr().out == ""
