from __future__ import annotations

import io
import json

from packproof import AuditReport, CheckConfig, audit_jsonl, audit_records


def audit(payload: object, config: CheckConfig | None = None) -> AuditReport:
    return audit_jsonl(io.StringIO(json.dumps(payload) + "\n"), config)


def test_clean_packed_record_with_block_causal_mask() -> None:
    matrix = [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
    ]
    report = audit(
        {
            "id": "clean",
            "input_ids": [101, 11, 12, 102, 101, 21, 102],
            "seq_lengths": [4, 3],
            "position_ids": [0, 1, 2, 3, 0, 1, 2],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1],
            "attention_mask_2d": matrix,
            "labels": [101, 11, 12, 102, 101, 21, 102],
            "loss_mask": [0, 1, 1, 1, 0, 1, 1],
        },
        CheckConfig(label_alignment="copy", require_eos=True, eos_id=102),
    )
    assert report.ok
    assert report.errors == 0
    assert report.records_checked == 1


def test_cross_segment_attention_and_positions_are_errors() -> None:
    report = audit(
        {
            "input_ids": [1, 2, 3, 4],
            "seq_lengths": [2, 2],
            "position_ids": [0, 1, 2, 3],
            "attention_mask_2d": [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 1],
            ],
        },
        CheckConfig(position_mode="reset"),
    )
    codes = {finding.code for finding in report.findings}
    assert "ATTN003" in codes
    assert "POS003" in codes


def test_auto_positions_accept_global_indexes() -> None:
    report = audit({"input_ids": [1, 2, 3, 4], "seq_lengths": [2, 2], "position_ids": [0, 1, 2, 3]})
    assert report.ok


def test_segment_id_repetition_is_rejected() -> None:
    report = audit({"input_ids": [1, 2, 3, 4], "segment_ids": [0, 0, 1, 0]})
    assert "SEG005" in {finding.code for finding in report.findings}


def test_cu_seqlens_and_length_mismatch() -> None:
    report = audit({"input_ids": [1, 2, 3], "cu_seqlens": [0, 2, 2, 3]})
    codes = {finding.code for finding in report.findings}
    assert "SEG002" in codes

    report = audit({"input_ids": [1, 2, 3], "seq_lengths": [2]})
    assert "SEG003" in {finding.code for finding in report.findings}


def test_masks_labels_and_loss_masks() -> None:
    report = audit(
        {
            "input_ids": [9, 1, 0],
            "attention_mask": [1, 0, 1],
            "labels": [8, 8, 0],
            "loss_mask": [1, 1, 0],
        },
        CheckConfig(pad_id=0, label_alignment="copy"),
    )
    codes = {finding.code for finding in report.findings}
    assert {"MASK004", "PAD001", "LABEL004", "LABEL005", "LOSS004"}.issubset(codes)


def test_next_label_alignment_does_not_cross_segment_boundary() -> None:
    report = audit(
        {
            "input_ids": [10, 11, 20, 21],
            "seq_lengths": [2, 2],
            "labels": [11, -100, 21, -100],
        },
        CheckConfig(label_alignment="next"),
    )
    assert report.ok


def test_requirements_and_eos() -> None:
    report = audit(
        {"input_ids": [1, 2]}, CheckConfig(require_segments=True, require_positions=True)
    )
    codes = {finding.code for finding in report.findings}
    assert "SEG006" in codes
    assert "POS005" not in codes  # only packed records require positions

    report = audit(
        {"input_ids": [1, 2], "seq_lengths": [2]}, CheckConfig(require_eos=True, eos_id=9)
    )
    assert "EOS001" in {finding.code for finding in report.findings}


def test_input_bounds_and_parse_errors() -> None:
    stream = io.StringIO("not json\n[]\n\n{" + '"input_ids":[1,2]}' + "\n")
    report = audit_jsonl(stream, CheckConfig(max_records=1, max_line_bytes=100))
    codes = {finding.code for finding in report.findings}
    assert {"INPUT001", "INPUT002", "INPUT011"}.issubset(codes)
    assert report.records_checked == 1
    assert report.lines_read == 4


def test_audit_records_preserves_line_numbers() -> None:
    report = audit_records([(7, {"input_ids": []})])
    assert report.findings[0].line == 7


def test_vocabulary_and_duplicate_ids() -> None:
    report = audit_jsonl(
        io.StringIO('{"id":"a","input_ids":[1,9]}\n{"id":"a","input_ids":[2]}\n'),
        CheckConfig(vocab_size=5),
    )
    codes = {finding.code for finding in report.findings}
    assert "INPUT010" in codes
    assert "META001" in codes


def test_attention_matrix_shape_and_values() -> None:
    report = audit(
        {
            "input_ids": [1, 2],
            "attention_mask_2d": [[1, 2], [0]],
        }
    )
    codes = {finding.code for finding in report.findings}
    assert "ATTN001" in codes

    report = audit({"input_ids": [1, 2], "attention_mask_2d": [[1, 0], [0, 3]]})
    assert "ATTN005" in {finding.code for finding in report.findings}


def test_max_records_is_a_hard_bound() -> None:
    stream = io.StringIO('{"input_ids":[1]}\n{"input_ids":[2]}\n')
    report = audit_jsonl(stream, CheckConfig(max_records=1))
    assert report.records_checked == 1
    assert report.truncated
    assert "INPUT009" in {finding.code for finding in report.findings}
