from pathlib import Path

import pytest

from tender_extract.evaluation import GoldCase, load_gold_dataset, normalize_value, score_case
from tender_extract.schema import (
    DocumentMetadata,
    EvidenceSpan,
    ExtractedField,
    ExtractionResult,
)


pytestmark = pytest.mark.unit


def _result(fields: dict[str, ExtractedField]) -> ExtractionResult:
    return ExtractionResult(
        metadata=DocumentMetadata(
            filename="sample.md",
            file_size=1,
            total_lines=1,
            total_chunks=1,
            processing_time=0.01,
        ),
        fields=fields,
    )


def _field(name: str, *values: str) -> ExtractedField:
    spans = [
        EvidenceSpan(
            value=value,
            start=0,
            end=len(value),
            confidence=0.9,
            source="test",
        )
        for value in values
    ]
    return ExtractedField(
        field_name=name,
        field_type=name,
        values=spans,
        primary_value=values[0] if values else None,
        confidence=0.9 if values else 0.0,
    )


def test_normalize_value_ignores_whitespace_and_fullwidth_punctuation():
    assert normalize_value(" 四川（水利） 公司 ") == normalize_value("四川(水利)公司")


def test_score_case_reports_false_positive_and_false_negative():
    case = GoldCase(
        case_id="case-1",
        document=Path("sample.md"),
        expected={
            "project_number": ["TEST-2026-001"],
            "bidder": ["四川测试有限公司"],
        },
    )
    result = _result(
        {
            "project_number": _field("project_number", "TEST-2026-001", "WRONG-002"),
        }
    )

    metrics, failures, exact = score_case(case, result)

    assert not exact
    assert metrics["project_number"].true_positive == 1
    assert metrics["project_number"].false_positive == 1
    assert metrics["project_number"].false_negative == 0
    assert metrics["bidder"].false_negative == 1
    assert {failure.field_name for failure in failures} == {"project_number", "bidder"}


def test_load_gold_dataset_resolves_documents_relative_to_jsonl(tmp_path: Path):
    fixture = tmp_path / "fixtures" / "sample.md"
    fixture.parent.mkdir()
    fixture.write_text("项目编号：TEST-2026-001", encoding="utf-8")
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(
        '{"id":"sample","document":"fixtures/sample.md","expected":{"project_number":"TEST-2026-001"}}\n',
        encoding="utf-8",
    )

    cases = load_gold_dataset(dataset)

    assert len(cases) == 1
    assert cases[0].document == fixture
    assert cases[0].expected["project_number"] == ["TEST-2026-001"]
