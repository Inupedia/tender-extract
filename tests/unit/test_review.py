import json
from pathlib import Path

import pytest

from tender_extract.evaluation import load_gold_dataset
from tender_extract.review import ReviewStore, build_review_items
from tender_extract.schema import (
    DocumentMetadata,
    EvidenceSpan,
    ExtractedField,
    ExtractionResult,
)


pytestmark = pytest.mark.unit


def _result(*fields: ExtractedField) -> ExtractionResult:
    return ExtractionResult(
        metadata=DocumentMetadata(
            filename="sample.md",
            file_size=100,
            total_lines=5,
            total_chunks=1,
            processing_time=0.01,
        ),
        fields={field.field_name: field for field in fields},
    )


def _field(
    name: str,
    value: str,
    confidence: float,
    *,
    source: str = "regex_enhanced",
    conflicts: list[str] | None = None,
) -> ExtractedField:
    return ExtractedField(
        field_name=name,
        field_type=name,
        primary_value=value,
        confidence=confidence,
        conflicts=conflicts or [],
        values=[
            EvidenceSpan(
                value=value,
                start=0,
                end=len(value),
                confidence=confidence,
                source=source,
                ref=f"原文证据：{value}",
            )
        ],
    )


def test_build_review_items_selects_low_conflict_and_llm_fields(tmp_path: Path):
    document = tmp_path / "sample.md"
    document.write_text("demo", encoding="utf-8")
    result = _result(
        _field("low", "低置信值", 0.6),
        _field("conflict", "冲突值", 0.95, conflicts=["存在冲突"]),
        _field("llm_only", "模型恢复值", 0.93, source="llm"),
        _field("safe", "高置信规则值", 0.95),
    )

    items = build_review_items(result, document, confidence_threshold=0.7)

    by_name = {item.field_name: item for item in items}
    assert set(by_name) == {"low", "conflict", "llm_only"}
    assert by_name["low"].reasons == ["low_confidence"]
    assert by_name["conflict"].reasons == ["conflict"]
    assert by_name["llm_only"].reasons == ["llm_recovered"]


def test_review_store_upsert_is_idempotent_and_preserves_resolution(tmp_path: Path):
    document = tmp_path / "sample.md"
    document.write_text("demo", encoding="utf-8")
    queue = ReviewStore(tmp_path / "review.jsonl")
    result = _result(_field("project_name", "测试项目", 0.6))

    first = queue.add_result(result, document, confidence_threshold=0.7)
    second = queue.add_result(result, document, confidence_threshold=0.7)

    assert len(first) == 1
    assert second == []
    assert len(queue.load()) == 1

    resolved = queue.resolve(first[0].id, "correct", ["修正项目"], reviewer="tester")
    assert resolved.status == "resolved"
    assert resolved.decision is not None
    assert resolved.decision.values == ["修正项目"]

    queue.add_result(result, document, confidence_threshold=0.7)
    persisted = queue.load()[0]
    assert persisted.status == "resolved"
    assert persisted.decision is not None
    assert persisted.decision.values == ["修正项目"]


def test_review_export_creates_gold_dataset_including_rejections(tmp_path: Path):
    document = tmp_path / "sample.md"
    document.write_text("项目名称：错误项目\n投标人：误识别公司", encoding="utf-8")
    queue = ReviewStore(tmp_path / "review.jsonl")
    result = _result(
        _field("project_name", "错误项目", 0.6),
        _field("bidder", "误识别公司", 0.6),
    )
    items = queue.add_result(result, document, confidence_threshold=0.7)
    by_name = {item.field_name: item for item in items}

    queue.resolve(by_name["project_name"].id, "correct", ["正确项目"])
    queue.resolve(by_name["bidder"].id, "reject")

    gold_path = tmp_path / "gold-reviewed.jsonl"
    count = queue.export_gold(gold_path)

    assert count == 1
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    assert payload["expected"]["project_name"] == ["正确项目"]
    assert payload["expected"]["bidder"] == []
    assert payload["tags"] == ["human-reviewed"]

    cases = load_gold_dataset(gold_path)
    assert len(cases) == 1
    assert cases[0].document == document.resolve()


def test_correct_requires_value(tmp_path: Path):
    document = tmp_path / "sample.md"
    document.write_text("demo", encoding="utf-8")
    queue = ReviewStore(tmp_path / "review.jsonl")
    item = queue.add_result(
        _result(_field("project_name", "候选项目", 0.6)),
        document,
        confidence_threshold=0.7,
    )[0]

    with pytest.raises(ValueError, match="correct requires"):
        queue.resolve(item.id, "correct")
