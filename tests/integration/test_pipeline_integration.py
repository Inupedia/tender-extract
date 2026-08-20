from pathlib import Path

import pytest

from tender_extract.pipeline import ExtractionPipeline
from tender_extract.schema import ChunkInfo, ProcessingConfig


pytestmark = pytest.mark.integration


def test_pipeline_rule_path_returns_structured_auditable_result(tmp_path: Path):
    path = tmp_path / "tender.md"
    path.write_text(
        "# 招标公告\n"
        "项目名称：测试水库除险加固工程\n"
        "项目编号：TEST-2026-001\n"
        "招标人：测试水务有限公司\n"
        "投标人：测试工程集团有限公司\n"
        "投标报价：100万元\n"
        "投标保证金：5万元\n"
        "投标截止时间：2026年9月30日\n",
        encoding="utf-8",
    )

    pipeline = ExtractionPipeline(
        ProcessingConfig(llm_provider="none", use_ocr=False, persist_llm_cache=False)
    )
    result = pipeline.extract_file(str(path))

    assert result.metadata.extraction_stats["original_format"] == "md"
    assert result.fields["project_number"].primary_value == "TEST-2026-001"
    assert float(result.fields["bid_amount"].values[0].normalized_value) == 1_000_000
    assert float(result.fields["deposit"].values[0].normalized_value) == 50_000
    assert result.llm_calls == 0
    assert result.chunks_processed >= 1
    assert any(span.ref for span in result.fields["bid_amount"].values)


def test_pipeline_applies_yaml_style_custom_pattern(tmp_path: Path):
    path = tmp_path / "custom.md"
    path.write_text("项目名称：自定义规则测试工程\n内部编号：ZX-321\n", encoding="utf-8")
    config = ProcessingConfig(
        llm_provider="none",
        use_ocr=False,
        persist_llm_cache=False,
        custom_patterns={
            "internal_code": [
                {"pattern": r"内部编号[：:]\s*([A-Z]{2}-\d{3})", "confidence": 0.98}
            ]
        },
    )

    result = ExtractionPipeline(config).extract_file(str(path))

    assert result.fields["internal_code"].primary_value == "ZX-321"
    assert result.fields["internal_code"].confidence == pytest.approx(0.98)


def test_pipeline_dedupe_stage_removes_exact_duplicate_chunks():
    pipeline = ExtractionPipeline(
        ProcessingConfig(
            llm_provider="none",
            use_ocr=False,
            enable_dedupe=True,
            persist_llm_cache=False,
        )
    )
    chunks = [
        ChunkInfo(
            chunk_id="a",
            content="重复章节",
            start_line=1,
            end_line=1,
            token_count=4,
            fingerprint="same",
        ),
        ChunkInfo(
            chunk_id="b",
            content="重复章节",
            start_line=2,
            end_line=2,
            token_count=4,
            fingerprint="same",
        ),
        ChunkInfo(
            chunk_id="c",
            content="唯一章节",
            start_line=3,
            end_line=3,
            token_count=4,
            fingerprint="unique",
        ),
    ]

    kept, deduped = pipeline._dedupe_chunks(chunks)

    assert [chunk.chunk_id for chunk in kept] == ["a", "c"]
    assert deduped == 1
