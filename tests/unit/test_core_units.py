import pytest

from tender_extract.config import build_processing_config
from tender_extract.dedupe import DeduplicationEngine
from tender_extract.field_registry import get_expected_fields_for_modules
from tender_extract.pipeline import _chunk_offset
from tender_extract.pii import redact_for_cloud_llm
from tender_extract.schema import ChunkInfo


pytestmark = pytest.mark.unit


def _chunk(chunk_id: str, content: str, *, start_line: int = 1, fingerprint: str = "") -> ChunkInfo:
    return ChunkInfo(
        chunk_id=chunk_id,
        content=content,
        start_line=start_line,
        end_line=start_line,
        token_count=max(len(content) // 4, 1),
        fingerprint=fingerprint,
    )


def test_dedupe_marks_exact_duplicate_by_fingerprint():
    chunks = [
        _chunk("a", "同一段落", fingerprint="same"),
        _chunk("b", "同一段落", fingerprint="same"),
        _chunk("c", "不同段落", fingerprint="different"),
    ]

    results = DeduplicationEngine(enable_lsh=False).process_chunks(chunks)

    assert results[0].is_duplicate is False
    assert results[1].is_duplicate is True
    assert results[1].duplicate_of == "a"
    assert results[2].is_duplicate is False


def test_chunk_offset_uses_line_anchor_for_repeated_text():
    full = "第一章\n重复模板\n第二章\n重复模板\n尾部"
    second = _chunk("second", "重复模板", start_line=4, fingerprint="second")

    offset = _chunk_offset(full, second)

    assert offset == full.rfind("重复模板")


def test_cloud_redaction_masks_id_phone_and_email():
    raw = "身份证110101199003078888，电话13800138000，邮箱demo@example.com"

    redacted = redact_for_cloud_llm(raw)

    assert "110101199003078888" not in redacted
    assert "13800138000" not in redacted
    assert "demo@example.com" not in redacted
    assert "****" in redacted


def test_yaml_runtime_config_keeps_custom_patterns_and_base_flags():
    config = build_processing_config(
        {
            "base": {"use_dedupe": True, "use_ocr": False, "confidence_threshold": 0.81},
            "patterns": {
                "internal_code": [
                    {"pattern": r"内部编号[：:]\s*([A-Z]{2}-\d{3})", "confidence": 0.97}
                ]
            },
        },
        llm="none",
    )

    assert config.enable_dedupe is True
    assert config.use_ocr is False
    assert config.confidence_threshold == pytest.approx(0.81)
    assert "internal_code" in config.custom_patterns


def test_registry_exposes_rule_and_llm_only_fields_for_modules():
    fields = get_expected_fields_for_modules({"basic_info", "financial_info"})

    assert "project_name" in fields
    assert "construction_site" in fields
    assert "bid_amount" in fields
    assert "payment_terms" in fields
