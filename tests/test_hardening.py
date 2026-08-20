from pathlib import Path

from tender_extract.audit_merge import AuditPreservingFieldMerger
from tender_extract.config import build_processing_config
from tender_extract.configurable_engine import ConfigurableExtractionEngine
from tender_extract.field_registry import FIELD_REGISTRY, get_expected_fields_for_modules
from tender_extract.llm_router import LLMRouter
from tender_extract.pii import redact_for_cloud_llm
from tender_extract.pipeline import ExtractionPipeline
from tender_extract.schema import (
    EvidenceSpan,
    ExtractedField,
    LLMRequest,
    LLMResponse,
    ProcessingConfig,
)


def test_yaml_patterns_are_applied_to_active_engine():
    cfg = build_processing_config(
        {
            "patterns": {
                "custom_reference": [
                    {
                        "pattern": r"内部编号[：:]\s*([A-Z]{2}-\d{3})",
                        "confidence": 0.99,
                    }
                ]
            }
        },
        llm="none",
    )
    engine = ConfigurableExtractionEngine(cfg.custom_patterns)
    result = engine.extract_all_fields("内部编号：AB-123")
    assert result["custom_reference"].primary_value == "AB-123"
    assert result["custom_reference"].confidence == 0.99


def test_registry_covers_router_fields_and_rule_fields():
    expected = get_expected_fields_for_modules({"basic_info", "financial_info"})
    assert "project_name" in expected
    assert "construction_site" in expected
    assert "bid_amount" in expected
    assert FIELD_REGISTRY["project_name"].has_rule_patterns is True
    assert FIELD_REGISTRY["construction_site"].has_rule_patterns is False


def test_cloud_llm_request_redacts_pii_before_prompt():
    router = LLMRouter(
        ProcessingConfig(
            llm_provider="siliconflow",
            llm_api_key="test-key",
            persist_llm_cache=False,
            redact_pii_for_cloud_llm=True,
        )
    )
    request = LLMRequest(
        chunk_text="项目经理张三，身份证号110101199003078888，电话13800138000，邮箱a@b.com",
        field_name="project_manager",
        field_type="project_manager",
    )
    outbound = router._prepare_request(request)
    assert "110101199003078888" not in outbound.chunk_text
    assert "13800138000" not in outbound.chunk_text
    assert "a@b.com" not in outbound.chunk_text
    assert "****" in outbound.chunk_text


def test_ollama_keeps_local_text_unredacted():
    router = LLMRouter(
        ProcessingConfig(
            llm_provider="ollama",
            persist_llm_cache=False,
            redact_pii_for_cloud_llm=True,
        )
    )
    request = LLMRequest(
        chunk_text="身份证号110101199003078888",
        field_name="project_manager",
        field_type="project_manager",
    )
    assert router._prepare_request(request).chunk_text == request.chunk_text


def test_redact_for_cloud_llm_masks_common_sensitive_values():
    text = redact_for_cloud_llm("110101199003078888 13800138000 user@example.com")
    assert "110101199003078888" not in text
    assert "13800138000" not in text
    assert "user@example.com" not in text


def test_audit_merger_keeps_alternative_candidates():
    field = ExtractedField(
        field_name="bid_amount",
        field_type="bid_amount",
        values=[
            EvidenceSpan(
                value="100万元",
                normalized_value="1000000.00",
                start=10,
                end=20,
                confidence=0.95,
                source="regex_enhanced",
            ),
            EvidenceSpan(
                value="80万元",
                normalized_value="800000.00",
                start=30,
                end=40,
                confidence=0.8,
                source="regex_enhanced",
            ),
        ],
        conflicts=["存在2个不同值"],
    )
    merged = AuditPreservingFieldMerger().resolve_conflicts({"bid_amount": field})["bid_amount"]
    assert merged.primary_value == "100万元"
    assert len(merged.values) == 2
    assert merged.conflicts == ["存在2个不同值"]


def test_pipeline_recovers_field_that_rules_never_created(tmp_path: Path, monkeypatch):
    path = tmp_path / "missing.md"
    path.write_text("# 项目概况\n项目名称：某水库除险加固工程\n建设地点位于测试县。\n", encoding="utf-8")
    pipeline = ExtractionPipeline(
        ProcessingConfig(
            llm_provider="siliconflow",
            llm_api_key="test-key",
            persist_llm_cache=False,
            recover_missing_fields_with_llm=True,
        )
    )

    def fake_extract(request: LLMRequest):
        if request.field_name == "construction_site":
            return LLMResponse(
                field_name=request.field_name,
                extracted_values=["测试县"],
                confidence=0.88,
                reasoning="测试",
            )
        return LLMResponse(
            field_name=request.field_name,
            extracted_values=[],
            confidence=0.0,
            reasoning="",
        )

    monkeypatch.setattr(pipeline.llm, "extract_with_llm", fake_extract)
    result = pipeline.extract_file(str(path))
    assert result.fields["construction_site"].primary_value == "测试县"
    assert result.fields["construction_site"].values[0].source == "llm"
    assert result.fields["construction_site"].values[0].start == -1
