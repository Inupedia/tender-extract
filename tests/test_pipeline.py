from pathlib import Path

from tender_extract.extraction_engine import ExtractionEngine, chinese_amount_to_yuan
from tender_extract.llm_providers import get_provider
from tender_extract.llm_router import LLMRouter
from tender_extract.personnel_extractor import PersonnelExtractor
from tender_extract.pii import mask_id_card, mask_result
from tender_extract.pipeline import ExtractionPipeline
from tender_extract.schema import (
    EvidenceSpan,
    ExtractedField,
    ExtractionResult,
    DocumentMetadata,
    PersonnelRecord,
    ProcessingConfig,
)


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example.md"


def test_example_md_extracts_core_fields():
    pipeline = ExtractionPipeline(ProcessingConfig(llm_provider="none"))
    result = pipeline.extract_file(str(EXAMPLE))
    assert result.fields["project_name"].primary_value.startswith("测试工程项目")
    assert result.fields["project_number"].primary_value == "TEST-2024-001"
    assert "测试工程集团" in (result.fields["bidder"].primary_value or "")
    amount = result.fields["bid_amount"]
    assert amount.values[0].normalized_value == "1000000.00"
    deposit = result.fields["deposit"]
    assert float(deposit.values[0].normalized_value) == 50000.0
    assert amount.confidence >= 0.9


def test_personnel_not_leaked_across_files(tmp_path):
    doc_a = tmp_path / "a.md"
    doc_b = tmp_path / "b.md"
    valid_id = _make_id("11010119900307")
    doc_a.write_text(
        f"项目名称：甲工程\n项目经理：赵一\n身份证号：{valid_id}\n",
        encoding="utf-8",
    )
    doc_b.write_text("项目名称：乙工程\n招标人：乙建设有限公司\n", encoding="utf-8")

    pipeline = ExtractionPipeline(ProcessingConfig(llm_provider="none", include_pii=True))
    result_a = pipeline.extract_file(str(doc_a))
    result_b = pipeline.extract_file(str(doc_b))
    assert any(p.name == "赵一" for p in result_a.personnel)
    assert all(p.name != "赵一" for p in result_b.personnel)


def test_id_card_masked_by_default(tmp_path):
    valid_id = _make_id("11010119900307")
    path = tmp_path / "p.md"
    path.write_text(f"项目经理：钱二\n身份证号：{valid_id}\n", encoding="utf-8")
    pipeline = ExtractionPipeline(ProcessingConfig(llm_provider="none", include_pii=False))
    result = pipeline.extract_file(str(path))
    assert result.personnel
    assert "****" in result.personnel[0].id_card
    assert valid_id not in result.personnel[0].id_card


def test_certificate_expiry_is_not_cert_number():
    text = "建造师证书编号：ABC123456789\n有效期至：2027年12月31日\n"
    certs = PersonnelExtractor().extract_certificates(text)
    assert certs
    assert certs[0].cert_number == "ABC123456789"
    assert "2027" in certs[0].expiry_date
    assert all(c.cert_type != "有效期" for c in certs)


def test_chinese_amount_conversion():
    assert chinese_amount_to_yuan("壹佰万元整") == 1_000_000
    assert chinese_amount_to_yuan("伍万元") == 50_000


def test_should_not_send_high_confidence_numbers_to_llm():
    router = LLMRouter(ProcessingConfig(llm_provider="none"))
    field = ExtractedField(
        field_name="project_number",
        field_type="project_number",
        values=[
            EvidenceSpan(
                value="TEST-2024-001",
                start=0,
                end=12,
                confidence=0.95,
                source="regex_enhanced",
            )
        ],
        primary_value="TEST-2024-001",
        confidence=0.95,
    )
    assert router.should_use_llm(field, 0.7) is False


def test_provider_aliases():
    assert get_provider("claude").id == "anthropic"
    assert get_provider("qwen").kind == "openai_compat"
    assert get_provider("deepseek").base_url.startswith("https://api.deepseek.com")


def test_mask_id_card():
    assert mask_id_card("110101199003078888") == "110101****8888"


def test_config_path_used(tmp_path):
    from tender_extract.config import build_processing_config, load_yaml

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("llm:\n  provider: none\nbase:\n  confidence_threshold: 0.42\n", encoding="utf-8")
    processing = build_processing_config(load_yaml(cfg), llm="none")
    assert processing.confidence_threshold == 0.42


def _make_id(prefix14: str) -> str:
    """prefix14 = 6-digit region + YYYYMMDD, then add 3 seq digits + checksum."""
    body = prefix14 + "852"
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_chars = "10X98765432"
    total = sum(int(body[i]) * weights[i] for i in range(17))
    return body + check_chars[total % 11]
