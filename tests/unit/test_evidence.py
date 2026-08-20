from pathlib import Path

import pytest

from tender_extract.document_parser import ParsedDocument
from tender_extract.evidence import EvidenceLocator
from tender_extract.schema import EvidenceSpan, ExtractedField


pytestmark = pytest.mark.unit


def _field(span: EvidenceSpan) -> dict[str, ExtractedField]:
    return {
        "project_number": ExtractedField(
            field_name="project_number",
            field_type="project_number",
            values=[span],
            primary_value=span.value,
            confidence=span.confidence,
        )
    }


def test_markdown_evidence_has_document_section_lines_and_source_span(tmp_path: Path):
    content = (
        "# 招标文件\n\n"
        "## 项目概况\n\n"
        "项目名称：青江水库除险加固工程\n"
        "项目编号：QJ-2026-001\n"
    )
    match = "项目编号：QJ-2026-001"
    start = content.index(match)
    value_start = content.index("QJ-2026-001")
    span = EvidenceSpan(
        value="QJ-2026-001",
        start=start,
        end=start + len(match),
        confidence=0.95,
        source="regex_enhanced",
    )
    parsed = ParsedDocument(
        content=content,
        filename="tender.md",
        original_format="md",
    )

    EvidenceLocator(parsed, tmp_path / "tender.md", "tender-v1").enrich_fields(_field(span))

    assert span.location is not None
    assert span.location.document_id == "tender-v1"
    assert span.location.page is None
    assert span.location.bbox is None
    assert span.location.section_path == ["招标文件", "项目概况"]
    assert span.location.line_start == 6
    assert span.location.line_end == 6
    assert span.location.source_text == "QJ-2026-001"
    assert span.location.source_start == value_start
    assert span.location.source_end == value_start + len("QJ-2026-001")
    assert span.start == value_start
    assert span.end == value_start + len("QJ-2026-001")


def test_unlocated_llm_span_recovers_offsets_from_value(tmp_path: Path):
    content = "# 项目概况\n\n建设地点：成都市青羊区\n"
    span = EvidenceSpan(
        value="成都市青羊区",
        start=-1,
        end=-1,
        confidence=0.82,
        source="llm",
        ref="建设地点：成都市青羊区",
    )
    fields = {
        "construction_site": ExtractedField(
            field_name="construction_site",
            field_type="construction_site",
            values=[span],
            primary_value=span.value,
            confidence=span.confidence,
        )
    }
    parsed = ParsedDocument(
        content=content,
        filename="site.md",
        original_format="md",
    )

    EvidenceLocator(parsed, tmp_path / "site.md", "site-doc").enrich_fields(fields)

    assert span.start == content.index("成都市青羊区")
    assert span.end == span.start + len("成都市青羊区")
    assert span.location is not None
    assert span.location.section_path == ["项目概况"]
    assert span.location.source_text == "成都市青羊区"
