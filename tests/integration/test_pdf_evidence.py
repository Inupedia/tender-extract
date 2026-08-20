from pathlib import Path

import pytest

from tender_extract.pipeline import ExtractionPipeline
from tender_extract.schema import ProcessingConfig


pytestmark = pytest.mark.integration


def test_pdf_extraction_attaches_page_and_bbox(tmp_path: Path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "evidence.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Tender Code: TEST-2026-001")
    doc.save(path)
    doc.close()

    config = ProcessingConfig(
        use_ocr=False,
        use_modules=False,
        include_pii=True,
        custom_patterns={
            "document_code": [
                {
                    "pattern": r"Tender Code:\s*([A-Z0-9-]+)",
                    "confidence": 0.95,
                    "description": "integration-pdf-code",
                }
            ]
        },
    )

    result = ExtractionPipeline(config).extract_file(
        str(path), document_id="pdf-evidence-doc"
    )

    assert result.metadata.total_pages == 1
    assert result.metadata.extraction_stats["page_count"] == 1

    span = result.fields["document_code"].values[0]
    assert span.location is not None
    assert span.location.document_id == "pdf-evidence-doc"
    assert span.location.page == 1
    assert span.location.source_text == "TEST-2026-001"
    assert span.location.bbox is not None
    assert span.location.bbox.x1 > span.location.bbox.x0
    assert span.location.bbox.y1 > span.location.bbox.y0
    assert span.location.bbox.page_width > 0
    assert span.location.bbox.page_height > 0
    assert span.location.bbox.coordinate_system == "pdf_points_top_left"

    stats = result.metadata.extraction_stats
    assert stats["evidence_span_count"] >= 1
    assert stats["evidence_page_count"] >= 1
    assert stats["evidence_bbox_count"] >= 1
