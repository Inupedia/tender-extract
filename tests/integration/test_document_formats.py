from pathlib import Path

import pytest

from tender_extract.document_parser import DocumentParser


pytestmark = pytest.mark.integration


def test_markdown_parser_round_trip(tmp_path: Path):
    path = tmp_path / "sample.md"
    path.write_text("# 招标公告\n\n项目名称：测试工程\n", encoding="utf-8")

    parsed = DocumentParser(use_ocr=False).parse(str(path))

    assert parsed.original_format == "md"
    assert parsed.filename == "sample.md"
    assert "项目名称：测试工程" in parsed.content


def test_pdf_parser_extracts_text_page(tmp_path: Path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Tender Integration PDF TEST-2026-001")
    doc.save(path)
    doc.close()

    parsed = DocumentParser(use_ocr=False).parse(str(path))

    assert parsed.original_format == "pdf"
    assert parsed.total_pages == 1
    assert parsed.metadata["text_pages"] == 1
    assert "Tender Integration PDF TEST-2026-001" in parsed.content


def test_docx_parser_preserves_heading_paragraph_and_table(tmp_path: Path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_heading("招标公告", level=1)
    document.add_paragraph("项目名称：DOCX测试工程")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "字段"
    table.cell(0, 1).text = "值"
    table.cell(1, 0).text = "项目编号"
    table.cell(1, 1).text = "DOCX-2026-001"
    document.save(path)

    parsed = DocumentParser(use_ocr=False).parse(str(path))

    assert parsed.original_format == "docx"
    assert "招标公告" in parsed.content
    assert "DOCX测试工程" in parsed.content
    assert "DOCX-2026-001" in parsed.content
    assert parsed.tables


def test_scanned_pdf_path_uses_ocr_hook_without_paddle_runtime(tmp_path: Path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    import tender_extract.document_parser as parser_module

    path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()

    monkeypatch.setattr(parser_module, "PADDLEOCR_AVAILABLE", True)
    parser = parser_module.DocumentParser(use_ocr=True)
    monkeypatch.setattr(parser, "_ocr_page", lambda page: "项目名称：OCR测试工程")

    parsed = parser.parse(str(path))

    assert parsed.metadata["ocr_pages"] == 1
    assert "OCR测试工程" in parsed.content
