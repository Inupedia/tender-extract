"""Structured source provenance for extracted tender fields."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .document_parser import ParsedDocument
from .schema import BoundingBox, EvidenceLocation, EvidenceSpan, ExtractedField

try:
    import pymupdf

    PYMUPDF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    try:
        import fitz as pymupdf

        PYMUPDF_AVAILABLE = True
    except ImportError:  # pragma: no cover - optional dependency
        pymupdf = None
        PYMUPDF_AVAILABLE = False


_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


class EvidenceLocator:
    """Attach document/page/section/bbox provenance to EvidenceSpan objects.

    Character offsets, line numbers and Markdown section paths are available for all
    parsed formats. Physical pages and bounding boxes are only emitted for PDFs when
    the source can be located reliably; unknown geometry stays null rather than being
    guessed.
    """

    def __init__(
        self,
        parsed: ParsedDocument,
        file_path: str | Path,
        document_id: str,
    ) -> None:
        self.parsed = parsed
        self.file_path = Path(file_path)
        self.document_id = document_id
        self.content = parsed.content
        self._headings = _build_heading_index(self.content)
        self._page_keys = [_search_key(text) for text in parsed.page_contents]

    def enrich_fields(self, fields: dict[str, ExtractedField]) -> None:
        pdf = None
        if (
            self.parsed.original_format == "pdf"
            and PYMUPDF_AVAILABLE
            and self.file_path.exists()
        ):
            try:
                pdf = pymupdf.open(str(self.file_path))
            except Exception:
                pdf = None
        try:
            for field in fields.values():
                for span in field.values:
                    self._enrich_span(span, pdf)
        finally:
            if pdf is not None:
                pdf.close()

    def _enrich_span(self, span: EvidenceSpan, pdf) -> None:
        start, end = _recover_offsets(span, self.content)
        if start >= 0 and end > start:
            span.start = start
            span.end = end
            source_text = self.content[start:end].strip() or None
            line_start = self.content.count("\n", 0, start) + 1
            line_end = self.content.count("\n", 0, end) + 1
            section_path = _section_at(self._headings, start)
        else:
            source_text = None
            line_start = None
            line_end = None
            section_path = []

        page = None
        bbox = None
        if self.parsed.original_format == "pdf":
            page = self._find_pdf_page(span, source_text)
            if page is not None and pdf is not None and 1 <= page <= len(pdf):
                bbox = _find_pdf_bbox(pdf[page - 1], span, source_text)

        span.location = EvidenceLocation(
            document_id=self.document_id,
            page=page,
            section_path=section_path,
            line_start=line_start,
            line_end=line_end,
            bbox=bbox,
            source_text=source_text,
            source_start=start,
            source_end=end,
        )

    def _find_pdf_page(self, span: EvidenceSpan, source_text: Optional[str]) -> Optional[int]:
        if not self._page_keys:
            return None
        candidates: list[str] = []
        for raw in (source_text, span.ref, span.value, span.normalized_value):
            key = _search_key(raw or "")
            if len(key) >= 3 and key not in candidates:
                candidates.append(key)
        candidates.sort(key=len, reverse=True)
        for candidate in candidates:
            for index, page_key in enumerate(self._page_keys, start=1):
                if candidate in page_key:
                    return index
        return None


def _recover_offsets(span: EvidenceSpan, content: str) -> tuple[int, int]:
    if 0 <= span.start < span.end <= len(content):
        return span.start, span.end

    for candidate in (span.value, span.normalized_value):
        value = (candidate or "").strip()
        if not value:
            continue
        start = content.find(value)
        if start != -1:
            return start, start + len(value)

    ref = (span.ref or "").strip()
    if ref:
        ref_start = content.find(ref)
        if ref_start != -1:
            for candidate in (span.value, span.normalized_value):
                value = (candidate or "").strip()
                local = ref.find(value) if value else -1
                if local != -1:
                    start = ref_start + local
                    return start, start + len(value)
            return ref_start, ref_start + len(ref)
    return -1, -1


def _build_heading_index(content: str) -> list[tuple[int, list[str]]]:
    headings: list[tuple[int, list[str]]] = []
    path: list[str] = []
    for match in _HEADING_RE.finditer(content):
        level = len(match.group(1))
        title = match.group(2).strip()
        path = path[: max(level - 1, 0)]
        while len(path) < level - 1:
            path.append("")
        path.append(title)
        headings.append((match.start(), [item for item in path if item]))
    return headings


def _section_at(headings: list[tuple[int, list[str]]], offset: int) -> list[str]:
    current: list[str] = []
    for position, path in headings:
        if position > offset:
            break
        current = path
    return list(current)


def _search_key(text: str) -> str:
    text = re.sub(r"[\s|`*_#>]+", "", text)
    return text.strip().casefold()


def _find_pdf_bbox(page, span: EvidenceSpan, source_text: Optional[str]) -> Optional[BoundingBox]:
    candidates: list[str] = []
    for raw in (span.value, source_text, span.normalized_value):
        value = (raw or "").strip()
        if not value or "\n" in value or len(value) > 160 or value in candidates:
            continue
        candidates.append(value)

    for candidate in candidates:
        try:
            rects = page.search_for(candidate)
        except Exception:
            rects = []
        if not rects:
            continue
        rect = rects[0]
        return BoundingBox(
            x0=round(float(rect.x0), 3),
            y0=round(float(rect.y0), 3),
            x1=round(float(rect.x1), 3),
            y1=round(float(rect.y1), 3),
            page_width=round(float(page.rect.width), 3),
            page_height=round(float(page.rect.height), 3),
        )
    return None
