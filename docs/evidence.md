# Structured evidence provenance

PR4 adds a stable source-location contract to every extracted `EvidenceSpan`.
The goal is to make an extracted value traceable back to the source document
without pretending that every file format has reliable physical-page geometry.

## Evidence contract

Each span keeps its existing extraction metadata and now carries an optional
`location` object:

```json
{
  "value": "QJ-2026-001",
  "start": 42,
  "end": 53,
  "confidence": 0.95,
  "source": "regex_enhanced",
  "location": {
    "document_id": "tender-v1",
    "page": 3,
    "section_path": ["招标文件", "项目概况"],
    "line_start": 28,
    "line_end": 28,
    "bbox": {
      "x0": 120.2,
      "y0": 246.7,
      "x1": 201.5,
      "y1": 259.4,
      "page_width": 595.0,
      "page_height": 842.0,
      "coordinate_system": "pdf_points_top_left"
    },
    "source_text": "QJ-2026-001",
    "source_start": 42,
    "source_end": 53
  }
}
```

`source_start` / `source_end` are offsets in the parser's unified Markdown
representation. The locator narrows historical regex spans from the whole regex
match to the actual extracted value whenever that value can be found inside the
match window. This makes UI highlighting much more precise while preserving
`ref` as surrounding context.

## Format behavior

### PDF

For text PDFs the locator provides:

- document id
- unified-Markdown character span
- line number
- inferred section path when headings are available
- 1-based physical page
- bounding box in PDF points when the extracted value can be found by PyMuPDF

For OCR pages the page number can still be recovered from OCR page text, but
`bbox` stays `null` unless reliable geometry exists. PR4 deliberately does not
invent OCR coordinates.

### Markdown / TXT

These formats provide document id, character span, line number, source text and
Markdown section path. `page` and `bbox` stay `null` because there is no stable
physical-page concept.

### DOCX

DOCX provides document id, character span, line number, source text and section
path after conversion to the unified Markdown representation. Physical Word page
numbers and coordinates are not deterministic without a rendering engine, so
`page` and `bbox` stay `null`.

## LLM evidence

LLM-recovered values historically use `start = -1` / `end = -1`. After all rule,
NER and LLM candidates have been merged, the same locator tries to find an exact
returned value (or its evidence context) in the parsed document. When successful,
the LLM span receives the same location object as deterministic extraction.
When it cannot be located reliably, offsets remain `-1` and uncertain dimensions
stay null.

This separation is intentional: model confidence is not treated as source-location
confidence.

## Tender package propagation

PR3 package extraction now passes each manifest `document.id` into the extraction
pipeline. Every package field source carries its structured evidence, and an
effective field exposes the winning source directly:

```json
{
  "field_name": "project_scope",
  "primary_value": "修订后的招标范围",
  "selected_from_document": "amendment-1",
  "selected_evidence": [
    {
      "document_id": "amendment-1",
      "page": 2,
      "section_path": ["补遗一"],
      "source_text": "修订后的招标范围",
      "source_start": 118,
      "source_end": 127
    }
  ],
  "sources": []
}
```

This is the contract a later UI can use for "click extracted value → open source
document → jump to page → highlight bbox".

## Human review

`ReviewEvidence` also retains the location object. Review tooling can therefore
show the exact source position next to a low-confidence, conflicting or LLM-led
candidate instead of asking a reviewer to search the whole document manually.

## Privacy

The normal extraction pipeline still defaults to `include_pii = false`.
`location.source_text` is passed through the same result-masking path as the old
`EvidenceSpan.value` and `ref`, so structured provenance cannot bypass existing
output redaction.

## Evidence coverage metrics

`DocumentMetadata.extraction_stats` now reports:

- `evidence_span_count`
- `evidence_located_count`
- `evidence_page_count`
- `evidence_bbox_count`

These metrics make evidence quality measurable instead of silently assuming every
extracted field is equally traceable.
