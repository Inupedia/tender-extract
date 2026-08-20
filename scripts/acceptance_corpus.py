#!/usr/bin/env python3
"""Run the extraction pipeline against real PDF examples and write a benchmark report."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pymupdf

from tender_extract.pipeline import ExtractionPipeline
from tender_extract.schema import ProcessingConfig


def _span_stats(result) -> tuple[int, int, int]:
    spans = [span for field in result.fields.values() for span in field.values]
    located = [span for span in spans if span.location is not None]
    page_located = [span for span in located if span.location.page is not None]
    bbox_located = [span for span in located if span.location.bbox is not None]
    return len(located), len(page_located), len(bbox_located)


def _field_details(result, limit: int = 10) -> dict[str, dict]:
    ranked = sorted(result.fields.items(), key=lambda item: item[1].confidence, reverse=True)
    output: dict[str, dict] = {}
    for name, field in ranked[:limit]:
        if not field.primary_value:
            continue
        primary = field.values[0] if field.values else None
        location = primary.location if primary is not None else None
        output[name] = {
            "value": field.primary_value,
            "confidence": round(field.confidence, 4),
            "source": primary.source if primary is not None else None,
            "page": location.page if location is not None else None,
            "has_bbox": bool(location and location.bbox),
            "source_text": location.source_text if location is not None else None,
            "section_path": location.section_path if location is not None else [],
        }
    return output


def _physical_pages(path: Path) -> int:
    with pymupdf.open(path) as doc:
        return len(doc)


def run_one(path: Path) -> dict:
    config = ProcessingConfig(
        llm_provider="none",
        use_ocr=False,
        include_pii=False,
        persist_llm_cache=False,
    )
    pipeline = ExtractionPipeline(config)
    started = time.perf_counter()
    result = pipeline.extract_file(str(path), document_id=path.name)
    wall_time = time.perf_counter() - started
    stats = result.metadata.extraction_stats
    located, page_located, bbox_located = _span_stats(result)
    pages = result.metadata.total_pages or _physical_pages(path)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "pages": pages,
        "format": stats.get("original_format"),
        "fields": len(result.fields),
        "avg_confidence": round(float(stats.get("avg_confidence") or 0.0), 4),
        "wall_time_seconds": round(wall_time, 4),
        "pipeline_time_seconds": round(float(result.metadata.processing_time), 4),
        "llm_calls": result.llm_calls,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "located_spans": located,
        "page_located_spans": page_located,
        "bbox_located_spans": bbox_located,
        "fields_detail": _field_details(result),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default="examples", help="Directory containing PDFs")
    parser.add_argument("--report", default="artifacts/acceptance.json")
    parser.add_argument("--min-pdfs", type=int, default=1)
    parser.add_argument(
        "--require-example",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply strict checks to the historical examples/example.pdf fixture",
    )
    args = parser.parse_args()

    examples = Path(args.examples)
    pdfs = sorted(examples.glob("*.pdf"))
    if len(pdfs) < args.min_pdfs:
        print(f"acceptance failed: expected at least {args.min_pdfs} PDFs, found {len(pdfs)}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    failures: list[str] = []
    for index, path in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] {path.name}", flush=True)
        try:
            row = run_one(path)
        except Exception as exc:  # acceptance must surface real parser failures
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {failures[-1]}", flush=True)
            continue
        rows.append(row)
        print(
            "  OK "
            f"pages={row['pages']} "
            f"fields={row['fields']} "
            f"confidence={row['avg_confidence']:.2f} "
            f"time={row['wall_time_seconds']:.3f}s "
            f"page_evidence={row['page_located_spans']} "
            f"bbox={row['bbox_located_spans']}",
            flush=True,
        )
        if row["errors"]:
            failures.append(f"{path.name}: pipeline errors: {row['errors']}")
        if row["fields"] == 0:
            failures.append(f"{path.name}: extracted zero fields")
        if row["located_spans"] == 0:
            failures.append(f"{path.name}: no structured evidence spans")

    if args.require_example:
        example = next((row for row in rows if row["file"] == "example.pdf"), None)
        if example is None:
            failures.append("example.pdf did not complete")
        else:
            if example["fields"] < 5:
                failures.append(f"example.pdf: expected >=5 fields, got {example['fields']}")
            if example["page_located_spans"] == 0:
                failures.append("example.pdf: no page-aware evidence")
            if example["bbox_located_spans"] == 0:
                failures.append("example.pdf: no bbox-aware evidence")

    total_pages = sum(row["pages"] for row in rows)
    total_time = sum(row["wall_time_seconds"] for row in rows)
    summary = {
        "pdf_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "total_pages": total_pages,
        "total_fields": sum(row["fields"] for row in rows),
        "total_wall_time_seconds": round(total_time, 4),
        "pages_per_second": round(total_pages / total_time, 2) if total_pages and total_time else None,
        "avg_seconds_per_pdf": round(statistics.mean(row["wall_time_seconds"] for row in rows), 4) if rows else None,
        "avg_fields_per_pdf": round(statistics.mean(row["fields"] for row in rows), 2) if rows else None,
        "failures": failures,
    }
    payload = {"summary": summary, "documents": rows}
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== acceptance summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
