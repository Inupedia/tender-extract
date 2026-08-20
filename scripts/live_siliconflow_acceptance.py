#!/usr/bin/env python3
"""Live SiliconFlow acceptance on the real PDF example.

This check intentionally complements the deterministic offline corpus:
- it exercises low-confidence review through the real ExtractionPipeline;
- it verifies strong rule fields are not overwritten by the LLM;
- it measures cold API calls and warm-cache reuse without recording secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from tender_extract.evaluation import normalize_value
from tender_extract.pipeline import ExtractionPipeline
from tender_extract.schema import ProcessingConfig

EXPECTED_STRONG_FIELDS = {
    "project_name": "合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目",
    "project_number": "2024BFFFZ01583",
    "tenderer": "合肥市公安局瑶海分局",
}


def _assert_expected(result) -> None:
    for field_name, expected in EXPECTED_STRONG_FIELDS.items():
        field = result.fields.get(field_name)
        if field is None or not field.primary_value:
            raise AssertionError(f"missing strong field: {field_name}")
        if normalize_value(field.primary_value) != normalize_value(expected):
            raise AssertionError(
                f"{field_name}: expected={expected!r}, actual={field.primary_value!r}"
            )


def _field_snapshot(result) -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}
    for name, field in sorted(result.fields.items()):
        if not field.primary_value:
            continue
        primary = field.values[0] if field.values else None
        snapshot[name] = {
            "value": field.primary_value,
            "confidence": round(field.confidence, 4),
            "source": primary.source if primary else None,
            "page": primary.location.page if primary and primary.location else None,
        }
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", default="examples/example.pdf")
    parser.add_argument("--report", default="artifacts/siliconflow-live.json")
    parser.add_argument(
        "--model",
        default=os.getenv("SILICONFLOW_MODEL") or "Qwen/Qwen3-8B",
    )
    args = parser.parse_args()

    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit("SILICONFLOW_API_KEY is required")

    cache_dir = Path(".cache/live-siliconflow")
    cache_file = cache_dir / "llm_cache.json"
    if cache_file.exists():
        cache_file.unlink()

    config = ProcessingConfig(
        llm_provider="siliconflow",
        llm_model=args.model,
        llm_api_key=api_key,
        confidence_threshold=0.95,
        recover_missing_fields_with_llm=False,
        use_ocr=False,
        include_pii=False,
        cache_dir=str(cache_dir),
        persist_llm_cache=True,
        redact_pii_for_cloud_llm=True,
    )
    pipeline = ExtractionPipeline(config)
    if not pipeline.llm.is_enabled():
        raise AssertionError("SiliconFlow client is not enabled")

    started = time.perf_counter()
    cold = pipeline.extract_file(args.document, document_id="live-example")
    cold_wall = time.perf_counter() - started
    _assert_expected(cold)

    cold_api_calls = pipeline.llm.total_calls
    cold_successful = pipeline.llm.successful_calls
    cold_failed = pipeline.llm.failed_calls
    cold_cache_hits = pipeline.llm.cache_hits
    if cold_api_calls < 1:
        raise AssertionError("real PDF did not exercise any live SiliconFlow call")
    if cold_successful < 1:
        raise AssertionError("SiliconFlow returned no successful response for the real PDF")

    started = time.perf_counter()
    warm = pipeline.extract_file(args.document, document_id="live-example")
    warm_wall = time.perf_counter() - started
    _assert_expected(warm)

    warm_cache_hits = pipeline.llm.cache_hits - cold_cache_hits
    warm_new_api_calls = pipeline.llm.total_calls - cold_api_calls
    if warm_cache_hits < 1:
        raise AssertionError("second real-PDF pass did not hit the LLM cache")
    if warm_new_api_calls != 0:
        raise AssertionError(
            f"warm pass unexpectedly made {warm_new_api_calls} new API calls"
        )

    stats = cold.metadata.extraction_stats
    report = {
        "provider": "siliconflow",
        "model": args.model,
        "document": Path(args.document).name,
        "pages": int(stats.get("page_count") or 0),
        "cold": {
            "wall_time_seconds": round(cold_wall, 4),
            "logical_llm_calls": cold.llm_calls,
            "network_api_calls": cold_api_calls,
            "successful_api_calls": cold_successful,
            "failed_api_calls": cold_failed,
            "cache_hits": cold_cache_hits,
            "fields": len(cold.fields),
            "avg_confidence": round(float(stats.get("avg_confidence") or 0.0), 4),
        },
        "warm": {
            "wall_time_seconds": round(warm_wall, 4),
            "logical_llm_calls": warm.llm_calls,
            "new_network_api_calls": warm_new_api_calls,
            "cache_hits": warm_cache_hits,
            "fields": len(warm.fields),
        },
        "strong_fields_preserved": EXPECTED_STRONG_FIELDS,
        "fields": _field_snapshot(cold),
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
