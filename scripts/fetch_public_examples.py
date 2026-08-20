#!/usr/bin/env python3
"""Download official public tender PDFs and create compact regression fixtures.

Full originals are kept in a cache directory for acceptance runs. To keep the
Git repository practical, large PDFs are reduced to their first N pages before
being written under examples/. The lock file records original URLs, sizes,
page counts and hashes so the corpus remains auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf

USER_AGENT = "tender-extract-acceptance/1.0 (+https://github.com/Inupedia/tender-extract)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, retries: int = 2) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.8",
            "Referer": url.rsplit("/", 1)[0] + "/",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response, target.open("wb") as out:
                shutil.copyfileobj(response, out)
            if target.stat().st_size < 1024 or target.read_bytes()[:5] != b"%PDF-":
                raise ValueError("response is not a valid PDF")
            return
        except Exception as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def materialize_fixture(original: Path, output: Path, max_full_bytes: int, excerpt_pages: int) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(original) as doc:
        original_pages = len(doc)
        if original.stat().st_size <= max_full_bytes:
            shutil.copy2(original, output)
            return original_pages, original_pages
        keep = min(original_pages, excerpt_pages)
        reduced = pymupdf.open()
        reduced.insert_pdf(doc, from_page=0, to_page=max(0, keep - 1))
        reduced.save(output, garbage=4, deflate=True, clean=True)
        reduced.close()
        return original_pages, keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="examples/public-sources.json")
    parser.add_argument("--cache-dir", default=".cache/public-full")
    parser.add_argument("--examples-dir", default="examples")
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--max-full-mb", type=float, default=3.0)
    parser.add_argument("--excerpt-pages", type=int, default=8)
    parser.add_argument("--lock", default="examples/public-corpus.lock.json")
    args = parser.parse_args()

    sources = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cache_dir = Path(args.cache_dir)
    examples_dir = Path(args.examples_dir)
    max_full_bytes = int(args.max_full_mb * 1024 * 1024)
    records: list[dict] = []
    failures: list[dict] = []

    for source in sources:
        if len(records) >= args.target_count:
            break
        source_id = source["id"]
        full = cache_dir / f"{source_id}.pdf"
        fixture = examples_dir / f"public-{len(records) + 1:02d}-{source_id}.pdf"
        print(f"fetch {source_id}", flush=True)
        try:
            if not full.exists():
                download(source["url"], full)
            with pymupdf.open(full) as doc:
                if len(doc) == 0:
                    raise ValueError("PDF has zero pages")
            original_pages, fixture_pages = materialize_fixture(
                full, fixture, max_full_bytes=max_full_bytes, excerpt_pages=args.excerpt_pages
            )
            record = {
                **source,
                "original_file": str(full),
                "original_bytes": full.stat().st_size,
                "original_pages": original_pages,
                "original_sha256": sha256(full),
                "fixture_file": str(fixture),
                "fixture_bytes": fixture.stat().st_size,
                "fixture_pages": fixture_pages,
                "fixture_sha256": sha256(fixture),
                "is_excerpt": fixture_pages < original_pages,
            }
            records.append(record)
            print(
                f"  ok original={record['original_pages']}p/{record['original_bytes']}B "
                f"fixture={record['fixture_pages']}p/{record['fixture_bytes']}B",
                flush=True,
            )
        except Exception as exc:
            failures.append({"id": source_id, "url": source["url"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  skip {source_id}: {failures[-1]['error']}", flush=True)

    lock = {
        "generated_from": str(args.manifest),
        "target_count": args.target_count,
        "materialized_count": len(records),
        "records": records,
        "failures": failures,
    }
    Path(args.lock).write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"materialized": len(records), "failed": len(failures)}, ensure_ascii=False))
    if len(records) < args.target_count:
        print(
            f"materialization failed: needed {args.target_count}, got {len(records)}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
