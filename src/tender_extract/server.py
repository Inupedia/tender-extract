"""HTTP service wrapper for tender-extract.

The server intentionally stays thin: extraction logic remains in ExtractionPipeline,
while this module adds upload validation, request-scoped runtime options, optional API
key protection, and stable HTTP endpoints for container deployment.
"""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .document_parser import PADDLEOCR_AVAILABLE, PYMUPDF_AVAILABLE, PYTHON_DOCX_AVAILABLE
from .pipeline import ExtractionPipeline
from .schema import ProcessingConfig

logger = logging.getLogger(__name__)

SERVER_VERSION = os.getenv("TENDER_SERVER_VERSION", "dev")
MAX_UPLOAD_MB = int(os.getenv("TENDER_SERVER_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
DEFAULT_LLM_PROVIDER = os.getenv("TENDER_SERVER_LLM_PROVIDER", "none")
DEFAULT_LLM_MODEL = os.getenv("TENDER_SERVER_LLM_MODEL") or None
DEFAULT_USE_OCR = os.getenv("TENDER_SERVER_USE_OCR", "false").lower() in {"1", "true", "yes", "on"}
CACHE_DIR = os.getenv("TENDER_SERVER_CACHE_DIR", "/data/cache")
API_KEY = os.getenv("TENDER_SERVER_API_KEY") or None
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}

app = FastAPI(
    title="tender-extract-server",
    version=SERVER_VERSION,
    description="中文采购与招投标文档结构化抽取服务",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Protect extraction endpoints only when TENDER_SERVER_API_KEY is configured."""

    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/healthz", tags=["system"])
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": SERVER_VERSION}


@app.get("/v1/info", tags=["system"])
def info() -> dict[str, object]:
    return {
        "name": "tender-extract-server",
        "version": SERVER_VERSION,
        "max_upload_mb": MAX_UPLOAD_MB,
        "default_llm_provider": DEFAULT_LLM_PROVIDER,
        "default_llm_model": DEFAULT_LLM_MODEL,
        "supported_formats": sorted(SUPPORTED_SUFFIXES),
        "capabilities": {
            "pdf": PYMUPDF_AVAILABLE,
            "docx": PYTHON_DOCX_AVAILABLE,
            "ocr": PADDLEOCR_AVAILABLE,
            "structured_evidence": True,
            "pii_redaction_default": True,
        },
    }


@app.post("/v1/extract", tags=["extraction"], dependencies=[Depends(_require_api_key)])
async def extract_document(
    file: Annotated[UploadFile, File(description="PDF / DOCX / TXT / Markdown document")],
    llm_provider: Annotated[str | None, Query(description="Override server default LLM provider")] = None,
    llm_model: Annotated[str | None, Query(description="Override server default model")] = None,
    confidence_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.7,
    include_pii: Annotated[bool, Query(description="Return unmasked PII; default false")] = False,
    use_ocr: Annotated[bool | None, Query(description="Request OCR when OCR dependencies are installed")] = None,
) -> JSONResponse:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type: {suffix or '(none)'}; supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    temp_path: Path | None = None
    total_bytes = 0
    request_id = uuid.uuid4().hex

    try:
        with tempfile.NamedTemporaryFile(prefix="tender-extract-", suffix=suffix, delete=False) as tmp:
            temp_path = Path(tmp.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds {MAX_UPLOAD_MB} MB upload limit",
                    )
                tmp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="empty upload")

        config = ProcessingConfig(
            llm_provider=llm_provider or DEFAULT_LLM_PROVIDER,
            llm_model=llm_model or DEFAULT_LLM_MODEL,
            confidence_threshold=confidence_threshold,
            include_pii=include_pii,
            use_ocr=DEFAULT_USE_OCR if use_ocr is None else use_ocr,
            cache_dir=CACHE_DIR,
            persist_llm_cache=True,
        )

        def _run() -> dict[str, object]:
            result = ExtractionPipeline(config).extract_file(str(temp_path), document_id=filename)
            return result.model_dump(mode="json")

        result = await run_in_threadpool(_run)
        return JSONResponse(
            {
                "request_id": request_id,
                "server_version": SERVER_VERSION,
                "result": result,
            },
            headers={"X-Request-ID": request_id},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("extraction failed request_id=%s filename=%s", request_id, filename)
        raise HTTPException(status_code=422, detail=f"extraction failed: {type(exc).__name__}") from exc
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
