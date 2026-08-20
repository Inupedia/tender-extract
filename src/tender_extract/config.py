"""加载 YAML 配置并用 CLI 覆盖。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from .schema import ProcessingConfig


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def build_processing_config(
    yaml_data: dict[str, Any],
    *,
    use_ner: Optional[bool] = None,
    llm: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    use_modules: Optional[bool] = None,
    include_pii: Optional[bool] = None,
    debug: bool = False,
) -> ProcessingConfig:
    base = yaml_data.get("base") or {}
    chunking = yaml_data.get("chunking") or {}
    llm_cfg = yaml_data.get("llm") or {}
    cache_cfg = yaml_data.get("cache") or {}
    patterns = yaml_data.get("patterns") or {}

    provider = llm if llm is not None else llm_cfg.get("provider", "none")
    if provider in (None, "", False):
        provider = "none"

    return ProcessingConfig(
        use_ner=use_ner if use_ner is not None else bool(base.get("use_ner", False)),
        llm_provider=str(provider).lower(),
        llm_model=model or llm_cfg.get("model"),
        llm_base_url=base_url or llm_cfg.get("base_url") or llm_cfg.get("ollama_base_url"),
        llm_api_key=api_key or llm_cfg.get("api_key"),
        confidence_threshold=float(base.get("confidence_threshold", 0.7)),
        max_chunk_tokens=int(chunking.get("max_chunk_size", chunking.get("max_tokens", 800))),
        overlap_tokens=int(chunking.get("overlap_size", chunking.get("overlap_tokens", 100))),
        cache_dir=cache_dir or cache_cfg.get("cache_dir", ".cache"),
        enable_dedupe=bool(base.get("use_dedupe", False)),
        enable_similarity_check=bool(base.get("use_similarity_check", False)),
        use_modules=use_modules if use_modules is not None else bool(base.get("use_modules", True)),
        include_pii=include_pii if include_pii is not None else bool(base.get("include_pii", False)),
        use_ocr=bool(base.get("use_ocr", True)),
        debug=debug,
        persist_llm_cache=bool(cache_cfg.get("enabled", True)),
        custom_patterns=patterns if isinstance(patterns, dict) else {},
        recover_missing_fields_with_llm=bool(llm_cfg.get("recover_missing_fields", True)),
        redact_pii_for_cloud_llm=bool(llm_cfg.get("redact_pii_for_cloud", True)),
    )


def resolve_config_path(cli_path: str) -> Path:
    path = Path(cli_path)
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[2] / "config" / "example.yaml"
    return fallback if fallback.exists() else path
