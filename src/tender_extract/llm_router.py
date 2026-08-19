"""按需 LLM 路由：OpenAI 兼容协议 + Anthropic + Ollama。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from .llm_providers import ProviderSpec, get_provider
from .schema import EvidenceSpan, ExtractedField, LLMRequest, LLMResponse, ProcessingConfig

logger = logging.getLogger(__name__)

NUMERIC_FIELD_TYPES = {
    "bid_amount", "deposit", "project_number", "contact_info",
    "bid_date", "registered_capital", "business_license",
}


class LLMRouter:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.spec: ProviderSpec = get_provider(config.llm_provider)
        self.model = config.llm_model or self.spec.default_model or None
        self.debug_mode = config.debug
        self.client: Any = None
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_hits = 0
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.cache_path = Path(config.cache_dir) / "llm_cache.json"
        self._initialize_client()
        if config.persist_llm_cache:
            self.load_cache(str(self.cache_path))

    def _env(self, *names: str) -> Optional[str]:
        for name in names:
            value = os.environ.get(name)
            if value:
                return value
        return None

    def _initialize_client(self) -> None:
        if self.spec.kind == "none":
            return

        api_key = self.config.llm_api_key or (
            self._env(self.spec.api_key_env, "LLM_API_KEY", "OPENAI_API_KEY")
            if self.spec.api_key_env else None
        )
        if self.spec.id == "gemini":
            api_key = self.config.llm_api_key or self._env(
                "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"
            )

        base_url = self.config.llm_base_url
        if not base_url and self.spec.base_url_env:
            base_url = self._env(self.spec.base_url_env, "LLM_BASE_URL")
        if not base_url:
            base_url = self.spec.base_url or None

        try:
            if self.spec.kind == "azure":
                from openai import AzureOpenAI

                endpoint = base_url or self._env("AZURE_OPENAI_ENDPOINT")
                if not api_key or not endpoint:
                    logger.warning("Azure OpenAI 需要 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT")
                    return
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version=self._env("AZURE_OPENAI_API_VERSION") or "2024-10-21",
                )
            elif self.spec.kind in {"openai_compat", "ollama"}:
                from openai import OpenAI

                if self.spec.kind == "ollama":
                    ollama_host = (base_url or "http://127.0.0.1:11434").rstrip("/")
                    if ollama_host.endswith("/v1"):
                        compat_url = ollama_host
                    else:
                        compat_url = ollama_host + "/v1"
                    self.client = OpenAI(api_key=api_key or "ollama", base_url=compat_url)
                else:
                    if not api_key and self.spec.id not in {"openai_compat"}:
                        logger.warning(
                            "LLM 提供商 %s 未配置密钥（环境变量 %s）",
                            self.spec.id, self.spec.api_key_env,
                        )
                        return
                    kwargs: dict[str, Any] = {}
                    if api_key:
                        kwargs["api_key"] = api_key
                    if base_url:
                        kwargs["base_url"] = base_url
                    self.client = OpenAI(**kwargs)
            elif self.spec.kind == "anthropic":
                if not api_key:
                    logger.warning("Anthropic 需要 ANTHROPIC_API_KEY")
                    return
                self.client = {"kind": "anthropic", "api_key": api_key, "base_url": base_url}
        except Exception as exc:
            logger.error("初始化 LLM 客户端失败: %s", exc)
            self.client = None

    def is_enabled(self) -> bool:
        return self.spec.kind != "none" and self.client is not None

    def should_use_llm(
        self, field: ExtractedField, confidence_threshold: float = 0.7
    ) -> bool:
        if not self.is_enabled():
            return False
        if field.confidence < confidence_threshold:
            return True
        if field.conflicts:
            return True
        return False

    def get_minimal_evidence_context(
        self, field: ExtractedField, chunk_text: str
    ) -> str:
        if field.values and field.values[0].ref:
            return field.values[0].ref
        if not field.values:
            return chunk_text[:800]

        spans = [(v.start, v.end) for v in field.values if v.end > v.start]
        if not spans:
            return chunk_text[:800]

        spans.sort()
        merged = [spans[0]]
        for start, end in spans[1:]:
            last = merged[-1]
            if start <= last[1] + 50:
                merged[-1] = (last[0], max(last[1], end))
            else:
                merged.append((start, end))

        contexts = []
        for start, end in merged:
            # 坐标必须是 chunk 内相对位置，不能是全文绝对位置
            if start >= len(chunk_text) or end > len(chunk_text) * 2:
                continue
            context_start = max(0, start - 120)
            context_end = min(len(chunk_text), end + 120)
            contexts.append(chunk_text[context_start:context_end])
        return "\n---\n".join(contexts) if contexts else chunk_text[:800]

    def extract_with_llm(self, request: LLMRequest) -> Optional[LLMResponse]:
        cache_key = self._cache_key(request)
        if cache_key in self.cache:
            self.cache_hits += 1
            return LLMResponse.model_validate(self.cache[cache_key])

        self.total_calls += 1
        prompt = self._build_prompt(request)
        if self.debug_mode:
            logger.info("LLM prompt (%s):\n%s", request.field_name, prompt)

        try:
            content = self._complete(prompt)
            if not content:
                self.failed_calls += 1
                return None
            parsed = self._parse_response(content, request.field_name)
            if parsed:
                self.successful_calls += 1
                self.cache[cache_key] = parsed.model_dump()
                return parsed
            self.failed_calls += 1
            return None
        except Exception as exc:
            self.failed_calls += 1
            logger.error("LLM 调用失败 %s: %s", request.field_name, exc)
            return None

    def _complete(self, prompt: str) -> Optional[str]:
        system = "你是中文招投标文档信息抽取助手。只返回 JSON，不要其它说明。"
        if self.spec.kind == "anthropic":
            return self._call_anthropic(system, prompt)
        return self._call_openai_compat(system, prompt)

    def _call_openai_compat(self, system: str, prompt: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system: str, prompt: str) -> Optional[str]:
        import httpx

        base = (self.client["base_url"] or "https://api.anthropic.com").rstrip("/")
        url = base + "/v1/messages"
        headers = {
            "x-api-key": self.client["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model or "claude-sonnet-4-5",
            "max_tokens": 800,
            "temperature": 0.1,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=60.0) as http:
            resp = http.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        parts = data.get("content") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts) or None

    def _build_prompt(self, request: LLMRequest) -> str:
        existing = "、".join(request.existing_values) if request.existing_values else "无"
        return (
            f"请从下列原文中抽取字段「{request.field_name}」（类型 {request.field_type}）。\n"
            f"已有候选：{existing}\n"
            "只返回 JSON：{\"extracted_values\": [\"值\"], \"confidence\": 0.0, \"reasoning\": \"简短理由\"}\n"
            "若无法确定，extracted_values 为空数组，confidence 为 0。\n\n"
            f"原文：\n{request.chunk_text}"
        )

    def _parse_response(self, content: str, field_name: str) -> Optional[LLMResponse]:
        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= 0:
            return None
        data = json.loads(text[start:end])
        values = data.get("extracted_values") or []
        if isinstance(values, str):
            values = [values]
        confidence = float(data.get("confidence", 0.5))
        confidence = min(max(confidence, 0.0), 1.0)
        return LLMResponse(
            field_name=field_name,
            extracted_values=[str(v).strip() for v in values if str(v).strip()],
            confidence=confidence,
            reasoning=data.get("reasoning") or "",
        )

    def merge_llm_results(
        self, field: ExtractedField, llm_response: LLMResponse
    ) -> ExtractedField:
        if not llm_response.extracted_values:
            return field
        # 规则已经高置信且无冲突时，不让 LLM 覆盖
        if field.confidence >= 0.9 and not field.conflicts:
            return field
        for value in llm_response.extracted_values:
            field.values.append(
                EvidenceSpan(
                    value=value,
                    start=0,
                    end=0,
                    confidence=llm_response.confidence,
                    source="llm",
                    pattern=f"{self.spec.id}:{self.model}",
                    ref=field.values[0].ref if field.values else None,
                )
            )
        field.values.sort(key=lambda x: x.confidence, reverse=True)
        field.primary_value = field.values[0].value
        field.confidence = field.values[0].confidence
        return field

    def _cache_key(self, request: LLMRequest) -> str:
        raw = f"{self.spec.id}|{self.model}|{request.field_name}|{request.chunk_text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def save_cache(self, filepath: Optional[str] = None) -> None:
        path = Path(filepath or self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_cache(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            return
        try:
            self.cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("加载 LLM 缓存失败: %s", exc)
