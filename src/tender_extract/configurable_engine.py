"""在增强规则引擎上叠加 YAML 自定义规则。"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .extraction_engine import ExtractionEngine

logger = logging.getLogger(__name__)


class ConfigurableExtractionEngine(ExtractionEngine):
    def __init__(self, custom_patterns: dict[str, list[dict[str, Any]]] | None = None) -> None:
        super().__init__()
        self._apply_custom_patterns(custom_patterns or {})

    def _clean_value(
        self, value: str, field_name: str, full_match: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        cleaned, unit, normalized = super()._clean_value(value, field_name, full_match)
        if field_name == "project_name" and cleaned:
            # 宽松项目名称规则不能跨行吞掉后续字段。保留第一行，并在常见字段标签前截断。
            cleaned = re.split(
                r"[\r\n]+|(?:项目编号|招标编号|项目地点|建设地点|招标人|采购人|投标人)[：:]",
                cleaned,
                maxsplit=1,
            )[0].strip()
            if len(cleaned) < 5:
                return None, unit, normalized
        return cleaned, unit, normalized

    def _apply_custom_patterns(self, custom_patterns: dict[str, list[dict[str, Any]]]) -> None:
        for field_name, definitions in custom_patterns.items():
            if not isinstance(definitions, list):
                continue
            compiled = self._compiled_patterns.setdefault(field_name, [])
            for definition in definitions:
                if isinstance(definition, str):
                    pattern_text = definition
                    confidence = 0.9
                    description = "yaml-custom"
                elif isinstance(definition, dict):
                    pattern_text = str(definition.get("pattern") or "")
                    confidence = float(definition.get("confidence", 0.9))
                    description = str(definition.get("description") or "yaml-custom")
                else:
                    continue
                if not pattern_text:
                    continue
                try:
                    compiled.append(
                        (
                            re.compile(pattern_text, re.IGNORECASE | re.MULTILINE),
                            max(0.0, min(confidence, 1.0)),
                            description,
                        )
                    )
                except re.error as exc:
                    logger.warning("YAML 正则编译失败 %s/%s: %s", field_name, description, exc)
