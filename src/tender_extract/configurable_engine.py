"""在增强规则引擎上叠加 YAML 自定义规则。"""
from __future__ import annotations

import logging
import re
from typing import Any

from .extraction_engine import ExtractionEngine

logger = logging.getLogger(__name__)


class ConfigurableExtractionEngine(ExtractionEngine):
    def __init__(self, custom_patterns: dict[str, list[dict[str, Any]]] | None = None) -> None:
        super().__init__()
        self._apply_custom_patterns(custom_patterns or {})

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
