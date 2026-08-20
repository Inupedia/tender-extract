"""在增强规则引擎上叠加 YAML 自定义规则。"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .extraction_engine import ExtractionEngine

logger = logging.getLogger(__name__)


_PROJECT_NAME_NEXT_FIELD = re.compile(
    r"(?:项目编号|采购项目编号|招标编号|标段编号|项目地点|建设地点|项目单位|招标人|采购人|投标人|采购代理机构|代理机构|采购方式|预算金额|项目预算|最高限价)[：:]"
)


class ConfigurableExtractionEngine(ExtractionEngine):
    def __init__(self, custom_patterns: dict[str, list[dict[str, Any]]] | None = None) -> None:
        super().__init__()
        self._apply_custom_patterns(custom_patterns or {})

    def _clean_value(
        self, value: str, field_name: str, full_match: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        cleaned, unit, normalized = super()._clean_value(value, field_name, full_match)
        if field_name == "project_name" and cleaned:
            # 真实采购文件常把项目名称在 PDF 中折成两行。先在明确的下一个字段标签前截断，
            # 再合并仅由版式造成的换行；这样既保留完整项目名，也不回退到 PR1 的跨字段吞噬问题。
            cleaned = _PROJECT_NAME_NEXT_FIELD.split(cleaned, maxsplit=1)[0]
            cleaned = re.sub(r"\s*[\r\n]+\s*", "", cleaned).strip()
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
