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
_TENDERER_BAD = re.compile(
    r"(?:https?://|www\.|\.gov\.|\.cn\b|渠道查询|信用中国|供应商的姓名|邮编|联系电话|收费管理暂行办法)",
    re.IGNORECASE,
)
_TENDERER_SUFFIX = re.compile(
    r"(?:有限公司|股份公司|集团|公司|企业|委员会|管理委员会|人民政府|政府|办公室|教育局|公安局|管理局|农业农村局|财政局|局|中心|服务中心|监测站|站|大学|学院|学校|医院|研究院|设计院|院|所|馆|厅|部|署|协会|学会)$"
)
_BID_AMOUNT_CONTEXT = re.compile(
    r"(?:投标(?:总)?报价|投标金额|项目预算|预算金额|采购预算|最高限价|最高投标限价|招标控制价|控制价|合同金额|中标金额|成交金额)"
)
_CONTACT_LABEL = re.compile(r"(?:电话|联系电话|办公电话|传真)[：:]")

_DEADLINE_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (
        re.compile(
            r"(?:提交投标文件截止时间|提交响应文件截止时间|响应文件提交截止时间|投标截止时间|截止/谈判时间)[：:]?\s*"
            r"(20\d{2}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*(?:日)?(?:\s+\d{1,2}\s*(?::|时)\s*\d{2}(?::\d{2})?)?)",
            re.IGNORECASE | re.MULTILINE,
        ),
        0.98,
        "明确截止时间",
    ),
    (
        re.compile(
            r"投标人应于\s*(20\d{2}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)",
            re.IGNORECASE | re.MULTILINE,
        ),
        0.97,
        "投标人应于-截止时间",
    ),
]


class ConfigurableExtractionEngine(ExtractionEngine):
    def __init__(self, custom_patterns: dict[str, list[dict[str, Any]]] | None = None) -> None:
        super().__init__()
        # `legal_representative` 只能表示法定代表人，不能把联系人/项目经理/授权代表混成同一字段。
        self._compiled_patterns["legal_representative"] = [
            item
            for item in self._compiled_patterns.get("legal_representative", [])
            if item[2] in {"法定代表人", "法人代表"}
        ]
        # 真实采购文件的截止时间表达比基础规则更多，明确标签应压过正文中的普通日期。
        self._compiled_patterns.setdefault("bid_date", []).extend(_DEADLINE_PATTERNS)
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

        if field_name == "tenderer" and cleaned:
            cleaned = re.sub(r"^(?:采购人|招标人|采购单位|招标单位)\s*[：:]\s*", "", cleaned).strip()
            if _TENDERER_BAD.search(cleaned):
                return None, unit, normalized
            if not _TENDERER_SUFFIX.search(cleaned):
                return None, unit, normalized

        if field_name == "legal_representative" and cleaned:
            if cleaned in {"法定代表", "法人代表", "授权代表", "受托人", "联系人", "联系地址"}:
                return None, unit, normalized
            if cleaned.endswith("老师"):
                return None, unit, normalized

        if field_name == "bid_amount" and cleaned:
            # 裸露的“20000万元”等数字常来自规格、评分或模板，不应直接当成报价。
            if not _BID_AMOUNT_CONTEXT.search(full_match or ""):
                return None, unit, normalized

        if field_name == "contact_info" and cleaned:
            # 无标签的长串 0 开头数字容易来自项目编号/预算编号；座机必须有联系方式上下文。
            if cleaned.startswith("0") and not _CONTACT_LABEL.search(full_match or ""):
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
