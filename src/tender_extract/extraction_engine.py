"""增强抽取引擎：分层正则 + 金额单位 + 证据片段。"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .patterns import FIELD_PATTERNS, compile_patterns
from .schema import EvidenceSpan, ExtractedField

logger = logging.getLogger(__name__)


class ExtractionEngine:
    def __init__(self) -> None:
        self._compiled_patterns: dict[str, list] = {}
        for field_name, patterns in FIELD_PATTERNS.items():
            self._compiled_patterns[field_name] = compile_patterns(patterns)

    def extract_all_fields(
        self, text: str, target_fields: Optional[list[str]] = None
    ) -> dict[str, ExtractedField]:
        results: dict[str, ExtractedField] = {}
        fields_to_extract = target_fields or list(self._compiled_patterns.keys())
        for field_name in fields_to_extract:
            if field_name not in self._compiled_patterns:
                continue
            field_result = self._extract_field(text, field_name)
            if field_result and field_result.values:
                results[field_name] = field_result
        return self._post_process(results)

    def _extract_field(self, text: str, field_name: str) -> Optional[ExtractedField]:
        compiled = self._compiled_patterns.get(field_name, [])
        if not compiled:
            return None

        all_values: list[EvidenceSpan] = []
        for pattern, confidence, description in compiled:
            for match in pattern.finditer(text):
                raw = match.group(1) if match.groups() else match.group(0)
                raw = (raw or "").strip()
                if not raw or len(raw) < 2:
                    continue
                cleaned, unit, normalized = self._clean_value(raw, field_name, match.group(0))
                if not cleaned:
                    continue
                if not self._validate_value(cleaned, field_name):
                    continue
                start, end = match.start(), match.end()
                ref_start = max(0, start - 80)
                ref_end = min(len(text), end + 80)
                all_values.append(
                    EvidenceSpan(
                        value=cleaned,
                        start=start,
                        end=end,
                        confidence=confidence,
                        source="regex_enhanced",
                        pattern=description,
                        ref=text[ref_start:ref_end].strip(),
                        unit=unit,
                        normalized_value=normalized,
                    )
                )

        if not all_values:
            return None
        all_values = self._deduplicate_values(all_values)
        all_values.sort(key=lambda x: x.confidence, reverse=True)
        return ExtractedField(
            field_name=field_name,
            field_type=field_name,
            values=all_values,
            primary_value=all_values[0].value,
            confidence=all_values[0].confidence,
            conflicts=self._detect_conflicts(all_values),
        )

    def _clean_value(
        self, value: str, field_name: str, full_match: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        value = value.strip()
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
        if field_name in ("bid_amount", "deposit"):
            return self._clean_amount(value, full_match)
        if field_name == "bid_date":
            return self._clean_date(value), None, None
        if field_name == "contact_info":
            return self._clean_contact(value), None, None
        if field_name in ("bidder", "tenderer"):
            return self._clean_company(value), None, None
        if field_name == "legal_representative":
            return self._clean_person_name(value), None, None
        if field_name == "project_number":
            return self._clean_id(value), None, None
        return value, None, None

    def _clean_amount(
        self, value: str, full_match: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        context = full_match or value
        if re.search(r"[壹贰叁肆伍陆柒捌玖拾佰仟万亿零]", value):
            cleaned = re.sub(r"[^\u4e00-\u9fff元整]", "", value)
            if len(cleaned) < 3:
                return None, None, None
            yuan = chinese_amount_to_yuan(cleaned)
            unit = "元"
            display = cleaned if cleaned.endswith("元") or cleaned.endswith("整") else cleaned + "元"
            normalized = f"{yuan:.2f}" if yuan is not None else None
            return display, unit, normalized

        unit = "万元" if ("万" in context and "元" in context) or re.search(r"\d[\d,.]+\s*万", context) else "元"
        if "万元" in context:
            unit = "万元"
        elif re.search(r"万\s*元", context):
            unit = "万元"

        compact = value.replace(",", "").replace("，", "")
        num_match = re.search(r"(\d+(?:\.\d{1,6})?)", compact)
        if not num_match:
            return None, None, None
        number = num_match.group(1)
        amount = float(number)
        yuan = amount * 10000 if unit == "万元" else amount
        display = f"{number}{unit}"
        return display, unit, f"{yuan:.2f}"

    def _clean_date(self, value: str) -> Optional[str]:
        value = re.sub(r"\s+", "", value)
        return value or None

    def _clean_contact(self, value: str) -> Optional[str]:
        value = re.sub(r"[\s\-—]", "", value)
        if re.match(r"^1[3-9]\d{9}$", value):
            return value
        if re.match(r"^0\d{2,3}\d{7,8}$", value):
            return value
        if "@" in value:
            return value
        return value if len(value) >= 6 else None

    def _clean_company(self, value: str) -> Optional[str]:
        value = re.sub(r"^[^a-zA-Z\u4e00-\u9fff]+", "", value)
        value = re.sub(r"[^a-zA-Z\u4e00-\u9fff（）()]+$", "", value)
        if not re.search(r"(公司|集团|企业|局|中心|院|所|社)$", value) and len(value) < 8:
            return None
        return value if len(value) >= 4 else None

    def _clean_person_name(self, value: str) -> Optional[str]:
        name = re.sub(r"[^\u4e00-\u9fa5]", "", value)
        return name if 2 <= len(name) <= 4 else None

    def _clean_id(self, value: str) -> Optional[str]:
        value = value.strip()
        return value if len(value) >= 5 else None

    def _validate_value(self, value: str, field_name: str) -> bool:
        if not value or len(value) < 2:
            return False
        if value in {"无", "暂无", "略", "详见", "见附件", "以上", "以下", "如下"}:
            return False
        if field_name in ("bid_amount", "deposit"):
            try:
                if not re.search(r"[壹贰叁肆伍陆柒捌玖]", value):
                    num = float(re.sub(r"[^\d.]", "", value) or "0")
                    if num <= 0:
                        return False
            except (ValueError, TypeError):
                return False
        elif field_name == "legal_representative":
            if not re.match(r"^[\u4e00-\u9fa5]{2,4}$", value):
                return False
        elif field_name == "project_number":
            if not re.search(r"[A-Za-z0-9]", value):
                return False
        return True

    def _deduplicate_values(self, values: list[EvidenceSpan]) -> list[EvidenceSpan]:
        seen: dict[str, EvidenceSpan] = {}
        for item in values:
            key = (item.normalized_value or item.value).strip().lower()
            if key not in seen or item.confidence > seen[key].confidence:
                seen[key] = item
        return list(seen.values())

    def _detect_conflicts(self, values: list[EvidenceSpan]) -> list[str]:
        if len(values) <= 1:
            return []
        unique = {v.normalized_value or v.value for v in values}
        if len(unique) <= 1:
            return []
        numeric = []
        for item in values:
            try:
                numeric.append(float(item.normalized_value or re.sub(r"[^\d.]", "", item.value)))
            except (ValueError, TypeError):
                continue
        if len(numeric) > 1 and max(numeric) / max(min(numeric), 0.01) > 5:
            return [f"数值差异过大: {min(numeric)} vs {max(numeric)}"]
        if len(unique) > 3:
            return [f"存在{len(unique)}个不同值"]
        return []

    def _post_process(
        self, results: dict[str, ExtractedField]
    ) -> dict[str, ExtractedField]:
        if "bid_amount" in results and "deposit" in results:
            try:
                amount = float(results["bid_amount"].values[0].normalized_value or 0)
                deposit = float(results["deposit"].values[0].normalized_value or 0)
                if amount > 0 and deposit > amount:
                    results["deposit"].confidence *= 0.5
                    results["deposit"].conflicts.append(
                        "保证金超过投标金额，可能存在单位不一致"
                    )
            except (ValueError, TypeError, IndexError, AttributeError):
                pass
        return results

    def get_low_confidence_fields(
        self, results: dict[str, ExtractedField], threshold: float = 0.7
    ) -> list[str]:
        return [
            name
            for name, field in results.items()
            if field.confidence < threshold or field.conflicts
        ]


_CN_DIGITS = {
    "零": 0, "壹": 1, "贰": 2, "叁": 3, "肆": 4,
    "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"拾": 10, "佰": 100, "仟": 1000, "十": 10, "百": 100, "千": 1000}


def chinese_amount_to_yuan(text: str) -> Optional[float]:
    """中文大写金额转元。支持 万/亿。"""
    text = re.sub(r"[元整人民币]", "", text)
    if not text:
        return None
    total = 0.0
    current_section = 0
    current_number = 0
    for char in text:
        if char in _CN_DIGITS:
            current_number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            if current_number == 0 and unit == 10:
                current_number = 1
            current_section += current_number * unit
            current_number = 0
        elif char == "万":
            current_section += current_number
            total += current_section * 10_000
            current_section = 0
            current_number = 0
        elif char == "亿":
            current_section += current_number
            total += current_section * 100_000_000
            current_section = 0
            current_number = 0
    total += current_section + current_number
    return total if total > 0 else None
