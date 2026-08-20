"""字段合并策略模块。

核心原则：选择 primary value 不等于删除其它候选。所有候选与冲突信息都保留，
以便后续审计、人工复核和证据追踪。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from .schema import ExtractedField, EvidenceSpan


class FieldMerger:
    """审计友好的字段合并器。"""

    def __init__(self):
        self.merge_strategies = {
            "bid_amount": self._merge_amount_field,
            "amount": self._merge_amount_field,
            "bid_date": self._merge_date_field,
            "date": self._merge_date_field,
            "project_number": self._merge_number_field,
            "number": self._merge_number_field,
            "contact_info": self._merge_contact_field,
            "contact": self._merge_contact_field,
            "deposit": self._merge_deposit_field,
            "default": self._merge_default_field,
        }

    def merge_fields(self, fields: dict[str, ExtractedField]) -> dict[str, ExtractedField]:
        return self.resolve_conflicts(fields)

    def resolve_conflicts(self, fields: dict[str, ExtractedField]) -> dict[str, ExtractedField]:
        resolved: dict[str, ExtractedField] = {}
        for name, original in fields.items():
            field = original.model_copy(deep=True)
            strategy = self.merge_strategies.get(field.field_type, self.merge_strategies["default"])
            resolved[name] = strategy(field)
        return resolved

    def _with_primary(self, field: ExtractedField, best: EvidenceSpan | None) -> ExtractedField:
        if best is None:
            return field
        # 保留所有候选，仅更新 primary；冲突说明也不抹掉。
        field.values = sorted(field.values, key=lambda item: item.confidence, reverse=True)
        field.primary_value = best.value
        field.confidence = best.confidence
        return field

    def _merge_amount_field(self, field: ExtractedField) -> ExtractedField:
        ordered = sorted(field.values, key=lambda item: item.confidence, reverse=True)
        for value in ordered:
            amount = self._parse_amount(value.value, value)
            if amount is not None and self._is_reasonable_amount(amount):
                return self._with_primary(field, value)
        return self._with_primary(field, ordered[0] if ordered else None)

    def _merge_deposit_field(self, field: ExtractedField) -> ExtractedField:
        ordered = sorted(field.values, key=lambda item: item.confidence, reverse=True)
        for value in ordered:
            amount = self._parse_amount(value.value, value)
            if amount is not None and self._is_reasonable_deposit(amount):
                return self._with_primary(field, value)
        return self._with_primary(field, ordered[0] if ordered else None)

    def _merge_date_field(self, field: ExtractedField) -> ExtractedField:
        ordered = sorted(field.values, key=lambda item: item.confidence, reverse=True)
        for value in ordered:
            parsed = self._parse_date(value.value)
            if parsed and self._is_reasonable_date(parsed):
                return self._with_primary(field, value)
        return self._with_primary(field, ordered[0] if ordered else None)

    def _merge_number_field(self, field: ExtractedField) -> ExtractedField:
        ordered = sorted(field.values, key=lambda item: item.confidence, reverse=True)
        for value in ordered:
            if self._is_valid_number_format(value.value):
                return self._with_primary(field, value)
        return self._with_primary(field, ordered[0] if ordered else None)

    def _merge_contact_field(self, field: ExtractedField) -> ExtractedField:
        ordered = sorted(field.values, key=lambda item: item.confidence, reverse=True)
        for value in ordered:
            if self._is_valid_contact_format(value.value):
                return self._with_primary(field, value)
        return self._with_primary(field, ordered[0] if ordered else None)

    def _merge_default_field(self, field: ExtractedField) -> ExtractedField:
        ordered = sorted(field.values, key=lambda item: item.confidence, reverse=True)
        return self._with_primary(field, ordered[0] if ordered else None)

    # 兼容旧调用：冲突解决现在只选择 primary，不销毁其它 evidence。
    def _resolve_field_conflicts(self, field: ExtractedField) -> ExtractedField:
        strategy = self.merge_strategies.get(field.field_type, self.merge_strategies["default"])
        return strategy(field)

    def _resolve_duplicate_values(self, field: ExtractedField) -> ExtractedField:
        unique: list[EvidenceSpan] = []
        seen: set[tuple[str, int, int, str]] = set()
        for value in field.values:
            key = (value.value, value.start, value.end, value.source)
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        field.values = unique
        return self._merge_default_field(field)

    def _resolve_numeric_conflicts(self, field: ExtractedField) -> ExtractedField:
        return self._merge_amount_field(field) if field.field_type != "deposit" else self._merge_deposit_field(field)

    def _resolve_by_confidence(self, field: ExtractedField) -> ExtractedField:
        return self._merge_default_field(field)

    def _parse_amount(self, value: str, span: EvidenceSpan | None = None) -> Optional[float]:
        if span is not None and span.normalized_value:
            try:
                return float(span.normalized_value)
            except (TypeError, ValueError):
                pass
        cleaned = re.sub(r"[^\d.]", "", value)
        if not cleaned:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return amount * 10000 if "万" in value else amount

    def _parse_date(self, value: str) -> Optional[str]:
        patterns = [
            r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?",
            r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})",
        ]
        for index, pattern in enumerate(patterns):
            match = re.search(pattern, value)
            if not match:
                continue
            try:
                if index == 0:
                    year, month, day = match.groups()
                else:
                    month, day, year = match.groups()
                date = datetime(int(year), int(month), int(day))
                return date.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    def _is_reasonable_amount(self, amount: float) -> bool:
        return 1.0 <= amount <= 10_000_000_000.0

    def _is_reasonable_deposit(self, amount: float) -> bool:
        return 1_000.0 <= amount <= 1_000_000_000.0

    def _is_reasonable_date(self, date_str: str) -> bool:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return False
        now = datetime.now()
        return now - timedelta(days=3650) <= date <= now + timedelta(days=1825)

    def _is_valid_number_format(self, value: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9\-_/]{5,30}", value) is not None

    def _is_valid_contact_format(self, value: str) -> bool:
        compact = re.sub(r"[\s\-]", "", value)
        if re.fullmatch(r"1[3-9]\d{9}", compact):
            return True
        if re.fullmatch(r"0\d{2,3}\d{7,8}", compact):
            return True
        return re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", value) is not None

    def _calculate_amount_reasonableness(self, amount: float) -> float:
        if 1_000 <= amount <= 1_000_000_000:
            return 1.0
        if 100 <= amount <= 10_000_000_000:
            return 0.8
        return 0.3

    def _calculate_deposit_reasonableness(self, amount: float) -> float:
        if 1_000 <= amount <= 10_000_000:
            return 1.0
        if 100 <= amount <= 100_000_000:
            return 0.8
        return 0.3
