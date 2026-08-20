"""审计友好的字段合并：选择 primary，但不丢弃候选证据。"""
from __future__ import annotations

from .merge import FieldMerger
from .schema import ExtractedField


class AuditPreservingFieldMerger(FieldMerger):
    def merge_fields(self, fields: dict[str, ExtractedField]) -> dict[str, ExtractedField]:
        return self.resolve_conflicts(fields)

    def resolve_conflicts(self, fields: dict[str, ExtractedField]) -> dict[str, ExtractedField]:
        resolved: dict[str, ExtractedField] = {}
        for name, original in fields.items():
            field = original.model_copy(deep=True)
            if not field.values:
                resolved[name] = field
                continue

            # 候选全部保留，仅根据置信度和基础合理性选择 primary。
            ordered = sorted(field.values, key=lambda item: item.confidence, reverse=True)
            best = self._choose_primary(field, ordered)
            field.values = ordered
            field.primary_value = best.value
            field.confidence = best.confidence
            resolved[name] = field
        return resolved

    def _choose_primary(self, field: ExtractedField, ordered):
        if field.field_type in {"bid_amount", "deposit"}:
            for value in ordered:
                amount = self._parse_amount(value.value, value)
                if amount is None:
                    continue
                if field.field_type == "bid_amount" and self._is_reasonable_amount(amount):
                    return value
                if field.field_type == "deposit" and self._is_reasonable_deposit(amount):
                    return value
        if field.field_type in {"bid_date", "date"}:
            for value in ordered:
                parsed = self._parse_date(value.value)
                if parsed and self._is_reasonable_date(parsed):
                    return value
        return ordered[0]
