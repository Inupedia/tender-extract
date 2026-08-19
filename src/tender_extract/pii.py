"""敏感信息脱敏。"""
from __future__ import annotations

import re

from .schema import CertificateRecord, ExtractionResult, PersonnelRecord


_ID_CARD = re.compile(
    r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
)


def mask_id_card(value: str) -> str:
    if not value or len(value) < 10:
        return value
    return value[:6] + "****" + value[-4:]


def mask_text(value: str) -> str:
    return _ID_CARD.sub(lambda m: mask_id_card(m.group(0)), value)


def mask_result(result: ExtractionResult) -> ExtractionResult:
    for field in result.fields.values():
        if field.primary_value:
            field.primary_value = mask_text(field.primary_value)
        for span in field.values:
            span.value = mask_text(span.value)
            if span.ref:
                span.ref = mask_text(span.ref)
    masked_people: list[PersonnelRecord] = []
    for person in result.personnel:
        data = person.model_dump()
        if data.get("id_card"):
            data["id_card"] = mask_id_card(data["id_card"])
        masked_people.append(PersonnelRecord.model_validate(data))
    result.personnel = masked_people
    result.certificates = [
        CertificateRecord.model_validate(c.model_dump()) for c in result.certificates
    ]
    return result
