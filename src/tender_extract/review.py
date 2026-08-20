"""Human-review queue and feedback export for extracted tender fields."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .schema import EvidenceLocation, ExtractionResult

ReviewAction = Literal["accept", "correct", "reject"]
ReviewStatus = Literal["pending", "resolved"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


class ReviewEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    ref: Optional[str] = None
    location: Optional[EvidenceLocation] = None


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: ReviewAction
    values: list[str] = Field(default_factory=list)
    reviewer: Optional[str] = None
    note: Optional[str] = None
    decided_at: str


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    document: str
    field_name: str
    candidate_values: list[str] = Field(default_factory=list)
    primary_value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    evidence: list[ReviewEvidence] = Field(default_factory=list)
    status: ReviewStatus = "pending"
    decision: Optional[ReviewDecision] = None
    created_at: str
    updated_at: str


def build_review_items(
    result: ExtractionResult,
    document: str | Path,
    confidence_threshold: float,
) -> list[ReviewItem]:
    """Create review items for low-confidence, conflicting, or LLM-led fields."""
    document_path = str(Path(document).resolve())
    items: list[ReviewItem] = []

    for field_name, extracted in sorted(result.fields.items()):
        reasons: list[str] = []
        if extracted.confidence < confidence_threshold:
            reasons.append("low_confidence")
        if extracted.conflicts:
            reasons.append("conflict")

        primary_span = extracted.values[0] if extracted.values else None
        if primary_span is not None and primary_span.source == "llm":
            reasons.append("llm_recovered")

        if not reasons:
            continue

        candidates = _unique_values(
            [span.normalized_value or span.value for span in extracted.values]
            + ([extracted.primary_value] if extracted.primary_value else [])
        )
        raw_fingerprint = json.dumps(
            {
                "document": document_path,
                "field": field_name,
                "candidates": candidates,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        item_id = hashlib.sha256(raw_fingerprint.encode("utf-8")).hexdigest()[:20]
        timestamp = _now_iso()
        items.append(
            ReviewItem(
                id=item_id,
                document=document_path,
                field_name=field_name,
                candidate_values=candidates,
                primary_value=extracted.primary_value,
                confidence=extracted.confidence,
                reasons=reasons,
                conflicts=list(extracted.conflicts),
                evidence=[
                    ReviewEvidence(
                        value=span.normalized_value or span.value,
                        source=span.source,
                        confidence=span.confidence,
                        ref=(span.ref[:800] if span.ref else None),
                        location=span.location,
                    )
                    for span in extracted.values[:5]
                ],
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
    return items


class ReviewStore:
    """JSONL-backed review store with deterministic upsert semantics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[ReviewItem]:
        if not self.path.exists():
            return []
        items: list[ReviewItem] = []
        for line_number, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(ReviewItem.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid review queue JSONL at {self.path}:{line_number}: {exc}"
                ) from exc
        return items

    def _save(self, items: list[ReviewItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        content = "\n".join(
            json.dumps(item.model_dump(), ensure_ascii=False, sort_keys=True)
            for item in sorted(items, key=lambda item: (item.document, item.field_name, item.id))
        )
        if content:
            content += "\n"
        temp.write_text(content, encoding="utf-8")
        temp.replace(self.path)

    def upsert(self, incoming: list[ReviewItem]) -> list[ReviewItem]:
        existing = {item.id: item for item in self.load()}
        added: list[ReviewItem] = []
        for item in incoming:
            current = existing.get(item.id)
            if current is None:
                existing[item.id] = item
                added.append(item)
                continue
            if current.status == "resolved":
                continue
            existing[item.id] = item.model_copy(
                update={
                    "created_at": current.created_at,
                    "updated_at": _now_iso(),
                }
            )
        self._save(list(existing.values()))
        return added

    def add_result(
        self,
        result: ExtractionResult,
        document: str | Path,
        confidence_threshold: float,
    ) -> list[ReviewItem]:
        return self.upsert(build_review_items(result, document, confidence_threshold))

    def list_items(self, status: str = "pending") -> list[ReviewItem]:
        items = self.load()
        if status == "all":
            return items
        if status not in {"pending", "resolved"}:
            raise ValueError("status must be pending, resolved, or all")
        return [item for item in items if item.status == status]

    def resolve(
        self,
        item_id: str,
        action: ReviewAction,
        values: Optional[list[str]] = None,
        reviewer: Optional[str] = None,
        note: Optional[str] = None,
    ) -> ReviewItem:
        items = self.load()
        target: Optional[ReviewItem] = None
        for item in items:
            if item.id == item_id:
                target = item
                break
        if target is None:
            raise KeyError(f"Review item not found: {item_id}")

        supplied = _unique_values(values or [])
        if action == "correct":
            if not supplied:
                raise ValueError("correct requires at least one corrected value")
            final_values = supplied
        elif action == "accept":
            if supplied:
                final_values = supplied
            elif target.primary_value:
                final_values = [target.primary_value]
            elif target.candidate_values:
                final_values = [target.candidate_values[0]]
            else:
                raise ValueError("accept requires a candidate or explicit value")
        elif action == "reject":
            final_values = []
        else:
            raise ValueError(f"Unsupported review action: {action}")

        timestamp = _now_iso()
        resolved = target.model_copy(
            update={
                "status": "resolved",
                "decision": ReviewDecision(
                    action=action,
                    values=final_values,
                    reviewer=reviewer,
                    note=note,
                    decided_at=timestamp,
                ),
                "updated_at": timestamp,
            }
        )
        updated = [resolved if item.id == item_id else item for item in items]
        self._save(updated)
        return resolved

    def export_gold(self, output_path: str | Path) -> int:
        """Export resolved decisions as partially-labelled Gold Dataset JSONL."""
        resolved = [
            item for item in self.load()
            if item.status == "resolved" and item.decision is not None
        ]
        latest_by_field: dict[tuple[str, str], ReviewItem] = {}
        for item in sorted(resolved, key=lambda item: item.updated_at):
            latest_by_field[(item.document, item.field_name)] = item

        grouped: dict[str, dict[str, list[str]]] = {}
        for (document, field_name), item in latest_by_field.items():
            assert item.decision is not None
            grouped.setdefault(document, {})[field_name] = list(item.decision.values)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for document, expected in sorted(grouped.items()):
            case_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()[:12]
            payload = {
                "id": f"review-{case_hash}",
                "document": document,
                "expected": expected,
                "tags": ["human-reviewed"],
            }
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        output.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        return len(lines)
