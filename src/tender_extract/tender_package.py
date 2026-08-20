"""Project-level tender package manifests, revision resolution and aggregation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .pipeline import ExtractionPipeline
from .schema import ExtractionResult, ProcessingConfig

DocumentRole = Literal[
    "tender", "amendment", "clarification", "bid", "attachment", "other"
]

ROLE_PRIORITY: dict[str, int] = {
    "other": 0,
    "attachment": 100,
    "tender": 200,
    "clarification": 300,
    "amendment": 400,
    "bid": 500,
}


class PackageDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    role: DocumentRole
    logical_name: Optional[str] = None
    revision: int = Field(default=1, ge=1)
    supersedes: list[str] = Field(default_factory=list)
    issued_at: Optional[str] = None
    bidder: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bidder(self) -> "PackageDocument":
        if self.role == "bid" and not (self.bidder or "").strip():
            raise ValueError("bid documents require bidder")
        return self


class TenderPackageManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    package_id: str = Field(min_length=1)
    project_name: Optional[str] = None
    documents: list[PackageDocument] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResolution(BaseModel):
    id: str
    active: bool
    reason: Optional[str] = None
    superseded_by: list[str] = Field(default_factory=list)


class PackageFieldSource(BaseModel):
    document_id: str
    role: DocumentRole
    revision: int
    bidder: Optional[str] = None
    primary_value: Optional[str] = None
    values: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EffectivePackageField(BaseModel):
    field_name: str
    primary_value: Optional[str] = None
    values: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    selected_from_document: str
    conflicts: list[str] = Field(default_factory=list)
    sources: list[PackageFieldSource] = Field(default_factory=list)


class PackageDocumentResult(BaseModel):
    document: PackageDocument
    active: bool
    inactive_reason: Optional[str] = None
    extraction: ExtractionResult


class PackageExtractionResult(BaseModel):
    package_id: str
    project_name: Optional[str] = None
    resolutions: dict[str, DocumentResolution]
    documents: dict[str, PackageDocumentResult]
    tender_fields: dict[str, EffectivePackageField] = Field(default_factory=dict)
    bidder_fields: dict[str, dict[str, EffectivePackageField]] = Field(default_factory=dict)
    llm_calls: int = 0


def load_package_manifest(path: str | Path) -> TenderPackageManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Tender package manifest does not exist: {manifest_path}")
    raw = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    manifest = TenderPackageManifest.model_validate(payload)

    base_dir = manifest_path.resolve().parent
    normalized: list[PackageDocument] = []
    for document in manifest.documents:
        document_path = Path(document.path)
        if not document_path.is_absolute():
            document_path = base_dir / document_path
        normalized.append(document.model_copy(update={"path": str(document_path.resolve())}))
    manifest = manifest.model_copy(update={"documents": normalized})
    validate_package_manifest(manifest, check_files=True)
    return manifest


def validate_package_manifest(
    manifest: TenderPackageManifest,
    *,
    check_files: bool = False,
) -> dict[str, DocumentResolution]:
    ids = [document.id for document in manifest.documents]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"Duplicate document ids: {', '.join(duplicates)}")

    by_id = {document.id: document for document in manifest.documents}
    if check_files:
        missing = [document.path for document in manifest.documents if not Path(document.path).exists()]
        if missing:
            raise FileNotFoundError(f"Package document does not exist: {missing[0]}")

    for document in manifest.documents:
        if document.id in document.supersedes:
            raise ValueError(f"Document {document.id} cannot supersede itself")
        for target in document.supersedes:
            if target not in by_id:
                raise ValueError(f"Document {document.id} supersedes unknown document {target}")

    _validate_supersedes_acyclic(manifest.documents)
    _validate_logical_revisions(manifest.documents)

    superseded_by: dict[str, list[str]] = defaultdict(list)
    for document in manifest.documents:
        for target in document.supersedes:
            superseded_by[target].append(document.id)

    resolutions: dict[str, DocumentResolution] = {}
    for document in manifest.documents:
        replacements = sorted(superseded_by.get(document.id, []))
        if replacements:
            resolutions[document.id] = DocumentResolution(
                id=document.id,
                active=False,
                reason="superseded",
                superseded_by=replacements,
            )
        else:
            resolutions[document.id] = DocumentResolution(id=document.id, active=True)

    logical_groups: dict[tuple[str, str, str], list[PackageDocument]] = defaultdict(list)
    for document in manifest.documents:
        if not document.logical_name:
            continue
        key = (document.role, document.bidder or "", document.logical_name)
        logical_groups[key].append(document)

    for documents in logical_groups.values():
        active = [document for document in documents if resolutions[document.id].active]
        if len(active) <= 1:
            continue
        latest_revision = max(document.revision for document in active)
        latest = [document for document in active if document.revision == latest_revision]
        if len(latest) > 1:
            ids_text = ", ".join(sorted(document.id for document in latest))
            raise ValueError(f"Ambiguous latest revision for logical document: {ids_text}")
        winner = latest[0]
        for document in active:
            if document.id == winner.id:
                continue
            resolutions[document.id] = DocumentResolution(
                id=document.id,
                active=False,
                reason="newer_revision",
                superseded_by=[winner.id],
            )
    return resolutions


def _validate_supersedes_acyclic(documents: list[PackageDocument]) -> None:
    graph = {document.id: list(document.supersedes) for document in documents}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(f"Supersedes cycle detected at document {node}")
        visiting.add(node)
        for target in graph.get(node, []):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _validate_logical_revisions(documents: list[PackageDocument]) -> None:
    seen: dict[tuple[str, str, str, int], str] = {}
    for document in documents:
        if not document.logical_name:
            continue
        key = (
            document.role,
            document.bidder or "",
            document.logical_name,
            document.revision,
        )
        previous = seen.get(key)
        if previous:
            raise ValueError(
                f"Duplicate revision {document.revision} for logical document "
                f"{document.logical_name}: {previous}, {document.id}"
            )
        seen[key] = document.id


def extract_package(
    manifest: TenderPackageManifest,
    config: ProcessingConfig,
) -> PackageExtractionResult:
    resolutions = validate_package_manifest(manifest, check_files=True)
    pipeline = ExtractionPipeline(config)
    documents: dict[str, PackageDocumentResult] = {}

    for document in manifest.documents:
        resolution = resolutions[document.id]
        extraction = pipeline.extract_file(document.path)
        documents[document.id] = PackageDocumentResult(
            document=document,
            active=resolution.active,
            inactive_reason=resolution.reason,
            extraction=extraction,
        )

    tender_sources: dict[str, list[PackageFieldSource]] = defaultdict(list)
    bidder_sources: dict[str, dict[str, list[PackageFieldSource]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for document in manifest.documents:
        resolution = resolutions[document.id]
        if not resolution.active:
            continue
        extraction = documents[document.id].extraction
        target: Optional[dict[str, list[PackageFieldSource]]] = None
        if document.role in {"tender", "amendment", "clarification"}:
            target = tender_sources
        elif document.role == "bid":
            target = bidder_sources[document.bidder or "__unscoped__"]
        elif document.role == "attachment":
            target = (
                bidder_sources[document.bidder]
                if document.bidder
                else tender_sources
            )
        else:
            target = tender_sources

        for field_name, field in extraction.fields.items():
            values = _unique_values(
                [span.normalized_value or span.value for span in field.values]
                + ([field.primary_value] if field.primary_value else [])
            )
            target[field_name].append(
                PackageFieldSource(
                    document_id=document.id,
                    role=document.role,
                    revision=document.revision,
                    bidder=document.bidder,
                    primary_value=field.primary_value,
                    values=values,
                    confidence=field.confidence,
                )
            )

    tender_fields = {
        field_name: _resolve_effective_field(field_name, sources)
        for field_name, sources in tender_sources.items()
    }
    bidder_fields = {
        bidder: {
            field_name: _resolve_effective_field(field_name, sources)
            for field_name, sources in field_map.items()
        }
        for bidder, field_map in bidder_sources.items()
    }

    return PackageExtractionResult(
        package_id=manifest.package_id,
        project_name=manifest.project_name,
        resolutions=resolutions,
        documents=documents,
        tender_fields=tender_fields,
        bidder_fields=bidder_fields,
        llm_calls=sum(item.extraction.llm_calls for item in documents.values()),
    )


def _resolve_effective_field(
    field_name: str,
    sources: list[PackageFieldSource],
) -> EffectivePackageField:
    ranked = sorted(
        sources,
        key=lambda source: (
            ROLE_PRIORITY.get(source.role, 0),
            source.revision,
            source.confidence,
            source.document_id,
        ),
        reverse=True,
    )
    selected = ranked[0]
    all_values = _unique_values([value for source in ranked for value in source.values])
    selected_values = _unique_values(
        selected.values + ([selected.primary_value] if selected.primary_value else [])
    )
    primary = selected.primary_value or (selected_values[0] if selected_values else None)
    conflicts = [value for value in all_values if value != primary]
    return EffectivePackageField(
        field_name=field_name,
        primary_value=primary,
        values=all_values,
        confidence=selected.confidence,
        selected_from_document=selected.document_id,
        conflicts=conflicts,
        sources=ranked,
    )


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
