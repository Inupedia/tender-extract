from pathlib import Path

import pytest

from tender_extract.tender_package import (
    PackageDocument,
    PackageFieldSource,
    TenderPackageManifest,
    _resolve_effective_field,
    load_package_manifest,
    validate_package_manifest,
)


pytestmark = pytest.mark.unit


def _doc(
    document_id: str,
    *,
    role: str = "tender",
    logical_name: str | None = None,
    revision: int = 1,
    supersedes: list[str] | None = None,
    bidder: str | None = None,
) -> PackageDocument:
    return PackageDocument(
        id=document_id,
        path=f"/{document_id}.md",
        role=role,
        logical_name=logical_name,
        revision=revision,
        supersedes=supersedes or [],
        bidder=bidder,
    )


def test_latest_logical_revision_becomes_effective():
    manifest = TenderPackageManifest(
        package_id="pkg-1",
        documents=[
            _doc("clarification-v1", role="clarification", logical_name="clarification", revision=1),
            _doc("clarification-v2", role="clarification", logical_name="clarification", revision=2),
        ],
    )

    resolutions = validate_package_manifest(manifest)

    assert resolutions["clarification-v1"].active is False
    assert resolutions["clarification-v1"].reason == "newer_revision"
    assert resolutions["clarification-v1"].superseded_by == ["clarification-v2"]
    assert resolutions["clarification-v2"].active is True


def test_explicit_supersedes_and_cycle_validation():
    manifest = TenderPackageManifest(
        package_id="pkg-1",
        documents=[
            _doc("v1"),
            _doc("v2", supersedes=["v1"]),
        ],
    )
    resolutions = validate_package_manifest(manifest)
    assert resolutions["v1"].reason == "superseded"
    assert resolutions["v1"].superseded_by == ["v2"]

    cyclic = TenderPackageManifest(
        package_id="pkg-cycle",
        documents=[
            _doc("a", supersedes=["b"]),
            _doc("b", supersedes=["a"]),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_package_manifest(cyclic)


def test_supersedes_cannot_cross_bidder_scope():
    manifest = TenderPackageManifest(
        package_id="pkg-bidders",
        documents=[
            _doc("bid-a-v1", role="bid", bidder="A公司"),
            _doc("bid-b-v2", role="bid", bidder="B公司", supersedes=["bid-a-v1"]),
        ],
    )

    with pytest.raises(ValueError, match="scope"):
        validate_package_manifest(manifest)


def test_amendment_field_wins_tender_but_keeps_conflict_provenance():
    resolved = _resolve_effective_field(
        "bid_deadline",
        [
            PackageFieldSource(
                document_id="tender-v1",
                role="tender",
                revision=1,
                primary_value="2026-09-01 10:00",
                values=["2026-09-01 10:00"],
                confidence=0.95,
            ),
            PackageFieldSource(
                document_id="amendment-1",
                role="amendment",
                revision=1,
                primary_value="2026-09-03 10:00",
                values=["2026-09-03 10:00"],
                confidence=0.90,
            ),
        ],
    )

    assert resolved.primary_value == "2026-09-03 10:00"
    assert resolved.selected_from_document == "amendment-1"
    assert resolved.conflicts == ["2026-09-01 10:00"]
    assert [source.document_id for source in resolved.sources] == ["amendment-1", "tender-v1"]


def test_manifest_paths_are_relative_to_manifest(tmp_path: Path):
    source = tmp_path / "招标文件.md"
    source.write_text("项目名称：青江水库除险加固工程\n", encoding="utf-8")
    manifest_path = tmp_path / "package.yaml"
    manifest_path.write_text(
        "package_id: pkg-relative\n"
        "documents:\n"
        "  - id: tender\n"
        "    path: 招标文件.md\n"
        "    role: tender\n",
        encoding="utf-8",
    )

    manifest = load_package_manifest(manifest_path)

    assert manifest.documents[0].path == str(source.resolve())
