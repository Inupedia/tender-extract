import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tender_extract.package_cli import app


pytestmark = pytest.mark.e2e
runner = CliRunner()


def test_package_validate_and_extract_effective_views(tmp_path: Path):
    documents = {
        "tender.md": "项目范围：原招标范围\n",
        "amendment.md": "项目范围：修订后的招标范围\n",
        "clarification-v1.md": "澄清事项：第一次澄清\n",
        "clarification-v2.md": "澄清事项：第二次澄清\n",
        "bid-a.md": "承诺：A公司施工方案\n",
        "bid-b.md": "承诺：B公司施工方案\n",
    }
    for name, content in documents.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    manifest = tmp_path / "package.yaml"
    manifest.write_text(
        "package_id: pkg-e2e\n"
        "project_name: 青江水库除险加固工程\n"
        "documents:\n"
        "  - id: tender-v1\n"
        "    path: tender.md\n"
        "    role: tender\n"
        "  - id: amendment-1\n"
        "    path: amendment.md\n"
        "    role: amendment\n"
        "  - id: clarification-v1\n"
        "    path: clarification-v1.md\n"
        "    role: clarification\n"
        "    logical_name: clarification\n"
        "    revision: 1\n"
        "  - id: clarification-v2\n"
        "    path: clarification-v2.md\n"
        "    role: clarification\n"
        "    logical_name: clarification\n"
        "    revision: 2\n"
        "  - id: bid-a\n"
        "    path: bid-a.md\n"
        "    role: bid\n"
        "    bidder: A公司\n"
        "  - id: bid-b\n"
        "    path: bid-b.md\n"
        "    role: bid\n"
        "    bidder: B公司\n",
        encoding="utf-8",
    )

    config = tmp_path / "config.yaml"
    config.write_text(
        "base:\n"
        "  confidence_threshold: 0.7\n"
        "  use_ocr: false\n"
        "  use_modules: false\n"
        "llm:\n"
        "  provider: none\n"
        "cache:\n"
        "  enabled: false\n"
        "patterns:\n"
        "  project_scope:\n"
        "    - pattern: '项目范围[：:]\\s*([^\\n]+)'\n"
        "      confidence: 0.9\n"
        "  clarification_note:\n"
        "    - pattern: '澄清事项[：:]\\s*([^\\n]+)'\n"
        "      confidence: 0.9\n"
        "  commitment:\n"
        "    - pattern: '承诺[：:]\\s*([^\\n]+)'\n"
        "      confidence: 0.9\n",
        encoding="utf-8",
    )

    validate_result = runner.invoke(app, ["validate", str(manifest)])
    assert validate_result.exit_code == 0, validate_result.stdout
    assert "校验通过" in validate_result.stdout

    output = tmp_path / "package-result.json"
    extract_result = runner.invoke(
        app,
        [
            "extract",
            str(manifest),
            "--config",
            str(config),
            "--out",
            str(output),
        ],
    )
    assert extract_result.exit_code == 0, extract_result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["package_id"] == "pkg-e2e"
    assert payload["resolutions"]["clarification-v1"]["active"] is False
    assert payload["resolutions"]["clarification-v1"]["reason"] == "newer_revision"
    assert payload["resolutions"]["clarification-v2"]["active"] is True

    project_scope = payload["tender_fields"]["project_scope"]
    assert project_scope["primary_value"] == "修订后的招标范围"
    assert project_scope["selected_from_document"] == "amendment-1"
    assert "原招标范围" in project_scope["conflicts"]

    clarification = payload["tender_fields"]["clarification_note"]
    assert clarification["primary_value"] == "第二次澄清"
    assert clarification["selected_from_document"] == "clarification-v2"

    assert payload["bidder_fields"]["A公司"]["commitment"]["primary_value"] == "A公司施工方案"
    assert payload["bidder_fields"]["B公司"]["commitment"]["primary_value"] == "B公司施工方案"
    assert "B公司施工方案" not in payload["bidder_fields"]["A公司"]["commitment"]["values"]
