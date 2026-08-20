import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tender_extract.cli import app


pytestmark = pytest.mark.e2e
runner = CliRunner()


def test_cli_extract_writes_json_from_real_pipeline(tmp_path: Path):
    source = tmp_path / "e2e.md"
    source.write_text(
        "# 招标文件\n"
        "项目名称：E2E水利测试工程\n"
        "项目编号：E2E-2026-001\n"
        "投标人：端到端测试工程有限公司\n"
        "投标报价：88万元\n"
        "投标保证金：4万元\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "base:\n  use_ocr: false\n  confidence_threshold: 0.7\nllm:\n  provider: none\ncache:\n  enabled: false\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "extract",
            str(source),
            "--out",
            str(out),
            "--config",
            str(config),
            "--no-modules",
        ],
    )

    assert result.exit_code == 0, result.stdout
    output_file = out / "e2e.md.json"
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["fields"]["project_number"]["primary_value"] == "E2E-2026-001"
    assert payload["metadata"]["extraction_stats"]["original_format"] == "md"
    assert payload["llm_calls"] == 0


def test_cli_missing_input_fails_with_nonzero_exit(tmp_path: Path):
    missing = tmp_path / "does-not-exist.md"

    result = runner.invoke(app, ["extract", str(missing)])

    assert result.exit_code == 1
    assert "输入路径不存在" in result.stdout
