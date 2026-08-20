import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tender_extract.cli import app


pytestmark = pytest.mark.e2e
runner = CliRunner()


def test_cli_eval_scores_real_pipeline_and_writes_report(tmp_path: Path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    source = fixture_dir / "sample.md"
    source.write_text(
        "# 招标文件\n"
        "项目编号：EVAL-2026-001\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(
        '{"id":"eval-basic","document":"fixtures/sample.md","expected":{"project_number":["EVAL-2026-001"]}}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "base:\n  use_ocr: false\n  confidence_threshold: 0.7\nllm:\n  provider: none\ncache:\n  enabled: false\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            str(dataset),
            "--config",
            str(config),
            "--report",
            str(report_path),
            "--fail-under",
            "1.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["micro"]["f1"] == 1.0
    assert payload["exact_case_accuracy"] == 1.0
    assert payload["llm"]["provider"] == "none"


def test_cli_eval_quality_gate_returns_exit_code_two(tmp_path: Path):
    source = tmp_path / "sample.md"
    source.write_text("项目编号：ACTUAL-2026-001\n", encoding="utf-8")
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(
        '{"id":"eval-fail","document":"sample.md","expected":{"project_number":["EXPECTED-2026-001"]}}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("base:\n  use_ocr: false\nllm:\n  provider: none\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "eval",
            str(dataset),
            "--config",
            str(config),
            "--fail-under",
            "0.5",
        ],
    )

    assert result.exit_code == 2
    assert "质量门禁失败" in result.stdout
