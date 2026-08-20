import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tender_extract.review import ReviewStore
from tender_extract.review_cli import app


pytestmark = pytest.mark.e2e
runner = CliRunner()


def test_review_run_resolve_and_export_gold(tmp_path: Path):
    source = tmp_path / "review.md"
    source.write_text(
        "# 项目概况\n项目范围：青江水库大坝及溢洪道除险加固\n",
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
        "      confidence: 0.6\n",
        encoding="utf-8",
    )
    queue_path = tmp_path / "review.jsonl"
    result_path = tmp_path / "result.json"

    run_result = runner.invoke(
        app,
        [
            "run",
            str(source),
            "--config",
            str(config),
            "--queue",
            str(queue_path),
            "--result",
            str(result_path),
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout
    assert result_path.exists()

    items = ReviewStore(queue_path).load()
    assert len(items) == 1
    assert items[0].field_name == "project_scope"
    assert "low_confidence" in items[0].reasons

    list_result = runner.invoke(
        app, ["list", "--queue", str(queue_path), "--status", "pending"]
    )
    assert list_result.exit_code == 0, list_result.stdout
    assert items[0].id in list_result.stdout

    resolve_result = runner.invoke(
        app,
        [
            "resolve",
            items[0].id,
            "--queue",
            str(queue_path),
            "--action",
            "correct",
            "--value",
            "青江水库枢纽除险加固",
            "--reviewer",
            "e2e",
        ],
    )
    assert resolve_result.exit_code == 0, resolve_result.stdout

    gold_path = tmp_path / "gold-reviewed.jsonl"
    export_result = runner.invoke(
        app, ["export", "--queue", str(queue_path), "--out", str(gold_path)]
    )
    assert export_result.exit_code == 0, export_result.stdout

    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    assert payload["document"] == str(source.resolve())
    assert payload["expected"]["project_scope"] == ["青江水库枢纽除险加固"]
    assert payload["tags"] == ["human-reviewed"]
