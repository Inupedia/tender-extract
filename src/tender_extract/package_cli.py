"""CLI for project-level tender package validation and extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import build_processing_config, load_yaml, resolve_config_path
from .tender_package import extract_package, load_package_manifest, validate_package_manifest

app = typer.Typer(help="项目级标书包：版本、修订、投标人隔离与聚合抽取")
console = Console()


def _load(manifest_path: str):
    try:
        return load_package_manifest(manifest_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]标书包错误：{exc}[/red]")
        raise typer.Exit(1) from exc


def _print_resolution(manifest, resolutions) -> None:
    table = Table(title=f"Tender Package: {manifest.package_id}")
    table.add_column("ID")
    table.add_column("Role")
    table.add_column("Rev", justify="right")
    table.add_column("Bidder")
    table.add_column("Status")
    table.add_column("Reason")
    for document in manifest.documents:
        resolution = resolutions[document.id]
        status = "[green]active[/green]" if resolution.active else "[yellow]inactive[/yellow]"
        table.add_row(
            document.id,
            document.role,
            str(document.revision),
            document.bidder or "-",
            status,
            resolution.reason or "-",
        )
    console.print(table)


@app.command("validate")
def validate(
    manifest_path: str = typer.Argument(..., help="package.yaml / package.json"),
):
    """校验文件存在性、版本唯一性、supersedes 引用和循环。"""
    manifest = _load(manifest_path)
    resolutions = validate_package_manifest(manifest, check_files=True)
    _print_resolution(manifest, resolutions)
    active = sum(1 for item in resolutions.values() if item.active)
    console.print(f"[green]校验通过：{len(manifest.documents)} 个文件，{active} 个 active[/green]")


@app.command("inspect")
def inspect(
    manifest_path: str = typer.Argument(..., help="package.yaml / package.json"),
):
    """查看标书包的 effective document view。"""
    manifest = _load(manifest_path)
    resolutions = validate_package_manifest(manifest, check_files=True)
    _print_resolution(manifest, resolutions)
    bidders = sorted({document.bidder for document in manifest.documents if document.bidder})
    console.print(Panel.fit(
        f"项目：{manifest.project_name or '-'}\n"
        f"文件：{len(manifest.documents)}\n"
        f"投标人：{', '.join(bidders) if bidders else '-'}",
        title=manifest.package_id,
    ))


@app.command("extract")
def extract(
    manifest_path: str = typer.Argument(..., help="package.yaml / package.json"),
    out: str = typer.Option("./out/package.json", "--out", "-o", help="项目级 JSON 输出"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c", help="抽取配置"),
    llm: str = typer.Option("none", "--llm", help="LLM 提供商，例如 siliconflow"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="覆盖 API Base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="覆盖 API Key（推荐环境变量）"),
    cache_dir: str = typer.Option(".cache/package", "--cache-dir", help="LLM 缓存目录"),
    use_ner: bool = typer.Option(False, "--use-ner", help="启用 NER"),
    debug: bool = typer.Option(False, "--debug", help="调试模式"),
):
    """抽取整个项目包并生成招标侧 effective fields + 按投标人隔离的字段。"""
    manifest = _load(manifest_path)
    config_path = resolve_config_path(config)
    yaml_data = load_yaml(config_path)
    processing = build_processing_config(
        yaml_data,
        use_ner=use_ner,
        llm=llm,
        model=model,
        base_url=base_url,
        api_key=api_key,
        cache_dir=cache_dir,
        include_pii=False,
        debug=debug,
    )
    Path(processing.cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        result = extract_package(manifest, processing)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]项目包抽取失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    active = sum(1 for item in result.resolutions.values() if item.active)
    console.print(Panel.fit(
        f"项目包：{result.package_id}\n"
        f"文件：{len(result.documents)}（active {active}）\n"
        f"招标侧 effective fields：{len(result.tender_fields)}\n"
        f"投标人：{len(result.bidder_fields)}\n"
        f"LLM calls：{result.llm_calls}\n"
        f"输出：{output}",
        title="项目包抽取完成",
    ))


if __name__ == "__main__":
    app()
