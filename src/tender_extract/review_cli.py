"""CLI for human review and feedback-loop operations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import build_processing_config, load_yaml, resolve_config_path
from .pipeline import ExtractionPipeline
from .review import ReviewStore
from .schema import ExtractionResult

app = typer.Typer(help="人工校核队列与 Gold Dataset 回流工具")
console = Console()


def _print_items(store: ReviewStore, status: str) -> int:
    items = store.list_items(status)
    table = Table(title=f"人工校核队列 ({status})")
    table.add_column("ID")
    table.add_column("状态")
    table.add_column("字段")
    table.add_column("置信度", justify="right")
    table.add_column("原因")
    table.add_column("候选值")
    for item in items:
        table.add_row(
            item.id,
            item.status,
            item.field_name,
            f"{item.confidence:.3f}",
            ",".join(item.reasons),
            " | ".join(item.candidate_values[:3]) or "-",
        )
    console.print(table)
    return len(items)


@app.command()
def run(
    document: str = typer.Argument(..., help="待抽取并生成校核项的文档"),
    queue: str = typer.Option(".review/queue.jsonl", "--queue", "-q", help="校核队列 JSONL"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c", help="抽取配置"),
    llm: str = typer.Option("none", "--llm", help="LLM 提供商，例如 siliconflow"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="覆盖 API Base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="覆盖 API Key（推荐环境变量）"),
    cache_dir: str = typer.Option(".cache/review", "--cache-dir", help="LLM 缓存目录"),
    result_out: Optional[str] = typer.Option(None, "--result", help="可选：保存完整抽取 JSON"),
):
    """运行真实抽取，并自动把需人工确认的字段写入 review queue。"""
    path = Path(document)
    if not path.exists():
        console.print(f"[red]文档不存在: {path}[/red]")
        raise typer.Exit(1)

    yaml_data = load_yaml(resolve_config_path(config))
    processing = build_processing_config(
        yaml_data,
        llm=llm,
        model=model,
        base_url=base_url,
        api_key=api_key,
        cache_dir=cache_dir,
        include_pii=False,
    )
    Path(processing.cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        result = ExtractionPipeline(processing).extract_file(str(path))
        store = ReviewStore(queue)
        added = store.add_result(result, path, processing.confidence_threshold)
    except Exception as exc:
        console.print(f"[red]校核任务生成失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    if result_out:
        output = Path(result_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"[green]抽取结果已保存: {output}[/green]")

    console.print(
        f"[green]完成[/green]: 抽取 {len(result.fields)} 个字段，"
        f"新增 {len(added)} 个待校核项，队列={queue}"
    )


@app.command()
def collect(
    result_json: str = typer.Argument(..., help="tender-extract 生成的 ExtractionResult JSON"),
    document: str = typer.Option(..., "--document", "-d", help="对应原始文档路径"),
    queue: str = typer.Option(".review/queue.jsonl", "--queue", "-q", help="校核队列 JSONL"),
    threshold: float = typer.Option(0.7, "--threshold", help="低置信度阈值"),
):
    """从已有抽取 JSON 生成/更新待校核项。"""
    if not 0.0 <= threshold <= 1.0:
        console.print("[red]--threshold 必须在 0 到 1 之间[/red]")
        raise typer.Exit(1)
    result_path = Path(result_json)
    document_path = Path(document)
    if not result_path.exists() or not document_path.exists():
        console.print("[red]result JSON 或原始文档不存在[/red]")
        raise typer.Exit(1)

    try:
        result = ExtractionResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        store = ReviewStore(queue)
        added = store.add_result(result, document_path, threshold)
    except Exception as exc:
        console.print(f"[red]生成校核项失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[green]新增 {len(added)} 个待校核项[/green]")


@app.command("list")
def list_reviews(
    queue: str = typer.Option(".review/queue.jsonl", "--queue", "-q"),
    status: str = typer.Option("pending", "--status", help="pending/resolved/all"),
):
    """查看待校核或已处理项目。"""
    try:
        count = _print_items(ReviewStore(queue), status)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"共 {count} 项")


@app.command()
def resolve(
    item_id: str = typer.Argument(..., help="Review item ID"),
    action: str = typer.Option(..., "--action", "-a", help="accept/correct/reject"),
    value: Optional[list[str]] = typer.Option(None, "--value", "-v", help="accept/correct 的最终值，可重复"),
    queue: str = typer.Option(".review/queue.jsonl", "--queue", "-q"),
    reviewer: Optional[str] = typer.Option(None, "--reviewer", help="校核人标识"),
    note: Optional[str] = typer.Option(None, "--note", help="校核备注"),
):
    """接受、修正或拒绝一个校核项。"""
    if action not in {"accept", "correct", "reject"}:
        console.print("[red]--action 必须是 accept/correct/reject[/red]")
        raise typer.Exit(1)
    try:
        item = ReviewStore(queue).resolve(
            item_id,
            action,  # type: ignore[arg-type]
            values=value,
            reviewer=reviewer,
            note=note,
        )
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    values = item.decision.values if item.decision else []
    console.print(
        f"[green]已处理[/green] {item.id} {item.field_name}: "
        f"{action} -> {values}"
    )


@app.command()
def export(
    queue: str = typer.Option(".review/queue.jsonl", "--queue", "-q"),
    out: str = typer.Option("eval/gold-reviewed.jsonl", "--out", "-o"),
):
    """把已人工确认的决策导出为可直接用于 tender-extract eval 的 Gold Dataset。"""
    try:
        cases = ReviewStore(queue).export_gold(out)
    except ValueError as exc:
        console.print(f"[red]导出失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]已导出 {cases} 个 Gold case: {out}[/green]")


if __name__ == "__main__":
    app()
