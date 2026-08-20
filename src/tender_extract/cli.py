"""命令行入口。"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import build_processing_config, load_yaml, resolve_config_path
from .document_parser import DocumentParser
from .evaluation import EvaluationReport, evaluate_dataset
from .llm_providers import list_providers
from .pipeline import ExtractionPipeline
from .schema import ExtractionResult

app = typer.Typer(help="面向中文标书的混合抽取流水线")
console = Console()
logger = logging.getLogger(__name__)

PRIORITY_FIELDS = [
    "project_name", "tenderer", "bidder", "bid_amount", "deposit",
    "legal_representative", "bid_date", "project_number", "contact_info",
]


def _configure_logging(verbose: bool, debug: bool) -> None:
    if not (verbose or debug):
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    parser = DocumentParser(use_ocr=False)
    supported = parser.get_supported_extensions()
    if input_path.is_file():
        return [input_path]
    if pattern in {"*", "*.*"}:
        return [p for p in input_path.iterdir() if p.suffix.lower() in supported]
    return list(input_path.glob(pattern))


def _save_result(result: ExtractionResult, out_path: Path) -> Path:
    output_file = out_path / f"{result.metadata.filename}.json"
    output_file.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def _print_file_summary(result: ExtractionResult) -> None:
    stats = result.metadata.extraction_stats
    console.print(
        f"  [bold green]完成[/bold green]: {len(result.fields)} 个字段, "
        f"平均置信度 {stats.get('avg_confidence', 0):.2f}, "
        f"耗时 {result.metadata.processing_time:.2f}s"
    )
    for name in PRIORITY_FIELDS:
        field = result.fields.get(name)
        if not field:
            continue
        color = "green" if field.confidence >= 0.9 else "yellow" if field.confidence >= 0.7 else "red"
        console.print(
            f"     {name}: [{color}]{field.primary_value}[/{color}] "
            f"(置信度={field.confidence:.2f})"
        )
    if result.personnel:
        console.print(f"  人员: {len(result.personnel)}")
        for person in result.personnel[:5]:
            extra = f" ({person.role})" if person.role else ""
            id_part = f" 身份证:{person.id_card}" if person.id_card else ""
            console.print(f"     {person.name}{extra}{id_part}")
    if result.certificates:
        console.print(f"  证书: {len(result.certificates)}")
        for cert in result.certificates[:3]:
            expiry = f" 有效期:{cert.expiry_date}" if cert.expiry_date else ""
            console.print(f"     [{cert.cert_type}] {cert.cert_number}{expiry}")
    for warning in result.warnings:
        console.print(f"  [yellow]警告: {warning}[/yellow]")


def _print_evaluation_summary(report: EvaluationReport) -> None:
    table = Table(title="抽取质量评测")
    table.add_column("字段")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("TP/FP/FN", justify="right")
    for field_name, metrics in sorted(report.per_field.items()):
        table.add_row(
            field_name,
            f"{metrics.precision:.3f}",
            f"{metrics.recall:.3f}",
            f"{metrics.f1:.3f}",
            f"{metrics.true_positive}/{metrics.false_positive}/{metrics.false_negative}",
        )
    console.print(table)
    console.print(Panel.fit(
        f"[bold]Micro F1[/bold]: {report.micro.f1:.3f}\n"
        f"Macro F1: {report.macro_f1:.3f}\n"
        f"Exact case accuracy: {report.exact_case_accuracy:.3f} "
        f"({report.exact_cases}/{report.cases})\n"
        f"LLM: {report.provider} / {report.model or '默认模型'}\n"
        f"LLM calls: {report.llm_calls}\n"
        f"Failed fields: {len(report.failures)}",
        title="评测摘要",
    ))
    for failure in report.failures[:20]:
        console.print(
            f"[yellow]{failure.case_id}/{failure.field_name}[/yellow] "
            f"expected={list(failure.expected)} predicted={list(failure.predicted)}"
        )


def _run_extract(
    input_path: str,
    out: str,
    config: str,
    use_ner: bool,
    llm: str,
    model: Optional[str],
    pattern: str,
    verbose: bool,
    debug: bool,
    use_modules: bool,
    include_pii: bool,
    base_url: Optional[str],
    api_key: Optional[str],
    cache_dir: str,
) -> None:
    _configure_logging(verbose, debug)
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
        use_modules=use_modules,
        include_pii=include_pii,
        debug=debug,
    )

    source = Path(input_path)
    if not source.exists():
        console.print(f"[red]错误：输入路径不存在 {input_path}[/red]")
        raise typer.Exit(1)

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    Path(processing.cache_dir).mkdir(parents=True, exist_ok=True)

    files = _collect_files(source, pattern)
    if not files:
        console.print("[yellow]未找到支持的文件（PDF / DOCX / MD / TXT）[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]找到 {len(files)} 个文件[/green]")
    if processing.llm_provider != "none":
        console.print(
            f"[blue]LLM: {processing.llm_provider} / {processing.llm_model or '默认模型'}[/blue]"
        )

    pipeline = ExtractionPipeline(processing)
    results: list[ExtractionResult] = []
    started = time.time()

    for index, file_path in enumerate(files, start=1):
        console.print(f"\n[bold cyan]处理文件 {index}/{len(files)}: {file_path.name}[/bold cyan]")
        try:
            result = pipeline.extract_file(str(file_path))
            results.append(result)
            saved = _save_result(result, out_path)
            _print_file_summary(result)
            console.print(f"[green]已保存: {saved}[/green]")
        except Exception as exc:
            console.print(f"[red]处理失败 {file_path}: {exc}[/red]")
            if debug:
                logger.exception("extract failed")

    elapsed = time.time() - started
    total_fields = sum(len(item.fields) for item in results)
    console.print(Panel.fit(
        f"[bold]处理完成[/bold]\n"
        f"文件数：{len(results)}\n"
        f"总字段数：{total_fields}\n"
        f"LLM 调用：{sum(item.llm_calls for item in results)}\n"
        f"总耗时：{elapsed:.2f}秒\n"
        f"平均每文件：{(elapsed / len(results) if results else 0):.2f}秒",
        title="抽取摘要",
    ))
    if not results:
        raise typer.Exit(1)


@app.command()
def extract(
    input_path: str = typer.Argument(..., help="输入文件或目录（PDF/DOCX/MD/TXT）"),
    out: str = typer.Option("./out", "--out", "-o", help="输出目录"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c", help="配置文件路径"),
    use_ner: bool = typer.Option(False, "--use-ner", help="启用 jieba NER 补充"),
    llm: str = typer.Option("none", "--llm", help="LLM 提供商，见 `tender-extract providers`"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="覆盖 API Base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="覆盖 API Key（优先用环境变量）"),
    pattern: str = typer.Option("*", "--pattern", "-p", help="目录匹配模式"),
    cache_dir: str = typer.Option(".cache", "--cache-dir", help="LLM 缓存目录"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
    debug: bool = typer.Option(False, "--debug", help="调试模式"),
    use_modules: bool = typer.Option(True, "--modules/--no-modules", help="启用模块化路由"),
    include_pii: bool = typer.Option(False, "--include-pii", help="输出完整身份证号（默认脱敏）"),
):
    """抽取标书信息（推荐入口，支持多格式与主流 LLM）。"""
    _run_extract(
        input_path, out, config, use_ner, llm, model, pattern,
        verbose, debug, use_modules, include_pii, base_url, api_key, cache_dir,
    )


@app.command("extract-v2")
def extract_v2(
    input_path: str = typer.Argument(..., help="输入文件或目录"),
    out: str = typer.Option("./out", "--out", "-o"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c"),
    use_ner: bool = typer.Option(False, "--use-ner"),
    llm: str = typer.Option("none", "--llm"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    pattern: str = typer.Option("*", "--pattern", "-p"),
    cache_dir: str = typer.Option(".cache", "--cache-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    debug: bool = typer.Option(False, "--debug"),
    use_modules: bool = typer.Option(True, "--modules/--no-modules"),
    include_pii: bool = typer.Option(False, "--include-pii"),
):
    """extract 的兼容别名。"""
    _run_extract(
        input_path, out, config, use_ner, llm, model, pattern,
        verbose, debug, use_modules, include_pii, base_url, api_key, cache_dir,
    )


@app.command("eval")
def evaluate(
    dataset: str = typer.Argument("./eval/gold.jsonl", help="Gold Dataset（JSONL）路径"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c", help="配置文件路径"),
    llm: str = typer.Option("none", "--llm", help="LLM 提供商，例如 siliconflow"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="覆盖 API Base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="覆盖 API Key（推荐使用环境变量）"),
    cache_dir: str = typer.Option(".cache/eval", "--cache-dir", help="评测 LLM 缓存目录"),
    report_path: Optional[str] = typer.Option(None, "--report", help="保存 JSON 评测报告"),
    fail_under: float = typer.Option(0.0, "--fail-under", help="Micro F1 低于阈值时返回退出码 2"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
    debug: bool = typer.Option(False, "--debug", help="调试模式"),
):
    """在 Gold Dataset 上评测真实抽取质量，可选 SiliconFlow LLM。"""
    if not 0.0 <= fail_under <= 1.0:
        console.print("[red]错误：--fail-under 必须在 0 到 1 之间[/red]")
        raise typer.Exit(1)

    _configure_logging(verbose, debug)
    config_path = resolve_config_path(config)
    yaml_data = load_yaml(config_path)
    processing = build_processing_config(
        yaml_data,
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
        report = evaluate_dataset(dataset, processing)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]评测失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    _print_evaluation_summary(report)
    if report_path:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"[green]评测报告已保存: {output}[/green]")

    if report.micro.f1 < fail_under:
        console.print(
            f"[red]质量门禁失败：Micro F1 {report.micro.f1:.3f} < {fail_under:.3f}[/red]"
        )
        raise typer.Exit(2)


@app.command()
def providers():
    """列出支持的 LLM 提供商。"""
    table = Table(title="LLM 提供商")
    table.add_column("ID")
    table.add_column("名称")
    table.add_column("默认模型")
    table.add_column("密钥环境变量")
    table.add_column("说明")
    for spec in list_providers():
        table.add_row(
            spec.id,
            spec.name,
            spec.default_model or "-",
            spec.api_key_env or spec.base_url_env or "-",
            spec.notes or spec.base_url,
        )
    console.print(table)
    console.print("任意 OpenAI 兼容服务可用 `--llm openai_compat --base-url ... --api-key ...`")


@app.command()
def info(file_path: str = typer.Argument(..., help="文件路径")):
    """显示文档结构信息。"""
    from .preprocess import MarkdownPreprocessor

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]错误：文件不存在 {file_path}[/red]")
        raise typer.Exit(1)
    parser = DocumentParser()
    parsed = parser.parse(str(path))
    preprocessor = MarkdownPreprocessor()
    structure = preprocessor.extract_structured_content(parsed.content)
    console.print(Panel.fit(
        f"[bold]文件信息[/bold]\n"
        f"文件名：{path.name}\n"
        f"格式：{parsed.original_format}\n"
        f"大小：{path.stat().st_size:,} 字节\n"
        f"总行数：{structure['total_lines']:,}\n"
        f"总章节数：{structure['total_chapters']}",
        title="文档分析",
    ))


@app.command()
def test(
    file_path: str = typer.Argument(..., help="测试文件路径"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c"),
):
    """对单个文件跑一遍流水线冒烟测试。"""
    _run_extract(
        file_path, "./out", config, False, "none", None, "*",
        True, False, True, False, None, None, ".cache",
    )


if __name__ == "__main__":
    app()
