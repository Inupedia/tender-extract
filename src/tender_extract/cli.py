"""
CLI命令行接口模块
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from .schema import ProcessingConfig, ExtractionResult, DocumentMetadata
from .preprocess import MarkdownPreprocessor
from .chunker import DocumentChunker, ChunkingConfig
from .rules import RuleExtractor
from .ner import NERExtractor
from .dedupe import DeduplicationEngine
from .llm_router import LLMRouter
from .merge import FieldMerger

app = typer.Typer(help="面向中文标书的混合抽取流水线")
console = Console()
logger = logging.getLogger(__name__)


class TenderExtractor:
    """标书抽取器主类"""
    
    def __init__(self, config: ProcessingConfig, debug_mode: bool = False):
        self.config = config
        self.debug_mode = debug_mode
        
        # 初始化各个组件
        self.preprocessor = MarkdownPreprocessor()
        self.chunker = DocumentChunker(ChunkingConfig(
            max_tokens=config.max_chunk_tokens,
            overlap_tokens=config.overlap_tokens,
            min_chunk_tokens=200
        ))
        self.rule_extractor = RuleExtractor("config/example.yaml")
        self.ner_extractor = NERExtractor(use_foolnltk=config.use_ner)
        self.dedupe_engine = DeduplicationEngine(
            similarity_threshold=0.8,
            enable_lsh=config.enable_similarity_check
        )
        # 读取配置文件获取Ollama地址
        import yaml
        try:
            with open("config/example.yaml", 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                ollama_base_url = yaml_config.get('llm', {}).get('ollama_base_url', None)
        except Exception as e:
            console.print(f"[yellow]警告：无法读取配置文件，使用默认Ollama地址: {e}[/yellow]")
            ollama_base_url = None
        
        self.llm_router = LLMRouter(
            provider=config.llm_provider,
            model=config.llm_model,
            base_url=ollama_base_url,
            debug_mode=debug_mode
        )
        self.field_merger = FieldMerger()
    
    def extract_document(self, file_path: str) -> ExtractionResult:
        """抽取单个文档"""
        start_time = time.time()
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 获取文件信息
        file_info = Path(file_path)
        file_size = file_info.stat().st_size
        
        # 预处理
        structure_info = self.preprocessor.extract_structured_content(content)
        
        # 切片
        chunks = self.chunker.chunk_document(content, file_info.name)
        
        # 去重
        if self.config.enable_dedupe:
            similarity_results = self.dedupe_engine.process_chunks(chunks)
            chunks = [chunk for i, chunk in enumerate(chunks) 
                     if not similarity_results[i].is_duplicate]
        
        # 混合抽取流水线：先用规则层吃掉确定性字段，只把低置信/冲突的小片段路由给LLM
        all_extractions = []
        llm_calls = 0
        
        for i, chunk in enumerate(chunks):
            # 显示chunk处理进度
            logger.info(f"📄 处理切片 {i+1}/{len(chunks)}: 长度={len(chunk.content)}字符")
            
            # 计算chunk在完整文档中的偏移量
            chunk_offset = self._find_chunk_offset(content, chunk.content, chunk.start_line)
            
            # 1. 规则抽取（高吞吐）
            rule_fields = self.rule_extractor.extract_fields(chunk.content)
            logger.info(f"   📋 规则抽取完成: 找到 {len(rule_fields)} 个字段")
            
            # 2. NER抽取（如果启用）
            if self.config.use_ner:
                ner_fields = self.ner_extractor.extract_entities(chunk.content, content)
                logger.info(f"   🏷️  NER抽取完成: 找到 {len(ner_fields)} 个实体")
                # 调整NER结果的位置信息并添加引用段落
                for field_name, field in ner_fields.items():
                    for evidence in field.values:
                        # 调整位置信息到完整文档中的绝对位置
                        evidence.start += chunk_offset
                        evidence.end += chunk_offset
                        
                        # 添加引用段落 - 基于内容匹配而不是位置
                        evidence.ref = self._find_ref_by_content(content, evidence.value)
                rule_fields = self.ner_extractor.merge_with_rules(ner_fields, rule_fields)
            
            # 3. 调整位置信息并添加引用段落
            for field_name, field in rule_fields.items():
                for evidence in field.values:
                    # 调整位置信息到完整文档中的绝对位置
                    evidence.start += chunk_offset
                    evidence.end += chunk_offset
                    
                    # 添加引用段落 - 基于内容匹配而不是位置
                    evidence.ref = self._find_ref_by_content(content, evidence.value)
            
            # 4. 按需LLM路由（仅处理低置信/冲突字段）
            llm_fields_count = 0
            for field_name, field in rule_fields.items():
                if self.llm_router.should_use_llm(field, self.config.confidence_threshold):
                    llm_fields_count += 1
                    logger.info(f"   🤖 使用LLM处理字段: {field_name} (置信度={field.confidence:.2f})")
                    
                    # 获取最小证据片段
                    minimal_context = self.llm_router.get_minimal_evidence_context(field, chunk.content)
                    
                    from .schema import LLMRequest
                    llm_request = LLMRequest(
                        chunk_text=minimal_context,  # 只发送最小证据片段
                        field_name=field_name,
                        field_type=field.field_type
                    )
                    
                    llm_response = self.llm_router.extract_with_llm(llm_request)
                    if llm_response:
                        field = self.llm_router.merge_llm_results(field, llm_response)
                        llm_calls += 1
                        logger.info(f"   ✅ LLM处理完成: {field_name} -> 新置信度={field.confidence:.2f}")
                    else:
                        logger.warning(f"   ⚠️  LLM处理失败: {field_name}")
            
            if llm_fields_count > 0:
                logger.info(f"   🤖 本切片LLM处理: {llm_fields_count} 个字段")
            
            all_extractions.append(rule_fields)
        
        # 记录LLM调用次数
        self.llm_calls = llm_calls
        logger.info(f"📊 总LLM调用次数: {llm_calls}")
        
        # 合并结果
        merged_fields = self._merge_all_extractions(all_extractions)
        
        # 解决冲突
        resolved_fields = self.field_merger.resolve_conflicts(merged_fields)
        
        # 构建结果
        processing_time = time.time() - start_time
        
        metadata = DocumentMetadata(
            filename=file_info.name,
            file_size=file_size,
            total_lines=structure_info['total_lines'],
            total_chunks=len(chunks),
            processing_time=processing_time,
            extraction_stats=self._get_extraction_stats(resolved_fields)
        )
        
        return ExtractionResult(
            metadata=metadata,
            fields=resolved_fields,
            chunks_processed=len(chunks),
            llm_calls=getattr(self, 'llm_calls', 0),
            cache_hits=self.llm_router.cache_hits,
            errors=[],
            warnings=[]
        )
    
    def _merge_all_extractions(self, extractions: List[dict]) -> dict:
        """合并所有抽取结果"""
        merged_fields = {}
        
        for extraction in extractions:
            for field_name, field in extraction.items():
                if field_name not in merged_fields:
                    merged_fields[field_name] = field
                else:
                    # 合并字段
                    existing_field = merged_fields[field_name]
                    existing_field.values.extend(field.values)
                    
                    # 重新计算置信度
                    if existing_field.values:
                        existing_field.values.sort(key=lambda x: x.confidence, reverse=True)
                        existing_field.primary_value = existing_field.values[0].value
                        existing_field.confidence = max(v.confidence for v in existing_field.values)
        
        return merged_fields
    
    def _get_extraction_stats(self, fields: dict) -> dict:
        """获取抽取统计信息"""
        stats = {
            'total_fields': len(fields),
            'fields_by_type': {},
            'avg_confidence': 0.0
        }
        
        total_confidence = 0.0
        
        for field in fields.values():
            field_type = field.field_type
            if field_type not in stats['fields_by_type']:
                stats['fields_by_type'][field_type] = 0
            stats['fields_by_type'][field_type] += 1
            
            total_confidence += field.confidence
        
        if stats['total_fields'] > 0:
            stats['avg_confidence'] = total_confidence / stats['total_fields']
        
        # 确保所有值都是基本类型
        for key, value in stats['fields_by_type'].items():
            stats['fields_by_type'][key] = int(value)
        
        return stats
    
    def _find_chunk_offset(self, full_content: str, chunk_content: str, start_line: int) -> int:
        """
        计算chunk在完整文档中的字符偏移量
        
        Args:
            full_content: 完整文档内容
            chunk_content: chunk内容
            start_line: chunk的起始行号
            
        Returns:
            字符偏移量
        """
        # 方法1：基于内容匹配（最准确）
        try:
            pos = full_content.find(chunk_content)
            if pos != -1:
                return pos
        except:
            pass
        
        # 方法2：基于行号计算（备用方案）
        if start_line > 1:
            lines = full_content.split('\n')
            if start_line <= len(lines):
                # 计算前面所有行的字符数
                offset = sum(len(line) + 1 for line in lines[:start_line - 1])  # +1 for newline
                return offset
        
        # 如果都失败了，返回0
        return 0
    
    def _find_ref_by_content(self, full_content: str, entity_value: str) -> str:
        """
        基于内容匹配查找引用段落，增强章节信息
        
        Args:
            full_content: 完整文档内容
            entity_value: 实体值
            
        Returns:
            包含实体的引用段落，增强章节信息
        """
        if not entity_value or not entity_value.strip():
            return ""
        
        # 在完整文档中查找实体
        pos = full_content.find(entity_value)
        if pos == -1:
            return entity_value  # 如果找不到，返回实体本身
        
        # 构建章节树以获取章节信息
        from .preprocess import MarkdownPreprocessor
        preprocessor = MarkdownPreprocessor()
        chapter_tree = preprocessor.build_chapter_tree(full_content)
        
        # 获取实体所在位置的章节路径
        entity_line = full_content[:pos].count('\n') + 1
        chapter_path = preprocessor.get_chapter_path(entity_line)
        
        # 从实体位置开始查找段落
        start_pos = pos
        end_pos = pos + len(entity_value)
        
        # 向前查找段落开始
        paragraph_start = start_pos
        while paragraph_start > 0:
            char = full_content[paragraph_start - 1]
            if char in ['\n', '\r']:
                # 检查是否是连续的空行（段落分隔）
                if paragraph_start > 1:
                    prev_char = full_content[paragraph_start - 2]
                    if prev_char in ['\n', '\r']:
                        break
                # 单个换行符，继续向前搜索
                paragraph_start -= 1
            else:
                paragraph_start -= 1
            
            # 限制向前搜索的范围
            if start_pos - paragraph_start > 500:
                break
        
        # 向后查找段落结束
        paragraph_end = end_pos
        while paragraph_end < len(full_content):
            char = full_content[paragraph_end]
            if char in ['\n', '\r']:
                # 检查是否是连续的空行（段落分隔）
                if paragraph_end + 1 < len(full_content):
                    next_char = full_content[paragraph_end + 1]
                    if next_char in ['\n', '\r']:
                        break
                # 单个换行符，继续向后搜索
                paragraph_end += 1
            else:
                paragraph_end += 1
            
            # 限制向后搜索的范围
            if paragraph_end - end_pos > 500:
                break
        
        # 提取段落内容
        paragraph = full_content[paragraph_start:paragraph_end].strip()
        
        # 清理段落内容
        paragraph = self._clean_paragraph_content(paragraph)
        
        # 确保段落包含实体
        if entity_value not in paragraph:
            # 如果清理后不包含实体，返回原始段落
            paragraph = full_content[paragraph_start:paragraph_end].strip()
        
        # 增强ref字段：添加章节信息
        enhanced_ref = self._enhance_ref_with_chapter_info(paragraph, chapter_path, entity_value)
        
        return enhanced_ref
    
    def _enhance_ref_with_chapter_info(self, paragraph: str, chapter_path: List[str], entity_value: str) -> str:
        """
        增强引用段落，添加章节信息
        
        Args:
            paragraph: 原始段落
            chapter_path: 章节路径
            entity_value: 实体值
            
        Returns:
            增强后的引用段落
        """
        if not chapter_path:
            return paragraph
        
        # 构建章节信息
        chapter_info = " → ".join(chapter_path)
        
        # 如果段落太短，直接返回带章节信息的段落
        if len(paragraph.strip()) < 50:
            return f"[章节: {chapter_info}] {paragraph}"
        
        # 尝试找到包含实体的完整句子
        sentences = self._split_into_sentences(paragraph)
        relevant_sentences = []
        
        for sentence in sentences:
            if entity_value in sentence:
                relevant_sentences.append(sentence)
        
        # 如果找到了相关句子，使用它们；否则使用整个段落
        if relevant_sentences:
            content = " ".join(relevant_sentences)
        else:
            content = paragraph
        
        # 构建增强的引用，包含更丰富的章节信息
        enhanced_ref = f"[章节路径: {chapter_info}] {content}"
        
        # 如果章节路径包含关键信息，进一步标注
        key_sections = ['投标函', '投标文件', '招标', '评标', '合同', '保证金', '报价']
        for key_section in key_sections:
            if any(key_section in chapter for chapter in chapter_path):
                enhanced_ref = f"[关键章节: {key_section} | 章节路径: {chapter_info}] {content}"
                break
        
        return enhanced_ref
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        import re
        
        # 中文句子分割模式
        sentence_pattern = r'[。！？；\n]+'
        sentences = re.split(sentence_pattern, text)
        
        # 过滤空句子并清理
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # 过滤太短的句子
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _is_numbered_list_item(self, entity_value: str) -> bool:
        """检查是否是序号列表项"""
        # 检查是否以数字开头，后面跟着空格或点
        import re
        return bool(re.match(r'^\d+[.\s]', entity_value.strip()))
    
    def _clean_paragraph_content(self, paragraph: str) -> str:
        """
        清理段落内容，移除表格等无关内容
        
        Args:
            paragraph: 原始段落内容
            
        Returns:
            清理后的段落内容
        """
        if not paragraph:
            return paragraph
        
        # 如果段落太短，直接返回
        if len(paragraph.strip()) < 10:
            return paragraph.strip()
        
        # 移除表格内容（以|开头的行）
        lines = paragraph.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过表格行
            if line.startswith('|') or line.startswith('---') or line.startswith('+'):
                continue
            # 跳过空行
            if not line:
                continue
            # 跳过只包含特殊字符的行
            if line.replace('-', '').replace('_', '').replace('=', '').replace('*', '').strip() == '':
                continue
            cleaned_lines.append(line)
        
        # 重新组合段落
        cleaned_paragraph = '\n'.join(cleaned_lines)
        
        # 移除过多的空白字符
        import re
        cleaned_paragraph = re.sub(r'\s+', ' ', cleaned_paragraph)
        
        # 如果清理后内容太短，返回原始内容
        if len(cleaned_paragraph.strip()) < 10:
            return paragraph.strip()
        
        return cleaned_paragraph.strip()


@app.command()
def extract(
    input_path: str = typer.Argument(..., help="输入文件或目录路径"),
    out: str = typer.Option("./out", "--out", "-o", help="输出目录"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c", help="配置文件路径"),
    use_ner: bool = typer.Option(False, "--use-ner", help="启用NER"),
    llm: str = typer.Option("none", "--llm", help="LLM提供商：none/ollama/openai"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM模型名称"),
    pattern: str = typer.Option("*.md", "--pattern", "-p", help="文件匹配模式"),
    cache_dir: str = typer.Option(".cache", "--cache-dir", help="缓存目录"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
    debug: bool = typer.Option(False, "--debug", help="LLM调试模式，显示完整提示词和响应")
):
    """抽取文档信息"""
    
    # 设置日志级别
    if verbose:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.print("[yellow]详细模式已启用，将显示LLM处理详情[/yellow]")
    
    if debug:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.print("[red]🔍 LLM调试模式已启用，将显示完整提示词和响应[/red]")
    
    # 检查输入路径
    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"[red]错误：输入路径不存在 {input_path}[/red]")
        sys.exit(1)
    
    # 创建输出目录
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # 构建配置
    processing_config = ProcessingConfig(
        use_ner=use_ner,
        llm_provider=llm,
        llm_model=model,
        confidence_threshold=0.7,
        max_chunk_tokens=800,
        overlap_tokens=100,
        cache_dir=str(cache_path),
        enable_dedupe=True,
        enable_similarity_check=True
    )
    
    # 初始化抽取器
    extractor = TenderExtractor(processing_config, debug_mode=debug)
    
    # 获取文件列表
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob(pattern))
    
    if not files:
        console.print(f"[yellow]警告：未找到匹配的文件 {pattern}[/yellow]")
        return
    
    console.print(f"[green]找到 {len(files)} 个文件[/green]")
    if llm != "none":
        console.print(f"[blue]LLM配置: {llm} - {model or '默认模型'}[/blue]")
    
    # 处理文件
    results = []
    total_llm_calls = 0
    total_cache_hits = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for i, file_path in enumerate(files):
            task = progress.add_task(f"处理 {file_path.name}...", total=None)
            
            try:
                console.print(f"\n[bold cyan]处理文件 {i+1}/{len(files)}: {file_path.name}[/bold cyan]")
                result = extractor.extract_document(str(file_path))
                results.append(result)
                
                # 累计LLM统计
                total_llm_calls += result.llm_calls
                total_cache_hits += result.cache_hits
                
                progress.update(task, description=f"完成 {file_path.name}")
                
                # 显示文件处理结果
                console.print(f"[green]✅ 文件处理完成: {file_path.name}[/green]")
                console.print(f"   📊 抽取字段: {result.metadata.extraction_stats.get('total_fields', 0)} 个")
                console.print(f"   🤖 LLM调用: {result.llm_calls} 次")
                console.print(f"   💾 缓存命中: {result.cache_hits} 次")
                console.print(f"   ⏱️  处理时间: {result.metadata.processing_time:.2f} 秒")
                
            except Exception as e:
                progress.update(task, description=f"失败 {file_path.name}")
                console.print(f"[red]❌ 处理文件失败 {file_path}: {e}[/red]")
    
    # 显示总体统计
    if llm != "none":
        console.print(f"\n[bold]📈 总体LLM统计[/bold]")
        console.print(f"   🤖 总LLM调用: {total_llm_calls} 次")
        console.print(f"   💾 总缓存命中: {total_cache_hits} 次")
        if total_llm_calls > 0:
            cache_rate = total_cache_hits / (total_llm_calls + total_cache_hits) * 100
            console.print(f"   📊 缓存命中率: {cache_rate:.1f}%")
    
    # 保存结果 - 只生成一个JSON文件
    for result in results:
        output_file = out_path / f"{result.metadata.filename}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]💾 结果已保存: {output_file}[/green]")
    
    # 显示统计信息
    _display_results_summary(results)


@app.command()
def info(
    file_path: str = typer.Argument(..., help="文件路径")
):
    """显示文件信息"""
    
    file_path = Path(file_path)
    if not file_path.exists():
        console.print(f"[red]错误：文件不存在 {file_path}[/red]")
        sys.exit(1)
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 预处理
    preprocessor = MarkdownPreprocessor()
    structure_info = preprocessor.extract_structured_content(content)
    
    # 显示信息
    console.print(Panel.fit(
        f"[bold]文件信息[/bold]\n"
        f"文件名：{file_path.name}\n"
        f"大小：{file_path.stat().st_size:,} 字节\n"
        f"总行数：{structure_info['total_lines']:,}\n"
        f"总章节数：{structure_info['total_chapters']}\n"
        f"估算token数：{structure_info['content_stats']['estimated_tokens']:,}",
        title="文档分析"
    ))
    
    # 显示关键章节
    if structure_info['key_sections']:
        table = Table(title="关键章节")
        table.add_column("标题", style="cyan")
        table.add_column("级别", style="magenta")
        table.add_column("行号范围", style="green")
        table.add_column("内容预览", style="yellow")
        
        for section in structure_info['key_sections'][:10]:  # 只显示前10个
            table.add_row(
                section['title'],
                str(section['level']),
                f"{section['start_line']}-{section['end_line']}",
                section['content_preview'][:50] + "..."
            )
        
        console.print(table)


@app.command()
def test(
    file_path: str = typer.Argument(..., help="测试文件路径"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c", help="配置文件路径")
):
    """测试抽取功能"""
    
    file_path = Path(file_path)
    if not file_path.exists():
        console.print(f"[red]错误：文件不存在 {file_path}[/red]")
        sys.exit(1)
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 测试各个组件
    console.print("[bold]测试各个组件...[/bold]")
    
    # 1. 测试预处理
    console.print("\n[cyan]1. 测试预处理[/cyan]")
    preprocessor = MarkdownPreprocessor()
    structure_info = preprocessor.extract_structured_content(content)
    console.print(f"✓ 预处理完成，共 {structure_info['total_chapters']} 个章节")
    
    # 2. 测试切片
    console.print("\n[cyan]2. 测试切片[/cyan]")
    chunker = DocumentChunker(ChunkingConfig())
    chunks = chunker.chunk_document(content, file_path.name)
    console.print(f"✓ 切片完成，共 {len(chunks)} 个切片")
    
    # 3. 测试规则抽取
    console.print("\n[cyan]3. 测试规则抽取[/cyan]")
    rule_extractor = RuleExtractor(config)
    
    # 测试前几个切片
    total_fields = 0
    for i, chunk in enumerate(chunks[:3]):  # 只测试前3个切片
        fields = rule_extractor.extract_fields(chunk.content, content)
        total_fields += len(fields)
        console.print(f"  切片 {i+1}: 抽取到 {len(fields)} 个字段")
    
    console.print(f"✓ 规则抽取完成，前3个切片共抽取 {total_fields} 个字段")
    
    # 4. 测试NER（如果可用）
    console.print("\n[cyan]4. 测试NER[/cyan]")
    ner_extractor = NERExtractor()
    if ner_extractor.use_foolnltk or ner_extractor.use_jieba:
        ner_fields = ner_extractor.extract_entities(content[:1000], content)  # 只测试前1000字符
        console.print(f"✓ NER完成，抽取到 {len(ner_fields)} 个实体类型")
    else:
        console.print("⚠ NER组件不可用")
    
    console.print("\n[green]✓ 所有测试完成！[/green]")


def _display_results_summary(results: List[ExtractionResult]):
    """显示结果摘要"""
    
    if not results:
        return
    
    # 统计信息
    total_files = len(results)
    total_fields = sum(len(r.fields) for r in results)
    total_llm_calls = sum(r.llm_calls for r in results)
    total_cache_hits = sum(r.cache_hits for r in results)
    avg_processing_time = sum(r.metadata.processing_time for r in results) / total_files
    
    # 字段类型统计
    field_types = {}
    for result in results:
        for field in result.fields.values():
            field_type = field.field_type
            if field_type not in field_types:
                field_types[field_type] = 0
            field_types[field_type] += 1
    
    # 显示摘要
    console.print(Panel.fit(
        f"[bold]处理完成！[/bold]\n"
        f"文件数：{total_files}\n"
        f"总字段数：{total_fields}\n"
        f"LLM调用次数：{total_llm_calls}\n"
        f"缓存命中次数：{total_cache_hits}\n"
        f"平均处理时间：{avg_processing_time:.2f}秒",
        title="处理摘要"
    ))
    
    # 显示字段类型分布
    if field_types:
        table = Table(title="字段类型分布")
        table.add_column("字段类型", style="cyan")
        table.add_column("数量", style="magenta")
        
        for field_type, count in sorted(field_types.items(), key=lambda x: x[1], reverse=True):
            table.add_row(field_type, str(count))
        
        console.print(table)


def _generate_summary(extraction_result: ExtractionResult, output_path: str):
    """生成标书关键信息摘要"""
    # 关键字段优先级
    priority_fields = [
        'project_name', 'bidder', 'tenderer', 'bid_amount', 
        'deposit', 'project_manager', 'legal_representative',
        'bid_date', 'project_number', 'contact_info',
        # 围串标检测关键字段
        'shareholder_info', 'subsidiary_info', 'business_license',
        'registered_capital', 'establishment_date', 'registered_address',
        'business_scope', 'qualification_cert', 'performance_record',
        'financial_info', 'bank_account', 'joint_venture',
        'technical_staff', 'equipment_info', 'bidding_consortium'
    ]
    
    field_display_names = {
        'project_name': '项目名称',
        'bidder': '投标人',
        'tenderer': '招标人',
        'bid_amount': '投标金额',
        'deposit': '保证金',
        'project_manager': '项目负责人',
        'legal_representative': '法定代表人',
        'bid_date': '投标日期',
        'project_number': '项目编号',
        'contact_info': '联系方式',
        # 围串标检测关键字段
        'shareholder_info': '股东信息',
        'subsidiary_info': '关联公司',
        'business_license': '营业执照',
        'registered_capital': '注册资本',
        'establishment_date': '成立日期',
        'registered_address': '注册地址',
        'business_scope': '经营范围',
        'qualification_cert': '资质证书',
        'performance_record': '业绩记录',
        'financial_info': '财务信息',
        'bank_account': '银行账户',
        'joint_venture': '联合体',
        'technical_staff': '技术人员',
        'equipment_info': '设备信息',
        'bidding_consortium': '投标联合体',
        'person': '相关人员',
        'company': '公司信息',
        'organization': '组织机构',
        'location': '地点信息',
    }
    
    summary = {
        "文档信息": {
            "文件名": extraction_result.metadata.filename,
            "文件大小": f"{extraction_result.metadata.file_size / 1024:.1f}KB",
            "处理时间": f"{extraction_result.metadata.processing_time:.2f}秒",
            "抽取字段数": extraction_result.metadata.extraction_stats.get('total_fields', 0),
            "平均置信度": f"{extraction_result.metadata.extraction_stats.get('avg_confidence', 0):.2f}"
        },
        "关键信息": {}
    }
    
    # 提取关键字段
    for field_name in priority_fields:
        if field_name in extraction_result.fields:
            field = extraction_result.fields[field_name]
            if field.values:
                best_value = field.values[0]
                summary["关键信息"][field_display_names.get(field_name, field_name)] = {
                    "值": best_value.value,
                    "置信度": f"{best_value.confidence:.2f}",
                    "来源": best_value.source
                }
    
    # 添加其他重要字段
    other_important_fields = ['person', 'company', 'organization', 'location']
    for field_name in other_important_fields:
        if field_name in extraction_result.fields:
            field = extraction_result.fields[field_name]
            if field.values:
                values = field.values[:3]  # 只取前3个值
                summary["关键信息"][field_display_names.get(field_name, field_name)] = {
                    "值": [v.value for v in values],
                    "置信度": f"{field.confidence:.2f}",
                    "来源": values[0].source
                }
    
    # 保存完整摘要
    summary_file = output_path.replace('.json', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 保存简洁摘要（纯文本）
    text_file = output_path.replace('.json', '_summary.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("标书关键信息摘要\n")
        f.write("=" * 50 + "\n\n")
        
        # 文档信息
        f.write("【文档信息】\n")
        doc_info = summary["文档信息"]
        for key, value in doc_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # 关键信息
        f.write("【关键信息】\n")
        key_info = summary["关键信息"]
        for key, info in key_info.items():
            f.write(f"{key}: {info['值']}\n")
            f.write(f"  置信度: {info['置信度']}, 来源: {info['来源']}\n")
            f.write("\n")
    
    return summary_file, text_file


@app.command()
def extract_v2(
    input_path: str = typer.Argument(..., help="输入文件或目录路径（支持 PDF/DOCX/MD/TXT）"),
    out: str = typer.Option("./out", "--out", "-o", help="输出目录"),
    config: str = typer.Option("./config/example.yaml", "--config", "-c", help="配置文件路径"),
    use_ner: bool = typer.Option(False, "--use-ner", help="启用NER"),
    llm: str = typer.Option("none", "--llm", help="LLM提供商：none/ollama/openai"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM模型名称"),
    pattern: str = typer.Option("*", "--pattern", "-p", help="文件匹配模式（默认所有支持格式）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
    debug: bool = typer.Option(False, "--debug", help="调试模式"),
    use_modules: bool = typer.Option(True, "--modules/--no-modules", help="启用模块化路由"),
):
    """
    增强版抽取（v2）：支持 PDF/DOCX 输入 + 模块化路由 + 增强正则

    核心改进：
    1. 解析层：自动检测文件格式，PDF/DOCX/TXT/MD 统一转 Markdown
    2. 路由层：按章节内容路由到专业模块（基础信息/财务/资质/评标等）
    3. 抽取层：增强正则模式库 + 分层置信度 + 多方法竞争
    4. 仅低置信片段路由 LLM
    """
    from .document_parser import DocumentParser
    from .module_router import ModuleRouter
    from .extraction_engine import ExtractionEngine

    # 设置日志
    if verbose or debug:
        import logging as _logging
        _logging.basicConfig(
            level=_logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # 初始化组件
    parser = DocumentParser()
    router = ModuleRouter()
    engine = ExtractionEngine()

    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        console.print(f"[red]错误：输入路径不存在 {input_path}[/red]")
        sys.exit(1)

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    # 获取文件列表
    supported_exts = parser.get_supported_extensions()
    if input_path_obj.is_file():
        files = [input_path_obj]
    else:
        if pattern == "*":
            files = [
                f for f in input_path_obj.iterdir()
                if f.suffix.lower() in supported_exts
            ]
        else:
            files = list(input_path_obj.glob(pattern))

    if not files:
        console.print(f"[yellow]未找到支持的文件。支持格式: {supported_exts}[/yellow]")
        return

    console.print(f"[green]找到 {len(files)} 个文件[/green]")
    console.print(f"[blue]支持格式: {', '.join(sorted(supported_exts))}[/blue]")
    if use_modules:
        console.print("[blue]模块化路由: 已启用[/blue]")

    results = []
    total_start = time.time()

    for i, file_path in enumerate(files):
        console.print(f"\n[bold cyan]═══ 处理文件 {i+1}/{len(files)}: {file_path.name} ═══[/bold cyan]")
        file_start = time.time()

        try:
            # ---- 步骤1: 文档解析 ----
            console.print("[dim]  ① 解析文档...[/dim]")
            parsed_doc = parser.parse(str(file_path))
            console.print(
                f"  [green]✓[/green] 解析完成: {parsed_doc.original_format} → Markdown "
                f"({len(parsed_doc.content)} 字符"
                f"{f', {parsed_doc.total_pages} 页' if parsed_doc.total_pages else ''})"
            )

            # ---- 步骤2: 切块 ----
            console.print("[dim]  ② 智能切块...[/dim]")
            chunker = DocumentChunker(ChunkingConfig(
                max_tokens=800,
                overlap_tokens=100,
                min_chunk_tokens=200
            ))
            chunks = chunker.chunk_document(parsed_doc.content, parsed_doc.filename)
            console.print(f"  [green]✓[/green] 切块完成: {len(chunks)} 个切片")

            # ---- 步骤3: 模块路由 ----
            if use_modules:
                console.print("[dim]  ③ 模块路由...[/dim]")
                routed_chunks = router.route_chunks(chunks)
                routing_summary = router.get_routing_summary(routed_chunks)
                for module_name, count in routing_summary['module_distribution'].items():
                    console.print(f"     [{module_name}]: {count} 个切片")
            else:
                routed_chunks = None

            # ---- 步骤4: 增强抽取 ----
            console.print("[dim]  ④ 增强正则抽取...[/dim]")
            all_fields = {}

            # 对所有 chunk 运行增强正则抽取引擎
            for chunk in chunks:
                chunk_fields = engine.extract_all_fields(chunk.content)
                for fname, ffield in chunk_fields.items():
                    if fname not in all_fields:
                        all_fields[fname] = ffield
                    else:
                        # 合并值，保留最高置信度
                        all_fields[fname].values.extend(ffield.values)
                        if ffield.confidence > all_fields[fname].confidence:
                            all_fields[fname].confidence = ffield.confidence
                            all_fields[fname].primary_value = ffield.primary_value

            # ---- 步骤5: NER 补充（可选）----
            if use_ner:
                console.print("[dim]  ⑤ NER抽取...[/dim]")
                ner_extractor = NERExtractor(use_foolnltk=True)
                ner_fields = ner_extractor.extract_entities(
                    parsed_doc.content[:5000], parsed_doc.content
                )
                for fname, ffield in ner_fields.items():
                    if fname not in all_fields:
                        all_fields[fname] = ffield
                    else:
                        all_fields[fname].values.extend(ffield.values)
                console.print(f"  [green]✓[/green] NER补充: {len(ner_fields)} 种实体")

            # ---- 步骤5b: 人员信息专项抽取 ----
            console.print("[dim]  ⑤b 人员信息抽取...[/dim]")
            from .personnel_extractor import PersonnelExtractor
            pe = PersonnelExtractor()
            personnel = pe.extract_personnel(parsed_doc.content)
            certificates = pe.extract_certificates(parsed_doc.content)
            if personnel:
                console.print(f"  [green]✓[/green] 人员信息: {len(personnel)} 人")
                for p in personnel[:5]:  # 显示前5个
                    info_parts = [p.name]
                    if p.role:
                        info_parts.append(f"({p.role})")
                    if p.id_card:
                        info_parts.append(f"身份证:{p.id_card[:6]}****{p.id_card[-4:]}")
                    console.print(f"     {''.join(info_parts)}")
            if certificates:
                console.print(f"  [green]✓[/green] 证书信息: {len(certificates)} 个")
                for c in certificates[:3]:
                    console.print(f"     [{c.cert_type}] {c.cert_number}")

            # ---- 步骤6: LLM 路由（仅低置信） ----
            low_conf_fields = engine.get_low_confidence_fields(all_fields)
            llm_calls = 0
            if llm != "none" and low_conf_fields:
                console.print(
                    f"[dim]  ⑥ LLM路由 ({len(low_conf_fields)} 个低置信字段)...[/dim]"
                )
                # 此处复用原有 LLM 路由逻辑
                import yaml
                try:
                    with open(config, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f)
                        ollama_base_url = yaml_config.get('llm', {}).get('ollama_base_url', None)
                except Exception:
                    ollama_base_url = None

                llm_router = LLMRouter(
                    provider=llm, model=model,
                    base_url=ollama_base_url, debug_mode=debug
                )
                for fname in low_conf_fields:
                    if fname in all_fields:
                        field = all_fields[fname]
                        if llm_router.should_use_llm(field, 0.7):
                            from .schema import LLMRequest
                            context = field.values[0].ref if field.values and field.values[0].ref else ""
                            llm_req = LLMRequest(
                                chunk_text=context or parsed_doc.content[:500],
                                field_name=fname,
                                field_type=field.field_type
                            )
                            llm_resp = llm_router.extract_with_llm(llm_req)
                            if llm_resp:
                                all_fields[fname] = llm_router.merge_llm_results(field, llm_resp)
                                llm_calls += 1
                if llm_calls:
                    console.print(f"  [green]✓[/green] LLM处理: {llm_calls} 次调用")

            # ---- 构建结果 ----
            processing_time = time.time() - file_start
            file_info = Path(file_path)

            metadata = DocumentMetadata(
                filename=file_info.name,
                file_size=file_info.stat().st_size,
                total_lines=len(parsed_doc.content.split('\n')),
                total_chunks=len(chunks),
                processing_time=processing_time,
                extraction_stats={
                    'total_fields': len(all_fields),
                    'fields_by_type': {f: 1 for f in all_fields.keys()},
                    'avg_confidence': (
                        sum(f.confidence for f in all_fields.values()) / len(all_fields)
                        if all_fields else 0
                    ),
                    'low_confidence_count': len(low_conf_fields),
                    'modules_used': (
                        list(routing_summary['module_distribution'].keys())
                        if use_modules else []
                    ),
                    'personnel_count': len(personnel) if personnel else 0,
                    'certificates_count': len(certificates) if certificates else 0,
                }
            )

            result = ExtractionResult(
                metadata=metadata,
                fields=all_fields,
                chunks_processed=len(chunks),
                llm_calls=llm_calls,
                cache_hits=0,
                errors=[],
                warnings=[]
            )
            results.append(result)

            # 显示摘要
            console.print(f"\n  [bold green]✅ 完成[/bold green]: "
                         f"{len(all_fields)} 个字段, "
                         f"平均置信度 {metadata.extraction_stats['avg_confidence']:.2f}, "
                         f"耗时 {processing_time:.2f}s")

            # 显示提取的关键字段
            priority_display = [
                'project_name', 'tenderer', 'bidder', 'bid_amount', 'deposit',
                'legal_representative', 'bid_date', 'project_number', 'contact_info'
            ]
            for fname in priority_display:
                if fname in all_fields:
                    f = all_fields[fname]
                    conf_color = "green" if f.confidence >= 0.9 else (
                        "yellow" if f.confidence >= 0.7 else "red"
                    )
                    console.print(
                        f"     {fname}: [{conf_color}]{f.primary_value}[/{conf_color}] "
                        f"(置信度={f.confidence:.2f})"
                    )

        except Exception as e:
            console.print(f"[red]❌ 处理失败 {file_path}: {e}[/red]")
            if debug:
                import traceback
                traceback.print_exc()

    # 保存结果
    for result in results:
        output_file = out_path / f"{result.metadata.filename}.json"
        # 构建完整输出（包含人员信息）
        output_data = result.dict()
        # 人员信息单独保存（如果有）
        if personnel:
            output_data['personnel'] = [
                {
                    'name': p.name,
                    'role': p.role,
                    'id_card': p.id_card,
                    'education': p.education,
                    'major': p.major,
                    'graduation_school': p.graduation_school,
                    'certificates': p.certificates,
                    'confidence': p.confidence,
                } for p in personnel
            ]
        if certificates:
            output_data['certificates'] = [
                {
                    'cert_type': c.cert_type,
                    'cert_number': c.cert_number,
                    'holder_name': c.holder_name,
                    'level': c.level,
                    'major': c.major,
                } for c in certificates
            ]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]💾 结果已保存: {output_file}[/green]")

    # 总体统计
    total_time = time.time() - total_start
    total_fields = sum(len(r.fields) for r in results)
    console.print(Panel.fit(
        f"[bold]处理完成！[/bold]\n"
        f"文件数：{len(results)}\n"
        f"总字段数：{total_fields}\n"
        f"总耗时：{total_time:.2f}秒\n"
        f"平均每文件：{total_time/len(results):.2f}秒" if results else "无结果",
        title="v2 处理摘要"
    ))


if __name__ == "__main__":
    app() 