"""
基于LangChain的智能文档切片模块
"""
import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .schema import ChunkInfo
from .preprocess import ChapterNode, MarkdownPreprocessor
import logging

logger = logging.getLogger(__name__)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-text-splitters未安装，将使用基础切片功能")


@dataclass
class ChunkingConfig:
    """切片配置"""
    max_tokens: int = 800
    overlap_tokens: int = 100
    min_chunk_tokens: int = 200
    preserve_chapters: bool = True
    chapter_priority: bool = True
    use_langchain: bool = True


class DocumentChunker:
    """基于LangChain的智能文档切片器"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.preprocessor = MarkdownPreprocessor()
        
        # 初始化LangChain文本分割器
        if LANGCHAIN_AVAILABLE and config.use_langchain:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.max_tokens,
                chunk_overlap=config.overlap_tokens,
                length_function=self._count_tokens,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
            
            # Markdown头部分割器
            self.markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "标题1"),
                    ("##", "标题2"),
                    ("###", "标题3"),
                    ("####", "标题4"),
                    ("#####", "标题5"),
                    ("######", "标题6"),
                ]
            )
        else:
            self.text_splitter = None
            self.markdown_splitter = None
        
    def chunk_document(self, content: str, filename: str) -> List[ChunkInfo]:
        """对文档进行智能切片"""
        # 预处理和构建章节树
        cleaned_content = self.preprocessor.clean_text(content)
        chapter_tree = self.preprocessor.build_chapter_tree(cleaned_content)
        
        chunks = []
        
        if self.config.chapter_priority and self.markdown_splitter:
            # 使用LangChain的Markdown分割器进行章节优先切片
            chunks = self._langchain_chapter_based_chunking(cleaned_content, filename)
        elif self.text_splitter:
            # 使用LangChain的递归字符分割器
            chunks = self._langchain_recursive_chunking(cleaned_content, filename)
        else:
            # 回退到基础切片
            chunks = self._fallback_chunking(chapter_tree, filename)
        
        # 添加指纹
        for chunk in chunks:
            chunk.fingerprint = self._calculate_fingerprint(chunk.content)
        
        logger.info(f"文档 {filename} 切片完成，共生成 {len(chunks)} 个切片")
        return chunks
    
    def _langchain_chapter_based_chunking(self, content: str, filename: str) -> List[ChunkInfo]:
        """使用LangChain的Markdown分割器进行章节优先切片"""
        chunks = []
        chunk_id = 0
        
        try:
            # 首先按Markdown头部分割
            md_splits = self.markdown_splitter.split_text(content)
            
            for split in md_splits:
                # 获取章节信息
                chapter_path = []
                if hasattr(split, 'metadata') and split.metadata:
                    for header_level, header_text in split.metadata.items():
                        if header_text:
                            chapter_path.append(header_text)
                
                # 如果分割后的文本仍然太长，进一步分割
                if self._count_tokens(split.page_content) > self.config.max_tokens:
                    sub_chunks = self.text_splitter.split_text(split.page_content)
                    
                    for sub_chunk in sub_chunks:
                        chunk = ChunkInfo(
                            chunk_id=f"{filename}_{chunk_id:04d}",
                            content=sub_chunk,
                            start_line=self._estimate_start_line(content, sub_chunk),
                            end_line=self._estimate_end_line(content, sub_chunk),
                            chapter_path=chapter_path.copy(),  # 保留章节信息
                            token_count=self._count_tokens(sub_chunk),
                            fingerprint=""
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    # 直接使用分割后的文本
                    chunk = ChunkInfo(
                        chunk_id=f"{filename}_{chunk_id:04d}",
                        content=split.page_content,
                        start_line=self._estimate_start_line(content, split.page_content),
                        end_line=self._estimate_end_line(content, split.page_content),
                        chapter_path=chapter_path.copy(),
                        token_count=self._count_tokens(split.page_content),
                        fingerprint=""
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
        except Exception as e:
            logger.warning(f"LangChain章节分割失败，回退到基础分割: {e}")
            return self._fallback_chunking(self.preprocessor.build_chapter_tree(content), filename)
        
        return chunks
    
    def _langchain_recursive_chunking(self, content: str, filename: str) -> List[ChunkInfo]:
        """使用LangChain的递归字符分割器"""
        chunks = []
        chunk_id = 0
        
        try:
            # 使用递归字符分割器
            splits = self.text_splitter.split_text(content)
            
            for split in splits:
                chunk = ChunkInfo(
                    chunk_id=f"{filename}_{chunk_id:04d}",
                    content=split,
                    start_line=self._estimate_start_line(content, split),
                    end_line=self._estimate_end_line(content, split),
                    chapter_path=[],  # 递归分割不保留章节信息
                    token_count=self._count_tokens(split),
                    fingerprint=""
                )
                chunks.append(chunk)
                chunk_id += 1
                
        except Exception as e:
            logger.warning(f"LangChain递归分割失败，回退到基础分割: {e}")
            return self._fallback_chunking(self.preprocessor.build_chapter_tree(content), filename)
        
        return chunks
    
    def _fallback_chunking(self, chapter_tree: ChapterNode, filename: str) -> List[ChunkInfo]:
        """回退到基础章节切片"""
        chunks = []
        chunk_id = 0
        
        def process_chapter(node: ChapterNode, chapter_path: List[str]):
            nonlocal chunk_id
            
            if not node.content.strip():
                return
            
            # 估算token数量
            estimated_tokens = self._estimate_tokens(node.content)
            
            if estimated_tokens <= self.config.max_tokens:
                # 章节内容可以直接作为一个切片
                chunk = ChunkInfo(
                    chunk_id=f"{filename}_{chunk_id:04d}",
                    content=node.content,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    chapter_path=chapter_path.copy(),
                    token_count=estimated_tokens,
                    fingerprint=""
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 章节内容过长，需要进一步切片
                sub_chunks = self._split_large_chapter(node, chapter_path, filename, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            
            # 递归处理子章节
            for child in node.children:
                new_path = chapter_path + [child.title]
                process_chapter(child, new_path)
        
        # 从根节点开始处理
        process_chapter(chapter_tree, [])
        
        return chunks
    
    def _split_large_chapter(self, node: ChapterNode, chapter_path: List[str], 
                           filename: str, start_chunk_id: int) -> List[ChunkInfo]:
        """分割大章节"""
        chunks = []
        
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', node.content)
        current_chunk = ""
        current_chunk_id = start_chunk_id
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # 检查添加当前段落是否会超过限制
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            if self._estimate_tokens(test_chunk) <= self.config.max_tokens:
                current_chunk = test_chunk
            else:
                # 保存当前chunk
                if current_chunk.strip():
                    chunk = ChunkInfo(
                        chunk_id=f"{filename}_{current_chunk_id:04d}",
                        content=current_chunk.strip(),
                        start_line=node.start_line,
                        end_line=node.end_line,
                        chapter_path=chapter_path.copy(),
                        token_count=self._estimate_tokens(current_chunk),
                        fingerprint=""
                    )
                    chunks.append(chunk)
                    current_chunk_id += 1
                
                # 开始新的chunk
                current_chunk = paragraph
        
        # 保存最后一个chunk
        if current_chunk.strip():
            chunk = ChunkInfo(
                chunk_id=f"{filename}_{current_chunk_id:04d}",
                content=current_chunk.strip(),
                start_line=node.start_line,
                end_line=node.end_line,
                chapter_path=chapter_path.copy(),
                token_count=self._estimate_tokens(current_chunk),
                fingerprint=""
            )
            chunks.append(chunk)
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """估算文本的token数量（简化版本）"""
        # 对于中文，大约每4个字符算1个token
        # 对于英文，大约每4个字符算1个token
        return len(text) // 4
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量（兼容旧版本）"""
        return self._count_tokens(text)
    
    def _estimate_start_line(self, full_content: str, chunk_content: str) -> int:
        """估算chunk在原文中的起始行号"""
        try:
            start_pos = full_content.find(chunk_content)
            if start_pos != -1:
                return full_content[:start_pos].count('\n') + 1
        except:
            pass
        return 1
    
    def _estimate_end_line(self, full_content: str, chunk_content: str) -> int:
        """估算chunk在原文中的结束行号"""
        try:
            start_pos = full_content.find(chunk_content)
            if start_pos != -1:
                end_pos = start_pos + len(chunk_content)
                return full_content[:end_pos].count('\n') + 1
        except:
            pass
        return 1
    
    def _calculate_fingerprint(self, content: str) -> str:
        """计算内容的指纹"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def merge_small_chunks(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """合并过小的切片"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # 检查是否可以合并
            combined_tokens = current_chunk.token_count + next_chunk.token_count
            
            if (combined_tokens <= self.config.max_tokens and 
                current_chunk.chapter_path == next_chunk.chapter_path):
                
                # 合并切片
                current_chunk.content += '\n' + next_chunk.content
                current_chunk.end_line = next_chunk.end_line
                current_chunk.token_count = combined_tokens
                current_chunk.fingerprint = self._calculate_fingerprint(current_chunk.content)
            else:
                # 不能合并，保存当前切片
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # 添加最后一个切片
        merged_chunks.append(current_chunk)
        
        logger.info(f"切片合并完成，从 {len(chunks)} 个合并为 {len(merged_chunks)} 个")
        return merged_chunks
    
    def get_chunk_statistics(self, chunks: List[ChunkInfo]) -> Dict[str, Any]:
        """获取切片统计信息"""
        if not chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in chunks]
        chapter_paths = [len(chunk.chapter_path) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'avg_chapter_depth': sum(chapter_paths) / len(chapter_paths),
            'unique_chapter_paths': len(set(tuple(chunk.chapter_path) for chunk in chunks)),
            'chunks_by_size': {
                'small': len([c for c in chunks if c.token_count < 400]),
                'medium': len([c for c in chunks if 400 <= c.token_count < 800]),
                'large': len([c for c in chunks if c.token_count >= 800])
            }
        }
    
    def find_reference_paragraph(self, full_content: str, start_pos: int, end_pos: int, 
                                max_context: int = 500) -> str:
        """
        根据实体的位置信息找到对应的完整引用段落
        
        Args:
            full_content: 完整文档内容
            start_pos: 实体起始位置
            end_pos: 实体结束位置
            max_context: 最大上下文字符数
            
        Returns:
            包含实体的完整段落
        """
        if start_pos < 0 or end_pos > len(full_content) or start_pos >= end_pos:
            return ""
        
        # 首先验证实体是否在指定位置
        entity_text = full_content[start_pos:end_pos]
        if not entity_text.strip():
            return ""
        
        # 方法1：基于位置信息查找段落
        paragraph = self._find_paragraph_by_position(full_content, start_pos, end_pos, max_context)
        
        # 验证段落是否包含实体
        if entity_text in paragraph:
            return self._clean_paragraph_content(paragraph)
        
        # 方法2：基于内容匹配查找段落（备用方案）
        paragraph = self._find_paragraph_by_content(full_content, entity_text, max_context)
        
        return self._clean_paragraph_content(paragraph)
    
    def _find_paragraph_by_position(self, full_content: str, start_pos: int, end_pos: int, max_context: int) -> str:
        """基于位置信息查找段落"""
        # 找到段落的开始位置
        paragraph_start = start_pos
        while paragraph_start > 0:
            char = full_content[paragraph_start - 1]
            # 如果遇到段落分隔符，停止向前搜索
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
            if start_pos - paragraph_start > max_context:
                break
        
        # 找到段落的结束位置
        paragraph_end = end_pos
        while paragraph_end < len(full_content):
            char = full_content[paragraph_end]
            # 如果遇到段落分隔符，停止向后搜索
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
            if paragraph_end - end_pos > max_context:
                break
        
        return full_content[paragraph_start:paragraph_end].strip()
    
    def _find_paragraph_by_content(self, full_content: str, entity_text: str, max_context: int) -> str:
        """基于内容匹配查找段落"""
        # 在完整文档中查找实体
        pos = full_content.find(entity_text)
        if pos == -1:
            return entity_text  # 如果找不到，返回实体本身
        
        # 从实体位置开始查找段落
        start_pos = pos
        end_pos = pos + len(entity_text)
        
        # 扩展搜索范围
        expanded_start = max(0, start_pos - max_context)
        expanded_end = min(len(full_content), end_pos + max_context)
        
        # 提取扩展后的内容
        expanded_content = full_content[expanded_start:expanded_end]
        
        # 在扩展内容中查找段落边界
        lines = expanded_content.split('\n')
        paragraph_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                paragraph_lines.append(line)
            elif paragraph_lines:  # 遇到空行且已有内容，停止
                break
        
        if paragraph_lines:
            return '\n'.join(paragraph_lines)
        else:
            return entity_text
    
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
        cleaned_paragraph = re.sub(r'\s+', ' ', cleaned_paragraph)
        
        # 如果清理后内容太短，返回原始内容
        if len(cleaned_paragraph.strip()) < 10:
            return paragraph.strip()
        
        return cleaned_paragraph.strip() 