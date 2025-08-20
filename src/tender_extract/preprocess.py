"""
Markdown 预处理和章节树构建
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChapterNode:
    """章节节点"""
    title: str
    level: int
    start_line: int
    end_line: int
    content: str
    children: List['ChapterNode']
    parent: Optional['ChapterNode'] = None


class MarkdownPreprocessor:
    """Markdown预处理器"""
    
    def __init__(self):
        self.md = MarkdownIt()
        self.chapter_tree: Optional[ChapterNode] = None
        
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 标准化标点符号
        text = text.replace('：', ':').replace('，', ',').replace('。', '.')
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff:.,;!?()（）【】""''\n\t]', '', text)
        
        return text.strip()
    
    def parse_markdown(self, content: str) -> SyntaxTreeNode:
        """解析Markdown"""
        try:
            tokens = self.md.parse(content)
            return SyntaxTreeNode(tokens)
        except Exception as e:
            logger.error(f"Markdown解析失败: {e}")
            # 降级处理：按行分割
            return self._fallback_parse(content)
    
    def _fallback_parse(self, content: str) -> SyntaxTreeNode:
        """降级解析：按行分割"""
        lines = content.split('\n')
        tokens = []
        
        for i, line in enumerate(lines):
            # 简单的标题检测
            if re.match(r'^#{1,6}\s+', line):
                level = len(re.match(r'^(#{1,6})', line).group(1))
                tokens.append({
                    'type': 'heading_open',
                    'tag': f'h{level}',
                    'map': [i, i + 1]
                })
                tokens.append({
                    'type': 'inline',
                    'content': line.lstrip('#').strip(),
                    'map': [i, i + 1]
                })
                tokens.append({
                    'type': 'heading_close',
                    'tag': f'h{level}',
                    'map': [i, i + 1]
                })
            else:
                tokens.append({
                    'type': 'paragraph_open',
                    'tag': 'p',
                    'map': [i, i + 1]
                })
                tokens.append({
                    'type': 'inline',
                    'content': line,
                    'map': [i, i + 1]
                })
                tokens.append({
                    'type': 'paragraph_close',
                    'tag': 'p',
                    'map': [i, i + 1]
                })
        
        return SyntaxTreeNode(tokens)
    
    def build_chapter_tree(self, content: str) -> ChapterNode:
        """构建章节树"""
        lines = content.split('\n')
        root = ChapterNode("根节点", 0, 0, len(lines) - 1, content, [])
        current_stack = [root]
        chapter_nodes = []  # 记录所有章节节点
        
        for i, line in enumerate(lines):
            # 检测标题
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # 创建新章节节点
                chapter = ChapterNode(
                    title=title,
                    level=level,
                    start_line=i,
                    end_line=i,
                    content="",
                    children=[],
                    parent=None
                )
                
                # 找到合适的父节点
                while current_stack and current_stack[-1].level >= level:
                    current_stack.pop()
                
                if current_stack:
                    chapter.parent = current_stack[-1]
                    current_stack[-1].children.append(chapter)
                
                current_stack.append(chapter)
                chapter_nodes.append(chapter)
        
        # 设置每个章节的结束行号
        for i, chapter in enumerate(chapter_nodes):
            if i < len(chapter_nodes) - 1:
                # 当前章节的结束行是下一个同级或更高级章节的开始行减1
                next_chapter = chapter_nodes[i + 1]
                if next_chapter.level <= chapter.level:
                    chapter.end_line = next_chapter.start_line - 1
                else:
                    # 下一个章节是子章节，需要找到同级的下一个章节
                    for j in range(i + 1, len(chapter_nodes)):
                        if chapter_nodes[j].level <= chapter.level:
                            chapter.end_line = chapter_nodes[j].start_line - 1
                            break
                    else:
                        # 没有找到同级章节，结束到文档末尾
                        chapter.end_line = len(lines) - 1
            else:
                # 最后一个章节，结束到文档末尾
                chapter.end_line = len(lines) - 1
        
        # 填充章节内容
        self._fill_chapter_content(root, lines)
        
        self.chapter_tree = root
        return root
    
    def _fill_chapter_content(self, node: ChapterNode, lines: List[str]):
        """填充章节内容"""
        if not node.children:
            # 叶子节点，包含从start_line到end_line的内容
            content_lines = lines[node.start_line:node.end_line + 1]
            node.content = '\n'.join(content_lines)
        else:
            # 非叶子节点，内容为子节点之间的内容
            content_parts = []
            last_end = node.start_line
            
            for child in node.children:
                if child.start_line > last_end:
                    content_lines = lines[last_end:child.start_line]
                    content_parts.append('\n'.join(content_lines))
                last_end = child.end_line + 1
                self._fill_chapter_content(child, lines)
            
            # 添加最后一个子节点之后的内容
            if last_end <= node.end_line:
                content_lines = lines[last_end:node.end_line + 1]
                content_parts.append('\n'.join(content_lines))
            
            node.content = '\n'.join(content_parts)
    
    def get_chapter_path(self, line_number: int) -> List[str]:
        """获取指定行号的章节路径"""
        if not self.chapter_tree:
            return []
        
        path = []
        self._find_chapter_path(self.chapter_tree, line_number, path)
        return path
    
    def _find_chapter_path(self, node: ChapterNode, line_number: int, path: List[str]) -> bool:
        """递归查找章节路径"""
        if node.start_line <= line_number <= node.end_line:
            if node.title != "根节点":
                path.append(node.title)
            
            for child in node.children:
                if self._find_chapter_path(child, line_number, path):
                    return True
            
            return True
        return False
    
    def extract_structured_content(self, content: str) -> Dict[str, Any]:
        """提取结构化内容"""
        # 清理文本
        cleaned_content = self.clean_text(content)
        
        # 构建章节树
        chapter_tree = self.build_chapter_tree(cleaned_content)
        
        # 提取关键信息
        structure_info = {
            'total_lines': len(cleaned_content.split('\n')),
            'total_chapters': self._count_chapters(chapter_tree),
            'chapter_hierarchy': self._serialize_chapter_tree(chapter_tree),
            'key_sections': self._extract_key_sections(chapter_tree),
            'content_stats': self._analyze_content_stats(cleaned_content)
        }
        
        return structure_info
    
    def _count_chapters(self, node: ChapterNode) -> int:
        """统计章节数量"""
        count = 1 if node.title != "根节点" else 0
        for child in node.children:
            count += self._count_chapters(child)
        return count
    
    def _serialize_chapter_tree(self, node: ChapterNode) -> Dict[str, Any]:
        """序列化章节树"""
        result = {
            'title': node.title,
            'level': node.level,
            'start_line': node.start_line,
            'end_line': node.end_line,
            'children': [self._serialize_chapter_tree(child) for child in node.children]
        }
        return result
    
    def _extract_key_sections(self, node: ChapterNode) -> List[Dict[str, Any]]:
        """提取关键章节"""
        key_sections = []
        key_keywords = [
            '招标', '投标', '评标', '资格', '技术', '商务', '合同', 
            '保证金', '报价', '时间', '地点', '联系方式'
        ]
        
        def check_key_section(title: str) -> bool:
            return any(keyword in title for keyword in key_keywords)
        
        def traverse(node: ChapterNode):
            if node.title != "根节点" and check_key_section(node.title):
                key_sections.append({
                    'title': node.title,
                    'level': node.level,
                    'start_line': node.start_line,
                    'end_line': node.end_line,
                    'content_preview': node.content[:200] + '...' if len(node.content) > 200 else node.content
                })
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return key_sections
    
    def _analyze_content_stats(self, content: str) -> Dict[str, Any]:
        """分析内容统计"""
        lines = content.split('\n')
        words = re.findall(r'\w+', content)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', content)
        
        return {
            'total_lines': len(lines),
            'total_words': len(words),
            'total_chinese_chars': len(chinese_chars),
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'estimated_tokens': len(words) + len(chinese_chars) // 2  # 粗略估算
        } 