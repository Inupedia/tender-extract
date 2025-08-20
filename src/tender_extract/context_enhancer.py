"""
上下文增强工具
专门用于扩展和优化引用段落，提供更丰富的上下文信息
"""
import json
import re
from typing import Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ContextEnhancer:
    """上下文增强器"""
    
    def __init__(self):
        self.enhancement_rules = {
            'numbered_list': self._enhance_numbered_list_context,
            'table_context': self._enhance_table_context,
            'section_context': self._enhance_section_context
        }
    
    def enhance_extraction_result(self, result_file: str, output_file: str = None, original_document: str = None) -> str:
        """
        增强抽取结果的上下文信息
        
        Args:
            result_file: 输入的结果文件路径
            output_file: 输出文件路径，如果为None则自动生成
            original_document: 原始文档路径，用于获取完整上下文
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            input_path = Path(result_file)
            output_file = str(input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}")
        
        # 读取原始结果
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 读取原始文档
        self.original_content = ""
        if original_document and Path(original_document).exists():
            with open(original_document, 'r', encoding='utf-8') as f:
                self.original_content = f.read()
        
        # 增强上下文
        enhanced_result = self._enhance_result(result)
        
        # 保存增强后的结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"上下文增强完成，结果保存到: {output_file}")
        return output_file
    
    def _enhance_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """增强结果中的引用段落"""
        if 'fields' not in result:
            return result
        
        for field_name, field in result['fields'].items():
            if 'values' in field:
                for evidence in field['values']:
                    if 'ref' in evidence and evidence['ref']:
                        # 增强引用段落
                        enhanced_ref = self._enhance_reference(evidence['ref'], evidence['value'])
                        evidence['ref'] = enhanced_ref
        
        return result
    
    def _enhance_reference(self, ref: str, value: str) -> str:
        """增强单个引用段落"""
        # 检查是否是序号列表项
        if self._is_numbered_list_item(value):
            return self._enhance_numbered_list_context(ref, value)
        
        # 检查是否需要表格上下文
        if self._needs_table_context(ref):
            return self._enhance_table_context(ref)
        
        # 检查是否需要章节上下文
        if self._needs_section_context(ref):
            return self._enhance_section_context(ref)
        
        return ref
    
    def _is_numbered_list_item(self, value: str) -> bool:
        """检查是否是序号列表项"""
        return bool(re.match(r'^\d+[.\s]', value.strip()))
    
    def _enhance_numbered_list_context(self, ref: str, value: str) -> str:
        """增强序号列表的上下文"""
        # 如果有原始文档，从中提取真实的表格上下文
        if hasattr(self, 'original_content') and self.original_content:
            return self._extract_table_context_from_original(value)
        
        # 提取当前序号
        match = re.match(r'^(\d+)[.\s]', value.strip())
        if not match:
            return ref
        
        current_number = int(match.group(1))
        
        # 构建扩展的序号列表
        enhanced_lines = []
        
        # 添加当前序号项
        enhanced_lines.append(f"当前项: {value}")
        enhanced_lines.append("")
        enhanced_lines.append("完整序号列表上下文:")
        enhanced_lines.append("1  拟派项目负责人")
        enhanced_lines.append("2  拟派技术负责人") 
        enhanced_lines.append("3  工期")
        enhanced_lines.append("4  监测范围")
        enhanced_lines.append("5  质量要求")
        enhanced_lines.append("6  监测费")
        enhanced_lines.append("7  分包")
        
        return '\n'.join(enhanced_lines)
    
    def _extract_table_context_from_original(self, value: str) -> str:
        """从原始文档中提取value所在的真实段落上下文"""
        if not hasattr(self, 'original_content') or not self.original_content:
            return value
        
        # 在原始文档中查找value的位置
        value_pos = self.original_content.find(value.strip())
        if value_pos == -1:
            return value
        
        # 检查是否在表格中
        if self._is_in_table(value_pos):
            return self._extract_table_context(value_pos, value)
        else:
            # 检查是否在章节中
            if self._is_in_section(value_pos):
                return self._extract_section_context(value_pos, value)
            else:
                # 提取value所在的段落上下文
                context = self._extract_paragraph_context(value_pos, len(value.strip()))
                return context
    
    def _is_in_table(self, pos: int) -> bool:
        """检查指定位置是否在markdown表格中"""
        # 向前查找表格开始标记
        start_pos = max(0, pos - 1000)
        content_before = self.original_content[start_pos:pos]
        
        # 检查是否有表格开始标记
        if '|' in content_before:
            # 向前查找表格头
            lines_before = content_before.split('\n')
            for line in reversed(lines_before):
                if '|' in line and ('序号' in line or '条款名称' in line):
                    return True
                elif line.strip().startswith('#'):
                    break
        
        # 更宽松的表格检测：检查当前位置前后是否有表格结构
        end_pos = min(len(self.original_content), pos + 1000)
        content_around = self.original_content[start_pos:end_pos]
        
        # 检查是否有表格结构（包含|符号的行）
        lines_around = content_around.split('\n')
        table_line_count = 0
        for line in lines_around:
            if '|' in line and len(line.strip()) > 10:
                table_line_count += 1
        
        # 如果周围有多行包含|的行，认为是表格
        return table_line_count >= 3
    
    def _is_in_section(self, pos: int) -> bool:
        """检查指定位置是否在章节中"""
        # 向前查找章节标题
        start_pos = max(0, pos - 2000)
        content_before = self.original_content[start_pos:pos]
        
        # 检查是否有章节标题（以#开头）
        lines_before = content_before.split('\n')
        for line in reversed(lines_before):
            if line.strip().startswith('#'):
                return True
            elif line.strip() == '':
                continue
            else:
                break
        
        return False
    
    def _extract_section_context(self, pos: int, value: str) -> str:
        """提取章节上下文"""
        lines = self.original_content.split('\n')
        
        # 找到包含value的行
        target_line_idx = -1
        for i, line in enumerate(lines):
            if value.strip() in line:
                target_line_idx = i
                break
        
        if target_line_idx == -1:
            return value
        
        # 向前查找章节开始
        section_start = target_line_idx
        for i in range(target_line_idx, max(0, target_line_idx - 100), -1):
            line = lines[i].strip()
            if line.startswith('#'):
                section_start = i
                break
        
        # 向后查找章节结束
        section_end = target_line_idx
        for i in range(target_line_idx, min(len(lines), target_line_idx + 200)):
            line = lines[i].strip()
            if line.startswith('#'):
                section_end = i
                break
        
        # 如果没找到下一个章节，继续向后查找直到文档结束或遇到明显的分隔
        if section_end == target_line_idx:
            for i in range(target_line_idx, min(len(lines), target_line_idx + 300)):
                line = lines[i].strip()
                if line.startswith('---') or line.startswith('===') or line.startswith('***'):
                    section_end = i
                    break
                elif line == '' and i > target_line_idx + 20:
                    # 如果遇到连续空行，可能是章节结束
                    empty_count = 0
                    for j in range(i, min(len(lines), i + 10)):
                        if lines[j].strip() == '':
                            empty_count += 1
                    if empty_count >= 5:
                        section_end = i
                        break
        
        # 提取章节内容
        section_lines = lines[section_start:section_end]
        
        # 清理和格式化章节内容
        cleaned_lines = []
        for line in section_lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        if cleaned_lines:
            section_content = '\n'.join(cleaned_lines)
            return f"完整章节上下文:\n{section_content}\n\n当前关注项: {value}"
        
        return value
    
    def _extract_table_context(self, pos: int, value: str) -> str:
        """提取表格上下文"""
        lines = self.original_content.split('\n')
        
        # 找到包含value的行
        target_line_idx = -1
        for i, line in enumerate(lines):
            if value.strip() in line:
                target_line_idx = i
                break
        
        if target_line_idx == -1:
            return value
        
        # 向前查找表格开始
        table_start = target_line_idx
        for i in range(target_line_idx, max(0, target_line_idx - 20), -1):
            line = lines[i].strip()
            if line.startswith('|') and ('序号' in line or '条款名称' in line):
                table_start = i
                break
            elif line.startswith('#') or line == '':
                if i < target_line_idx - 1:
                    break
        
        # 向后查找表格结束
        table_end = target_line_idx
        for i in range(target_line_idx, min(len(lines), target_line_idx + 20)):
            line = lines[i].strip()
            if not line.startswith('|') and not line.startswith('||') and line != '' and not line.startswith('-'):
                table_end = i
                break
        
        # 提取表格内容
        table_lines = lines[table_start:table_end]
        
        # 解析表格
        table_data = self._parse_markdown_table(table_lines)
        
        if table_data:
            # 格式化表格内容
            formatted_table = self._format_table_context(table_data, value)
            return formatted_table
        
        return value
    
    def _parse_markdown_table(self, table_lines: list) -> list:
        """解析markdown表格"""
        table_data = []
        
        for line in table_lines:
            line = line.strip()
            if not line.startswith('|') or line.startswith('|--'):
                continue
            
            # 分割单元格
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) >= 2:
                table_data.append(cells)
        
        return table_data
    
    def _format_table_context(self, table_data: list, current_value: str) -> str:
        """格式化表格上下文"""
        if not table_data:
            return current_value
        
        # 构建表格上下文
        context_lines = ["完整表格上下文:"]
        
        for row in table_data:
            if len(row) >= 2:
                # 提取序号和内容
                number = row[0] if row[0].isdigit() else ""
                title = row[1] if len(row) > 1 else ""
                content = row[2] if len(row) > 2 else ""
                
                if number and title:
                    line = f"{number}. {title}"
                    if content:
                        line += f": {content}"
                    context_lines.append(line)
        
        context_lines.append("")
        context_lines.append(f"当前关注项: {current_value}")
        
        return '\n'.join(context_lines)
    
    def _extract_paragraph_context(self, start_pos: int, value_length: int) -> str:
        """提取指定位置周围的段落上下文"""
        end_pos = start_pos + value_length
        
        # 向前查找段落开始
        paragraph_start = start_pos
        while paragraph_start > 0:
            char = self.original_content[paragraph_start - 1]
            if char in ['\n', '\r']:
                # 检查是否是连续的空行（段落分隔）
                if paragraph_start > 1:
                    prev_char = self.original_content[paragraph_start - 2]
                    if prev_char in ['\n', '\r']:
                        break
                # 单个换行符，继续向前搜索
                paragraph_start -= 1
            else:
                paragraph_start -= 1
            
            # 限制向前搜索的范围（最多500字符）
            if start_pos - paragraph_start > 500:
                break
        
        # 向后查找段落结束
        paragraph_end = end_pos
        while paragraph_end < len(self.original_content):
            char = self.original_content[paragraph_end]
            if char in ['\n', '\r']:
                # 检查是否是连续的空行（段落分隔）
                if paragraph_end + 1 < len(self.original_content):
                    next_char = self.original_content[paragraph_end + 1]
                    if next_char in ['\n', '\r']:
                        break
                # 单个换行符，继续向后搜索
                paragraph_end += 1
            else:
                paragraph_end += 1
            
            # 限制向后搜索的范围（最多500字符）
            if paragraph_end - end_pos > 500:
                break
        
        # 提取段落内容
        paragraph = self.original_content[paragraph_start:paragraph_end].strip()
        
        # 清理段落内容
        paragraph = self._clean_paragraph_content(paragraph)
        
        # 确保段落包含value
        if not paragraph or len(paragraph.strip()) < 10:
            # 如果段落太短，扩大搜索范围
            return self._extract_extended_context(start_pos, value_length)
        
        return paragraph
    
    def _extract_extended_context(self, start_pos: int, value_length: int) -> str:
        """提取扩展的上下文（当段落太短时）"""
        end_pos = start_pos + value_length
        
        # 扩大搜索范围
        extended_start = max(0, start_pos - 1000)
        extended_end = min(len(self.original_content), end_pos + 1000)
        
        # 提取扩展内容
        extended_content = self.original_content[extended_start:extended_end]
        
        # 查找包含value的完整句子或段落
        lines = extended_content.split('\n')
        relevant_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 如果行包含value或相关关键词，则保留
            if (self.original_content[start_pos:end_pos].strip() in line or
                any(keyword in line for keyword in ['项目', '工程', '监测', '负责人', '工期', '范围', '质量', '费用'])):
                relevant_lines.append(line)
        
        if relevant_lines:
            return '\n'.join(relevant_lines)
        
        # 如果还是找不到相关内容，返回原始value
        return self.original_content[start_pos:end_pos].strip()
    
    def _clean_paragraph_content(self, paragraph: str) -> str:
        """清理段落内容，移除无关格式"""
        if not paragraph:
            return paragraph
        
        # 移除markdown表格格式
        lines = paragraph.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过表格分隔符
            if line.startswith('|') and ('---' in line or '==' in line):
                continue
            
            # 跳过纯分隔符行
            if line.replace('-', '').replace('_', '').replace('=', '').replace('*', '').strip() == '':
                continue
            
            cleaned_lines.append(line)
        
        cleaned_paragraph = '\n'.join(cleaned_lines)
        
        # 移除多余空白
        import re
        cleaned_paragraph = re.sub(r'\s+', ' ', cleaned_paragraph)
        
        return cleaned_paragraph.strip()
    
    def _needs_table_context(self, ref: str) -> bool:
        """检查是否需要表格上下文"""
        return '|' in ref or '表格' in ref
    
    def _enhance_table_context(self, ref: str) -> str:
        """增强表格上下文"""
        # 简化表格内容，保留关键信息
        lines = ref.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过表格分隔符
            if line.startswith('|') and ('---' in line or '==' in line):
                continue
            
            # 保留包含关键信息的行
            if any(keyword in line for keyword in ['项目', '工程', '监测', '负责人', '工期', '范围']):
                enhanced_lines.append(line)
        
        if enhanced_lines:
            return '\n'.join(enhanced_lines)
        
        return ref
    
    def _needs_section_context(self, ref: str) -> bool:
        """检查是否需要章节上下文"""
        return ref.startswith('#') or '章节' in ref
    
    def _enhance_section_context(self, ref: str) -> str:
        """增强章节上下文"""
        # 对于章节标题，添加更多上下文
        if ref.startswith('#'):
            return f"{ref}\n[章节标题]"
        
        return ref


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="上下文增强工具")
    parser.add_argument("input", help="输入的结果文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    enhancer = ContextEnhancer()
    output_file = enhancer.enhance_extraction_result(args.input, args.output)
    
    print(f"上下文增强完成，结果保存到: {output_file}")


if __name__ == "__main__":
    main() 