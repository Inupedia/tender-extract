"""
正则表达式和关键词启发式抽取模块
"""
import re
import yaml
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import ahocorasick
    PYAHOCORASICK_AVAILABLE = True
except ImportError:
    PYAHOCORASICK_AVAILABLE = False
    logger.warning("pyahocorasick未安装，关键词匹配功能将不可用")

from .schema import EvidenceSpan, ExtractedField


@dataclass
class PatternMatch:
    """模式匹配结果"""
    value: str
    start: int
    end: int
    pattern: str
    field_type: str
    confidence: float
    context: str = ""  # 添加上下文信息，用于LLM路由判断


class RuleExtractor:
    """规则抽取器"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.patterns = self._compile_patterns()
        self.keyword_trie = self._build_keyword_trie()
        
        # 初始化智能抽取模式
        self.smart_patterns = self._init_smart_patterns()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, str, float]]]:
        """编译正则表达式模式"""
        patterns = {}
        
        if 'patterns' not in self.config:
            return patterns
        
        for field_type, pattern_list in self.config['patterns'].items():
            patterns[field_type] = []
            
            for pattern_info in pattern_list:
                if isinstance(pattern_info, dict):
                    pattern_str = pattern_info.get('pattern', '')
                    description = pattern_info.get('description', '')
                    confidence = pattern_info.get('confidence', 0.9)
                else:
                    pattern_str = pattern_info
                    description = ''
                    confidence = 0.9
                
                try:
                    compiled_pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                    patterns[field_type].append((compiled_pattern, description, confidence))
                except re.error as e:
                    logger.warning(f"正则表达式编译失败: {pattern_str}, 错误: {e}")
        
        return patterns
    
    def _build_keyword_trie(self):
        """构建关键词Trie树"""
        if not PYAHOCORASICK_AVAILABLE:
            return None
        
        trie = ahocorasick.Automaton()
        
        if 'synonyms' not in self.config:
            return trie
        
        for field_type, keywords in self.config['synonyms'].items():
            for keyword in keywords:
                trie.add_word(keyword, (field_type, keyword))
        
        trie.make_automaton()
        return trie
    
    def _init_smart_patterns(self) -> Dict[str, List[re.Pattern]]:
        """初始化智能抽取模式"""
        smart_patterns = {
            'project_name': [
                re.compile(r'项目名称[：:]\s*([^，。\n]{5,100})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'工程名称[：:]\s*([^，。\n]{5,100})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'([^，。\n]{5,50}工程[^，。\n]{0,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'([^，。\n]{5,50}项目[^，。\n]{0,50})', re.IGNORECASE | re.MULTILINE),
            ],
            'bidder': [
                re.compile(r'投标人[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'投标单位[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'([^，。\n]{5,50}公司[^，。\n]{0,30})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'([^，。\n]{5,50}集团[^，。\n]{0,30})', re.IGNORECASE | re.MULTILINE),
            ],
            'tenderer': [
                re.compile(r'招标人[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'招标单位[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
            ],
            'bid_amount': [
                # 精确的金额匹配
                re.compile(r'投标报价[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'投标金额[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'愿意以人民币[（(]大写[)）]\s*([^，。\n]{5,50})[^，。\n]*的投标总报价', re.IGNORECASE | re.MULTILINE),
                re.compile(r'投标总报价[^，。\n]*人民币[（(]大写[)）]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                # 监测费匹配
                re.compile(r'监测费[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\|\s*\d+\s*\|\s*监测费\s*\|\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
                # 更精确的数字金额
                re.compile(r'(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
            ],
            'deposit': [
                re.compile(r'投标保证金[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'履约保证金[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'保证金[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
            ],
            'contact_info': [
                # 更精确的电话号码匹配
                re.compile(r'电话[：:]\s*(\d{3,4}[-\s]?\d{7,8})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'手机[：:]\s*(\d{11})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'联系电话[：:]\s*(\d{3,4}[-\s]?\d{7,8})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'邮箱[：:]\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'地址[：:]\s*([^，。\n]{10,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'邮编[：:]\s*(\d{6})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'网址[：:]\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'传真[：:]\s*(\d{3,4}[-\s]?\d{7,8})', re.IGNORECASE | re.MULTILINE),
            ],
            'project_manager': [
                re.compile(r'项目负责人[：:]\s*([^，。\n]{2,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'拟派项目负责人[：:]\s*([^，。\n]{2,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'项目经理[：:]\s*([^，。\n]{2,20})', re.IGNORECASE | re.MULTILINE),
            ],
            'legal_representative': [
                # 更精确的法定代表人匹配
                re.compile(r'法定代表人[：:]\s*([^，。\n]{2,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'法人代表[：:]\s*([^，。\n]{2,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'法人[：:]\s*([^，。\n]{2,20})', re.IGNORECASE | re.MULTILINE),
            ],
            'bid_date': [
                re.compile(r'投标日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'投标时间[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'(\d{4}年\d{1,2}月\d{1,2}日)', re.IGNORECASE | re.MULTILINE),
            ],
            'project_number': [
                re.compile(r'项目编号[：:]\s*([A-Z0-9\-_]{5,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'招标编号[：:]\s*([A-Z0-9\-_]{5,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'合同编号[：:]\s*([A-Z0-9\-_]{5,20})', re.IGNORECASE | re.MULTILINE),
            ],
            # 围串标检测关键字段
            'shareholder_info': [
                re.compile(r'股东[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'持股比例[：:]\s*([^，。\n]{5,30})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'股权结构[：:]\s*([^，。\n]{5,100})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'实际控制人[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
            ],
            'subsidiary_info': [
                re.compile(r'子公司[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'关联公司[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'控股公司[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'参股公司[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
            ],
            'business_license': [
                re.compile(r'营业执照[：:]\s*([A-Z0-9]{10,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'统一社会信用代码[：:]\s*([A-Z0-9]{18})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'注册号[：:]\s*([A-Z0-9]{10,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'组织机构代码[：:]\s*([A-Z0-9]{9})', re.IGNORECASE | re.MULTILINE),
            ],
            'registered_capital': [
                re.compile(r'注册资本[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'实收资本[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'资本金[：:]\s*(\d+(?:\.\d+)?)\s*(?:万元|万|元)', re.IGNORECASE | re.MULTILINE),
            ],
            'establishment_date': [
                re.compile(r'成立日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'注册日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)', re.IGNORECASE | re.MULTILINE),
                re.compile(r'设立日期[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)', re.IGNORECASE | re.MULTILINE),
            ],
            'registered_address': [
                re.compile(r'注册地址[：:]\s*([^，。\n]{10,100})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'公司地址[：:]\s*([^，。\n]{10,100})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'办公地址[：:]\s*([^，。\n]{10,100})', re.IGNORECASE | re.MULTILINE),
            ],
            'business_scope': [
                re.compile(r'经营范围[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'业务范围[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'经营内容[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
            ],
            'qualification_cert': [
                re.compile(r'资质证书[：:]\s*([A-Z0-9\-]{5,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'资格证书[：:]\s*([A-Z0-9\-]{5,20})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'等级证书[：:]\s*([A-Z0-9\-]{5,20})', re.IGNORECASE | re.MULTILINE),
            ],
            'performance_record': [
                re.compile(r'业绩记录[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'项目业绩[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'工程业绩[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
            ],
            'financial_info': [
                re.compile(r'财务状况[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'财务信息[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'资产状况[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
            ],
            'bank_account': [
                re.compile(r'银行账户[：:]\s*([A-Z0-9\-]{10,30})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'开户行[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'账号[：:]\s*([A-Z0-9\-]{10,30})', re.IGNORECASE | re.MULTILINE),
            ],
            'joint_venture': [
                re.compile(r'合资企业[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'合作企业[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'联营企业[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
            ],
            'technical_staff': [
                re.compile(r'技术人员[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'技术团队[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'专业技术人员[：:]\s*([^，。\n]{5,50})', re.IGNORECASE | re.MULTILINE),
            ],
            'equipment_info': [
                re.compile(r'设备信息[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'设备清单[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'主要设备[：:]\s*([^，。\n]{10,200})', re.IGNORECASE | re.MULTILINE),
            ],
            'bidding_consortium': [
                re.compile(r'投标联合体[：:]\s*([^，。\n]{5,100})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'联合投标[：:]\s*([^，。\n]{5,100})', re.IGNORECASE | re.MULTILINE),
                re.compile(r'联合体成员[：:]\s*([^，。\n]{5,100})', re.IGNORECASE | re.MULTILINE),
            ],
        }
        
        return smart_patterns
    
    def extract_fields(self, text: str, full_content: str = None) -> Dict[str, ExtractedField]:
        """从文本中提取字段"""
        extracted_fields = {}
        
        # 如果没有提供完整内容，使用当前文本作为完整内容
        if full_content is None:
            full_content = text
        
        # 1. 正则表达式抽取
        pattern_matches = self._extract_with_patterns(text)
        
        # 2. 关键词抽取
        keyword_matches = self._extract_with_keywords(text) if self.keyword_trie else []
        
        # 3. 智能抽取
        smart_matches = self._extract_with_smart_patterns(text)
        
        # 4. 合并结果
        all_matches = pattern_matches + keyword_matches + smart_matches
        
        # 4. 按字段类型分组
        for match in all_matches:
            if match.field_type not in extracted_fields:
                extracted_fields[match.field_type] = ExtractedField(
                    field_name=match.field_type,
                    field_type=match.field_type,
                    values=[],
                    confidence=0.0
                )
            
            # 计算在完整文档中的绝对位置
            # 这里需要根据text在full_content中的位置来调整
            # 暂时使用相对位置，后续可以在调用时传入正确的偏移量
            absolute_start = match.start
            absolute_end = match.end
            
            # 创建证据片段
            source = 'smart_extractor' if match.confidence == 0.95 else 'regex'
            evidence = EvidenceSpan(
                value=match.value,
                start=absolute_start,
                end=absolute_end,
                confidence=match.confidence,
                source=source,
                pattern=match.pattern
            )
            
            extracted_fields[match.field_type].values.append(evidence)
        
        # 5. 计算整体置信度和主要值
        for field in extracted_fields.values():
            if field.values:
                # 按置信度排序
                field.values.sort(key=lambda x: x.confidence, reverse=True)
                field.primary_value = field.values[0].value
                field.confidence = max(v.confidence for v in field.values)
                
                # 检查冲突
                field.conflicts = self._detect_conflicts(field.values)
        
        return extracted_fields
    
    def _extract_with_patterns(self, text: str) -> List[PatternMatch]:
        """使用正则表达式提取"""
        matches = []
        
        for field_type, pattern_list in self.patterns.items():
            for pattern, description, confidence in pattern_list:
                for match in pattern.finditer(text):
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # 清理和验证值
                    cleaned_value = self._clean_extracted_value(value, field_type)
                    if cleaned_value and len(cleaned_value.strip()) > 0:
                        # 检查是否是有意义的值
                        if self._is_meaningful_value(cleaned_value, field_type):
                            matches.append(PatternMatch(
                                value=cleaned_value,
                                start=match.start(),
                                end=match.end(),
                                pattern=pattern.pattern,
                                field_type=field_type,
                                confidence=confidence
                            ))
        
        return matches
    
    def _extract_with_keywords(self, text: str) -> List[PatternMatch]:
        """使用关键词提取"""
        matches = []
        
        if not self.keyword_trie:
            return matches
        
        try:
            for end_index, (field_type, keyword) in self.keyword_trie.iter(text):
                start_index = end_index - len(keyword) + 1
                
                # 检查关键词上下文
                context = self._get_keyword_context(text, start_index, end_index)
                if context:
                    matches.append(PatternMatch(
                        value=context,
                        start=start_index,
                        end=end_index + len(context),
                        pattern=f"keyword:{keyword}",
                        field_type=field_type,
                        confidence=0.8  # 关键词匹配的置信度
                    ))
        except Exception as e:
            logger.warning(f"关键词匹配出错: {e}")
        
        return matches
    
    def _get_keyword_context(self, text: str, start: int, end: int, 
                            context_size: int = 100) -> Optional[str]:
        """获取关键词上下文"""
        # 扩展上下文范围
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        context = text[context_start:context_end]
        
        # 尝试提取结构化的值
        # 例如：关键词后面的冒号、等号等分隔符
        patterns = [
            r'[：:]\s*([^，。\n]{1,50})',  # 冒号分隔
            r'[=＝]\s*([^，。\n]{1,50})',  # 等号分隔
            r'[为是]\s*([^，。\n]{1,50})',  # "为"或"是"分隔
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _clean_extracted_value(self, value: str, field_type: str) -> Optional[str]:
        """清理和验证提取的值"""
        if not value:
            return None
        
        value = value.strip()
        
        # 根据字段类型进行特定的清理
        if field_type == 'amount' or field_type == 'bid_amount':
            # 金额清理
            # 1. 先尝试提取数字金额
            num_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:万元|万|元)', value)
            if num_match:
                return num_match.group(1) + '万元'  # 保留单位信息
            
            # 2. 如果没有单位，只保留数字
            value = re.sub(r'[^\d.]', '', value)
            if not re.match(r'^\d+(\.\d+)?$', value):
                return None
        
        elif field_type == 'date':
            # 日期清理
            value = re.sub(r'[年月日]', '-', value)
            value = re.sub(r'-+', '-', value)
            value = value.strip('-')
        
        elif field_type == 'contact':
            # 联系方式清理
            if field_type == 'contact' and 'email' in field_type:
                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
                    return None
        
        return value if value else None
    
    def _is_meaningful_value(self, value: str, field_type: str) -> bool:
        """检查值是否有意义"""
        if not value or len(value.strip()) < 2:
            return False
        
        # 过滤掉一些明显无意义的值
        meaningless_values = {
            '无', '无无', '无无无', '无无无无', '无无无无无',
            '零', '零零', '零零零',
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾',
            '日起', '之日起', '日起之',
            '四川省', '中国', '成都', '成都市', '青羊区', '北路', '东路', '南路', '西路',
            '公司', '集团', '企业', '有限', '股份', '责任',
            '工程', '项目', '建设', '施工', '安装', '维护',
            '部', '局', '处', '科', '室', '中心', '站',
            '经理', '主任', '工程师', '技术员', '负责人',
            '证书', '资质', '许可证', '执照'
        }
        
        if value.strip() in meaningless_values:
            return False
        
        # 检查是否包含太多重复字符
        if len(set(value)) < len(value) * 0.3:
            return False
        
        return True
    
    def _extract_with_smart_patterns(self, text: str) -> List[PatternMatch]:
        """使用智能模式提取"""
        matches = []
        
        for field_type, patterns in self.smart_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # 清理和验证值
                    cleaned_value = self._clean_smart_value(value, field_type)
                    if cleaned_value and self._is_meaningful_value(cleaned_value, field_type):
                        matches.append(PatternMatch(
                            value=cleaned_value,
                            start=match.start(),
                            end=match.end(),
                            pattern=pattern.pattern,
                            field_type=field_type,
                            confidence=0.95  # 智能抽取的高置信度
                        ))
        
        return matches
    
    def _clean_smart_value(self, value: str, field_type: str) -> Optional[str]:
        """清理智能抽取的值"""
        if not value:
            return None
        
        value = value.strip()
        
        # 根据字段类型进行特定清理
        if field_type == 'bid_amount':
            # 金额清理 - 更精确的处理
            # 1. 先尝试提取数字金额
            num_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:万元|万|元)', value)
            if num_match:
                return num_match.group(1) + '万元'  # 保留单位信息
            
            # 2. 处理大写金额，但过滤掉明显错误的内容
            if any(char in value for char in ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾', '佰', '仟', '万', '亿']):
                # 过滤掉包含"u u"等明显错误的内容
                if 'u u' in value or 'UIE' in value:
                    return None
                # 只保留有意义的大写金额部分
                cleaned = re.sub(r'[^零壹贰叁肆伍陆柒捌玖拾佰仟万亿]', '', value)
                if len(cleaned) >= 2:
                    return cleaned
            return None
        
        elif field_type == 'deposit':
            # 保证金清理
            cleaned = re.sub(r'[^\d.]', '', value)
            if cleaned and re.match(r'^\d+(\.\d+)?$', cleaned):
                return cleaned
        
        elif field_type == 'contact_info':
            # 联系方式清理 - 更精确
            # 1. 电话号码
            phone_match = re.search(r'(\d{3,4}[-\s]?\d{7,8})', value)
            if phone_match:
                return phone_match.group(1)
            
            # 2. 手机号
            mobile_match = re.search(r'(\d{11})', value)
            if mobile_match:
                return mobile_match.group(1)
            
            # 3. 邮箱
            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', value)
            if email_match:
                return email_match.group(0)
            
            # 4. 邮编
            zip_match = re.search(r'(\d{6})', value)
            if zip_match:
                return zip_match.group(1)
        
        elif field_type == 'legal_representative':
            # 法定代表人清理 - 过滤掉明显错误的内容
            if any(char in value for char in ['答字', '白死', 'UIE', 'u u']):
                return None
            # 只保留中文姓名
            name_match = re.search(r'([\u4e00-\u9fa5]{2,4})', value)
            if name_match:
                return name_match.group(1)
        
        elif field_type == 'bank_account':
            # 银行账户清理 - 过滤掉明显错误的内容
            if 'UIE' in value or 'u u' in value:
                return None
            # 提取账号
            account_match = re.search(r'(\d{10,20})', value)
            if account_match:
                return account_match.group(1)
        
        elif field_type == 'bid_date':
            # 日期清理
            date_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)', value)
            if date_match:
                date_str = date_match.group(1)
                # 标准化日期格式
                date_str = re.sub(r'[年月日]', '-', date_str)
                date_str = re.sub(r'-+', '-', date_str)
                return date_str.strip('-')
        
        return value
    
    def _detect_conflicts(self, values: List[EvidenceSpan]) -> List[str]:
        """检测冲突"""
        conflicts = []
        
        if len(values) <= 1:
            return conflicts
        
        # 检查重复值
        unique_values = set(v.value for v in values)
        if len(unique_values) < len(values):
            conflicts.append("存在重复值")
        
        # 检查值范围冲突（针对数值类型）
        numeric_values = []
        for value in values:
            try:
                if 'amount' in value.pattern or 'deposit' in value.pattern:
                    numeric_values.append(float(value.value))
            except ValueError:
                continue
        
        if len(numeric_values) > 1:
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            if max_val / min_val > 10:  # 数值差异过大
                conflicts.append(f"数值差异过大: {min_val} vs {max_val}")
        
        return conflicts
    
    def get_field_priority(self, field_type: str) -> str:
        """获取字段优先级"""
        if 'field_priority' not in self.config:
            return 'medium'
        
        for priority, fields in self.config['field_priority'].items():
            if field_type in fields:
                return priority
        
        return 'medium'
    
    def get_confidence_threshold(self, priority: str) -> float:
        """获取置信度阈值"""
        if 'confidence_thresholds' not in self.config:
            return 0.7
        
        return self.config['confidence_thresholds'].get(priority, 0.7)
    
    def validate_extraction(self, field: ExtractedField) -> bool:
        """验证抽取结果"""
        if not field.values:
            return False
        
        # 检查置信度
        priority = self.get_field_priority(field.field_type)
        threshold = self.get_confidence_threshold(priority)
        
        if field.confidence < threshold:
            return False
        
        # 检查冲突
        if field.conflicts:
            return False
        
        return True
    
    def get_extraction_stats(self, fields: Dict[str, ExtractedField]) -> Dict[str, Any]:
        """获取抽取统计信息"""
        stats = {
            'total_fields': len(fields),
            'fields_by_type': {},
            'fields_by_priority': {'high': 0, 'medium': 0, 'low': 0},
            'total_values': 0,
            'avg_confidence': 0.0,
            'conflicts_count': 0
        }
        
        total_confidence = 0.0
        
        for field in fields.values():
            # 按类型统计
            field_type = field.field_type
            if field_type not in stats['fields_by_type']:
                stats['fields_by_type'][field_type] = 0
            stats['fields_by_type'][field_type] += 1
            
            # 按优先级统计
            priority = self.get_field_priority(field.field_type)
            stats['fields_by_priority'][priority] += 1
            
            # 值统计
            stats['total_values'] += len(field.values)
            total_confidence += field.confidence
            
            # 冲突统计
            if field.conflicts:
                stats['conflicts_count'] += 1
        
        if stats['total_fields'] > 0:
            stats['avg_confidence'] = total_confidence / stats['total_fields']
        
        return stats 