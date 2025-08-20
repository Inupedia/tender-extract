"""
中文NER模块
"""
import re
from typing import List, Dict, Any, Optional
from .schema import EvidenceSpan, ExtractedField
import logging

logger = logging.getLogger(__name__)

# foolnltk已移除，使用jieba和正则表达式进行NER
FOOLNLTK_AVAILABLE = False

try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba未安装，将使用基础NER")


class NERExtractor:
    """命名实体识别抽取器"""
    
    def __init__(self, use_foolnltk: bool = False):
        self.use_foolnltk = False  # foolnltk已移除
        self.use_jieba = JIEBA_AVAILABLE
        
        # 实体类型映射
        self.entity_mapping = {
            'nr': 'person',      # 人名
            'ns': 'location',    # 地名
            'nt': 'organization', # 机构名
            'nz': 'other',       # 其他专名
        }
        
        # 招标投标相关的实体类型
        self.tender_entities = {
            'company': r'[^\s，。；：]{2,20}(?:公司|集团|企业|有限|股份|责任)',
            'project': r'[^\s，。；：]{5,50}(?:工程|项目|建设|施工|安装|维护)',
            'department': r'[^\s，。；：]{2,20}(?:部|局|处|科|室|中心|站)',
            'position': r'[^\s，。；：]{2,10}(?:经理|主任|工程师|技术员|负责人)',
            'certificate': r'[A-Z0-9]{5,20}(?:证书|资质|许可证|执照)',
        }
        
    def extract_entities(self, text: str, full_content: str = None) -> Dict[str, ExtractedField]:
        """提取命名实体"""
        extracted_fields = {}
        
        # 如果没有提供完整内容，使用当前文本作为完整内容
        if full_content is None:
            full_content = text
        
        # foolnltk已移除，直接使用jieba和正则表达式
        
        # 2. 使用jieba（如果可用）
        if self.use_jieba:
            jieba_entities = self._extract_with_jieba(text)
            for entity_type, entities in jieba_entities.items():
                if entity_type not in extracted_fields:
                    extracted_fields[entity_type] = ExtractedField(
                        field_name=entity_type,
                        field_type=entity_type,
                        values=[],
                        confidence=0.0
                    )
                extracted_fields[entity_type].values.extend(entities)
        
        # 3. 使用正则表达式提取招标投标相关实体
        tender_entities = self._extract_tender_entities(text)
        for entity_type, entities in tender_entities.items():
            if entity_type not in extracted_fields:
                extracted_fields[entity_type] = ExtractedField(
                    field_name=entity_type,
                    field_type=entity_type,
                    values=[],
                    confidence=0.0
                )
            extracted_fields[entity_type].values.extend(entities)
        
        # 4. 计算置信度和去重
        for field in extracted_fields.values():
            if field.values:
                # 去重
                unique_entities = self._deduplicate_entities(field.values)
                field.values = unique_entities
                
                # 按置信度排序
                field.values.sort(key=lambda x: x.confidence, reverse=True)
                field.primary_value = field.values[0].value
                field.confidence = max(v.confidence for v in field.values)
        
        return extracted_fields
    

    
    def _extract_with_jieba(self, text: str) -> Dict[str, List[EvidenceSpan]]:
        """使用jieba提取实体"""
        entities = {}
        
        try:
            # 使用jieba进行词性标注
            words = pseg.cut(text)
            
            for word, flag in words:
                if flag in self.entity_mapping:
                    entity_type = self.entity_mapping[flag]
                    
                    # 检查是否是有意义的实体
                    if not self._is_meaningful_jieba_entity(word, entity_type):
                        continue
                    
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    # 查找实体在原文中的位置
                    start = text.find(word)
                    if start != -1:
                        evidence = EvidenceSpan(
                            value=word,
                            start=start,
                            end=start + len(word),
                            confidence=0.8,  # jieba的置信度
                            source='jieba',
                            pattern=f"pos:{flag}"
                        )
                        entities[entity_type].append(evidence)
        
        except Exception as e:
            logger.error(f"jieba实体提取失败: {e}")
        
        return entities
    
    def _extract_tender_entities(self, text: str) -> Dict[str, List[EvidenceSpan]]:
        """提取招标投标相关实体"""
        entities = {}
        
        for entity_type, pattern in self.tender_entities.items():
            entities[entity_type] = []
            
            for match in re.finditer(pattern, text):
                value = match.group(0)
                
                # 验证实体质量
                if self._validate_tender_entity(value, entity_type):
                    # 检查是否是有意义的实体
                    if self._is_meaningful_entity(value, entity_type):
                        evidence = EvidenceSpan(
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.7,  # 正则表达式的置信度
                            source='regex',
                            pattern=f"tender:{entity_type}"
                        )
                        entities[entity_type].append(evidence)
        
        return entities
    
    def _validate_tender_entity(self, value: str, entity_type: str) -> bool:
        """验证招标投标实体的质量"""
        if not value or len(value) < 2:
            return False
        
        # 根据实体类型进行特定验证
        if entity_type == 'company':
            # 公司名称验证
            company_suffixes = ['公司', '集团', '企业', '有限', '股份', '责任']
            return any(suffix in value for suffix in company_suffixes)
        
        elif entity_type == 'project':
            # 项目名称验证
            project_keywords = ['工程', '项目', '建设', '施工', '安装', '维护']
            return any(keyword in value for keyword in project_keywords)
        
        elif entity_type == 'department':
            # 部门名称验证
            dept_suffixes = ['部', '局', '处', '科', '室', '中心', '站']
            return any(suffix in value for suffix in dept_suffixes)
        
        elif entity_type == 'position':
            # 职位名称验证
            position_suffixes = ['经理', '主任', '工程师', '技术员', '负责人']
            return any(suffix in value for suffix in position_suffixes)
        
        elif entity_type == 'certificate':
            # 证书编号验证
            return re.match(r'^[A-Z0-9]{5,20}', value) is not None
        
        return True
    
    def _is_meaningful_entity(self, value: str, entity_type: str) -> bool:
        """检查实体是否有意义"""
        if not value or len(value.strip()) < 3:
            return False
        
        # 过滤掉一些明显无意义的实体
        meaningless_entities = {
            '四川省', '中国', '成都', '成都市', '青羊区', '北路', '东路', '南路', '西路',
            '日起', '之日起', '日起之',
            '无', '无无', '无无无',
            '零', '零零', '零零零',
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾',
        }
        
        if value.strip() in meaningless_entities:
            return False
        
        # 检查是否包含太多重复字符
        if len(set(value)) < len(value) * 0.4:
            return False
        
        # 检查是否是有意义的标书相关实体
        if entity_type == 'company':
            # 公司名称应该包含具体的公司名
            if len(value) < 4 or value in ['公司', '集团', '企业']:
                return False
        
        elif entity_type == 'project':
            # 项目名称应该包含具体的项目信息
            if len(value) < 5 or value in ['工程', '项目', '建设']:
                return False
        
        return True
    
    def _is_meaningful_jieba_entity(self, word: str, entity_type: str) -> bool:
        """检查jieba提取的实体是否有意义"""
        if not word or len(word.strip()) < 2:
            return False
        
        # 过滤掉一些明显无意义的词
        meaningless_words = {
            '四川省', '中国', '成都', '成都市', '青羊区', '北路', '东路', '南路', '西路',
            '日起', '之日起', '日起之',
            '无', '无无', '无无无',
            '零', '零零', '零零零',
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾',
            '大唐', '天唐', '唐际孜水', '三库', '汇报', '开工日期',
            '卡号', '号码', '编号', '序号', '第', '第第',
            '向家坝', '水土保持', '伍拾', '通知书', '须知', '熊峰',
            '浣花', '北路', '青羊区', '成都市', '四川省',
            '电建', '勘测', '设计', '研究院', '有限公司',
            '建设', '开发', '有限责任', '公司',
            '灌区', '北总干渠', '一期', '二步', '工程',
            '监测', '招标', '投标', '文件',
        }
        
        if word.strip() in meaningless_words:
            return False
        
        # 检查是否包含太多重复字符
        if len(set(word)) < len(word) * 0.5:
            return False
        
        # 检查是否是有意义的标书相关实体
        if entity_type == 'location':
            # 地名应该是有意义的地点
            if len(word) < 3 or word in ['四川省', '中国', '成都']:
                return False
        
        elif entity_type == 'person':
            # 人名应该是具体的姓名，过滤掉一些明显不是人名的词
            if len(word) < 2 or word in ['日起', '之日起', '向家坝', '水土保持', '伍拾', '通知书', '须知']:
                return False
        
        elif entity_type == 'organization':
            # 机构名应该是具体的机构
            if len(word) < 3:
                return False
        
        return True
    
    def _deduplicate_entities(self, entities: List[EvidenceSpan]) -> List[EvidenceSpan]:
        """去重实体"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 使用值和位置作为唯一标识
            key = (entity.value, entity.start, entity.end)
            
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # 如果发现重复，保留置信度更高的
                for i, existing in enumerate(unique_entities):
                    existing_key = (existing.value, existing.start, existing.end)
                    if key == existing_key and entity.confidence > existing.confidence:
                        unique_entities[i] = entity
                        break
        
        return unique_entities
    
    def merge_with_rules(self, ner_fields: Dict[str, ExtractedField], 
                        rule_fields: Dict[str, ExtractedField]) -> Dict[str, ExtractedField]:
        """将NER结果与规则结果合并"""
        merged_fields = rule_fields.copy()
        
        for field_name, ner_field in ner_fields.items():
            if field_name in merged_fields:
                # 合并到现有字段
                existing_field = merged_fields[field_name]
                existing_field.values.extend(ner_field.values)
                
                # 重新计算置信度
                if existing_field.values:
                    existing_field.values.sort(key=lambda x: x.confidence, reverse=True)
                    existing_field.primary_value = existing_field.values[0].value
                    existing_field.confidence = max(v.confidence for v in existing_field.values)
            else:
                # 添加新字段
                merged_fields[field_name] = ner_field
        
        return merged_fields
    
    def get_entity_statistics(self, fields: Dict[str, ExtractedField]) -> Dict[str, Any]:
        """获取实体统计信息"""
        stats = {
            'total_entities': 0,
            'entities_by_type': {},
            'entities_by_source': {'foolnltk': 0, 'jieba': 0, 'regex': 0},
            'avg_confidence': 0.0,
            'unique_entities': 0
        }
        
        total_confidence = 0.0
        unique_values = set()
        
        for field in fields.values():
            stats['total_entities'] += len(field.values)
            
            # 按类型统计
            field_type = field.field_type
            if field_type not in stats['entities_by_type']:
                stats['entities_by_type'][field_type] = 0
            stats['entities_by_type'][field_type] += len(field.values)
            
            # 按来源统计
            for value in field.values:
                source = value.source
                if source in stats['entities_by_source']:
                    stats['entities_by_source'][source] += 1
                
                total_confidence += value.confidence
                unique_values.add(value.value)
        
        if stats['total_entities'] > 0:
            stats['avg_confidence'] = total_confidence / stats['total_entities']
        
        stats['unique_entities'] = len(unique_values)
        
        return stats 