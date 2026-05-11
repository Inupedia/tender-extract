"""
增强版抽取引擎：多方法竞争 + 置信度路由

成熟方案核心策略：
1. 多方法并行运行（Regex/NER/关键词）
2. 按置信度选择最佳结果
3. 仅低置信/冲突片段路由给 LLM
4. 结果可审计（保留证据和来源）
"""
import re
import logging
from typing import Dict, List, Optional, Tuple
from .schema import ExtractedField, EvidenceSpan
from .patterns import (
    FIELD_PATTERNS, AMOUNT_PATTERNS, DEPOSIT_PATTERNS,
    DATE_PATTERNS, CONTACT_PATTERNS, COMPANY_PATTERNS,
    PERSON_PATTERNS, ID_PATTERNS, TABLE_PATTERNS,
    PROJECT_NAME_PATTERNS, PatternDef, compile_patterns
)

logger = logging.getLogger(__name__)


class ExtractionEngine:
    """
    增强版抽取引擎

    与原有 RuleExtractor 的区别：
    1. 使用分层置信度模式库（patterns.py）
    2. 支持多方法竞争（同一字段多种抽取方式）
    3. 表格感知抽取
    4. 更精确的中文金额/日期处理
    5. 去重和冲突检测
    """

    def __init__(self):
        self._compiled_patterns: Dict[str, list] = {}
        self._compile_all_patterns()

    def _compile_all_patterns(self):
        """预编译所有模式"""
        for field_name, patterns in FIELD_PATTERNS.items():
            self._compiled_patterns[field_name] = compile_patterns(patterns)

    def extract_all_fields(
        self, text: str, target_fields: Optional[List[str]] = None
    ) -> Dict[str, ExtractedField]:
        """
        从文本中抽取所有字段

        Args:
            text: 待抽取文本
            target_fields: 可选，限定抽取的字段列表

        Returns:
            字段名 -> ExtractedField 的映射
        """
        results: Dict[str, ExtractedField] = {}

        fields_to_extract = target_fields or list(self._compiled_patterns.keys())

        for field_name in fields_to_extract:
            if field_name not in self._compiled_patterns:
                continue

            field_result = self._extract_field(text, field_name)
            if field_result and field_result.values:
                results[field_name] = field_result

        # 后处理：去重和冲突检测
        results = self._post_process(results)

        return results

    def _extract_field(self, text: str, field_name: str) -> Optional[ExtractedField]:
        """抽取单个字段"""
        compiled = self._compiled_patterns.get(field_name, [])
        if not compiled:
            return None

        all_values: List[EvidenceSpan] = []

        for pattern, confidence, description in compiled:
            matches = pattern.finditer(text)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                value = value.strip()

                if not value or len(value) < 2:
                    continue

                # 清理值
                cleaned = self._clean_value(value, field_name)
                if not cleaned:
                    continue

                # 验证值
                if not self._validate_value(cleaned, field_name):
                    continue

                evidence = EvidenceSpan(
                    value=cleaned,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    source='regex_enhanced',
                    pattern=description
                )
                all_values.append(evidence)

        if not all_values:
            return None

        # 去重（相同值只保留置信度最高的）
        all_values = self._deduplicate_values(all_values)

        # 按置信度排序
        all_values.sort(key=lambda x: x.confidence, reverse=True)

        return ExtractedField(
            field_name=field_name,
            field_type=field_name,
            values=all_values,
            primary_value=all_values[0].value,
            confidence=all_values[0].confidence,
            conflicts=self._detect_conflicts(all_values)
        )

    def _clean_value(self, value: str, field_name: str) -> Optional[str]:
        """根据字段类型清理提取的值"""
        if not value:
            return None

        # 通用清理
        value = value.strip()
        value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)

        if field_name in ('bid_amount', 'deposit'):
            return self._clean_amount(value)
        elif field_name == 'bid_date':
            return self._clean_date(value)
        elif field_name == 'contact_info':
            return self._clean_contact(value)
        elif field_name in ('bidder', 'tenderer'):
            return self._clean_company(value)
        elif field_name == 'legal_representative':
            return self._clean_person_name(value)
        elif field_name == 'project_number':
            return self._clean_id(value)

        return value

    def _clean_amount(self, value: str) -> Optional[str]:
        """清理金额"""
        # 大写金额直接返回
        if re.search(r'[壹贰叁肆伍陆柒捌玖拾佰仟万亿零]', value):
            cleaned = re.sub(r'[^\u4e00-\u9fff元整]', '', value)
            if len(cleaned) >= 3:
                return cleaned
            return None

        # 数字金额标准化
        # 移除千分位逗号
        value = value.replace(',', '')
        # 提取数字部分（支持最多4位小数，覆盖万元单位场景）
        num_match = re.search(r'(\d+(?:\.\d{1,6})?)', value)
        if num_match:
            return num_match.group(1)

        return None

    def _clean_date(self, value: str) -> Optional[str]:
        """清理日期"""
        # 标准化中文日期分隔符
        value = re.sub(r'\s+', '', value)  # 移除空格
        # 保留中文日期格式
        if re.search(r'\d{4}年\d{1,2}月\d{1,2}日', value):
            return value
        # ISO格式
        if re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', value):
            return value
        return value

    def _clean_contact(self, value: str) -> Optional[str]:
        """清理联系方式"""
        value = re.sub(r'[\s\-—]', '', value)
        # 验证手机号
        if re.match(r'^1[3-9]\d{9}$', value):
            return value
        # 验证座机
        if re.match(r'^0\d{2,3}\d{7,8}$', value):
            return value
        # 邮箱
        if '@' in value:
            return value
        # 其他类型保留原值
        return value if len(value) >= 6 else None

    def _clean_company(self, value: str) -> Optional[str]:
        """清理公司名"""
        # 移除前后多余字符
        value = re.sub(r'^[^a-zA-Z\u4e00-\u9fff]+', '', value)
        value = re.sub(r'[^a-zA-Z\u4e00-\u9fff（）()]+$', '', value)
        # 验证至少包含公司类后缀
        if not re.search(r'(公司|集团|企业|局|中心|院|所|社)$', value):
            # 可能被截断了，如果够长还是保留
            if len(value) < 8:
                return None
        return value if len(value) >= 4 else None

    def _clean_person_name(self, value: str) -> Optional[str]:
        """清理人名"""
        # 只保留中文字符
        name = re.sub(r'[^\u4e00-\u9fa5]', '', value)
        if 2 <= len(name) <= 4:
            return name
        return None

    def _clean_id(self, value: str) -> Optional[str]:
        """清理编号"""
        value = value.strip()
        if len(value) >= 5:
            return value
        return None

    def _validate_value(self, value: str, field_name: str) -> bool:
        """验证值的有效性"""
        if not value or len(value) < 2:
            return False

        # 排除明显无意义的值
        noise_words = {'无', '暂无', '略', '详见', '见附件', '以上', '以下', '如下'}
        if value in noise_words:
            return False

        # 字段特定验证
        if field_name in ('bid_amount', 'deposit'):
            # 金额不能为0
            try:
                if not re.search(r'[壹贰叁肆伍陆柒捌玖]', value):
                    num = float(value.replace(',', ''))
                    if num <= 0:
                        return False
            except (ValueError, TypeError):
                pass

        elif field_name == 'legal_representative':
            # 人名必须是2-4个中文字符
            if not re.match(r'^[\u4e00-\u9fa5]{2,4}$', value):
                return False

        elif field_name == 'project_number':
            # 编号必须包含字母或数字
            if not re.search(r'[A-Za-z0-9]', value):
                return False

        return True

    def _deduplicate_values(self, values: List[EvidenceSpan]) -> List[EvidenceSpan]:
        """去重：相同值只保留置信度最高的"""
        seen: Dict[str, EvidenceSpan] = {}
        for v in values:
            normalized = v.value.strip().lower()
            if normalized not in seen or v.confidence > seen[normalized].confidence:
                seen[normalized] = v
        return list(seen.values())

    def _detect_conflicts(self, values: List[EvidenceSpan]) -> List[str]:
        """检测值冲突"""
        conflicts = []
        if len(values) <= 1:
            return conflicts

        unique_values = set(v.value for v in values)
        if len(unique_values) > 1:
            # 检查是否是数值类型冲突
            numeric_vals = []
            for v in values:
                try:
                    numeric_vals.append(float(v.value.replace(',', '')))
                except (ValueError, TypeError):
                    pass

            if len(numeric_vals) > 1:
                if max(numeric_vals) / max(min(numeric_vals), 0.01) > 5:
                    conflicts.append(
                        f"数值差异过大: {min(numeric_vals)} vs {max(numeric_vals)}"
                    )
            elif len(unique_values) > 3:
                conflicts.append(f"存在{len(unique_values)}个不同值")

        return conflicts

    def _post_process(
        self, results: Dict[str, ExtractedField]
    ) -> Dict[str, ExtractedField]:
        """后处理：跨字段验证和校正"""
        # 投标金额 vs 保证金：保证金不应超过投标金额
        if 'bid_amount' in results and 'deposit' in results:
            try:
                amount_val = float(
                    results['bid_amount'].primary_value.replace(',', '')
                )
                deposit_val = float(
                    results['deposit'].primary_value.replace(',', '')
                )
                if deposit_val > amount_val:
                    results['deposit'].confidence *= 0.5
                    results['deposit'].conflicts.append(
                        "保证金超过投标金额，可能存在单位不一致"
                    )
            except (ValueError, TypeError, AttributeError):
                pass

        return results

    def get_confidence_summary(
        self, results: Dict[str, ExtractedField]
    ) -> Dict[str, float]:
        """获取各字段置信度摘要"""
        return {
            name: field.confidence
            for name, field in results.items()
        }

    def get_low_confidence_fields(
        self, results: Dict[str, ExtractedField], threshold: float = 0.7
    ) -> List[str]:
        """获取低置信度字段（需要 LLM 处理的）"""
        return [
            name for name, field in results.items()
            if field.confidence < threshold or field.conflicts
        ]
