"""
字段合并策略模块
"""
from typing import List, Dict, Any, Optional
from .schema import ExtractedField, EvidenceSpan
import logging

logger = logging.getLogger(__name__)


class FieldMerger:
    """字段合并器"""
    
    def __init__(self):
        self.merge_strategies = {
            'amount': self._merge_amount_field,
            'date': self._merge_date_field,
            'number': self._merge_number_field,
            'contact': self._merge_contact_field,
            'deposit': self._merge_deposit_field,
            'default': self._merge_default_field
        }
    
    def merge_fields(self, fields: Dict[str, ExtractedField]) -> Dict[str, ExtractedField]:
        """合并字段"""
        merged_fields = {}
        
        for field_name, field in fields.items():
            # 选择合并策略
            strategy = self.merge_strategies.get(field.field_type, self.merge_strategies['default'])
            merged_field = strategy(field)
            merged_fields[field_name] = merged_field
        
        return merged_fields
    
    def _merge_amount_field(self, field: ExtractedField) -> ExtractedField:
        """合并金额字段"""
        if not field.values:
            return field
        
        # 按置信度排序
        sorted_values = sorted(field.values, key=lambda x: x.confidence, reverse=True)
        
        # 尝试找到最合理的金额
        best_value = None
        best_confidence = 0.0
        
        for value in sorted_values:
            try:
                # 解析金额
                amount = self._parse_amount(value.value)
                if amount is not None:
                    # 检查金额的合理性
                    if self._is_reasonable_amount(amount):
                        best_value = value
                        best_confidence = value.confidence
                        break
            except (ValueError, TypeError):
                continue
        
        # 如果没有找到合理的金额，使用置信度最高的
        if best_value is None and sorted_values:
            best_value = sorted_values[0]
            best_confidence = sorted_values[0].confidence
        
        # 创建合并后的字段
        merged_field = ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=[best_value] if best_value else [],
            primary_value=best_value.value if best_value else None,
            confidence=best_confidence,
            conflicts=field.conflicts
        )
        
        return merged_field
    
    def _merge_date_field(self, field: ExtractedField) -> ExtractedField:
        """合并日期字段"""
        if not field.values:
            return field
        
        # 按置信度排序
        sorted_values = sorted(field.values, key=lambda x: x.confidence, reverse=True)
        
        # 尝试找到最合理的日期
        best_value = None
        best_confidence = 0.0
        
        for value in sorted_values:
            try:
                # 解析日期
                date = self._parse_date(value.value)
                if date is not None:
                    # 检查日期的合理性
                    if self._is_reasonable_date(date):
                        best_value = value
                        best_confidence = value.confidence
                        break
            except (ValueError, TypeError):
                continue
        
        # 如果没有找到合理的日期，使用置信度最高的
        if best_value is None and sorted_values:
            best_value = sorted_values[0]
            best_confidence = sorted_values[0].confidence
        
        # 创建合并后的字段
        merged_field = ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=[best_value] if best_value else [],
            primary_value=best_value.value if best_value else None,
            confidence=best_confidence,
            conflicts=field.conflicts
        )
        
        return merged_field
    
    def _merge_number_field(self, field: ExtractedField) -> ExtractedField:
        """合并编号字段"""
        if not field.values:
            return field
        
        # 按置信度排序
        sorted_values = sorted(field.values, key=lambda x: x.confidence, reverse=True)
        
        # 尝试找到最合理的编号
        best_value = None
        best_confidence = 0.0
        
        for value in sorted_values:
            # 检查编号格式
            if self._is_valid_number_format(value.value):
                best_value = value
                best_confidence = value.confidence
                break
        
        # 如果没有找到合理的编号，使用置信度最高的
        if best_value is None and sorted_values:
            best_value = sorted_values[0]
            best_confidence = sorted_values[0].confidence
        
        # 创建合并后的字段
        merged_field = ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=[best_value] if best_value else [],
            primary_value=best_value.value if best_value else None,
            confidence=best_confidence,
            conflicts=field.conflicts
        )
        
        return merged_field
    
    def _merge_contact_field(self, field: ExtractedField) -> ExtractedField:
        """合并联系方式字段"""
        if not field.values:
            return field
        
        # 按置信度排序
        sorted_values = sorted(field.values, key=lambda x: x.confidence, reverse=True)
        
        # 尝试找到最合理的联系方式
        best_value = None
        best_confidence = 0.0
        
        for value in sorted_values:
            # 检查联系方式格式
            if self._is_valid_contact_format(value.value):
                best_value = value
                best_confidence = value.confidence
                break
        
        # 如果没有找到合理的联系方式，使用置信度最高的
        if best_value is None and sorted_values:
            best_value = sorted_values[0]
            best_confidence = sorted_values[0].confidence
        
        # 创建合并后的字段
        merged_field = ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=[best_value] if best_value else [],
            primary_value=best_value.value if best_value else None,
            confidence=best_confidence,
            conflicts=field.conflicts
        )
        
        return merged_field
    
    def _merge_deposit_field(self, field: ExtractedField) -> ExtractedField:
        """合并保证金字段"""
        if not field.values:
            return field
        
        # 按置信度排序
        sorted_values = sorted(field.values, key=lambda x: x.confidence, reverse=True)
        
        # 尝试找到最合理的保证金
        best_value = None
        best_confidence = 0.0
        
        for value in sorted_values:
            try:
                # 解析保证金
                amount = self._parse_amount(value.value)
                if amount is not None:
                    # 检查保证金的合理性
                    if self._is_reasonable_deposit(amount):
                        best_value = value
                        best_confidence = value.confidence
                        break
            except (ValueError, TypeError):
                continue
        
        # 如果没有找到合理的保证金，使用置信度最高的
        if best_value is None and sorted_values:
            best_value = sorted_values[0]
            best_confidence = sorted_values[0].confidence
        
        # 创建合并后的字段
        merged_field = ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=[best_value] if best_value else [],
            primary_value=best_value.value if best_value else None,
            confidence=best_confidence,
            conflicts=field.conflicts
        )
        
        return merged_field
    
    def _merge_default_field(self, field: ExtractedField) -> ExtractedField:
        """默认合并策略"""
        if not field.values:
            return field
        
        # 按置信度排序
        sorted_values = sorted(field.values, key=lambda x: x.confidence, reverse=True)
        
        # 使用置信度最高的值
        best_value = sorted_values[0]
        
        # 创建合并后的字段
        merged_field = ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=[best_value],
            primary_value=best_value.value,
            confidence=best_value.confidence,
            conflicts=field.conflicts
        )
        
        return merged_field
    
    def _parse_amount(self, value: str) -> Optional[float]:
        """解析金额"""
        import re
        
        # 移除非数字字符
        cleaned = re.sub(r'[^\d.]', '', value)
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _parse_date(self, value: str) -> Optional[str]:
        """解析日期"""
        import re
        from datetime import datetime
        
        # 常见的日期格式
        date_patterns = [
            r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?',
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, value)
            if match:
                try:
                    if len(match.groups()) == 3:
                        year, month, day = match.groups()
                        # 验证日期有效性
                        datetime(int(year), int(month), int(day))
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    continue
        
        return None
    
    def _is_reasonable_amount(self, amount: float) -> bool:
        """检查金额是否合理"""
        # 金额应该在合理范围内（1元到100亿元）
        return 1.0 <= amount <= 10000000000.0
    
    def _is_reasonable_date(self, date_str: str) -> bool:
        """检查日期是否合理"""
        from datetime import datetime, timedelta
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            now = datetime.now()
            
            # 日期应该在合理范围内（过去10年到未来5年）
            min_date = now - timedelta(days=3650)  # 10年前
            max_date = now + timedelta(days=1825)  # 5年后
            
            return min_date <= date <= max_date
        except ValueError:
            return False
    
    def _is_valid_number_format(self, value: str) -> bool:
        """检查编号格式是否有效"""
        import re
        
        # 编号应该包含字母和数字的组合
        if re.match(r'^[A-Z0-9\-_]{5,20}$', value):
            return True
        
        return False
    
    def _is_valid_contact_format(self, value: str) -> bool:
        """检查联系方式格式是否有效"""
        import re
        
        # 电话号码格式
        phone_pattern = r'^\d{3,4}[-\s]?\d{7,8}$'
        if re.match(phone_pattern, value):
            return True
        
        # 手机号码格式
        mobile_pattern = r'^1[3-9]\d{9}$'
        if re.match(mobile_pattern, value):
            return True
        
        # 邮箱格式
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, value):
            return True
        
        return False
    
    def _is_reasonable_deposit(self, amount: float) -> bool:
        """检查保证金是否合理"""
        # 保证金通常在项目金额的1%-10%之间
        # 这里我们假设项目金额在10万到100亿之间
        min_deposit = 1000.0  # 最低1000元
        max_deposit = 1000000000.0  # 最高10亿元
        
        return min_deposit <= amount <= max_deposit
    
    def resolve_conflicts(self, fields: Dict[str, ExtractedField]) -> Dict[str, ExtractedField]:
        """解决字段冲突"""
        resolved_fields = {}
        
        for field_name, field in fields.items():
            if not field.conflicts:
                resolved_fields[field_name] = field
                continue
            
            # 解决冲突
            resolved_field = self._resolve_field_conflicts(field)
            resolved_fields[field_name] = resolved_field
        
        return resolved_fields
    
    def _resolve_field_conflicts(self, field: ExtractedField) -> ExtractedField:
        """解决单个字段的冲突"""
        if not field.conflicts:
            return field
        
        # 根据冲突类型选择解决策略
        for conflict in field.conflicts:
            if "重复值" in conflict:
                field = self._resolve_duplicate_values(field)
            elif "数值差异过大" in conflict:
                field = self._resolve_numeric_conflicts(field)
            else:
                # 默认策略：选择置信度最高的值
                field = self._resolve_by_confidence(field)
        
        return field
    
    def _resolve_duplicate_values(self, field: ExtractedField) -> ExtractedField:
        """解决重复值冲突"""
        # 去重，保留置信度最高的
        unique_values = []
        seen = set()
        
        for value in field.values:
            if value.value not in seen:
                seen.add(value.value)
                unique_values.append(value)
        
        # 按置信度排序
        unique_values.sort(key=lambda x: x.confidence, reverse=True)
        
        return ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=unique_values,
            primary_value=unique_values[0].value if unique_values else None,
            confidence=unique_values[0].confidence if unique_values else 0.0,
            conflicts=[]
        )
    
    def _resolve_numeric_conflicts(self, field: ExtractedField) -> ExtractedField:
        """解决数值冲突"""
        # 对于数值冲突，选择最合理的值
        numeric_values = []
        
        for value in field.values:
            try:
                if 'amount' in value.pattern or 'deposit' in value.pattern:
                    amount = self._parse_amount(value.value)
                    if amount is not None:
                        numeric_values.append((value, amount))
            except (ValueError, TypeError):
                continue
        
        if numeric_values:
            # 选择最合理的数值
            best_value = None
            best_score = 0.0
            
            for value, amount in numeric_values:
                # 计算合理性分数
                if 'amount' in value.pattern:
                    score = self._calculate_amount_reasonableness(amount)
                else:
                    score = self._calculate_deposit_reasonableness(amount)
                
                if score > best_score:
                    best_score = score
                    best_value = value
            
            if best_value:
                return ExtractedField(
                    field_name=field.field_name,
                    field_type=field.field_type,
                    values=[best_value],
                    primary_value=best_value.value,
                    confidence=best_value.confidence,
                    conflicts=[]
                )
        
        # 如果无法解决，使用置信度最高的值
        return self._resolve_by_confidence(field)
    
    def _resolve_by_confidence(self, field: ExtractedField) -> ExtractedField:
        """按置信度解决冲突"""
        if not field.values:
            return field
        
        # 选择置信度最高的值
        best_value = max(field.values, key=lambda x: x.confidence)
        
        return ExtractedField(
            field_name=field.field_name,
            field_type=field.field_type,
            values=[best_value],
            primary_value=best_value.value,
            confidence=best_value.confidence,
            conflicts=[]
        )
    
    def _calculate_amount_reasonableness(self, amount: float) -> float:
        """计算金额合理性分数"""
        # 基于金额范围的合理性评分
        if 1000 <= amount <= 1000000000:  # 1千到10亿
            return 1.0
        elif 100 <= amount < 1000 or 1000000000 < amount <= 10000000000:  # 100-1000或10亿-100亿
            return 0.8
        else:
            return 0.3
    
    def _calculate_deposit_reasonableness(self, amount: float) -> float:
        """计算保证金合理性分数"""
        # 基于保证金范围的合理性评分
        if 1000 <= amount <= 10000000:  # 1千到1千万
            return 1.0
        elif 100 <= amount < 1000 or 10000000 < amount <= 100000000:  # 100-1000或1千万-1亿
            return 0.8
        else:
            return 0.3 