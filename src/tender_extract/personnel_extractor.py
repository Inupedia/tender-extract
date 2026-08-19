"""
人员信息专项抽取模块

针对标书中分散在不同位置的人员信息：
- 身份证号码
- 姓名 + 职务/角色
- 毕业证/学历信息
- 专业资格证书（建造师、工程师等）
- 证书编号和有效期
- 社保信息
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PersonnelInfo:
    """人员信息结构"""
    name: str
    role: str = ""  # 角色/职务
    id_card: str = ""  # 身份证号
    education: str = ""  # 学历
    major: str = ""  # 专业
    graduation_school: str = ""  # 毕业院校
    graduation_date: str = ""  # 毕业日期
    certificates: List[Dict[str, str]] = field(default_factory=list)  # 证书列表
    contact: str = ""  # 联系方式
    confidence: float = 0.0


@dataclass
class CertificateInfo:
    """证书信息"""
    cert_type: str  # 证书类型
    cert_number: str  # 证书编号
    holder_name: str = ""  # 持有人
    issue_date: str = ""  # 发证日期
    expiry_date: str = ""  # 有效期至
    issuer: str = ""  # 发证机构
    level: str = ""  # 等级
    major: str = ""  # 专业


class PersonnelExtractor:
    """
    人员信息抽取器

    处理标书中常见的人员信息形式：
    1. 人员资质汇总表
    2. 身份证扫描件（OCR文本）
    3. 学历证书信息
    4. 专业资格证书
    5. 社保证明
    """

    # 身份证号码模式（18位或15位）
    ID_CARD_PATTERNS = [
        # 明确标注的身份证号
        (re.compile(
            r'(?:身份证[号码]*|证件号[码]?|身份号码)[：:]\s*'
            r'([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])'
        ), 0.95, "标注身份证号"),
        # 独立的18位身份证号
        (re.compile(
            r'([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])'
        ), 0.80, "独立身份证号"),
        # 15位老版身份证号
        (re.compile(
            r'(?:身份证[号码]*)[：:]\s*([1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3})'
        ), 0.75, "15位身份证号"),
    ]

    # 人员姓名+角色模式
    NAME_ROLE_PATTERNS = [
        (re.compile(r'项目(?:经理|负责人)[：:]\s*([\u4e00-\u9fa5]{2,4})'), "项目经理"),
        (re.compile(r'技术负责人[：:]\s*([\u4e00-\u9fa5]{2,4})'), "技术负责人"),
        (re.compile(r'安全员[：:]\s*([\u4e00-\u9fa5]{2,4})'), "安全员"),
        (re.compile(r'质(?:检|量)员[：:]\s*([\u4e00-\u9fa5]{2,4})'), "质检员"),
        (re.compile(r'施工员[：:]\s*([\u4e00-\u9fa5]{2,4})'), "施工员"),
        (re.compile(r'造价[师员][：:]\s*([\u4e00-\u9fa5]{2,4})'), "造价师"),
        (re.compile(r'总工[程]?师[：:]\s*([\u4e00-\u9fa5]{2,4})'), "总工程师"),
        (re.compile(r'(?:拟派|拟任|拟聘)\s*(?:项目经理|负责人)[：:]\s*([\u4e00-\u9fa5]{2,4})'), "拟派项目经理"),
        (re.compile(r'姓名[：:]\s*([\u4e00-\u9fa5]{2,4})'), "人员"),
    ]

    # 学历信息模式
    EDUCATION_PATTERNS = [
        (re.compile(r'(?:学历|文化程度)[：:]\s*(博士|硕士|本科|大专|中专|高中|研究生)'), 0.90),
        (re.compile(r'(?:毕业院校|毕业学校|学校)[：:]\s*([^，。\n]{4,30})'), 0.85),
        (re.compile(r'(?:所学专业|专业)[：:]\s*([^，。\n]{2,20})'), 0.85),
        (re.compile(r'(?:毕业时间|毕业日期)[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]?\d{0,2}[日]?)'), 0.85),
    ]

    # 证书模式
    CERTIFICATE_PATTERNS = [
        # 建造师（各种写法）
        (re.compile(
            r'(?:一级|二级|1级|2级)?\s*(?:注册)?\s*建造师[^，。\n]*?'
            r'(?:证书)?(?:编号|号)[：:]\s*([^\s，。\n]{6,25})'
        ), "建造师", 0.95),
        (re.compile(
            r'建造师.*?注册编号[：:]\s*([^\s，。\n]{6,25})'
        ), "建造师", 0.95),

        # 工程师职称（各种写法）
        (re.compile(
            r'(?:高级|中级|初级|正高级)\s*工程师[^，。\n]*?'
            r'(?:证书)?(?:编号|号)[：:]\s*([^\s，。\n]{6,25})'
        ), "工程师职称", 0.90),

        # 安全生产考核（含中文括号和字母）
        (re.compile(
            r'安全生产考核[^，。\n]*?(?:证书)?(?:编号|号)[：:]\s*([^\s，。\n]{6,30})'
        ), "安全B证", 0.90),
        (re.compile(
            r'安全[BC]证[^，。\n]*?(?:编号|号)?[：:]\s*([^\s，。\n]{6,30})'
        ), "安全证", 0.85),
        (re.compile(
            r'[（(][A-C]类?[)）]?安全[^，。\n]*?(?:编号|号)?[：:]\s*([^\s，。\n]{6,30})'
        ), "安全证", 0.85),

        # 通用证书编号
        (re.compile(
            r'(?:证书|资格证|执业证|资质证)[^，。\n]*?(?:编号|号)[：:]\s*([^\s，。\n]{6,30})'
        ), "资格证书", 0.80),
    ]

    EXPIRY_PATTERNS = [
        re.compile(r'有效期至[：:]?\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)'),
        re.compile(r'有效期[^：:]*[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)'),
    ]

    # 表格行中的人员信息模式
    TABLE_PERSONNEL_PATTERNS = [
        # | 序号 | 姓名 | 职务 | 身份证号 | 证书编号 |
        re.compile(
            r'\|\s*\d+\s*\|\s*([\u4e00-\u9fa5]{2,4})\s*\|\s*([^|]+)\s*\|\s*'
            r'([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])\s*\|'
        ),
        # | 姓名 | 身份证号 | 学历 | 专业 |
        re.compile(
            r'\|\s*([\u4e00-\u9fa5]{2,4})\s*\|\s*'
            r'([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])\s*\|\s*'
            r'([^|]*)\s*\|\s*([^|]*)\s*\|'
        ),
    ]

    def extract_personnel(self, text: str) -> List[PersonnelInfo]:
        """
        从文本中抽取所有人员信息

        Args:
            text: 文档文本（可以是 OCR 结果）

        Returns:
            人员信息列表
        """
        personnel_list = []

        # 1. 从表格中提取（最结构化）
        table_personnel = self._extract_from_tables(text)
        personnel_list.extend(table_personnel)

        # 2. 从标注文本中提取
        text_personnel = self._extract_from_text(text)
        personnel_list.extend(text_personnel)

        # 3. 提取散落的身份证号
        id_cards = self._extract_id_cards(text)
        for id_card, confidence in id_cards:
            # 检查是否已有此身份证号的人员
            existing = [p for p in personnel_list if p.id_card == id_card]
            if not existing:
                # 尝试关联附近的姓名
                name = self._find_nearby_name(text, id_card)
                personnel_list.append(PersonnelInfo(
                    name=name or "未知",
                    id_card=id_card,
                    confidence=confidence
                ))

        # 4. 去重合并
        personnel_list = self._merge_personnel(personnel_list)

        return personnel_list

    def extract_certificates(self, text: str) -> List[CertificateInfo]:
        """提取证书信息（去重）。有效期写入 expiry_date，不作为证书编号。"""
        certificates = []
        seen_numbers = set()

        for pattern, cert_type, confidence in self.CERTIFICATE_PATTERNS:
            for match in pattern.finditer(text):
                cert_number = match.group(1).strip()
                if len(cert_number) < 6 or cert_number in seen_numbers:
                    continue
                if re.search(r'\d{4}[-/年]', cert_number) and "证" not in cert_type:
                    continue
                seen_numbers.add(cert_number)
                cert = CertificateInfo(
                    cert_type=cert_type,
                    cert_number=cert_number,
                )
                cert.holder_name = self._find_nearby_name(
                    text, cert_number, search_range=200
                ) or ""
                nearby = text[max(0, match.start() - 80):match.end() + 120]
                cert.expiry_date = self._find_expiry_in_context(nearby)
                certificates.append(cert)

        return certificates

    def _find_expiry_in_context(self, context: str) -> str:
        for pattern in self.EXPIRY_PATTERNS:
            match = pattern.search(context)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_from_tables(self, text: str) -> List[PersonnelInfo]:
        """从表格行中提取人员信息"""
        personnel = []

        for pattern in self.TABLE_PERSONNEL_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 3:
                    name = groups[0].strip()
                    if len(groups) == 3:
                        # 姓名 | 职务 | 身份证号
                        role = groups[1].strip()
                        id_card = groups[2].strip()
                        personnel.append(PersonnelInfo(
                            name=name, role=role, id_card=id_card,
                            confidence=0.95
                        ))
                    elif len(groups) >= 4:
                        # 姓名 | 身份证号 | 学历 | 专业
                        id_card = groups[1].strip()
                        education = groups[2].strip()
                        major = groups[3].strip()
                        personnel.append(PersonnelInfo(
                            name=name, id_card=id_card,
                            education=education, major=major,
                            confidence=0.95
                        ))

        return personnel

    def _extract_from_text(self, text: str) -> List[PersonnelInfo]:
        """从普通文本中提取人员信息"""
        personnel = []

        for pattern, role in self.NAME_ROLE_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip()
                if self._is_valid_name(name):
                    person = PersonnelInfo(name=name, role=role, confidence=0.85)

                    # 在姓名附近搜索更多信息
                    context = text[max(0, match.start()-100):match.end()+500]
                    person.id_card = self._find_id_in_context(context)
                    person.education = self._find_education_in_context(context)
                    person.major = self._find_major_in_context(context)

                    personnel.append(person)

        return personnel

    def _extract_id_cards(self, text: str) -> List[Tuple[str, float]]:
        """提取所有身份证号"""
        results = []
        seen = set()

        for pattern, confidence, desc in self.ID_CARD_PATTERNS:
            for match in pattern.finditer(text):
                id_card = match.group(1).strip().upper()
                if id_card not in seen and self._validate_id_card(id_card):
                    seen.add(id_card)
                    results.append((id_card, confidence))

        return results

    def _validate_id_card(self, id_card: str) -> bool:
        """验证身份证号码校验位"""
        if len(id_card) != 18:
            return len(id_card) == 15

        # 18位校验
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_chars = '10X98765432'
        try:
            total = sum(int(id_card[i]) * weights[i] for i in range(17))
            expected = check_chars[total % 11]
            return id_card[17].upper() == expected
        except (ValueError, IndexError):
            return False

    def _find_nearby_name(
        self, text: str, target: str, search_range: int = 300
    ) -> Optional[str]:
        """在目标文本附近查找人名"""
        pos = text.find(target)
        if pos == -1:
            return None

        # 向前搜索
        context = text[max(0, pos - search_range):pos]
        # 查找最近的中文姓名
        name_pattern = re.compile(r'([\u4e00-\u9fa5]{2,4})')
        names = name_pattern.findall(context)
        if names:
            # 返回最后一个（最近的）
            candidate = names[-1]
            if self._is_valid_name(candidate):
                return candidate

        return None

    def _find_id_in_context(self, context: str) -> str:
        """在上下文中查找身份证号"""
        pattern = re.compile(
            r'([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])'
        )
        match = pattern.search(context)
        if match:
            id_card = match.group(1).upper()
            if self._validate_id_card(id_card):
                return id_card
        return ""

    def _find_education_in_context(self, context: str) -> str:
        """在上下文中查找学历"""
        pattern = re.compile(r'(博士|硕士|本科|大专|中专|研究生)')
        match = pattern.search(context)
        return match.group(1) if match else ""

    def _find_major_in_context(self, context: str) -> str:
        """在上下文中查找专业"""
        pattern = re.compile(r'(?:专业|所学专业)[：:]\s*([^，。\n]{2,20})')
        match = pattern.search(context)
        return match.group(1).strip() if match else ""

    def _is_valid_name(self, name: str) -> bool:
        """验证是否是有效的中文人名"""
        if not name or len(name) < 2 or len(name) > 4:
            return False
        # 排除常见的非人名词语
        non_names = {
            '公司', '集团', '项目', '工程', '建设', '设计',
            '负责人', '代表人', '经理', '工程师', '技术员',
            '单位', '部门', '机构', '中心', '证书', '资质',
        }
        if name in non_names:
            return False
        # 必须全是中文
        if not re.match(r'^[\u4e00-\u9fa5]+$', name):
            return False
        return True

    def _merge_personnel(self, personnel: List[PersonnelInfo]) -> List[PersonnelInfo]:
        """合并重复的人员记录（按身份证号或姓名去重）"""
        merged = {}

        for person in personnel:
            # 优先用身份证号作为主键
            key = person.id_card if person.id_card else person.name

            if key in merged:
                existing = merged[key]
                # 合并信息（保留非空值）
                if not existing.name or existing.name == "未知":
                    existing.name = person.name
                if not existing.id_card and person.id_card:
                    existing.id_card = person.id_card
                if not existing.role and person.role:
                    existing.role = person.role
                if not existing.education and person.education:
                    existing.education = person.education
                if not existing.major and person.major:
                    existing.major = person.major
                if not existing.graduation_school and person.graduation_school:
                    existing.graduation_school = person.graduation_school
                if person.confidence > existing.confidence:
                    existing.confidence = person.confidence
                existing.certificates.extend(person.certificates)
            else:
                # 检查是否有同名人员已按身份证号存储
                name_exists = False
                for k, v in merged.items():
                    if v.name == person.name and person.name != "未知":
                        # 合并到已有记录
                        if not v.id_card and person.id_card:
                            v.id_card = person.id_card
                        if not v.role and person.role:
                            v.role = person.role
                        if not v.education and person.education:
                            v.education = person.education
                        if not v.major and person.major:
                            v.major = person.major
                        name_exists = True
                        break
                if not name_exists:
                    merged[key] = person

        return list(merged.values())

    def format_personnel_summary(self, personnel: List[PersonnelInfo]) -> str:
        """格式化人员信息摘要"""
        if not personnel:
            return "未找到人员信息"

        lines = []
        for i, person in enumerate(personnel, 1):
            line = f"{i}. {person.name}"
            if person.role:
                line += f" ({person.role})"
            if person.id_card:
                # 脱敏显示
                masked = person.id_card[:6] + "****" + person.id_card[-4:]
                line += f" 身份证:{masked}"
            if person.education:
                line += f" {person.education}"
            if person.major:
                line += f"/{person.major}"
            lines.append(line)

        return "\n".join(lines)
