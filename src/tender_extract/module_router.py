"""
模块化路由层：按章节内容将文档片段路由到专业抽取模块

成熟方案核心思想（TextIn方案）：
1. 不要全文直抽 — 先按标题切块
2. 按关键词将每个块路由到对应的模块
3. 每个模块独立定义输入/输出 Schema
4. 模块之间互不干扰，便于扩展和调优
"""
import re
import logging
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from .schema import ChunkInfo

logger = logging.getLogger(__name__)


@dataclass
class ModuleDefinition:
    """模块定义"""
    module_id: str
    name: str
    description: str
    keywords: List[str]  # 路由关键词
    target_fields: List[str]  # 该模块负责抽取的字段
    priority: int = 5  # 优先级，数字越小优先级越高


@dataclass
class RoutedChunk:
    """路由后的切片"""
    chunk: ChunkInfo
    module_id: str
    module_name: str
    match_score: float  # 路由匹配分数
    matched_keywords: List[str] = field(default_factory=list)


# 预定义的标书模块
TENDER_MODULES = [
    ModuleDefinition(
        module_id="basic_info",
        name="基础信息",
        description="项目基本信息、招标公告、项目概况",
        keywords=[
            "项目名称", "项目编号", "招标编号", "工程名称",
            "项目概况", "招标公告", "招标范围", "项目背景",
            "建设地点", "工程地点", "项目地址", "建设规模",
            "工期", "工期要求", "计划工期", "合同工期",
        ],
        target_fields=[
            "project_name", "project_number", "project_scope",
            "construction_site", "project_duration"
        ],
        priority=1
    ),
    ModuleDefinition(
        module_id="bidder_info",
        name="投标人信息",
        description="投标人/招标人基本信息、联系方式",
        keywords=[
            "投标人", "投标单位", "投标方", "投标公司",
            "招标人", "招标单位", "招标方", "招标代理",
            "法定代表人", "法人代表", "授权代表",
            "联系人", "联系电话", "联系方式", "通讯地址",
            "传真", "邮箱", "邮编", "地址",
        ],
        target_fields=[
            "bidder", "tenderer", "legal_representative",
            "contact_info", "address"
        ],
        priority=2
    ),
    ModuleDefinition(
        module_id="financial_info",
        name="财务信息",
        description="金额、报价、保证金等财务相关",
        keywords=[
            "投标报价", "投标金额", "投标总报价", "报价",
            "投标保证金", "履约保证金", "保证金",
            "招标控制价", "预算金额", "项目金额", "合同金额",
            "人民币", "万元", "元整",
            "大写", "小写", "金额",
            "付款方式", "付款条件", "支付方式",
        ],
        target_fields=[
            "bid_amount", "deposit", "control_price",
            "budget_amount", "payment_terms"
        ],
        priority=2
    ),
    ModuleDefinition(
        module_id="qualification",
        name="资格要求",
        description="投标人资质、资格条件、证照要求",
        keywords=[
            "资格条件", "资质要求", "投标人资格",
            "营业执照", "统一社会信用代码", "注册资本",
            "成立日期", "经营范围", "业务范围",
            "资质证书", "资格证书", "等级证书",
            "资质等级", "施工资质", "设计资质",
            "安全生产许可证", "质量管理体系",
            "ISO", "认证",
        ],
        target_fields=[
            "business_license", "registered_capital",
            "establishment_date", "business_scope",
            "qualification_cert"
        ],
        priority=3
    ),
    ModuleDefinition(
        module_id="evaluation",
        name="评标办法",
        description="评标方法、评分标准、评审规则",
        keywords=[
            "评标办法", "评标方法", "评审办法",
            "评分标准", "评标标准", "评审标准",
            "评分项", "分值", "权重", "得分",
            "技术评分", "商务评分", "综合评分",
            "最低评标价", "综合评估法", "经评审的最低投标价法",
            "评标委员会", "评标专家",
        ],
        target_fields=[
            "evaluation_method", "scoring_criteria",
            "technical_score", "commercial_score"
        ],
        priority=4
    ),
    ModuleDefinition(
        module_id="submission",
        name="投标递交",
        description="投标文件递交、开标时间、截止时间",
        keywords=[
            "投标文件", "递交", "投标截止",
            "开标时间", "开标地点", "投标截止时间",
            "投标有效期", "响应文件", "递交截止",
            "投标日期", "投标时间",
        ],
        target_fields=[
            "bid_date", "bid_deadline", "opening_date",
            "validity_period"
        ],
        priority=3
    ),
    ModuleDefinition(
        module_id="personnel",
        name="人员要求",
        description="项目负责人、技术人员、团队要求、身份证、证书",
        keywords=[
            "项目经理", "项目负责人", "技术负责人",
            "总工程师", "安全员", "质检员", "施工员", "造价师",
            "拟派人员", "主要人员", "团队", "人员配备",
            "职称", "资格证书", "执业资格",
            "建造师", "工程师", "技术员",
            "身份证", "证件号", "学历", "毕业",
            "安全生产考核", "安全B证", "安全C证",
            "注册证书", "执业证书", "从业资格",
            "社保", "社会保险",
        ],
        target_fields=[
            "project_manager", "technical_staff",
            "team_requirements", "personnel_info"
        ],
        priority=3
    ),
    ModuleDefinition(
        module_id="company_info",
        name="企业信息",
        description="公司注册、股东、关联企业等围串标检测信息",
        keywords=[
            "股东", "持股比例", "股权结构", "实际控制人",
            "子公司", "关联公司", "控股", "参股",
            "注册地址", "公司地址", "办公地址",
            "银行账户", "开户行", "账号",
            "财务状况", "资产", "负债",
        ],
        target_fields=[
            "shareholder_info", "subsidiary_info",
            "registered_address", "bank_account",
            "financial_info"
        ],
        priority=5
    ),
    ModuleDefinition(
        module_id="performance",
        name="业绩记录",
        description="类似项目业绩、工程经验",
        keywords=[
            "业绩", "项目业绩", "工程业绩",
            "类似项目", "类似工程", "同类项目",
            "合同业绩", "中标业绩", "施工业绩",
            "业绩证明", "竣工验收",
        ],
        target_fields=[
            "performance_record", "similar_projects"
        ],
        priority=5
    ),
]


class ModuleRouter:
    """
    模块路由器

    核心策略：
    1. 先对每个 chunk 进行关键词扫描
    2. 根据匹配分数路由到最佳模块
    3. 一个 chunk 可以路由到多个模块（如果内容跨模块）
    4. 未匹配任何模块的 chunk 进入通用抽取
    """

    def __init__(self, modules: Optional[List[ModuleDefinition]] = None):
        self.modules = modules or TENDER_MODULES
        self._build_keyword_index()

    def _build_keyword_index(self):
        """构建关键词到模块的索引"""
        self.keyword_to_modules: Dict[str, List[str]] = {}
        for module in self.modules:
            for keyword in module.keywords:
                if keyword not in self.keyword_to_modules:
                    self.keyword_to_modules[keyword] = []
                self.keyword_to_modules[keyword].append(module.module_id)

    def route_chunks(self, chunks: List[ChunkInfo]) -> List[RoutedChunk]:
        """
        将切片路由到对应的模块

        Args:
            chunks: 文档切片列表

        Returns:
            路由后的切片列表（一个 chunk 可能出现多次，分属不同模块）
        """
        routed_chunks = []

        for chunk in chunks:
            routes = self._route_single_chunk(chunk)
            if routes:
                routed_chunks.extend(routes)
            else:
                # 未匹配任何模块，分配到通用模块
                routed_chunks.append(RoutedChunk(
                    chunk=chunk,
                    module_id="general",
                    module_name="通用抽取",
                    match_score=0.0,
                    matched_keywords=[]
                ))

        logger.info(
            f"路由完成: {len(chunks)} 个切片 → {len(routed_chunks)} 个路由任务"
        )
        return routed_chunks

    def _route_single_chunk(self, chunk: ChunkInfo) -> List[RoutedChunk]:
        """对单个切片进行路由"""
        module_scores: Dict[str, tuple] = {}  # module_id -> (score, matched_keywords)

        text = chunk.content
        # 同时检查章节路径
        chapter_text = " ".join(chunk.chapter_path)
        combined_text = f"{chapter_text} {text}"

        for module in self.modules:
            score, matched = self._calculate_module_score(combined_text, module)
            if score > 0:
                module_scores[module.module_id] = (score, matched)

        if not module_scores:
            return []

        # 选择得分最高的模块（可以多选，如果多个模块得分接近）
        max_score = max(s for s, _ in module_scores.values())
        threshold = max_score * 0.6  # 得分超过最高分60%的模块都算匹配

        routes = []
        for module_id, (score, matched) in module_scores.items():
            if score >= threshold:
                module = self._get_module(module_id)
                routes.append(RoutedChunk(
                    chunk=chunk,
                    module_id=module_id,
                    module_name=module.name if module else module_id,
                    match_score=score,
                    matched_keywords=matched
                ))

        # 按匹配分数排序
        routes.sort(key=lambda r: r.match_score, reverse=True)
        return routes

    def _calculate_module_score(
        self, text: str, module: ModuleDefinition
    ) -> tuple:
        """计算文本与模块的匹配分数"""
        matched_keywords = []
        score = 0.0

        for keyword in module.keywords:
            count = text.count(keyword)
            if count > 0:
                matched_keywords.append(keyword)
                # 关键词越长越精确，给更高权重
                keyword_weight = min(len(keyword) / 4, 3.0)
                score += count * keyword_weight

        # 用模块优先级加权（高优先级模块略微加分）
        priority_bonus = (10 - module.priority) / 10.0
        score *= (1 + priority_bonus * 0.1)

        return score, matched_keywords

    def _get_module(self, module_id: str) -> Optional[ModuleDefinition]:
        """获取模块定义"""
        for module in self.modules:
            if module.module_id == module_id:
                return module
        return None

    def get_module_target_fields(self, module_id: str) -> List[str]:
        """获取模块的目标字段列表"""
        module = self._get_module(module_id)
        if module:
            return module.target_fields
        return []

    def get_routing_summary(self, routed_chunks: List[RoutedChunk]) -> Dict[str, Any]:
        """获取路由统计摘要"""
        module_counts: Dict[str, int] = {}
        module_keywords: Dict[str, Set[str]] = {}

        for rc in routed_chunks:
            module_counts[rc.module_name] = module_counts.get(rc.module_name, 0) + 1
            if rc.module_name not in module_keywords:
                module_keywords[rc.module_name] = set()
            module_keywords[rc.module_name].update(rc.matched_keywords)

        return {
            'total_routes': len(routed_chunks),
            'module_distribution': module_counts,
            'top_keywords_per_module': {
                k: list(v)[:5] for k, v in module_keywords.items()
            }
        }
