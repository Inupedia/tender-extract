"""Pydantic 输出模型。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """PDF 页面坐标，单位为 PDF points，原点位于页面左上角。"""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    coordinate_system: str = "pdf_points_top_left"


class EvidenceLocation(BaseModel):
    """证据在源文档中的结构化定位；无法可靠确定的维度保持为空。"""

    document_id: str
    page: Optional[int] = Field(None, ge=1, description="PDF 物理页码，1-based")
    section_path: list[str] = Field(default_factory=list)
    line_start: Optional[int] = Field(None, ge=1)
    line_end: Optional[int] = Field(None, ge=1)
    bbox: Optional[BoundingBox] = None
    source_text: Optional[str] = None
    source_start: int = Field(-1, description="统一 Markdown 中的起始字符位置")
    source_end: int = Field(-1, description="统一 Markdown 中的结束字符位置")


class EvidenceSpan(BaseModel):
    """证据片段，包含字段值和原文定位。"""

    model_config = ConfigDict(extra="ignore")

    value: str = Field(..., description="提取的字段值")
    start: int = Field(..., description="在当前文本中的起始位置；无法可靠定位时为 -1")
    end: int = Field(..., description="在当前文本中的结束位置；无法可靠定位时为 -1")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    source: str = Field(..., description="来源：regex/ner/llm")
    pattern: Optional[str] = Field(None, description="匹配的模式说明")
    ref: Optional[str] = Field(None, description="对应的原文片段")
    unit: Optional[str] = Field(None, description="金额等单位，如 元/万元")
    normalized_value: Optional[str] = Field(None, description="规范化值，金额为人民币元")
    location: Optional[EvidenceLocation] = Field(None, description="源文档结构化定位")


class ExtractedField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field_name: str
    field_type: str
    values: list[EvidenceSpan] = Field(default_factory=list)
    primary_value: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    conflicts: list[str] = Field(default_factory=list)


class PersonnelRecord(BaseModel):
    name: str
    role: str = ""
    id_card: str = ""
    education: str = ""
    major: str = ""
    graduation_school: str = ""
    graduation_date: str = ""
    certificates: list[dict[str, str]] = Field(default_factory=list)
    contact: str = ""
    confidence: float = 0.0


class CertificateRecord(BaseModel):
    cert_type: str
    cert_number: str
    holder_name: str = ""
    issue_date: str = ""
    expiry_date: str = ""
    issuer: str = ""
    level: str = ""
    major: str = ""


class DocumentMetadata(BaseModel):
    filename: str
    file_size: int
    total_lines: int
    total_chunks: int
    total_pages: int = Field(0, ge=0, description="源文档物理页数；非分页格式为0")
    processing_time: float
    extraction_stats: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: DocumentMetadata
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    personnel: list[PersonnelRecord] = Field(default_factory=list)
    certificates: list[CertificateRecord] = Field(default_factory=list)
    chunks_processed: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LLMRequest(BaseModel):
    chunk_text: str
    field_name: str
    field_type: str
    context: Optional[str] = None
    existing_values: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    field_name: str
    extracted_values: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


class ChunkInfo(BaseModel):
    chunk_id: str
    content: str
    start_line: int
    end_line: int
    chapter_path: list[str] = Field(default_factory=list)
    token_count: int = 0
    fingerprint: str = ""


class ProcessingConfig(BaseModel):
    """运行时配置。CLI 参数覆盖文件配置。"""

    model_config = ConfigDict(extra="ignore")

    use_ner: bool = False
    llm_provider: str = "none"
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    confidence_threshold: float = 0.7
    max_chunk_tokens: int = 800
    overlap_tokens: int = 100
    cache_dir: str = ".cache"
    enable_dedupe: bool = False
    enable_similarity_check: bool = False
    use_modules: bool = True
    include_pii: bool = False
    use_ocr: bool = True
    debug: bool = False
    persist_llm_cache: bool = True
    custom_patterns: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    recover_missing_fields_with_llm: bool = True
    redact_pii_for_cloud_llm: bool = True
