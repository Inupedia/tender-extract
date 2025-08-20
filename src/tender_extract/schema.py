"""
Pydantic 输出模型定义
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class EvidenceSpan(BaseModel):
    """证据片段，包含字段值和引用定位"""
    value: str = Field(..., description="提取的字段值")
    start: int = Field(..., description="在原文中的起始位置")
    end: int = Field(..., description="在原文中的结束位置")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    source: str = Field(..., description="来源：rule/ner/llm")
    pattern: Optional[str] = Field(None, description="匹配的正则表达式或模式")
    ref: Optional[str] = Field(None, description="对应的完整引用段落")


class ExtractedField(BaseModel):
    """提取的字段"""
    field_name: str = Field(..., description="字段名称")
    field_type: str = Field(..., description="字段类型")
    values: List[EvidenceSpan] = Field(default_factory=list, description="提取的值列表")
    primary_value: Optional[str] = Field(None, description="主要值（置信度最高的）")
    confidence: float = Field(0.0, description="整体置信度", ge=0.0, le=1.0)
    conflicts: List[str] = Field(default_factory=list, description="冲突信息")


class DocumentMetadata(BaseModel):
    """文档元数据"""
    filename: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小（字节）")
    total_lines: int = Field(..., description="总行数")
    total_chunks: int = Field(..., description="总切片数")
    processing_time: float = Field(..., description="处理时间（秒）")
    extraction_stats: Dict[str, Any] = Field(default_factory=dict, description="抽取统计")


class ExtractionResult(BaseModel):
    """抽取结果"""
    metadata: DocumentMetadata = Field(..., description="文档元数据")
    fields: Dict[str, ExtractedField] = Field(default_factory=dict, description="提取的字段")
    chunks_processed: int = Field(0, description="处理的切片数")
    llm_calls: int = Field(0, description="LLM调用次数")
    cache_hits: int = Field(0, description="缓存命中次数")
    errors: List[str] = Field(default_factory=list, description="错误信息")
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LLMRequest(BaseModel):
    """LLM请求模型"""
    chunk_text: str = Field(..., description="文本片段")
    field_name: str = Field(..., description="字段名称")
    field_type: str = Field(..., description="字段类型")
    context: Optional[str] = Field(None, description="上下文信息")
    existing_values: List[str] = Field(default_factory=list, description="已存在的值")


class LLMResponse(BaseModel):
    """LLM响应模型"""
    field_name: str = Field(..., description="字段名称")
    extracted_values: List[str] = Field(default_factory=list, description="提取的值")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, description="推理过程")
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list, description="证据片段")


class ChunkInfo(BaseModel):
    """切片信息"""
    chunk_id: str = Field(..., description="切片ID")
    content: str = Field(..., description="切片内容")
    start_line: int = Field(..., description="起始行号")
    end_line: int = Field(..., description="结束行号")
    chapter_path: List[str] = Field(default_factory=list, description="章节路径")
    token_count: int = Field(0, description="token数量")
    fingerprint: str = Field("", description="内容指纹")


class ProcessingConfig(BaseModel):
    """处理配置"""
    use_ner: bool = Field(False, description="是否使用NER")
    llm_provider: str = Field("none", description="LLM提供商：none/ollama/openai")
    llm_model: Optional[str] = Field(None, description="LLM模型名称")
    confidence_threshold: float = Field(0.7, description="置信度阈值")
    max_chunk_tokens: int = Field(800, description="最大切片token数")
    overlap_tokens: int = Field(100, description="重叠token数")
    cache_dir: str = Field(".cache", description="缓存目录")
    enable_dedupe: bool = Field(True, description="是否启用去重")
    enable_similarity_check: bool = Field(True, description="是否启用相似度检查") 