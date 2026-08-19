"""
tender-extract: 面向中文标书的混合抽取流水线

架构：解析层 → 切块层 → 路由层 → 抽取层(Regex+NER+LLM) → 合并层
"""

__version__ = "0.3.0"
__author__ = "Tender Extract Team"

from .schema import (
    EvidenceSpan,
    ExtractedField,
    DocumentMetadata,
    ExtractionResult,
    ProcessingConfig,
    PersonnelRecord,
    CertificateRecord,
)
from .preprocess import MarkdownPreprocessor
from .chunker import DocumentChunker, ChunkingConfig
from .rules import RuleExtractor
from .ner import NERExtractor
from .dedupe import DeduplicationEngine
from .llm_router import LLMRouter
from .merge import FieldMerger
from .pipeline import ExtractionPipeline
from .document_parser import DocumentParser, ParsedDocument
from .module_router import ModuleRouter, RoutedChunk, TENDER_MODULES
from .extraction_engine import ExtractionEngine
from .patterns import FIELD_PATTERNS
from .personnel_extractor import PersonnelExtractor, PersonnelInfo, CertificateInfo

TenderExtractor = ExtractionPipeline

__all__ = [
    "EvidenceSpan",
    "ExtractedField",
    "DocumentMetadata",
    "ExtractionResult",
    "ProcessingConfig",
    "PersonnelRecord",
    "CertificateRecord",
    "MarkdownPreprocessor",
    "DocumentChunker",
    "ChunkingConfig",
    "RuleExtractor",
    "NERExtractor",
    "DeduplicationEngine",
    "LLMRouter",
    "FieldMerger",
    "TenderExtractor",
    "ExtractionPipeline",
    "DocumentParser",
    "ParsedDocument",
    "ModuleRouter",
    "RoutedChunk",
    "TENDER_MODULES",
    "ExtractionEngine",
    "FIELD_PATTERNS",
    "PersonnelExtractor",
    "PersonnelInfo",
    "CertificateInfo",
]
