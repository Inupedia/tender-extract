"""
tender-extract: 面向中文标书的混合抽取流水线

架构：解析层 → 切块层 → 路由层 → 抽取层(Regex+NER+LLM) → 合并层
"""

__version__ = "0.2.0"
__author__ = "Tender Extract Team"

from .schema import (
    EvidenceSpan,
    ExtractedField,
    DocumentMetadata,
    ExtractionResult,
    ProcessingConfig
)

from .preprocess import MarkdownPreprocessor
from .chunker import DocumentChunker, ChunkingConfig
from .rules import RuleExtractor
from .ner import NERExtractor
from .dedupe import DeduplicationEngine
from .llm_router import LLMRouter
from .merge import FieldMerger
from .cli import TenderExtractor
from .document_parser import DocumentParser, ParsedDocument
from .module_router import ModuleRouter, RoutedChunk, TENDER_MODULES
from .extraction_engine import ExtractionEngine
from .patterns import FIELD_PATTERNS

__all__ = [
    "EvidenceSpan",
    "ExtractedField",
    "DocumentMetadata",
    "ExtractionResult",
    "ProcessingConfig",
    "MarkdownPreprocessor",
    "DocumentChunker",
    "ChunkingConfig",
    "RuleExtractor",
    "NERExtractor",
    "DeduplicationEngine",
    "LLMRouter",
    "FieldMerger",
    "TenderExtractor",
    # New modules
    "DocumentParser",
    "ParsedDocument",
    "ModuleRouter",
    "RoutedChunk",
    "TENDER_MODULES",
    "ExtractionEngine",
    "FIELD_PATTERNS",
] 