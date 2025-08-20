"""
tender-extract: 面向中文标书的混合抽取流水线
"""

__version__ = "0.1.0"
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
    "TenderExtractor"
] 