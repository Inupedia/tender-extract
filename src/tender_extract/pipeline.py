"""统一抽取流水线：解析 → 切块 → 模块路由 → 规则抽取 → 按需 LLM → 合并。"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .chunker import ChunkingConfig, DocumentChunker
from .document_parser import DocumentParser
from .extraction_engine import ExtractionEngine
from .llm_router import LLMRouter
from .merge import FieldMerger
from .module_router import ModuleRouter
from .ner import NERExtractor
from .personnel_extractor import PersonnelExtractor
from .pii import mask_result
from .schema import (
    CertificateRecord,
    ChunkInfo,
    DocumentMetadata,
    ExtractedField,
    ExtractionResult,
    LLMRequest,
    PersonnelRecord,
    ProcessingConfig,
)

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """单文档抽取。每次调用独立持有人员/证书，不会跨文件串数据。"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.parser = DocumentParser(use_ocr=config.use_ocr)
        self.chunker = DocumentChunker(
            ChunkingConfig(
                max_tokens=config.max_chunk_tokens,
                overlap_tokens=config.overlap_tokens,
                min_chunk_tokens=200,
            )
        )
        self.router = ModuleRouter()
        self.engine = ExtractionEngine()
        self.personnel_extractor = PersonnelExtractor()
        self.merger = FieldMerger()
        self.llm = LLMRouter(config)
        self.ner = NERExtractor(use_foolnltk=config.use_ner) if config.use_ner else None

    def extract_file(self, file_path: str) -> ExtractionResult:
        started = time.time()
        warnings: list[str] = []
        errors: list[str] = []
        path = Path(file_path)

        parsed = self.parser.parse(str(path))
        content = parsed.content
        chunks = self.chunker.chunk_document(content, parsed.filename)

        routed = []
        routing_summary = {"module_distribution": {}}
        if self.config.use_modules:
            routed = self.router.route_chunks(chunks)
            routing_summary = self.router.get_routing_summary(routed)

        # 1) 全文抽取，避免切块切断「项目名称：」这类标注
        fields = self.engine.extract_all_fields(content)

        # 2) 按模块对切片补抽目标字段，路由结果真正参与抽取
        if routed:
            for item in routed:
                targets = self.router.get_module_target_fields(item.module_id)
                known = [name for name in targets if name in self.engine._compiled_patterns]
                chunk_fields = self.engine.extract_all_fields(
                    item.chunk.content,
                    target_fields=known or None,
                )
                offset = _chunk_offset(content, item.chunk)
                _lift_offsets(chunk_fields, offset, content)
                fields = _merge_field_maps(fields, chunk_fields)

        if self.ner:
            ner_fields = self.ner.extract_entities(content, content)
            fields = self.ner.merge_with_rules(ner_fields, fields)

        personnel = self.personnel_extractor.extract_personnel(content)
        certificates = self.personnel_extractor.extract_certificates(content)

        low_conf = self.engine.get_low_confidence_fields(
            fields, self.config.confidence_threshold
        )
        llm_calls = 0
        if self.llm.is_enabled() and low_conf:
            for name in low_conf:
                field = fields.get(name)
                if field is None or not self.llm.should_use_llm(
                    field, self.config.confidence_threshold
                ):
                    continue
                context = self.llm.get_minimal_evidence_context(field, content)
                response = self.llm.extract_with_llm(
                    LLMRequest(
                        chunk_text=context,
                        field_name=name,
                        field_type=field.field_type,
                        existing_values=[v.value for v in field.values[:5]],
                    )
                )
                if response:
                    fields[name] = self.llm.merge_llm_results(field, response)
                    llm_calls += 1
                else:
                    warnings.append(f"LLM 未返回字段 {name}")

        fields = self.merger.resolve_conflicts(fields)
        fields = self.engine._post_process(fields)

        personnel_records = [
            PersonnelRecord(
                name=p.name,
                role=p.role,
                id_card=p.id_card,
                education=p.education,
                major=p.major,
                graduation_school=p.graduation_school,
                graduation_date=p.graduation_date,
                certificates=p.certificates,
                contact=p.contact,
                confidence=p.confidence,
            )
            for p in personnel
        ]
        certificate_records = [
            CertificateRecord(
                cert_type=c.cert_type,
                cert_number=c.cert_number,
                holder_name=c.holder_name,
                issue_date=c.issue_date,
                expiry_date=c.expiry_date,
                issuer=c.issuer,
                level=c.level,
                major=c.major,
            )
            for c in certificates
        ]

        stats = {
            "total_fields": len(fields),
            "fields_by_type": {name: 1 for name in fields},
            "avg_confidence": (
                sum(f.confidence for f in fields.values()) / len(fields) if fields else 0.0
            ),
            "low_confidence_count": len(low_conf),
            "modules_used": list(routing_summary.get("module_distribution", {}).keys()),
            "personnel_count": len(personnel_records),
            "certificates_count": len(certificate_records),
            "original_format": parsed.original_format,
            "ocr_pages": (parsed.metadata or {}).get("ocr_pages", 0),
        }

        result = ExtractionResult(
            metadata=DocumentMetadata(
                filename=path.name,
                file_size=path.stat().st_size,
                total_lines=len(content.split("\n")),
                total_chunks=len(chunks),
                processing_time=time.time() - started,
                extraction_stats=stats,
            ),
            fields=fields,
            personnel=personnel_records,
            certificates=certificate_records,
            chunks_processed=len(chunks),
            llm_calls=llm_calls,
            cache_hits=self.llm.cache_hits,
            errors=errors,
            warnings=warnings,
        )
        if not self.config.include_pii:
            result = mask_result(result)
        if self.config.persist_llm_cache and llm_calls:
            self.llm.save_cache()
        return result


def _chunk_offset(full_content: str, chunk: ChunkInfo) -> int:
    pos = full_content.find(chunk.content)
    if pos != -1:
        return pos
    if chunk.start_line > 1:
        lines = full_content.split("\n")
        if chunk.start_line <= len(lines):
            return sum(len(line) + 1 for line in lines[: chunk.start_line - 1])
    return 0


def _lift_offsets(fields: dict[str, ExtractedField], offset: int, full_content: str) -> None:
    if offset <= 0:
        return
    for field in fields.values():
        for span in field.values:
            span.start += offset
            span.end += offset
            if span.end <= len(full_content):
                start = max(0, span.start - 80)
                end = min(len(full_content), span.end + 80)
                span.ref = full_content[start:end].strip()


def _merge_field_maps(
    left: dict[str, ExtractedField], right: dict[str, ExtractedField]
) -> dict[str, ExtractedField]:
    merged = dict(left)
    for name, field in right.items():
        if name not in merged:
            merged[name] = field
            continue
        existing = merged[name]
        existing.values.extend(field.values)
        existing.values.sort(key=lambda x: x.confidence, reverse=True)
        existing.primary_value = existing.values[0].value
        existing.confidence = existing.values[0].confidence
        existing.conflicts = list({*existing.conflicts, *field.conflicts})
    return merged
