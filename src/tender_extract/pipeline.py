"""统一抽取流水线：解析 → 切块 → 模块路由 → 规则抽取 → 按需 LLM → 合并。"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .chunker import ChunkingConfig, DocumentChunker
from .configurable_engine import ConfigurableExtractionEngine
from .dedupe import DeduplicationEngine
from .document_parser import DocumentParser
from .field_registry import get_expected_fields_for_modules
from .llm_router import LLMRouter
from .merge import FieldMerger
from .module_router import ModuleRouter, RoutedChunk
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
        self.engine = ConfigurableExtractionEngine(config.custom_patterns)
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
        chunks, deduped_count = self._dedupe_chunks(chunks)

        routed: list[RoutedChunk] = []
        routing_summary = {"module_distribution": {}}
        if self.config.use_modules:
            routed = self.router.route_chunks(chunks)
            routing_summary = self.router.get_routing_summary(routed)

        # 全文规则抽取保证关键标注不因切块边界丢失；YAML 规则已叠加到 active engine。
        fields = self.engine.extract_all_fields(content)

        # 模块补抽只执行该模块真实存在的规则；不能用 None 回退为“全规则”。
        for item in routed:
            targets = self.router.get_module_target_fields(item.module_id)
            known = [name for name in targets if name in self.engine._compiled_patterns]
            if not known:
                continue
            chunk_fields = self.engine.extract_all_fields(item.chunk.content, target_fields=known)
            offset = _chunk_offset(content, item.chunk)
            _lift_offsets(chunk_fields, offset, content)
            fields = _merge_field_maps(fields, chunk_fields)

        if self.ner:
            ner_fields = self.ner.extract_entities(content, content)
            fields = self.ner.merge_with_rules(ner_fields, fields)

        personnel = self.personnel_extractor.extract_personnel(content)
        certificates = self.personnel_extractor.extract_certificates(content)

        low_conf = self.engine.get_low_confidence_fields(fields, self.config.confidence_threshold)
        llm_calls = 0

        # 已命中但低置信/冲突：LLM 复核。
        if self.llm.is_enabled():
            for name in list(low_conf):
                field = fields.get(name)
                if field is None or not self.llm.should_use_llm(field, self.config.confidence_threshold):
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
                    fields[name] = self.llm.merge_llm_results(field, response, context)
                    llm_calls += 1
                else:
                    warnings.append(f"LLM 未返回字段 {name}")

        # 规则完全未命中的字段：由 Router/Field Registry 给出明确 extraction plan，再让 LLM 恢复。
        module_ids = {item.module_id for item in routed if item.module_id != "general"}
        expected_fields = get_expected_fields_for_modules(module_ids)
        missing_fields = sorted(expected_fields.difference(fields))
        if self.llm.is_enabled() and self.config.recover_missing_fields_with_llm:
            for name in missing_fields:
                if not self.llm.should_recover_missing(name):
                    continue
                context = _missing_field_context(routed, self.router, name, content)
                response = self.llm.extract_with_llm(
                    LLMRequest(
                        chunk_text=context,
                        field_name=name,
                        field_type=name,
                        existing_values=[],
                    )
                )
                if not response or not response.extracted_values:
                    continue
                placeholder = ExtractedField(field_name=name, field_type=name)
                fields[name] = self.llm.merge_llm_results(placeholder, response, context)
                llm_calls += 1

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
            "missing_fields_considered": len(missing_fields),
            "modules_used": list(routing_summary.get("module_distribution", {}).keys()),
            "deduped_chunks": deduped_count,
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

    def _dedupe_chunks(self, chunks: list[ChunkInfo]) -> tuple[list[ChunkInfo], int]:
        if not self.config.enable_dedupe or len(chunks) < 2:
            return chunks, 0
        engine = DeduplicationEngine(enable_lsh=self.config.enable_similarity_check)
        results = engine.process_chunks(chunks)
        kept = [chunk for chunk, result in zip(chunks, results) if not result.is_duplicate]
        return kept, len(chunks) - len(kept)


def _missing_field_context(
    routed: list[RoutedChunk], router: ModuleRouter, field_name: str, full_content: str
) -> str:
    pieces: list[str] = []
    seen: set[str] = set()
    for item in routed:
        if field_name not in router.get_module_target_fields(item.module_id):
            continue
        text = item.chunk.content.strip()
        if text and text not in seen:
            pieces.append(text)
            seen.add(text)
        if sum(len(piece) for piece in pieces) >= 2400:
            break
    return "\n---\n".join(pieces)[:2400] or full_content[:1200]


def _chunk_offset(full_content: str, chunk: ChunkInfo) -> int:
    # 优先使用 chunker 给出的行位置作为搜索锚点，降低重复模板总是命中第一次出现的概率。
    lines = full_content.split("\n")
    anchor = 0
    if chunk.start_line > 1 and chunk.start_line <= len(lines):
        anchor = sum(len(line) + 1 for line in lines[: chunk.start_line - 1])
    pos = full_content.find(chunk.content, anchor)
    if pos != -1:
        return pos
    pos = full_content.find(chunk.content)
    return pos if pos != -1 else anchor


def _lift_offsets(fields: dict[str, ExtractedField], offset: int, full_content: str) -> None:
    if offset <= 0:
        return
    for field in fields.values():
        for span in field.values:
            if span.start < 0 or span.end < 0:
                continue
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
        by_key = {(v.value, v.start, v.end, v.source): v for v in existing.values}
        for value in field.values:
            by_key.setdefault((value.value, value.start, value.end, value.source), value)
        existing.values = sorted(by_key.values(), key=lambda x: x.confidence, reverse=True)
        if existing.values:
            existing.primary_value = existing.values[0].value
            existing.confidence = existing.values[0].confidence
        existing.conflicts = list(dict.fromkeys([*existing.conflicts, *field.conflicts]))
    return merged
