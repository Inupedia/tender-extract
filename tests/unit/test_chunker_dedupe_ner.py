import pytest

import tender_extract.ner as ner_module
from tender_extract.chunker import ChunkingConfig, DocumentChunker
from tender_extract.dedupe import DeduplicationEngine, SimilarityResult
from tender_extract.ner import NERExtractor
from tender_extract.preprocess import ChapterNode
from tender_extract.schema import ChunkInfo, EvidenceSpan, ExtractedField


pytestmark = pytest.mark.unit


def _chunk(
    chunk_id: str,
    content: str,
    *,
    token_count: int | None = None,
    chapter_path: list[str] | None = None,
    fingerprint: str = "",
    start_line: int = 1,
    end_line: int = 1,
) -> ChunkInfo:
    return ChunkInfo(
        chunk_id=chunk_id,
        content=content,
        start_line=start_line,
        end_line=end_line,
        chapter_path=chapter_path or [],
        token_count=token_count if token_count is not None else max(len(content) // 4, 1),
        fingerprint=fingerprint,
    )


def _span(value: str, confidence: float, start: int = 0) -> EvidenceSpan:
    return EvidenceSpan(
        value=value,
        start=start,
        end=start + len(value),
        confidence=confidence,
        source="regex",
    )


def test_chunker_fallback_splits_large_plain_document_and_adds_fingerprints():
    chunker = DocumentChunker(
        ChunkingConfig(
            max_tokens=4,
            overlap_tokens=0,
            use_langchain=False,
            chapter_priority=False,
        )
    )
    content = (
        "这是第一段较长的内容用于触发基础章节切片。\n\n"
        "这是第二段较长的内容用于继续切片。"
    )

    chunks = chunker.chunk_document(content, "fallback.md")

    assert len(chunks) >= 2
    assert all(chunk.fingerprint for chunk in chunks)
    assert all(chunk.chunk_id.startswith("fallback.md_") for chunk in chunks)


def test_chunker_split_merge_statistics_and_reference_helpers():
    chunker = DocumentChunker(
        ChunkingConfig(max_tokens=4, overlap_tokens=0, use_langchain=False)
    )
    node = ChapterNode(
        title="测试章",
        level=1,
        start_line=2,
        end_line=8,
        content="第一段内容比较长需要切片。\n\n第二段内容同样比较长需要切片。",
        children=[],
    )

    split = chunker._split_large_chapter(node, ["测试章"], "doc", 3)
    assert len(split) >= 2
    assert split[0].chunk_id == "doc_0003"
    assert split[0].chapter_path == ["测试章"]

    mergeable = [
        _chunk("a", "甲", token_count=1, chapter_path=["A"], start_line=1, end_line=1),
        _chunk("b", "乙", token_count=2, chapter_path=["A"], start_line=2, end_line=2),
        _chunk("c", "丙", token_count=4, chapter_path=["B"], start_line=3, end_line=3),
    ]
    merged = chunker.merge_small_chunks(mergeable)
    assert len(merged) == 2
    assert "乙" in merged[0].content
    assert merged[0].end_line == 2
    assert merged[0].fingerprint

    stats = chunker.get_chunk_statistics(merged)
    assert stats["total_chunks"] == 2
    assert stats["min_tokens"] <= stats["max_tokens"]
    assert stats["unique_chapter_paths"] == 2
    assert chunker.get_chunk_statistics([]) == {}

    full = "前言\n\n项目名称：测试水库工程，建设地点成都。\n联系人张三。\n\n附录"
    start = full.index("测试水库工程")
    end = start + len("测试水库工程")
    reference = chunker.find_reference_paragraph(full, start, end)
    assert "项目名称" in reference
    assert "测试水库工程" in reference
    assert chunker.find_reference_paragraph(full, -1, end) == ""
    assert chunker._find_paragraph_by_content(full, "不存在实体", 20) == "不存在实体"

    dirty = "说明文字足够长\n| 表格 | 内容 |\n---\n正常结论文字也足够长"
    cleaned = chunker._clean_paragraph_content(dirty)
    assert "表格" not in cleaned
    assert "正常结论" in cleaned


def test_chunker_recursive_path_and_line_estimators():
    class FakeSplitter:
        def split_text(self, content):
            return ["第一行", "第三行"]

    chunker = DocumentChunker(
        ChunkingConfig(max_tokens=20, overlap_tokens=0, use_langchain=False)
    )
    chunker.text_splitter = FakeSplitter()
    content = "第一行\n第二行\n第三行"

    chunks = chunker._langchain_recursive_chunking(content, "x.md")

    assert [chunk.start_line for chunk in chunks] == [1, 3]
    assert [chunk.end_line for chunk in chunks] == [1, 3]
    assert chunker._count_tokens("12345678") == 2
    assert len(chunker._calculate_fingerprint("abc")) == 32


def test_dedupe_similarity_lsh_merge_and_stats():
    engine = DeduplicationEngine(similarity_threshold=0.75, lsh_threshold=0.5, enable_lsh=True)
    a = _chunk("a", "成都水库工程施工招标", token_count=10, chapter_path=["招标公告"], fingerprint="a")
    b = _chunk("b", "成都水库工程施工招标文件", token_count=11, chapter_path=["招标公告"], fingerprint="b")
    c = _chunk("c", "完全不同的财务资料", token_count=4, chapter_path=["财务"], fingerprint="c")

    assert engine._basic_similarity("abc", "abc") == 1.0
    assert engine._basic_similarity("", "") == 1.0
    assert engine._calculate_path_similarity([], []) == 1.0
    assert engine._calculate_path_similarity(["A"], []) == 0.0
    assert engine._calculate_path_similarity(["A", "B"], ["B", "C"]) == pytest.approx(1 / 3)
    assert engine._calculate_token_similarity(0, 0) == 1.0
    assert engine._calculate_token_similarity(0, 4) == 0.0
    assert engine._calculate_token_similarity(5, 10) == 0.5
    assert engine._calculate_similarity(a, b) > engine._calculate_similarity(a, c)

    results = engine.process_chunks([a, b, c])
    assert len(results) == 3
    assert results[1].similar_chunks
    cached = engine._find_similar_chunks(b, [a])
    assert cached == engine._find_similar_chunks(b, [a])

    assert engine._get_shingles("abcdef", 3) == ["abc", "bcd", "cde", "def"]
    if engine.enable_lsh:
        engine.build_lsh_index([a, b])
        matches = engine.query_lsh(a)
        assert "a" in matches
    else:
        assert engine.query_lsh(a) == []

    field_a = ExtractedField(
        field_name="project_name",
        field_type="project_name",
        values=[_span("甲工程", 0.8, 1)],
        primary_value="甲工程",
        confidence=0.8,
        conflicts=["a"],
    )
    field_b = ExtractedField(
        field_name="project_name",
        field_type="project_name",
        values=[_span("乙工程", 0.95, 10), _span("甲工程", 0.8, 1)],
        primary_value="乙工程",
        confidence=0.95,
        conflicts=["b"],
    )
    merged = engine._merge_fields(field_a, field_b)
    assert merged.primary_value == "乙工程"
    assert len(merged.values) == 2
    assert merged.conflicts == ["a", "b"]

    extraction_results = [
        {"project_name": field_a},
        {"project_name": field_b},
    ]
    similarity_results = [
        SimilarityResult("a", [("b", 0.9)], False),
        SimilarityResult("b", [], False),
    ]
    combined = engine.merge_duplicate_extractions(extraction_results, similarity_results)
    assert combined[0]["project_name"].primary_value == "乙工程"

    stats = engine.get_deduplication_stats(
        [
            SimilarityResult("a", [("b", 0.95), ("c", 0.75), ("d", 0.5)], False),
            SimilarityResult("b", [], True, duplicate_of="a"),
        ]
    )
    assert stats["total_chunks"] == 2
    assert stats["duplicate_chunks"] == 1
    assert stats["unique_chunks"] == 1
    assert stats["similarity_distribution"] == {"high": 1, "medium": 1, "low": 1}
    assert stats["avg_similarity"] == pytest.approx((0.95 + 0.75 + 0.5) / 3)


def test_ner_regex_jieba_helpers_merge_and_stats(monkeypatch):
    extractor = NERExtractor()
    extractor.use_jieba = False
    text = (
        "四川测试建设有限公司负责成都水库建设工程，"
        "项目管理部由项目经理负责，证书ABC12345证书。"
    )

    fields = extractor.extract_entities(text)

    assert "company" in fields
    assert fields["company"].primary_value
    assert "project" in fields
    assert extractor._validate_tender_entity("测试建设有限公司", "company")
    assert extractor._validate_tender_entity("成都水库建设工程", "project")
    assert extractor._validate_tender_entity("项目管理部", "department")
    assert extractor._validate_tender_entity("项目经理", "position")
    assert extractor._validate_tender_entity("ABC12345证书", "certificate")
    assert extractor._validate_tender_entity("x", "company") is False
    assert extractor._is_meaningful_entity("公司", "company") is False
    assert extractor._is_meaningful_entity("测试建设有限公司", "company") is True
    assert extractor._is_meaningful_jieba_entity("成都", "location") is False
    assert extractor._is_meaningful_jieba_entity("成都市青羊区", "location") is True

    duplicate_low = _span("张三", 0.6, 5)
    duplicate_high = _span("张三", 0.9, 5)
    unique = extractor._deduplicate_entities([duplicate_low, duplicate_high])
    assert len(unique) == 1
    assert unique[0].confidence == 0.9

    monkeypatch.setattr(
        ner_module.pseg,
        "cut",
        lambda text: iter([("张三", "nr"), ("成都市青羊区", "ns"), ("公司", "nt")]),
    )
    extractor.use_jieba = True
    jieba_entities = extractor._extract_with_jieba("张三在成都市青羊区公司任职")
    assert "person" in jieba_entities
    assert "location" in jieba_entities
    assert "organization" not in jieba_entities

    rule_field = ExtractedField(
        field_name="company",
        field_type="company",
        values=[_span("规则公司", 0.95, 0)],
        primary_value="规则公司",
        confidence=0.95,
    )
    merged = extractor.merge_with_rules(fields, {"company": rule_field})
    assert merged["company"].confidence >= 0.7
    assert len(merged["company"].values) >= 2

    stats = extractor.get_entity_statistics(merged)
    assert stats["total_entities"] > 0
    assert stats["unique_entities"] > 0
    assert stats["entities_by_source"]["regex"] > 0
    assert stats["avg_confidence"] > 0
