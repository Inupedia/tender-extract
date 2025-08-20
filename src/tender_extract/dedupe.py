"""
去重和相似度检测模块
"""
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .schema import ChunkInfo, ExtractedField
import logging

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz未安装，将使用基础相似度检测")

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    MinHash = None
    MinHashLSH = None
    logger.warning("datasketch未安装，MinHash LSH功能将不可用")


@dataclass
class SimilarityResult:
    """相似度检测结果"""
    chunk_id: str
    similar_chunks: List[Tuple[str, float]]  # (chunk_id, similarity_score)
    is_duplicate: bool
    duplicate_of: Optional[str] = None


class DeduplicationEngine:
    """去重引擎"""
    
    def __init__(self, similarity_threshold: float = 0.8, 
                 lsh_threshold: float = 0.5,
                 enable_lsh: bool = True):
        self.similarity_threshold = similarity_threshold
        self.lsh_threshold = lsh_threshold
        self.enable_lsh = enable_lsh and DATASKETCH_AVAILABLE
        self.use_rapidfuzz = RAPIDFUZZ_AVAILABLE
        
        # 缓存已处理的指纹
        self.fingerprint_cache = {}
        self.similarity_cache = {}
        
        # LSH索引
        self.lsh_index = None
        if self.enable_lsh:
            self.lsh_index = MinHashLSH(threshold=lsh_threshold, num_perm=128)
    
    def process_chunks(self, chunks: List[ChunkInfo]) -> List[SimilarityResult]:
        """处理切片去重"""
        results = []
        
        # 1. 精确去重（基于指纹）
        exact_duplicates = self._find_exact_duplicates(chunks)
        
        # 2. 相似度检测
        for i, chunk in enumerate(chunks):
            if chunk.chunk_id in exact_duplicates:
                # 精确重复
                duplicate_of = exact_duplicates[chunk.chunk_id]
                results.append(SimilarityResult(
                    chunk_id=chunk.chunk_id,
                    similar_chunks=[],
                    is_duplicate=True,
                    duplicate_of=duplicate_of
                ))
            else:
                # 相似度检测
                similar_chunks = self._find_similar_chunks(chunk, chunks[:i])
                is_duplicate = any(score >= self.similarity_threshold 
                                 for _, score in similar_chunks)
                
                results.append(SimilarityResult(
                    chunk_id=chunk.chunk_id,
                    similar_chunks=similar_chunks,
                    is_duplicate=is_duplicate
                ))
        
        return results
    
    def _find_exact_duplicates(self, chunks: List[ChunkInfo]) -> Dict[str, str]:
        """查找精确重复"""
        duplicates = {}
        seen_fingerprints = {}
        
        for chunk in chunks:
            fingerprint = chunk.fingerprint
            
            if fingerprint in seen_fingerprints:
                # 发现重复
                original_chunk_id = seen_fingerprints[fingerprint]
                duplicates[chunk.chunk_id] = original_chunk_id
            else:
                seen_fingerprints[fingerprint] = chunk.chunk_id
        
        return duplicates
    
    def _find_similar_chunks(self, target_chunk: ChunkInfo, 
                           other_chunks: List[ChunkInfo]) -> List[Tuple[str, float]]:
        """查找相似切片"""
        similar_chunks = []
        
        # 使用缓存
        cache_key = f"{target_chunk.chunk_id}_{len(other_chunks)}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        for other_chunk in other_chunks:
            # 跳过已确认为重复的切片
            if other_chunk.chunk_id in self.fingerprint_cache:
                continue
            
            similarity = self._calculate_similarity(target_chunk, other_chunk)
            
            if similarity >= self.similarity_threshold * 0.8:  # 稍微降低阈值以捕获更多候选
                similar_chunks.append((other_chunk.chunk_id, similarity))
        
        # 按相似度排序
        similar_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # 缓存结果
        self.similarity_cache[cache_key] = similar_chunks
        
        return similar_chunks
    
    def _calculate_similarity(self, chunk1: ChunkInfo, chunk2: ChunkInfo) -> float:
        """计算两个切片的相似度"""
        # 1. 基于内容的相似度
        content_similarity = self._calculate_content_similarity(
            chunk1.content, chunk2.content
        )
        
        # 2. 基于章节路径的相似度
        path_similarity = self._calculate_path_similarity(
            chunk1.chapter_path, chunk2.chapter_path
        )
        
        # 3. 基于token数量的相似度
        token_similarity = self._calculate_token_similarity(
            chunk1.token_count, chunk2.token_count
        )
        
        # 综合相似度（加权平均）
        weights = [0.6, 0.3, 0.1]  # 内容权重最高
        similarities = [content_similarity, path_similarity, token_similarity]
        
        weighted_similarity = sum(w * s for w, s in zip(weights, similarities))
        
        return weighted_similarity
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        if self.use_rapidfuzz:
            # 使用rapidfuzz计算相似度
            return fuzz.ratio(content1, content2) / 100.0
        else:
            # 基础相似度计算
            return self._basic_similarity(content1, content2)
    
    def _basic_similarity(self, text1: str, text2: str) -> float:
        """基础相似度计算"""
        # 将文本转换为字符集合
        chars1 = set(text1)
        chars2 = set(text2)
        
        if not chars1 and not chars2:
            return 1.0
        
        intersection = chars1.intersection(chars2)
        union = chars1.union(chars2)
        
        return len(intersection) / len(union)
    
    def _calculate_path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """计算章节路径相似度"""
        if not path1 and not path2:
            return 1.0
        
        if not path1 or not path2:
            return 0.0
        
        # 计算路径的Jaccard相似度
        set1 = set(path1)
        set2 = set(path2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)
    
    def _calculate_token_similarity(self, tokens1: int, tokens2: int) -> float:
        """计算token数量相似度"""
        if tokens1 == 0 and tokens2 == 0:
            return 1.0
        
        if tokens1 == 0 or tokens2 == 0:
            return 0.0
        
        # 使用相对差异计算相似度
        max_tokens = max(tokens1, tokens2)
        min_tokens = min(tokens1, tokens2)
        
        return min_tokens / max_tokens
    
    def build_lsh_index(self, chunks: List[ChunkInfo]):
        """构建LSH索引"""
        if not self.enable_lsh:
            return
        
        for chunk in chunks:
            minhash = self._create_minhash(chunk.content)
            self.lsh_index.insert(chunk.chunk_id, minhash)
    
    def _create_minhash(self, text: str) -> MinHash:
        """创建MinHash"""
        minhash = MinHash(num_perm=128)
        
        # 将文本分割为shingles（字符n-gram）
        shingles = self._get_shingles(text, k=3)
        
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def _get_shingles(self, text: str, k: int = 3) -> List[str]:
        """获取k-shingles"""
        shingles = []
        for i in range(len(text) - k + 1):
            shingles.append(text[i:i+k])
        return shingles
    
    def query_lsh(self, chunk: ChunkInfo) -> List[str]:
        """查询LSH索引"""
        if not self.enable_lsh or not self.lsh_index:
            return []
        
        minhash = self._create_minhash(chunk.content)
        return list(self.lsh_index.query(minhash))
    
    def merge_duplicate_extractions(self, extractions: List[Dict[str, ExtractedField]], 
                                  similarity_results: List[SimilarityResult]) -> List[Dict[str, ExtractedField]]:
        """合并重复抽取结果"""
        merged_extractions = []
        processed_duplicates = set()
        
        for i, (extraction, similarity_result) in enumerate(zip(extractions, similarity_results)):
            if similarity_result.is_duplicate and similarity_result.duplicate_of:
                # 这是一个重复，跳过
                processed_duplicates.add(i)
                continue
            
            if i in processed_duplicates:
                # 已经被处理为重复，跳过
                continue
            
            # 合并相似切片的结果
            merged_extraction = extraction.copy()
            
            # 查找所有相似的切片
            similar_indices = []
            for similar_chunk_id, similarity_score in similarity_result.similar_chunks:
                if similarity_score >= self.similarity_threshold:
                    # 找到对应的索引
                    for j, other_result in enumerate(similarity_results):
                        if other_result.chunk_id == similar_chunk_id:
                            similar_indices.append(j)
                            break
            
            # 合并相似切片的结果
            for similar_idx in similar_indices:
                if similar_idx < len(extractions):
                    similar_extraction = extractions[similar_idx]
                    merged_extraction = self._merge_extraction_results(
                        merged_extraction, similar_extraction
                    )
            
            merged_extractions.append(merged_extraction)
        
        return merged_extractions
    
    def _merge_extraction_results(self, extraction1: Dict[str, ExtractedField], 
                                extraction2: Dict[str, ExtractedField]) -> Dict[str, ExtractedField]:
        """合并两个抽取结果"""
        merged = extraction1.copy()
        
        for field_name, field2 in extraction2.items():
            if field_name in merged:
                # 合并字段
                field1 = merged[field_name]
                merged_field = self._merge_fields(field1, field2)
                merged[field_name] = merged_field
            else:
                # 添加新字段
                merged[field_name] = field2
        
        return merged
    
    def _merge_fields(self, field1: ExtractedField, field2: ExtractedField) -> ExtractedField:
        """合并两个字段"""
        # 合并值列表
        all_values = field1.values + field2.values
        
        # 去重（基于值和位置）
        unique_values = []
        seen = set()
        
        for value in all_values:
            key = (value.value, value.start, value.end)
            if key not in seen:
                seen.add(key)
                unique_values.append(value)
        
        # 按置信度排序
        unique_values.sort(key=lambda x: x.confidence, reverse=True)
        
        # 创建合并后的字段
        merged_field = ExtractedField(
            field_name=field1.field_name,
            field_type=field1.field_type,
            values=unique_values,
            primary_value=unique_values[0].value if unique_values else None,
            confidence=max(v.confidence for v in unique_values) if unique_values else 0.0,
            conflicts=field1.conflicts + field2.conflicts
        )
        
        return merged_field
    
    def get_deduplication_stats(self, similarity_results: List[SimilarityResult]) -> Dict[str, Any]:
        """获取去重统计信息"""
        stats = {
            'total_chunks': len(similarity_results),
            'duplicate_chunks': 0,
            'unique_chunks': 0,
            'similarity_distribution': {
                'high': 0,    # >= 0.9
                'medium': 0,  # 0.7-0.9
                'low': 0      # < 0.7
            },
            'avg_similarity': 0.0
        }
        
        total_similarity = 0.0
        similarity_count = 0
        
        for result in similarity_results:
            if result.is_duplicate:
                stats['duplicate_chunks'] += 1
            else:
                stats['unique_chunks'] += 1
            
            # 统计相似度分布
            for _, similarity_score in result.similar_chunks:
                total_similarity += similarity_score
                similarity_count += 1
                
                if similarity_score >= 0.9:
                    stats['similarity_distribution']['high'] += 1
                elif similarity_score >= 0.7:
                    stats['similarity_distribution']['medium'] += 1
                else:
                    stats['similarity_distribution']['low'] += 1
        
        if similarity_count > 0:
            stats['avg_similarity'] = total_similarity / similarity_count
        
        return stats 