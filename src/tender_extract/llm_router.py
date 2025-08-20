"""
LLM路由模块
"""
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from .schema import LLMRequest, LLMResponse, EvidenceSpan, ExtractedField
import logging

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai未安装，OpenAI功能将不可用")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama未安装，Ollama功能将不可用")


class LLMRouter:
    """LLM路由器"""
    
    def __init__(self, provider: str = "none", model: Optional[str] = None, 
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 debug_mode: bool = False):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.debug_mode = debug_mode
        
        # 初始化客户端
        self.client = None
        self._initialize_client()
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        
        # 统计
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    def _initialize_client(self):
        """初始化LLM客户端"""
        if self.provider == "openai" and OPENAI_AVAILABLE:
            if self.api_key:
                openai.api_key = self.api_key
            if self.base_url:
                openai.api_base = self.base_url
            self.client = openai
        elif self.provider == "ollama" and OLLAMA_AVAILABLE:
            if self.base_url:
                self.client = ollama.Client(host=self.base_url)
            else:
                self.client = ollama.Client()
        else:
            logger.warning(f"LLM提供商 {self.provider} 不可用或未配置")
    
    def should_use_llm(self, field: ExtractedField, confidence_threshold: float = 0.7) -> bool:
        """按需LLM路由策略 - 仅当规则层低置信或冲突时使用LLM"""
        if self.provider == "none" or not self.client:
            return False
        
        # 1. 置信度低于阈值（核心策略）
        if field.confidence < confidence_threshold:
            return True
        
        # 2. 存在冲突（多个不同值）
        if field.conflicts:
            return True
        
        # 3. 值质量检查
        if field.values:
            primary_value = field.values[0].value
            # 值长度异常
            if len(primary_value) < 2 or len(primary_value) > 200:
                return True
            # 包含特殊字符
            if re.search(r'[<>{}[\]()]', primary_value):
                return True
            # 全是数字或符号
            if re.match(r'^[\d\s\-_\.]+$', primary_value):
                return True
        
        # 4. 高优先级字段的严格检查
        high_priority_fields = ['amount', 'date', 'number', 'deposit', 'contact', 'bid_amount']
        if field.field_type in high_priority_fields and field.confidence < 0.8:
            return True
        
        return False
    
    def get_minimal_evidence_context(self, field: ExtractedField, chunk_text: str) -> str:
        """获取最小证据片段，用于LLM处理"""
        if not field.values:
            return chunk_text[:500]  # 默认返回前500字符
        
        # 获取所有证据片段的位置
        spans = []
        for evidence in field.values:
            spans.append((evidence.start, evidence.end))
        
        # 合并重叠的片段
        merged_spans = self._merge_overlapping_spans(spans)
        
        # 提取最小上下文
        contexts = []
        for start, end in merged_spans:
            # 扩展上下文（前后各100字符）
            context_start = max(0, start - 100)
            context_end = min(len(chunk_text), end + 100)
            context = chunk_text[context_start:context_end]
            contexts.append(context)
        
        return "\n---\n".join(contexts)
    
    def _merge_overlapping_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并重叠的片段"""
        if not spans:
            return []
        
        # 按起始位置排序
        spans.sort(key=lambda x: x[0])
        
        merged = [spans[0]]
        for current in spans[1:]:
            last = merged[-1]
            
            # 如果当前片段与最后一个片段重叠或相邻
            if current[0] <= last[1] + 50:  # 允许50字符的间隔
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def extract_with_llm(self, request: LLMRequest) -> Optional[LLMResponse]:
        """使用LLM进行抽取"""
        # 检查缓存
        cache_key = self._generate_cache_key(request)
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.info(f"LLM缓存命中: {request.field_name}")
            return self.cache[cache_key]
        
        self.total_calls += 1
        
        # 显示LLM调用进度
        logger.info(f"🔄 LLM调用 #{self.total_calls}: 字段={request.field_name}, 类型={request.field_type}")
        logger.info(f"📤 发送内容长度: {len(request.chunk_text)} 字符")
        
        try:
            if self.provider == "openai":
                response = self._call_openai(request)
            elif self.provider == "ollama":
                response = self._call_ollama(request)
            else:
                logger.error(f"不支持的LLM提供商: {self.provider}")
                return None
            
            if response:
                self.successful_calls += 1
                logger.info(f"✅ LLM调用成功: {request.field_name} -> 置信度={response.confidence:.2f}")
                # 缓存结果
                self.cache[cache_key] = response
                return response
            else:
                self.failed_calls += 1
                logger.error(f"❌ LLM调用失败: {request.field_name}")
                return None
                
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"❌ LLM调用异常: {request.field_name} - {e}")
            return None
    
    def _call_openai(self, request: LLMRequest) -> Optional[LLMResponse]:
        """调用OpenAI API"""
        try:
            # 构建提示词
            prompt = self._build_openai_prompt(request)
            
            # 显示发送的提示词
            logger.info(f"📤 发送给OpenAI的提示词:")
            logger.info(f"字段: {request.field_name}")
            logger.info(f"内容预览: {request.chunk_text[:200]}...")
            
            if self.debug_mode:
                logger.info("🔍 [调试模式] 完整提示词:")
                logger.info("-" * 50)
                logger.info(prompt)
                logger.info("-" * 50)
            
            # 调用API
            response = self.client.ChatCompletion.create(
                model=self.model or "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的文档信息抽取助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 显示返回的响应
            content = response.choices[0].message.content
            logger.info(f"📥 OpenAI返回内容:")
            logger.info(f"响应长度: {len(content)} 字符")
            logger.info(f"响应预览: {content[:300]}...")
            
            if self.debug_mode:
                logger.info("🔍 [调试模式] 完整响应:")
                logger.info("-" * 50)
                logger.info(content)
                logger.info("-" * 50)
            
            # 解析响应
            return self._parse_openai_response(content, request.field_name)
            
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return None
    
    def _call_ollama(self, request: LLMRequest) -> Optional[LLMResponse]:
        """调用Ollama API"""
        try:
            # 构建提示词
            prompt = self._build_ollama_prompt(request)
            
            # 显示发送的提示词
            logger.info(f"📤 发送给Ollama的提示词:")
            logger.info(f"字段: {request.field_name}")
            logger.info(f"内容预览: {request.chunk_text[:200]}...")
            
            if self.debug_mode:
                logger.info("🔍 [调试模式] 完整提示词:")
                logger.info("-" * 50)
                logger.info(prompt)
                logger.info("-" * 50)
            
            # 调用API
            response = self.client.chat(
                model=self.model or "deepseek-r1:32b",
                messages=[
                    {"role": "system", "content": "你是一个专业的文档信息抽取助手。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 显示返回的响应
            content = response['message']['content']
            logger.info(f"📥 Ollama返回内容:")
            logger.info(f"响应长度: {len(content)} 字符")
            logger.info(f"响应预览: {content[:300]}...")
            
            if self.debug_mode:
                logger.info("🔍 [调试模式] 完整响应:")
                logger.info("-" * 50)
                logger.info(content)
                logger.info("-" * 50)
            
            # 解析响应
            return self._parse_ollama_response(content, request.field_name)
            
        except Exception as e:
            logger.error(f"Ollama API调用失败: {e}")
            return None
    
    def _build_openai_prompt(self, request: LLMRequest) -> str:
        """构建OpenAI提示词"""
        prompt = f"""
请从以下文本中提取{request.field_name}字段的信息。

文本内容：
{request.chunk_text}

字段类型：{request.field_type}
上下文：{request.context or '无'}

如果已存在值：{', '.join(request.existing_values) if request.existing_values else '无'}

请以JSON格式返回结果，包含以下字段：
- extracted_values: 提取的值列表
- confidence: 置信度（0-1）
- reasoning: 推理过程
- evidence_spans: 证据片段列表，每个包含value, start, end, confidence

请确保返回的是有效的JSON格式。
"""
        return prompt
    
    def _build_ollama_prompt(self, request: LLMRequest) -> str:
        """构建Ollama提示词"""
        prompt = f"""
请从以下文本中提取{request.field_name}字段的信息。

文本内容：
{request.chunk_text}

字段类型：{request.field_type}
上下文：{request.context or '无'}

如果已存在值：{', '.join(request.existing_values) if request.existing_values else '无'}

请以JSON格式返回结果，包含以下字段：
- extracted_values: 提取的值列表
- confidence: 置信度（0-1）
- reasoning: 推理过程
- evidence_spans: 证据片段列表，每个包含value, start, end, confidence

请确保返回的是有效的JSON格式。
"""
        return prompt
    
    def _parse_openai_response(self, content: str, field_name: str) -> Optional[LLMResponse]:
        """解析OpenAI响应"""
        try:
            # 尝试提取JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("未找到JSON响应")
                return None
            
            json_str = content[json_start:json_end]
            data = json.loads(json_str)
            
            # 构建响应
            evidence_spans = []
            for span_data in data.get('evidence_spans', []):
                evidence_span = EvidenceSpan(
                    value=span_data.get('value', ''),
                    start=span_data.get('start', 0),
                    end=span_data.get('end', 0),
                    confidence=span_data.get('confidence', 0.5),
                    source='llm',
                    pattern='openai'
                )
                evidence_spans.append(evidence_span)
            
            return LLMResponse(
                field_name=field_name,
                extracted_values=data.get('extracted_values', []),
                confidence=data.get('confidence', 0.5),
                reasoning=data.get('reasoning', ''),
                evidence_spans=evidence_spans
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"响应解析失败: {e}")
            return None
    
    def _parse_ollama_response(self, content: str, field_name: str) -> Optional[LLMResponse]:
        """解析Ollama响应"""
        try:
            # 尝试提取JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("未找到JSON响应")
                return None
            
            json_str = content[json_start:json_end]
            data = json.loads(json_str)
            
            # 构建响应
            evidence_spans = []
            for span_data in data.get('evidence_spans', []):
                evidence_span = EvidenceSpan(
                    value=span_data.get('value', ''),
                    start=span_data.get('start', 0),
                    end=span_data.get('end', 0),
                    confidence=span_data.get('confidence', 0.5),
                    source='llm',
                    pattern='ollama'
                )
                evidence_spans.append(evidence_span)
            
            return LLMResponse(
                field_name=field_name,
                extracted_values=data.get('extracted_values', []),
                confidence=data.get('confidence', 0.5),
                reasoning=data.get('reasoning', ''),
                evidence_spans=evidence_spans
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"响应解析失败: {e}")
            return None
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{request.chunk_text}_{request.field_name}_{request.field_type}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def merge_llm_results(self, field: ExtractedField, llm_response: LLMResponse) -> ExtractedField:
        """合并LLM结果到字段"""
        # 添加LLM提取的值
        for value in llm_response.extracted_values:
            evidence_span = EvidenceSpan(
                value=value,
                start=0,  # LLM无法提供精确位置
                end=0,
                confidence=llm_response.confidence,
                source='llm',
                pattern=f"{self.provider}:{self.model}"
            )
            field.values.append(evidence_span)
        
        # 重新计算置信度
        if field.values:
            field.values.sort(key=lambda x: x.confidence, reverse=True)
            field.primary_value = field.values[0].value
            field.confidence = max(v.confidence for v in field.values)
        
        return field
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'provider': self.provider,
            'model': self.model,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        logger.info("LLM缓存已清空")
    
    def save_cache(self, filepath: str):
        """保存缓存到文件"""
        try:
            cache_data = {
                'cache': self.cache,
                'cache_hits': self.cache_hits,
                'statistics': self.get_statistics()
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"缓存已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def load_cache(self, filepath: str):
        """从文件加载缓存"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.cache = cache_data.get('cache', {})
            self.cache_hits = cache_data.get('cache_hits', 0)
            logger.info(f"缓存已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载缓存失败: {e}") 