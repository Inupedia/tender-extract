"""主流 LLM 厂商预设。

绝大多数厂商提供 OpenAI 兼容 Chat Completions。Anthropic 走原生 Messages API。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Kind = Literal["openai_compat", "azure", "anthropic", "ollama", "none"]


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    kind: Kind
    default_model: str
    api_key_env: str = ""
    base_url: str = ""
    base_url_env: str = ""
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "none": ProviderSpec("none", "不使用 LLM", "none", ""),
    "openai": ProviderSpec(
        "openai", "OpenAI", "openai_compat", "gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
    ),
    "azure": ProviderSpec(
        "azure", "Azure OpenAI", "azure", "gpt-4o-mini",
        api_key_env="AZURE_OPENAI_API_KEY",
        base_url_env="AZURE_OPENAI_ENDPOINT",
        notes="模型名填部署名；另支持 AZURE_OPENAI_API_VERSION",
    ),
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic Claude", "anthropic", "claude-sonnet-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
    ),
    "gemini": ProviderSpec(
        "gemini", "Google Gemini", "openai_compat", "gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        notes="也识别 GOOGLE_API_KEY",
    ),
    "ollama": ProviderSpec(
        "ollama", "Ollama（本地）", "ollama", "qwen2.5:14b",
        base_url="http://127.0.0.1:11434",
        base_url_env="OLLAMA_BASE_URL",
    ),
    "deepseek": ProviderSpec(
        "deepseek", "DeepSeek", "openai_compat", "deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    "qwen": ProviderSpec(
        "qwen", "阿里云通义千问", "openai_compat", "qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "dashscope": ProviderSpec(
        "dashscope", "阿里云 DashScope", "openai_compat", "qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "moonshot": ProviderSpec(
        "moonshot", "Moonshot / Kimi", "openai_compat", "moonshot-v1-auto",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
    ),
    "kimi": ProviderSpec(
        "kimi", "Kimi", "openai_compat", "moonshot-v1-auto",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
    ),
    "zhipu": ProviderSpec(
        "zhipu", "智谱 GLM", "openai_compat", "glm-4-flash",
        api_key_env="ZHIPUAI_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    ),
    "glm": ProviderSpec(
        "glm", "智谱 GLM", "openai_compat", "glm-4-flash",
        api_key_env="ZHIPUAI_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    ),
    "doubao": ProviderSpec(
        "doubao", "火山方舟 Doubao", "openai_compat", "doubao-pro-32k",
        api_key_env="ARK_API_KEY",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    "volcengine": ProviderSpec(
        "volcengine", "火山方舟", "openai_compat", "doubao-pro-32k",
        api_key_env="ARK_API_KEY",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    "hunyuan": ProviderSpec(
        "hunyuan", "腾讯混元", "openai_compat", "hunyuan-turbo",
        api_key_env="HUNYUAN_API_KEY",
        base_url="https://api.hunyuan.cloud.tencent.com/v1",
    ),
    "baichuan": ProviderSpec(
        "baichuan", "百川", "openai_compat", "Baichuan4-Turbo",
        api_key_env="BAICHUAN_API_KEY",
        base_url="https://api.baichuan-ai.com/v1",
    ),
    "minimax": ProviderSpec(
        "minimax", "MiniMax", "openai_compat", "MiniMax-Text-01",
        api_key_env="MINIMAX_API_KEY",
        base_url="https://api.minimax.chat/v1",
    ),
    "yi": ProviderSpec(
        "yi", "零一万物", "openai_compat", "yi-lightning",
        api_key_env="YI_API_KEY",
        base_url="https://api.lingyiwanwu.com/v1",
    ),
    "lingyi": ProviderSpec(
        "lingyi", "零一万物", "openai_compat", "yi-lightning",
        api_key_env="YI_API_KEY",
        base_url="https://api.lingyiwanwu.com/v1",
    ),
    "stepfun": ProviderSpec(
        "stepfun", "阶跃星辰", "openai_compat", "step-2-mini",
        api_key_env="STEPFUN_API_KEY",
        base_url="https://api.stepfun.com/v1",
    ),
    "siliconflow": ProviderSpec(
        "siliconflow", "硅基流动", "openai_compat", "Qwen/Qwen2.5-7B-Instruct",
        api_key_env="SILICONFLOW_API_KEY",
        base_url="https://api.siliconflow.cn/v1",
    ),
    "openrouter": ProviderSpec(
        "openrouter", "OpenRouter", "openai_compat", "openai/gpt-4o-mini",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    "groq": ProviderSpec(
        "groq", "Groq", "openai_compat", "llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
    ),
    "together": ProviderSpec(
        "together", "Together AI", "openai_compat", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
    ),
    "mistral": ProviderSpec(
        "mistral", "Mistral", "openai_compat", "mistral-small-latest",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
    ),
    "xai": ProviderSpec(
        "xai", "xAI Grok", "openai_compat", "grok-2-latest",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
    ),
    "grok": ProviderSpec(
        "grok", "xAI Grok", "openai_compat", "grok-2-latest",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
    ),
    "fireworks": ProviderSpec(
        "fireworks", "Fireworks", "openai_compat", "accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key_env="FIREWORKS_API_KEY",
        base_url="https://api.fireworks.ai/inference/v1",
    ),
    "perplexity": ProviderSpec(
        "perplexity", "Perplexity", "openai_compat", "sonar",
        api_key_env="PERPLEXITY_API_KEY",
        base_url="https://api.perplexity.ai",
    ),
    "openai_compat": ProviderSpec(
        "openai_compat", "任意 OpenAI 兼容接口", "openai_compat", "gpt-4o-mini",
        api_key_env="LLM_API_KEY",
        base_url_env="LLM_BASE_URL",
        notes="需同时提供 --base-url 或 LLM_BASE_URL",
    ),
}


def get_provider(provider_id: str) -> ProviderSpec:
    key = (provider_id or "none").lower().strip()
    aliases = {
        "claude": "anthropic",
        "google": "gemini",
        "chatgpt": "openai",
        "gpt": "openai",
        "tongyi": "qwen",
        "qianwen": "qwen",
        "glm4": "zhipu",
        "ark": "doubao",
        "local": "ollama",
    }
    key = aliases.get(key, key)
    if key not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"未知 LLM 提供商: {provider_id}。可选: {known}")
    return PROVIDERS[key]


def list_providers() -> list[ProviderSpec]:
    seen: set[str] = set()
    result: list[ProviderSpec] = []
    for spec in PROVIDERS.values():
        if spec.id in {"dashscope", "kimi", "glm", "volcengine", "lingyi", "grok"}:
            continue
        if spec.name in seen:
            continue
        seen.add(spec.name)
        result.append(spec)
    return result
