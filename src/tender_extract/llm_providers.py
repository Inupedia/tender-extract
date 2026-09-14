"""LLM provider registry.

Most hosted vendors expose an OpenAI-compatible Chat Completions API, so they share
one adapter. Native adapters are kept only where the protocol/authentication differs.
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
    aliases: tuple[str, ...] = ()
    api_key_env_aliases: tuple[str, ...] = ()
    auth_required: bool = True
    local: bool = False


PROVIDERS: dict[str, ProviderSpec] = {
    "none": ProviderSpec(
        "none", "不使用 LLM", "none", "", auth_required=False, local=True,
    ),
    "openai": ProviderSpec(
        "openai", "OpenAI", "openai_compat", "gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        aliases=("chatgpt", "gpt"),
    ),
    "azure": ProviderSpec(
        "azure", "Azure OpenAI", "azure", "gpt-4o-mini",
        api_key_env="AZURE_OPENAI_API_KEY",
        base_url_env="AZURE_OPENAI_ENDPOINT",
        notes="模型名填写 Azure deployment name；可用 AZURE_OPENAI_API_VERSION 覆盖 API 版本",
        aliases=("azure_openai", "azure-openai"),
    ),
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic Claude", "anthropic", "claude-sonnet-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        aliases=("claude",),
    ),
    "gemini": ProviderSpec(
        "gemini", "Google Gemini", "openai_compat", "gemini-3.8-flash",
        api_key_env="GEMINI_API_KEY",
        api_key_env_aliases=("GOOGLE_API_KEY",),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        aliases=("google",),
    ),
    "deepseek": ProviderSpec(
        "deepseek", "DeepSeek", "openai_compat", "deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    "qwen": ProviderSpec(
        "qwen", "阿里云通义千问 / DashScope", "openai_compat", "qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        aliases=("dashscope", "tongyi", "qianwen"),
    ),
    "moonshot": ProviderSpec(
        "moonshot", "Moonshot / Kimi", "openai_compat", "moonshot-v1-auto",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
        aliases=("kimi",),
    ),
    "zhipu": ProviderSpec(
        "zhipu", "智谱 GLM", "openai_compat", "glm-4-flash",
        api_key_env="ZHIPUAI_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        aliases=("glm", "glm4"),
    ),
    "doubao": ProviderSpec(
        "doubao", "火山方舟 Doubao", "openai_compat", "doubao-pro-32k",
        api_key_env="ARK_API_KEY",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        aliases=("ark", "volcengine"),
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
        aliases=("lingyi",),
    ),
    "stepfun": ProviderSpec(
        "stepfun", "阶跃星辰", "openai_compat", "step-2-mini",
        api_key_env="STEPFUN_API_KEY",
        base_url="https://api.stepfun.com/v1",
    ),
    "siliconflow": ProviderSpec(
        "siliconflow", "硅基流动", "openai_compat", "Qwen/Qwen3-8B",
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
        aliases=("grok",),
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
    "nvidia": ProviderSpec(
        "nvidia", "NVIDIA NIM", "openai_compat", "meta/llama-3.3-70b-instruct",
        api_key_env="NVIDIA_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        aliases=("nim", "nvidia_nim"),
    ),
    "ollama": ProviderSpec(
        "ollama", "Ollama（本地）", "ollama", "qwen2.5:14b",
        base_url="http://127.0.0.1:11434",
        base_url_env="OLLAMA_BASE_URL",
        aliases=("local",),
        auth_required=False,
        local=True,
    ),
    "vllm": ProviderSpec(
        "vllm", "vLLM（本地 / 自托管）", "openai_compat", "",
        base_url="http://127.0.0.1:8000/v1",
        base_url_env="VLLM_BASE_URL",
        auth_required=False,
        local=True,
    ),
    "lmstudio": ProviderSpec(
        "lmstudio", "LM Studio（本地）", "openai_compat", "",
        base_url="http://127.0.0.1:1234/v1",
        base_url_env="LMSTUDIO_BASE_URL",
        aliases=("lm-studio", "lm_studio"),
        auth_required=False,
        local=True,
    ),
    "openai_compat": ProviderSpec(
        "openai_compat", "任意 OpenAI-compatible 接口", "openai_compat", "",
        api_key_env="LLM_API_KEY",
        base_url_env="LLM_BASE_URL",
        notes="通过 --base-url / LLM_BASE_URL 指定 endpoint；适用于 vLLM、TGI、代理网关和其他兼容服务",
        aliases=("openai-compatible", "openai_compatible", "custom"),
        auth_required=False,
    ),
}


ALIASES: dict[str, str] = {
    alias: spec.id
    for spec in PROVIDERS.values()
    for alias in spec.aliases
}


def get_provider(provider_id: str) -> ProviderSpec:
    key = (provider_id or "none").lower().strip()
    key = ALIASES.get(key, key)
    if key not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"未知 LLM 提供商: {provider_id}。可选: {known}")
    return PROVIDERS[key]


def list_providers() -> list[ProviderSpec]:
    return list(PROVIDERS.values())


def provider_public_dict(spec: ProviderSpec) -> dict[str, object]:
    """Serialize non-secret provider metadata for CLI/HTTP discovery."""

    return {
        "id": spec.id,
        "name": spec.name,
        "kind": spec.kind,
        "default_model": spec.default_model or None,
        "api_key_env": spec.api_key_env or None,
        "base_url": spec.base_url or None,
        "base_url_env": spec.base_url_env or None,
        "aliases": list(spec.aliases),
        "auth_required": spec.auth_required,
        "local": spec.local,
        "notes": spec.notes or None,
    }
