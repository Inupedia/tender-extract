from tender_extract.llm_providers import get_provider, list_providers, provider_public_dict
from tender_extract.llm_router import LLMRouter
from tender_extract.schema import LLMRequest, ProcessingConfig


def test_registry_exposes_common_hosted_and_local_providers() -> None:
    provider_ids = {spec.id for spec in list_providers()}
    assert {
        "openai",
        "azure",
        "anthropic",
        "gemini",
        "deepseek",
        "qwen",
        "moonshot",
        "zhipu",
        "doubao",
        "siliconflow",
        "openrouter",
        "groq",
        "together",
        "mistral",
        "xai",
        "nvidia",
        "ollama",
        "vllm",
        "lmstudio",
        "openai_compat",
    }.issubset(provider_ids)


def test_provider_aliases_resolve_to_canonical_specs() -> None:
    assert get_provider("claude").id == "anthropic"
    assert get_provider("dashscope").id == "qwen"
    assert get_provider("kimi").id == "moonshot"
    assert get_provider("glm").id == "zhipu"
    assert get_provider("grok").id == "xai"
    assert get_provider("lm-studio").id == "lmstudio"
    assert get_provider("custom").id == "openai_compat"


def test_provider_discovery_contains_metadata_but_no_secret_values() -> None:
    payload = provider_public_dict(get_provider("deepseek"))
    assert payload["id"] == "deepseek"
    assert payload["api_key_env"] == "DEEPSEEK_API_KEY"
    assert payload["auth_required"] is True
    assert "api_key" not in payload


def test_generic_openai_compatible_requires_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    router = LLMRouter(
        ProcessingConfig(
            llm_provider="openai_compat",
            llm_model="custom-model",
            persist_llm_cache=False,
        )
    )
    assert router.is_enabled() is False


def test_vllm_works_without_key_and_is_treated_as_local(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    router = LLMRouter(
        ProcessingConfig(
            llm_provider="vllm",
            llm_model="Qwen/Qwen3-8B",
            llm_base_url="http://127.0.0.1:8000/v1",
            persist_llm_cache=False,
            redact_pii_for_cloud_llm=True,
        )
    )
    assert router.is_enabled() is True

    request = LLMRequest(
        chunk_text="身份证号110101199003078888",
        field_name="project_manager",
        field_type="project_manager",
    )
    assert router._prepare_request(request).chunk_text == request.chunk_text
