from pathlib import Path

from fastapi.testclient import TestClient

import tender_extract.server as server
from tender_extract.server import app


client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "version" in payload


def test_info_reports_runtime_capabilities() -> None:
    response = client.get("/v1/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "tender-extract-server"
    assert payload["capabilities"]["pdf"] is True
    assert payload["capabilities"]["structured_evidence"] is True
    assert payload["capabilities"]["pii_redaction_default"] is True
    assert payload["capabilities"]["multi_llm_provider"] is True
    assert payload["provider_discovery"] == "/v1/providers"


def test_provider_discovery_lists_common_providers() -> None:
    response = client.get("/v1/providers")
    assert response.status_code == 200
    payload = response.json()
    providers = {item["id"]: item for item in payload["providers"]}

    assert {
        "openai",
        "anthropic",
        "gemini",
        "deepseek",
        "qwen",
        "siliconflow",
        "openrouter",
        "ollama",
        "vllm",
        "openai_compat",
    }.issubset(providers)
    assert providers["deepseek"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert providers["vllm"]["local"] is True
    assert providers["vllm"]["auth_required"] is False
    assert payload["custom_openai_compatible"]["provider"] == "openai_compat"


def test_extract_real_example_pdf() -> None:
    path = ROOT / "examples" / "example.pdf"
    with path.open("rb") as handle:
        response = client.post(
            "/v1/extract?llm_provider=none&use_ocr=false",
            files={"file": (path.name, handle, "application/pdf")},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert response.headers["x-request-id"] == payload["request_id"]
    assert payload["llm_provider"] == "none"
    result = payload["result"]
    assert result["metadata"]["total_pages"] == 10
    assert result["fields"]["project_number"]["primary_value"] == "2024BFFFZ01583"
    assert result["fields"]["tenderer"]["primary_value"] == "合肥市公安局瑶海分局"
    assert result["fields"]["project_name"]["primary_value"].startswith("合肥市公安局瑶海分局雪亮工程")
    evidence = result["fields"]["project_number"]["values"][0]["location"]
    assert evidence["document_id"] == "example.pdf"
    assert evidence["page"] == 1


def test_rejects_unsupported_upload() -> None:
    response = client.post(
        "/v1/extract",
        files={"file": ("sample.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_rejects_unknown_llm_provider() -> None:
    response = client.post(
        "/v1/extract?llm_provider=not-a-provider",
        files={"file": ("sample.txt", b"project name: demo", "text/plain")},
    )
    assert response.status_code == 400
    assert "未知 LLM 提供商" in response.json()["detail"]


def test_optional_api_key_protects_extraction(monkeypatch) -> None:
    monkeypatch.setattr(server, "API_KEY", "test-secret")

    missing = client.post(
        "/v1/extract",
        files={"file": ("sample.txt", b"project name: demo", "text/plain")},
    )
    assert missing.status_code == 401

    allowed = client.post(
        "/v1/extract?llm_provider=none",
        headers={"X-API-Key": "test-secret"},
        files={"file": ("sample.txt", b"project name: demo", "text/plain")},
    )
    assert allowed.status_code == 200
