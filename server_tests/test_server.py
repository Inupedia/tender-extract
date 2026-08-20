from pathlib import Path

from fastapi.testclient import TestClient

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
