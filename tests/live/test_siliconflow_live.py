import os

import pytest

from tender_extract.llm_router import LLMRouter
from tender_extract.schema import LLMRequest, ProcessingConfig


pytestmark = pytest.mark.live


def test_siliconflow_live_extracts_known_field():
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        pytest.skip("SILICONFLOW_API_KEY is required for live integration tests")

    router = LLMRouter(
        ProcessingConfig(
            llm_provider="siliconflow",
            llm_model=os.getenv("SILICONFLOW_MODEL") or "Qwen/Qwen2.5-7B-Instruct",
            llm_api_key=api_key,
            persist_llm_cache=False,
            redact_pii_for_cloud_llm=True,
        )
    )
    assert router.is_enabled()

    response = router.extract_with_llm(
        LLMRequest(
            chunk_text="以下内容来自测试标书。建设地点：成都市青羊区。",
            field_name="construction_site",
            field_type="construction_site",
            existing_values=[],
        )
    )

    assert response is not None
    assert response.extracted_values
    assert any("成都" in value for value in response.extracted_values)
    assert response.confidence > 0
