import pytest

from tender_extract.configurable_engine import ConfigurableExtractionEngine


pytestmark = pytest.mark.unit


def test_project_name_does_not_swallow_following_fields():
    text = (
        "项目名称：青江水库除险加固工程\n"
        "项目编号：QJ-2026-001\n"
        "招标人：四川青江水利建设有限公司\n"
    )

    result = ConfigurableExtractionEngine().extract_all_fields(text)

    project_name = result["project_name"]
    assert project_name.primary_value == "青江水库除险加固工程"
    assert {item.value for item in project_name.values} == {"青江水库除险加固工程"}


def test_project_name_preserves_pdf_wrapped_continuation_line():
    # 来自公开采购文件中常见的 PDF 排版：项目名称在视觉上是一项，但文本层被折成两行。
    text = (
        "项目名称：2024 年-2026 年市属景观照明设施\n"
        "维护项目\n"
        "项目编号：TAHP-ZB-2023-1790\n"
        "采购人：北京市城市管理委员会\n"
    )

    result = ConfigurableExtractionEngine().extract_all_fields(text)

    project_name = result["project_name"]
    assert project_name.primary_value == "2024 年-2026 年市属景观照明设施维护项目"
    assert "项目编号" not in project_name.primary_value
    assert "采购人" not in project_name.primary_value
