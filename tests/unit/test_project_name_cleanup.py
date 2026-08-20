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
