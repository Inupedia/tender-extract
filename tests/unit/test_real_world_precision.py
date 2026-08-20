import pytest

from tender_extract.configurable_engine import ConfigurableExtractionEngine


pytestmark = pytest.mark.unit


def test_legal_representative_does_not_reuse_contact_or_project_manager_patterns():
    text = (
        "联系人：顾老师\n"
        "项目经理：张三\n"
        "法定代表人：李四\n"
    )
    result = ConfigurableExtractionEngine().extract_all_fields(text)
    assert result["legal_representative"].primary_value == "李四"
    assert {item.value for item in result["legal_representative"].values} == {"李四"}


def test_tenderer_rejects_instruction_and_url_noise_but_keeps_public_body():
    engine = ConfigurableExtractionEngine()
    noisy = engine.extract_all_fields(
        "采购人通过www.gsxt.gov.cn等渠道查询：供应商的姓名或者名称、地址、邮编、联系人及联系电话\n"
    )
    assert "tenderer" not in noisy

    valid = engine.extract_all_fields("采购人：上海市奉贤区人口综合管理服务中心\n")
    assert valid["tenderer"].primary_value == "上海市奉贤区人口综合管理服务中心"


def test_unlabelled_large_amount_is_not_promoted_to_bid_amount():
    engine = ConfigurableExtractionEngine()
    assert "bid_amount" not in engine.extract_all_fields("设备容量要求为20000万元等值参数")
    explicit = engine.extract_all_fields("预算金额：9,323,695.00元\n")
    assert explicit["bid_amount"].values[0].normalized_value == "9323695.00"


def test_explicit_deadline_beats_unrelated_document_date():
    text = (
        "本办法自2023年1月1日起施行。\n"
        "四、提交投标文件截止时间、开标时间和地点\n"
        "提交投标文件截止时间：2026 年4 月29 日10:00（北京时间）\n"
    )
    result = ConfigurableExtractionEngine().extract_all_fields(text)
    assert result["bid_date"].primary_value == "2026年4月29日10:00"
    assert result["bid_date"].confidence == 0.98


def test_split_line_deadline_is_joined_by_date_cleaner():
    text = "投标截止时间：2026-4-30\n9:30，迟到或不符合规定的投标文件恕不接受。\n"
    result = ConfigurableExtractionEngine().extract_all_fields(text)
    assert result["bid_date"].primary_value == "2026-4-309:30"
