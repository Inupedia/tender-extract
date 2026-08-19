from tender_extract.module_router import ModuleRouter
from tender_extract.schema import ChunkInfo


def test_financial_chunk_routes_to_financial_module():
    router = ModuleRouter()
    chunk = ChunkInfo(
        chunk_id="c1",
        content="投标报价：人民币 100 万元。投标保证金 2 万元。",
        start_line=1,
        end_line=2,
    )
    routed = router.route_chunks([chunk])
    names = {item.module_name for item in routed}
    assert "财务信息" in names
    fields = router.get_module_target_fields("financial_info")
    assert "bid_amount" in fields
    assert "deposit" in fields
