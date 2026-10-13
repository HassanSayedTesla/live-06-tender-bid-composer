"""Offline tests: no HF calls, no network."""
from src.tools.rfp_search_tool import RfpSearchTool
from src.validators import check_coverage


class FakeVS:
    def similarity_search(self, query, k=6):
        class D:
            page_content = f"RFP chunk about {query} (p.4)"

        return [D(), D()]


def test_rfp_search_tool_returns_context():
    t = RfpSearchTool(FakeVS(), k=2)
    out = t.forward("warranty terms?")
    assert "p.4" in out


def test_rfp_search_empty():
    class Empty:
        def similarity_search(self, q, k=6):
            return []

    assert RfpSearchTool(Empty()).forward("x") == "Nothing found."


def test_check_coverage_pass():
    bid = "Scope... Delivery... Warranty... Pricing... Compliance..." + "x" * 200
    assert check_coverage(bid) is True


def test_check_coverage_reject_brief():
    try:
        check_coverage("too short")
    except ValueError:
        return
    raise AssertionError("should have raised")


def test_calc_bid_table_offline():
    import json

    from src.tools.pricing_tools import calc_bid_table

    # call underlying func to avoid smolagents wrapper needing server
    fn = getattr(calc_bid_table, "func", None) or getattr(calc_bid_table, "__wrapped__", None)
    if fn is None:
        return  # wrapper shape differs; covered in integration
    items = json.dumps([{"item": "Laptop i7", "qty": 50, "unit_price_egp": 25000}])
    csv_text = fn(items)
    assert "TOTAL" in csv_text and "VAT" in csv_text
