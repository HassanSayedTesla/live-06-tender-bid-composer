"""Function tools (@tool) for pricing. Ch1.4 pattern. HF/OSS only."""
from __future__ import annotations

import json

from smolagents import tool

from src.config import VAT_RATE


@tool
def calc_bid_table(items_json: str) -> str:
    """Compute bid pricing table with VAT.

    Args:
        items_json (str): JSON list of {"item": str, "qty": number, "unit_price_egp": number}.

    Returns:
        str: CSV string with columns item,qty,unit_price_egp,line_total_egp plus VAT/total summary rows.
    """
    import pandas as pd

    items = json.loads(items_json)
    df = pd.DataFrame(items)
    df["qty"] = pd.to_numeric(df["qty"])
    df["unit_price_egp"] = pd.to_numeric(df["unit_price_egp"])
    df["line_total_egp"] = df["qty"] * df["unit_price_egp"]
    subtotal = float(df["line_total_egp"].sum())
    vat = round(subtotal * VAT_RATE, 2)
    total = round(subtotal + vat, 2)
    summary = pd.DataFrame(
        [
            {"item": "SUBTOTAL", "qty": "", "unit_price_egp": "", "line_total_egp": subtotal},
            {"item": f"VAT {VAT_RATE:.0%}", "qty": "", "unit_price_egp": "", "line_total_egp": vat},
            {"item": "TOTAL", "qty": "", "unit_price_egp": "", "line_total_egp": total},
        ]
    )
    out = pd.concat([df, summary], ignore_index=True)
    return out.to_csv(index=False)


@tool
def check_compliance_item(requirement: str, company_profile: str) -> str:
    """Check a single RFP requirement against the company profile.

    Args:
        requirement (str): One RFP requirement sentence.
        company_profile (str): Free-text company capabilities.

    Returns:
        str: 'PASS: ...' or 'FAIL: ...' with a one-line rationale.
    """
    req_words = set(requirement.lower().split())
    prof = company_profile.lower()
    hits = [w for w in req_words if len(w) > 4 and w in prof]
    if hits:
        return f"PASS: matched keywords {sorted(set(hits))[:5]}"
    return "FAIL: no evidence in company profile — flag for reviewer"
