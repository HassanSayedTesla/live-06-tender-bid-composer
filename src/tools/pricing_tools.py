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


_STOPWORDS = {
    "the", "and", "for", "with", "must", "shall", "should", "will", "have",
    "has", "are", "was", "were", "been", "being", "this", "that", "these",
    "those", "from", "into", "upon", "about", "above", "below", "than",
    "then", "when", "where", "which", "while", "who", "whom", "whose",
    "what", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "too", "very", "can",
    "could", "may", "might", "would", "also", "not", "nor", "but", "yet",
    "per", "via", "our", "your", "their", "its", "his", "her", "they",
    "them", "you", "we", "he", "she", "it", "of", "to", "in", "on", "at",
    "by", "an", "a", "as", "is", "be", "or", "if", "do", "does", "did",
    "required", "requirement", "requirements", "provide", "provided",
    "including", "include", "includes", "etc", "eg", "ie",
}


def _content_tokens(text: str) -> set[str]:
    import re

    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 3 and w not in _STOPWORDS
    }


@tool
def check_compliance_item(requirement: str, company_profile: str) -> str:
    """Heuristic pre-check of one RFP requirement against the company profile.

    Token-overlap heuristic (stopwords removed); the compliance agent and the
    final-answer judge make the semantic decision. PASS needs >=40% of the
    requirement's content words evidenced in the profile.

    Args:
        requirement (str): One RFP requirement sentence.
        company_profile (str): Free-text company capabilities.

    Returns:
        str: 'PASS: ...' or 'FAIL: ...' with matched/missing evidence words.
    """
    req_tokens = _content_tokens(requirement)
    prof_tokens = _content_tokens(company_profile)
    if not req_tokens:
        return "PASS: no substantive keywords to verify"
    hits = req_tokens & prof_tokens
    missing = req_tokens - prof_tokens
    coverage = len(hits) / len(req_tokens)
    if coverage >= 0.4:
        return f"PASS: {coverage:.0%} keyword coverage, matched {sorted(hits)[:6]}"
    return (
        f"FAIL: only {coverage:.0%} keyword coverage — "
        f"missing evidence for {sorted(missing)[:6]}; flag for reviewer"
    )
