"""Specialized worker agents. Ch3.1 + Ch1/Ch2 patterns. Open models only."""
from __future__ import annotations

from smolagents import CodeAgent

from src.config import MAX_STEPS_WORKER, PLANNING_INTERVAL
from src.models import build_chat_model


def _hf_model():
    return build_chat_model("worker")


def _web_search_tool():
    """OSS web search via smolagents WebSearchTool (DuckDuckGo lite, no extra deps)."""
    from smolagents import WebSearchTool

    return WebSearchTool()


def build_workers(rfp_search_tool, pricing_tools: list | None = None):
    """Create the 4 worker agents. Returns dict of CodeAgents."""
    from src.callbacks import STEP_CALLBACKS

    model = _hf_model()
    pricing_tools = pricing_tools or []
    # Keep worker outputs small: they are replayed into the manager's prompt,
    # which must stay under Groq free-tier TPM limits.
    compact = (
        " Keep your final answer compact (max ~250 words): only the facts, "
        "figures, tables and citations the manager needs."
    )

    rfp_reader = CodeAgent(
        tools=[rfp_search_tool],
        model=model,
        instructions=(
            "You extract RFP facts. Search thoroughly with different terms. "
            "Always cite page/section when present in the chunk." + compact
        ),
        name="rfp_reader_agent",
        description="Expert in reading RFPs and extracting requirements, terms, delivery",
        planning_interval=PLANNING_INTERVAL,
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
        step_callbacks=STEP_CALLBACKS,
    )
    company_agent = CodeAgent(
        tools=[_web_search_tool()],
        model=model,
        instructions="You research the bidder company and market background." + compact,
        name="company_agent",
        description="Expert in company background and market research",
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
        step_callbacks=STEP_CALLBACKS,
    )
    pricing_agent = CodeAgent(
        tools=pricing_tools,
        model=model,
        instructions=(
            "You build pricing tables with calc_bid_table. Always include VAT." + compact
        ),
        name="pricing_agent",
        description="Expert in bid pricing and VAT calculation",
        additional_authorized_imports=["pandas", "json"],
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
        step_callbacks=STEP_CALLBACKS,
    )
    compliance_agent = CodeAgent(
        tools=[],
        model=model,
        instructions=(
            "You check every must-have: scope, delivery, warranty, eligibility, docs."
            + compact
        ),
        name="compliance_agent",
        description="Expert in RFP compliance checking",
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
        step_callbacks=STEP_CALLBACKS,
    )
    return {
        "rfp_reader_agent": rfp_reader,
        "company_agent": company_agent,
        "pricing_agent": pricing_agent,
        "compliance_agent": compliance_agent,
    }
