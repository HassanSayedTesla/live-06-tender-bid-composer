"""Specialized worker agents. Ch3.1 + Ch1/Ch2 patterns. HF-only."""
from __future__ import annotations

from smolagents import CodeAgent, InferenceClientModel

from src.config import MAX_STEPS_WORKER, PLANNING_INTERVAL, WORKER_MODEL_ID


def _hf_model(model_id: str = WORKER_MODEL_ID):
    return InferenceClientModel(model_id=model_id)


def _web_search_tool():
    """OSS web search: prefer smolagents WebSearchTool, fallback to DuckDuckGo."""
    try:
        from smolagents import WebSearchTool

        return WebSearchTool()
    except Exception:
        from smolagents import Tool

        from duckduckgo_search import DDGS

        class DuckSearch(Tool):
            name = "web_search"
            description = "Search the web for company/market background."
            inputs = {"query": {"type": "string", "description": "Search query"}}
            output_type = "string"

            def forward(self, query: str) -> str:
                with DDGS() as ddg:
                    hits = list(ddg.text(query, max_results=5))
                return "\n".join(
                    f"- {h.get('title')}: {h.get('body')} ({h.get('href')})" for h in hits
                ) or "Nothing found."

        return DuckSearch()


def build_workers(rfp_search_tool, pricing_tools: list | None = None):
    """Create the 4 worker agents. Returns dict of CodeAgents."""
    model = _hf_model()
    pricing_tools = pricing_tools or []

    rfp_reader = CodeAgent(
        tools=[rfp_search_tool],
        model=model,
        instructions=(
            "You extract RFP facts. Search thoroughly with different terms. "
            "Always cite page/section when present in the chunk."
        ),
        name="rfp_reader_agent",
        description="Expert in reading RFPs and extracting requirements, terms, delivery",
        planning_interval=PLANNING_INTERVAL,
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
    )
    company_agent = CodeAgent(
        tools=[_web_search_tool()],
        model=model,
        instructions="You research the bidder company and market background.",
        name="company_agent",
        description="Expert in company background and market research",
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
    )
    pricing_agent = CodeAgent(
        tools=pricing_tools,
        model=model,
        instructions="You build pricing tables with calc_bid_table. Always include VAT.",
        name="pricing_agent",
        description="Expert in bid pricing and VAT calculation",
        additional_authorized_imports=["pandas", "json"],
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
    )
    compliance_agent = CodeAgent(
        tools=[],
        model=model,
        instructions="You check every must-have: scope, delivery, warranty, eligibility, docs.",
        name="compliance_agent",
        description="Expert in RFP compliance checking",
        max_steps=MAX_STEPS_WORKER,
        verbosity_level=1,
    )
    return {
        "rfp_reader_agent": rfp_reader,
        "company_agent": company_agent,
        "pricing_agent": pricing_agent,
        "compliance_agent": compliance_agent,
    }
