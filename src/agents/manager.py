"""Manager orchestrator. Ch3.1 pattern. HF default, Groq optional."""
from __future__ import annotations

from smolagents import CodeAgent

from src.config import MAX_STEPS_MANAGER
from src.models import build_chat_model


def build_manager(managed_agents: list) -> CodeAgent:
    return CodeAgent(
        tools=[],
        model=build_chat_model("manager"),
        instructions=(
            "You are a tender bid manager. Delegate: rfp_reader extracts "
            "requirements, company_agent profiles bidder, pricing_agent prices, "
            "compliance_agent verdicts. Synthesize a bid with sections: "
            "Scope, Delivery, Warranty, Pricing, Compliance — with citations."
        ),
        managed_agents=managed_agents,
        max_steps=MAX_STEPS_MANAGER,
        verbosity_level=1,
    )


DEFAULT_TASK = (
    "Compose a compliant bid draft from the RFP. Extract requirements, "
    "profile the company, compute pricing with VAT, and give a "
    "COMPLIANT/REJECT verdict with reasons and citations."
)
