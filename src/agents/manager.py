"""Manager orchestrator. Ch3.1 pattern. HF-only."""
from __future__ import annotations

from smolagents import CodeAgent, InferenceClientModel

from src.config import MANAGER_MODEL_ID, MAX_STEPS_MANAGER


def build_manager(managed_agents: list, model_id: str = MANAGER_MODEL_ID) -> CodeAgent:
    return CodeAgent(
        tools=[],
        model=InferenceClientModel(model_id=model_id),
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
