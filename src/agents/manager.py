"""Manager orchestrator. Ch3.1 pattern. HF default, Groq optional."""
from __future__ import annotations

from smolagents import CodeAgent

from src.config import MAX_STEPS_MANAGER
from src.models import build_chat_model


def build_manager(
    managed_agents: list,
    step_callbacks=None,
    final_answer_checks=None,
) -> CodeAgent:
    return CodeAgent(
        tools=[],
        model=build_chat_model("manager"),
        instructions=(
            "You are a tender bid manager. You MUST delegate to workers BEFORE "
            "giving a final answer:\n"
            "1. rfp_reader_agent — extract requirements, organization names, "
            "capacities, budgets, deadlines, warranty minimums, with citations.\n"
            "2. company_agent — bidder background.\n"
            "3. pricing_agent — itemized pricing via calc_bid_table (EGP, 14% VAT), "
            "aligned with the RFP budget and currency.\n"
            "4. compliance_agent — check EVERY mandatory requirement (warranty "
            "duration, delivery time, technical specs, submission documents).\n"
            "Then synthesize the bid with sections: Scope, Delivery, Warranty, "
            "Pricing, Compliance.\n"
            "Hard rules:\n"
            "- Use ONLY facts returned by the workers. Cite real RFP section "
            "titles/pages from rfp_reader output; NEVER invent clause numbers, "
            "standards, or requirements that are not in the RFP.\n"
            "- Name the actual project, issuing organization, and bidder company "
            "from the RFP and company profile. No placeholders like [Project Title].\n"
            "- Pricing must come from pricing_agent's calc_bid_table output.\n"
            "- Warranty and delivery must meet or exceed the RFP minimums; if the "
            "company cannot meet a mandatory requirement, verdict REJECT with reasons."
        ),
        managed_agents=managed_agents,
        max_steps=MAX_STEPS_MANAGER,
        verbosity_level=1,
        step_callbacks=step_callbacks,
        final_answer_checks=final_answer_checks,
    )


DEFAULT_TASK = (
    "Compose a compliant bid draft from the RFP. Extract requirements, "
    "profile the company, compute pricing with VAT, and give a "
    "COMPLIANT/REJECT verdict with reasons and citations."
)
