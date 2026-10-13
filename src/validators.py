"""Final-answer validators. Ch3.4 pattern, HF-only judge."""
from __future__ import annotations

REQUIRED_SECTIONS = ["scope", "delivery", "warranty", "pricing", "compliance"]

VALIDATION_PROMPT = """
Reasoning process: {}

Agent's final answer: {}

Does the final answer logically follow from the reasoning and solve the user's question?

Respond only TRUE or FALSE. No other text.
"""


def check_coverage(final_answer, agent_memory=None) -> bool:
    text = str(final_answer).lower()
    if len(str(final_answer)) < 200:
        raise ValueError("Bid draft is too brief (<200 chars)")
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    if missing:
        raise ValueError(f"Bid missing sections: {missing}")
    return True


def check_reasoning_accuracy(final_answer, agent_memory=None) -> bool:
    from smolagents import ChatMessage

    from src.models import build_chat_model

    model = build_chat_model("worker")
    try:
        reasoning = agent_memory.get_succinct_steps()
    except Exception:
        reasoning = str(agent_memory)[:2000]
    prompt = VALIDATION_PROMPT.format(reasoning, final_answer)
    out = model([ChatMessage(role="user", content=prompt)])
    verdict = str(getattr(out, "content", out)).strip().upper()
    if "FALSE" in verdict:
        raise ValueError("HF judge verdict FALSE: reasoning does not support answer")
    return True


FINAL_ANSWER_CHECKS = [check_coverage]
# Add HF judge in prod (costs an extra call): [check_coverage, check_reasoning_accuracy]
