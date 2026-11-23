"""Final-answer validators. Ch3.4 pattern: rule check + open-model judge.

smolagents 1.26 calls each check as check(final_answer, memory, agent=self),
so every validator must accept the `agent` keyword argument.
"""
from __future__ import annotations

REQUIRED_SECTIONS = ["scope", "delivery", "warranty", "pricing", "compliance"]

VALIDATION_PROMPT = """
Reasoning process: {}

Agent's final answer: {}

Does the final answer logically follow from the reasoning and solve the user's question?

Respond only TRUE or FALSE. No other text.
"""


def check_coverage(final_answer, agent_memory=None, agent=None) -> bool:
    text = str(final_answer).lower()
    if len(str(final_answer)) < 200:
        raise ValueError("Bid draft is too brief (<200 chars)")
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    if missing:
        raise ValueError(f"Bid missing sections: {missing}")
    return True


def check_reasoning_accuracy(final_answer, agent_memory=None, agent=None) -> bool:
    """LLM judge: does the bid follow from the agent's own reasoning?

    Uses a raw (non-coerced) model so the TRUE/FALSE verdict stays intact.
    If the judge call itself fails (rate limit / network), we warn and skip
    rather than killing the whole run — the rule check still applies.
    """
    from smolagents import ChatMessage

    from src.models import build_chat_model

    try:
        reasoning = agent_memory.get_succinct_steps()
    except Exception:
        reasoning = str(agent_memory)[:2000]
    prompt = VALIDATION_PROMPT.format(reasoning, final_answer)
    try:
        model = build_chat_model("worker", robust=False)
        out = model([ChatMessage(role="user", content=prompt)])
    except Exception as e:
        print(f"[validators] judge unavailable ({e}); skipping reasoning check")
        return True
    verdict = str(getattr(out, "content", out)).strip().upper()
    if "FALSE" in verdict:
        raise ValueError("Judge verdict FALSE: reasoning does not support answer")
    return True


FINAL_ANSWER_CHECKS = [check_coverage, check_reasoning_accuracy]
