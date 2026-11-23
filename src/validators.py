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
Check specifically: are the facts, figures, and citations in the answer actually
present in the reasoning (not invented)? Are there unfilled placeholders?

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


# --- anti-hallucination guards ----------------------------------------------

def check_no_placeholders(final_answer, agent_memory=None, agent=None) -> bool:
    """Reject unfilled template placeholders like [Project Title]."""
    import re

    text = str(final_answer)
    # Bracketed text that is not a numeric reference or a markdown link.
    holes = re.findall(r"\[(?!\d+\])([^\]\n]{3,60})\](?!\()", text)
    if holes:
        raise ValueError(f"Bid contains unfilled placeholders: {holes[:5]}")
    return True


def make_grounding_check(chunks):
    """Factory: the bid must reference the actual RFP's entities and key figures.

    Catches generic template bids that never mention the issuing organization
    or the numeric specs (capacity, budget, deadlines) found in the documents.
    """
    import re

    text = re.sub(r"\s+", " ", " ".join(c.page_content for c in chunks))
    generic_bigrams = {
        "scope of work", "project overview", "technical requirements",
        "evaluation criteria", "submission requirements", "delivery schedule",
        "quality management", "management systems", "environmental management",
    }
    bigrams = {
        m.lower()
        for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    } - generic_bigrams
    specs = set(
        re.findall(
            r"\b\d[\d,\.]*\s*(?:kWp|kW|MW|EGP|USD|weeks|years|days|%)\b"
            r"|\b(?:EGP|USD)\s*[\d,\.]+",
            text,
        )
    )
    specs = {re.sub(r"\s+", " ", s).strip().lower() for s in specs}

    def check_grounding(final_answer, agent_memory=None, agent=None) -> bool:
        ans = re.sub(r"\s+", " ", str(final_answer)).lower()
        if bigrams:
            hits = [b for b in bigrams if b in ans]
            if not hits:
                raise ValueError(
                    "Bid is not grounded in the RFP: none of its organizations/"
                    f"places ({sorted(bigrams)[:6]}) appear. Delegate to "
                    "rfp_reader_agent and use its findings."
                )
        if specs:
            norm = [re.sub(r"[\s,]", "", s) for s in specs]
            ans_norm = re.sub(r"[\s,]", "", ans)
            if not any(s in ans_norm for s in norm):
                raise ValueError(
                    "Bid ignores the RFP's key figures "
                    f"({sorted(specs)[:6]}). Use the extracted requirements."
                )
        return True

    return check_grounding


REQUIRED_WORKERS = ("rfp_reader_agent", "pricing_agent")


def check_delegation(final_answer, agent_memory=None, agent=None) -> bool:
    """The manager must actually call the RFP reader and pricing workers."""
    import json

    try:
        trace = json.dumps(agent_memory.get_succinct_steps(), default=str)
    except Exception:
        trace = str(agent_memory)
    missing = [w for w in REQUIRED_WORKERS if w not in trace]
    if missing:
        raise ValueError(
            f"Manager answered without delegating to {missing}. Call the worker "
            "agents first and build the bid from their outputs."
        )
    return True


def build_final_checks(chunks) -> list:
    """Per-run validator stack (grounding check is bound to the run's RFP)."""
    return [
        check_coverage,
        check_no_placeholders,
        make_grounding_check(chunks),
        check_delegation,
        check_reasoning_accuracy,
    ]
