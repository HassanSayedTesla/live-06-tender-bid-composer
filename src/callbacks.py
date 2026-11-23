"""Step callbacks for live observability + context-size control. Ch2.7 pattern."""
from __future__ import annotations

from smolagents import ActionStep, PlanningStep

from src.config import MAX_OBSERVATION_CHARS


def planning_callback(step, agent=None, log=None):
    text = f"AGENT PLANNING\n{'=' * 50}\n{step.plan[:300]}"
    if len(step.plan) > 300:
        text += "\n... (plan truncated)"
    if log is not None:
        log.append({"type": "planning", "text": text})
    else:
        print(text)


def action_callback(step, agent=None, log=None):
    # Observations are replayed into every later prompt; cap them so the
    # manager/workers stay under Groq free-tier TPM limits (413 errors).
    obs = getattr(step, "observations", None)
    if isinstance(obs, str) and len(obs) > MAX_OBSERVATION_CHARS:
        step.observations = obs[:MAX_OBSERVATION_CHARS] + "\n...[truncated for context budget]"
    text = f"Step {step.step_number}: Taking action"
    if getattr(step, "is_final_answer", False):
        try:
            total = step.token_usage.total_tokens
            text += f"\nTotal tokens used: {total}"
        except Exception:
            pass
    if log is not None:
        log.append({"type": "action", "text": text})
    else:
        print(text)


STEP_CALLBACKS = {PlanningStep: planning_callback, ActionStep: action_callback}
