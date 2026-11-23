"""Chat-model factory: HF Inference (default) or Groq free tier (open models).

Why: HF Inference Providers free credits run out fast (HTTP 402).
Groq hosts the same open families (Llama/Qwen/DeepSeek-distill) with a
generous free tier, reached via its OpenAI-compatible endpoint.

Env:
  MODEL_PROVIDER=hf|groq        (default hf)
  HF_TOKEN                      (hf path, via InferenceClientModel)
  GROQ_API_KEY                  (groq path, free at console.groq.com)
  WORKER_MODEL_ID / MANAGER_MODEL_ID            (hf model ids)
  GROQ_WORKER_MODEL_ID / GROQ_MANAGER_MODEL_ID  (groq model ids)
"""
from __future__ import annotations

import json
import os
import re

from smolagents import OpenAIServerModel

GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Verified against https://console.groq.com/docs/models (Oct 2026).
# deepseek-r1-distill-llama-70b was decommissioned — do not use it.
# Manager uses llama-3.3-70b-versatile: free tier allows ~12k TPM vs only
# 8k TPM for gpt-oss-120b, which a manager prompt (system + worker outputs)
# exceeds -> HTTP 413 rate_limit_exceeded. gpt-oss still selectable via env.
DEFAULT_GROQ_WORKER_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_MANAGER_MODEL = "llama-3.3-70b-versatile"


# gpt-oss "harmony" leftovers occasionally leak into chat content.
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
# CodeAgent default tags (smolagents 1.26): all agents here use CodeAgent.
_CODE_OPEN, _CODE_CLOSE = "<code>", "</code>"


def coerce_final_answer_code(text: str) -> str:
    """Rescue *explicit* final answers that gpt-oss emits without code tags.

    smolagents CodeAgent only accepts <code> tags, ```python fences, or
    directly-parseable Python. gpt-oss-120b sometimes replies with a bare
    final_answer(...) call or <final_answer>...</final_answer> XML; both are
    unambiguous final answers, so we wrap them instead of burning a retry.

    Anything else (plain prose, no final-answer intent) is returned UNCHANGED
    so smolagents raises a parsing error and the agent retries with feedback.
    Fail loud, never silently turn a confused reply into a "final" answer.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if _CODE_OPEN in text or re.search(r"```(?:python|py)", text):
        return text
    if re.search(r"<code_block>|tool_call|Calling tools", text):
        return text
    cleaned = _HARMONY_TOKEN_RE.sub("", text).strip()
    if not cleaned:
        return text
    # Bare final_answer(...) call without tags.
    if re.search(r"final_answer\s*\(", cleaned):
        return f"{_CODE_OPEN}\n{cleaned}\n{_CODE_CLOSE}"
    # <final_answer>...</final_answer> XML block.
    xml = re.search(r"<final_answer>(.*?)</final_answer>", cleaned, re.DOTALL)
    if xml:
        payload = json.dumps(xml.group(1).strip(), ensure_ascii=False)
        return f"{_CODE_OPEN}\nfinal_answer({payload})\n{_CODE_CLOSE}"
    # No explicit final-answer intent: leave untouched -> parser error -> retry.
    return text


class RobustOpenAIServerModel(OpenAIServerModel):
    """OpenAIServerModel that sanitizes outputs for CodeAgent parsing and
    rides out Groq free-tier rate limits (429, and 413 for TPM overflow)."""

    def generate(self, messages: list, stop_sequences: list | None = None, **kwargs):
        import time

        from openai import APIStatusError

        for attempt in range(3):
            try:
                message = super().generate(
                    messages, stop_sequences=stop_sequences, **kwargs
                )
                break
            except APIStatusError as e:
                # Groq returns 413 when a request exceeds the TPM quota.
                if e.status_code not in (413, 429) or attempt == 2:
                    raise
                time.sleep(20 * (attempt + 1))  # TPM window resets each minute
        message.content = coerce_final_answer_code(message.content)
        return message


def build_chat_model(role: str = "worker", robust: bool = True):
    """Return a smolagents chat model for 'worker' or 'manager' role.

    robust=False skips output coercion — used by the LLM judge, whose raw
    TRUE/FALSE reply must not be wrapped as a final_answer code block.
    """
    provider = os.getenv("MODEL_PROVIDER", "hf").strip().lower()
    if provider == "groq":
        if role == "manager":
            model_id = os.getenv("GROQ_MANAGER_MODEL_ID", DEFAULT_GROQ_MANAGER_MODEL)
        else:
            model_id = os.getenv("GROQ_WORKER_MODEL_ID", DEFAULT_GROQ_WORKER_MODEL)
        cls = RobustOpenAIServerModel if robust else OpenAIServerModel
        return cls(
            model_id=model_id,
            api_base=GROQ_API_BASE,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    from smolagents import InferenceClientModel

    from src.config import MANAGER_MODEL_ID, WORKER_MODEL_ID

    return InferenceClientModel(model_id=MANAGER_MODEL_ID if role == "manager" else WORKER_MODEL_ID)
