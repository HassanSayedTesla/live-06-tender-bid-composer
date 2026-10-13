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

import os

GROQ_API_BASE = "https://api.groq.com/openai/v1"


def build_chat_model(role: str = "worker"):
    """Return a smolagents chat model for 'worker' or 'manager' role."""
    provider = os.getenv("MODEL_PROVIDER", "hf").strip().lower()
    if provider == "groq":
        from smolagents import OpenAIServerModel

        if role == "manager":
            model_id = os.getenv("GROQ_MANAGER_MODEL_ID", "deepseek-r1-distill-llama-70b")
        else:
            model_id = os.getenv("GROQ_WORKER_MODEL_ID", "llama-3.3-70b-versatile")
        return OpenAIServerModel(
            model_id=model_id,
            api_base=GROQ_API_BASE,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    from smolagents import InferenceClientModel

    from src.config import MANAGER_MODEL_ID, WORKER_MODEL_ID

    return InferenceClientModel(model_id=MANAGER_MODEL_ID if role == "manager" else WORKER_MODEL_ID)
