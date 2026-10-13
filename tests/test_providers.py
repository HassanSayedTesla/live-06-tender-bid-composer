"""Provider + embedding-mode tests. Offline: no network, no downloads."""
import os

from src.models import build_chat_model


def _with_env(monkeypatch, **kv):
    for k, v in kv.items():
        monkeypatch.setenv(k, v)
    for k in (
        "MODEL_PROVIDER",
        "GROQ_API_KEY",
        "GROQ_WORKER_MODEL_ID",
        "GROQ_MANAGER_MODEL_ID",
        "WORKER_MODEL_ID",
        "MANAGER_MODEL_ID",
    ):
        if k not in kv:
            monkeypatch.delenv(k, raising=False)


def test_default_provider_is_hf(monkeypatch):
    _with_env(monkeypatch)
    from smolagents import InferenceClientModel

    assert isinstance(build_chat_model("worker"), InferenceClientModel)
    assert isinstance(build_chat_model("manager"), InferenceClientModel)


def test_groq_provider_builds_openai_server_model(monkeypatch):
    _with_env(monkeypatch, MODEL_PROVIDER="groq", GROQ_API_KEY="fake")
    from smolagents import OpenAIServerModel

    worker = build_chat_model("worker")
    manager = build_chat_model("manager")
    assert isinstance(worker, OpenAIServerModel)
    assert isinstance(manager, OpenAIServerModel)
    assert worker.model_id == "llama-3.3-70b-versatile"
    assert manager.model_id == "deepseek-r1-distill-llama-70b"


def test_local_embedder_is_default(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODE", raising=False)
    import src.config as config

    assert config.EMBEDDING_MODE == "local"
    assert os.getenv("EMBEDDING_MODE", "local") == "local"
