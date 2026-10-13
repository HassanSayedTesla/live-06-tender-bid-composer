---
title: Live 06 Tender Bid Composer
emoji: 📑
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: smolagents tender bid composer (manager + workers, BGE+FAISS, validators). HF-only.
---

# LIVE SYSTEM / 06 — Tender Bid Composer (smolagents, HF-only)

Upload RFP docs → manager delegates to 4 workers → bid draft with pricing, citations, compliance verdict.

Open models only. Embeddings run **locally** (sentence-transformers, zero credits).
LLM provider is switchable: `MODEL_PROVIDER=hf` (HF Inference, needs `HF_TOKEN`)
or `MODEL_PROVIDER=groq` (Groq free tier — Llama/Qwen/DeepSeek-distill, needs `GROQ_API_KEY`).

## Badges
`smolagents` · `CodeAgent` · `Managed Agents` · `FAISS` · `Planning + Callbacks` · `Validators` · `Streamlit` · `HF + Groq (open models)`

## Quickstart
```powershell
pip install -r requirements.txt
# Groq path (recommended on free tier — free key at console.groq.com):
$env:MODEL_PROVIDER="groq"; $env:GROQ_API_KEY="gsk_xxx"
# ...or HF path:
# $env:MODEL_PROVIDER="hf"; $env:HF_TOKEN="hf_xxx"
python -c "from src.pipeline import run_pipeline; print(run_pipeline('data/samples')['verdict'])"
streamlit run app.py
```

## Structure
- `src/ingest.py` — Ch2.1 loader + splitter
- `src/vectorstore.py` — Ch2.2 FAISS (local embeddings default, HF endpoint optional)
- `src/models.py` — provider factory: HF `InferenceClientModel` vs Groq `OpenAIServerModel`
- `src/tools/rfp_search_tool.py` — Ch2.4 `Tool` subclass
- `src/tools/pricing_tools.py` — Ch1.4 `@tool` pricing + VAT
- `src/agents/workers.py` — Ch3.1 workers, `planning_interval=3`, `additional_authorized_imports=["pandas"]`
- `src/agents/manager.py` — Ch3.1 `managed_agents`
- `src/callbacks.py` — Ch2.7 `step_callbacks`
- `src/validators.py` — Ch3.4 `final_answer_checks` (rule + HF judge)
- `src/memory.py` — Ch3.3 `get_succinct_steps` / `return_full_code`
- `app.py` — Streamlit demo with Verdict + Trace + Memory tabs

## Tests (offline, no HF calls)
```
pytest -q
```

## Deploy on HF Spaces (Docker SDK)
1. Create a Space with SDK `Docker` (blank Docker template).
2. Push this repo content to the Space repo, or `Create new Space → Import from GitHub` with this repo URL.
3. In Space **Settings → Variables and secrets**, add secret `GROQ_API_KEY` (free at console.groq.com) and variable `MODEL_PROVIDER=groq`. (HF path also works if your account has Inference credits: secret `HF_TOKEN`, `MODEL_PROVIDER=hf`.)
4. The Space builds `Dockerfile` (apt deps from `packages.txt`, pip from `requirements.txt`) and serves Streamlit on port `8501`.

## Streamlit Community Cloud secrets
```toml
MODEL_PROVIDER = "groq"
GROQ_API_KEY = "gsk_xxx"
```
