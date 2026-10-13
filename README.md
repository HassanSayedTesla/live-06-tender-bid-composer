# LIVE SYSTEM / 06 — Tender Bid Composer (smolagents, HF-only)

Upload RFP docs → manager delegates to 4 workers → bid draft with pricing, citations, compliance verdict.

100% Hugging Face + open source. Single secret: `HF_TOKEN`.

## Badges
`smolagents` · `CodeAgent` · `Managed Agents` · `BGE + FAISS` · `Planning + Callbacks` · `Validators` · `Streamlit` · `HF Spaces`

## Quickstart
```powershell
pip install -r requirements.txt
$env:HF_TOKEN="hf_xxx"
python -c "from src.pipeline import run_pipeline; print(run_pipeline('data/samples')['verdict'])"
streamlit run app.py
```

## Structure
- `src/ingest.py` — Ch2.1 loader + splitter
- `src/vectorstore.py` — Ch2.2 BGE + FAISS (endpoint with local fallback)
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
