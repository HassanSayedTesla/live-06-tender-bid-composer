"""End-to-end pipeline: ingest -> FAISS -> workers -> manager -> validate."""
from __future__ import annotations

from pathlib import Path

from src.agents.manager import DEFAULT_TASK, build_manager
from src.agents.workers import build_workers
from src.callbacks import STEP_CALLBACKS
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_ID, RETRIEVAL_K
from src.ingest import load_documents, split_documents
from src.memory import save_agent_memory
from src.tools.pricing_tools import calc_bid_table, check_compliance_item
from src.tools.rfp_search_tool import RfpSearchTool
from src.validators import FINAL_ANSWER_CHECKS, check_coverage


def run_pipeline(source_dir: str | Path, task: str = DEFAULT_TASK, log: list | None = None):
    log = log if log is not None else []
    docs = load_documents(source_dir)
    if not docs:
        raise FileNotFoundError(f"No PDF/TXT/MD documents in {source_dir}")
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Lazy heavy imports so `pytest --collect-only` stays fast
    from src.vectorstore import build_faiss

    vs = build_faiss(chunks, model_id=EMBEDDING_MODEL_ID)
    search_tool = RfpSearchTool(vs, k=RETRIEVAL_K)
    workers = build_workers(search_tool, pricing_tools=[calc_bid_table, check_compliance_item])
    manager = build_manager(list(workers.values()))
    # Attach observability + validation (Ch2.7 + Ch3.4)
    manager.step_callbacks = STEP_CALLBACKS
    manager.final_answer_checks = FINAL_ANSWER_CHECKS

    result = manager.run(task)
    try:
        check_coverage(result, manager.memory)
        verdict = "COMPLIANT ✓"
    except Exception as e:
        verdict = f"REJECT ✗ — {e}"
    mem_path = save_agent_memory(manager)
    return {"result": result, "verdict": verdict, "memory": str(mem_path), "log": log}
