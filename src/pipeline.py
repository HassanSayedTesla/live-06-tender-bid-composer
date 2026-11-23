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
from src.validators import build_final_checks, extract_rfp_facts, run_verdict_checks


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
    # Attach observability + validation at construction (Ch2.7 + Ch3.4).
    # smolagents 1.26 builds a CallbackRegistry from step_callbacks in the
    # constructor; overwriting the attr would break step finalization.
    checks = build_final_checks(chunks)
    manager = build_manager(
        list(workers.values()),
        step_callbacks=STEP_CALLBACKS,
        final_answer_checks=checks,
    )

    # Inject extracted RFP facts so the manager cannot write from memory.
    entities, specs = extract_rfp_facts(chunks)
    facts = ""
    if entities or specs:
        facts = (
            "\n\nRFP key facts extracted from the uploaded documents — use ONLY "
            "these and the workers' findings; do not invent clauses, standards, "
            f"or figures. Organizations/places: {', '.join(entities[:8]) or 'n/a'}. "
            f"Key figures: {', '.join(specs[:8]) or 'n/a'}."
        )
    result = manager.run(task + facts)
    # Re-run the FULL stack: smolagents' max-steps path skips final_answer_checks.
    verdict = run_verdict_checks(result, manager.memory, checks, agent=manager)
    # Keep memory inside the run's own directory (session-isolated in the UI).
    mem_path = save_agent_memory(manager, Path(source_dir) / "agent_memory.json")
    return {"result": result, "verdict": verdict, "memory": str(mem_path), "log": log}
