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

    # Inject extracted RFP facts as a checklist + an explicit delegation workflow.
    # gpt-oss skips general instructions; numbered steps in the task itself work.
    entities, specs = extract_rfp_facts(chunks)
    facts = ""
    if entities or specs:
        facts = (
            "\n\nKey-fact CHECKLIST from the uploaded RFP (verify and expand via "
            "rfp_reader_agent; do not invent anything beyond the RFP and worker "
            f"outputs). Organizations/places: {', '.join(entities[:8]) or 'n/a'}. "
            f"Key figures: {', '.join(specs[:8]) or 'n/a'}."
        )
    workflow = (
        "\n\nExecute these steps IN ORDER as code actions before any final answer:\n"
        "1. rfp = rfp_reader_agent(task='Extract ALL RFP requirements: scope items, "
        "technical specs, capacity, budget, currency, deadlines, warranty MINIMUMS, "
        "submission documents. Quote section titles and page numbers.')\n"
        "2. price = pricing_agent(task='Build an itemized pricing table with "
        "calc_bid_table in the RFP currency (EGP) with 14% VAT, within the RFP budget.')\n"
        "3. comp = compliance_agent(task='Check every mandatory RFP requirement "
        "(warranty duration, delivery time, technical specs, documents) against the "
        "company profile and rfp findings. List PASS/FAIL per requirement.')\n"
        "4. final_answer: synthesize the bid ONLY from rfp/price/comp outputs, with "
        "sections Scope, Delivery, Warranty, Pricing, Compliance and real citations. "
        "Warranty and delivery must meet the RFP minimums or verdict REJECT."
    )
    result = manager.run(task + facts + workflow)
    # Re-run the FULL stack: smolagents' max-steps path skips final_answer_checks.
    verdict = run_verdict_checks(result, manager.memory, checks, agent=manager)
    # Keep memory inside the run's own directory (session-isolated in the UI).
    mem_path = save_agent_memory(manager, Path(source_dir) / "agent_memory.json")
    return {"result": result, "verdict": verdict, "memory": str(mem_path), "log": log}
