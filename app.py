"""Streamlit demo: LIVE SYSTEM / 06 Tender Bid Composer (HF-only)."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Tender Bid Composer (smolagents, HF-only)", layout="wide")
st.title("Tender Bid Composer")
st.caption("smolagents · HF Inference · BGE + FAISS · Manager + 4 workers · Validators · HF-only, no closed APIs")

sample_dir = Path("data/samples")
uploaded = st.file_uploader("Upload RFP (PDF/TXT/MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
company = st.text_area("Company profile", value=(sample_dir / "company_profile.txt").read_text(encoding="utf-8") if (sample_dir / "company_profile.txt").exists() else "Hassan Tech Supplies, Cairo. 3 years laptop supply, warranty service, tax registered.")
task = st.text_area("Task", value="Compose a compliant bid draft with Scope, Delivery, Warranty, Pricing, Compliance + citations.")

if st.button("Compose Bid", type="primary"):
    workdir = Path("data/work")
    workdir.mkdir(parents=True, exist_ok=True)
    for f in workdir.glob("*"):
        if f.is_file():
            f.unlink()
    if uploaded:
        for uf in uploaded:
            (workdir / uf.name).write_bytes(uf.getbuffer())
    else:
        for f in sample_dir.glob("*"):
            if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md"):
                (workdir / f.name).write_bytes(f.read_bytes())
    (workdir / "company_profile.txt").write_text(company, encoding="utf-8")

    from src.pipeline import run_pipeline

    log: list = []
    with st.spinner("Manager delegating to workers (HF Inference)..."):
        try:
            out = run_pipeline(workdir, task=task, log=log)
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.stop()
    st.subheader(f"Verdict: {out['verdict']}")
    tabs = st.tabs(["Bid Draft", "Trace", "Memory"])
    with tabs[0]:
        st.markdown(str(out["result"]))
    with tabs[1]:
        for entry in log:
            with st.expander(entry["type"]):
                st.text(entry["text"])
        st.info("Callbacks: PlanningStep/ActionStep (Ch2.7). Tokens shown on final step.")
    with tabs[2]:
        mem = Path(out["memory"])
        if mem.exists():
            st.download_button("Download agent_memory.json", mem.read_bytes(), file_name="agent_memory.json")
            st.code(mem.read_text(encoding="utf-8")[:4000])
