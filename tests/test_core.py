"""Offline tests for ingest, splitting, FAISS wiring, coercion, compliance."""
import json

from src.ingest import Document, _split_text, load_documents, split_documents
from src.models import coerce_final_answer_code
from src.tools.pricing_tools import check_compliance_item
from src.vectorstore import FaissStore


# --- ingest / splitter -----------------------------------------------------

def test_split_text_respects_chunk_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = _split_text(text, 300, ["\n\n", "\n", ". ", " ", ""])
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_split_documents_keeps_metadata_and_overlap():
    docs = [Document("a" * 500 + "\n\n" + "b" * 500, {"source": "rfp.txt", "page": 2})]
    chunks = split_documents(docs, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(c.metadata["source"] == "rfp.txt" for c in chunks)
    assert chunks[1].page_content.startswith("a" * 50)  # overlap tail


def test_load_documents_txt(tmp_path):
    (tmp_path / "rfp.txt").write_text("Delivery within 30 days.", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Warranty\n2 years onsite.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 2
    assert {d.metadata["source"] for d in docs} == {"rfp.txt", "notes.md"}


def test_load_documents_missing_dir(tmp_path):
    assert load_documents(tmp_path / "nope") == []


# --- FAISS wiring (fake embedder, no downloads) -----------------------------

class FakeEmbedder:
    """Deterministic 3-d embeddings keyed on words."""

    def _vec(self, text):
        t = text.lower()
        return [float("warranty" in t), float("delivery" in t), float("price" in t)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_faiss_store_retrieves_nearest():
    docs = [
        Document("Warranty: 2 years onsite support.", {"source": "rfp", "page": 1}),
        Document("Delivery within 30 days of PO.", {"source": "rfp", "page": 2}),
        Document("Price per unit in EGP.", {"source": "rfp", "page": 3}),
    ]
    store = FaissStore.from_documents(docs, FakeEmbedder())
    hits = store.similarity_search("warranty terms?", k=1)
    assert len(hits) == 1
    assert "Warranty" in hits[0].page_content
    assert hits[0].metadata["page"] == 1


# --- output coercion: rescue explicit answers, fail loud otherwise ----------

def test_coerce_xml_final_answer():
    out = coerce_final_answer_code("<final_answer># Bid\nScope...</final_answer>")
    assert out.startswith("<code>") and out.endswith("</code>")
    assert "final_answer(" in out and "# Bid" in out


def test_coerce_bare_final_answer_call():
    out = coerce_final_answer_code('final_answer("done")')
    assert "<code>" in out and 'final_answer("done")' in out


def test_coerce_strips_harmony_tokens():
    out = coerce_final_answer_code("<|start|>final_answer(\"ok\")<|end|>")
    assert "<|start|>" not in out and "<code>" in out


def test_coerce_leaves_plain_prose_untouched():
    prose = "I think we should first read the RFP and then price the items."
    assert coerce_final_answer_code(prose) == prose  # parser will reject -> retry


def test_coerce_leaves_existing_code_untouched():
    code = "<code>\nfinal_answer('x')\n</code>"
    assert coerce_final_answer_code(code) == code


# --- compliance heuristic ----------------------------------------------------

def _compliance_fn():
    fn = getattr(check_compliance_item, "func", None) or getattr(
        check_compliance_item, "__wrapped__", None
    )
    return fn or check_compliance_item


def test_compliance_pass_with_evidence():
    fn = _compliance_fn()
    out = fn(
        "Supplier must provide laptop warranty service and tax registration",
        "Hassan Tech Supplies: laptop supply, warranty service, tax registered.",
    )
    assert out.startswith("PASS")


def test_compliance_fail_without_evidence():
    fn = _compliance_fn()
    out = fn(
        "Supplier must hold ISO 9001 certification and audited financial statements",
        "We sell laptops and printers.",
    )
    assert out.startswith("FAIL")


def test_compliance_no_substring_false_positive():
    fn = _compliance_fn()
    # Old substring matcher passed on 'certification' alone; token coverage must not.
    out = fn(
        "Bidder must submit ISO 9001 certification audited annually by accredited body",
        "Our certification team handles paperwork.",
    )
    assert out.startswith("FAIL")


# --- anti-hallucination guards -----------------------------------------------

SOLAR_RFP = (
    "Falcon Logistics invites qualified renewable-energy contractors to propose "
    "a rooftop solar photovoltaic system for a warehouse facility in 10th of "
    "Ramadan City, Egypt. RFP-2026-005. Estimated budget EGP 3,500,000. "
    "Target commissioning within 16 weeks of contract award. "
    "Design a nominal 150 kWp solar PV installation. "
    "Minimum five-year workmanship warranty; equipment warranties specified separately."
)


def _solar_chunks():
    return [Document(SOLAR_RFP, {"source": "rfp_solar.pdf", "page": 1})]


def test_placeholders_rejected():
    from src.validators import check_no_placeholders

    try:
        check_no_placeholders("Bid Draft - [Project Title]\nPrepared by: [Your Company Name]")
    except ValueError as e:
        assert "placeholders" in str(e)
        return
    raise AssertionError("should have raised")


def test_markdown_links_and_refs_not_placeholders():
    from src.validators import check_no_placeholders

    assert check_no_placeholders("See [ISO 9001](https://iso.org) and reference [1].") is True


def test_grounding_rejects_generic_template():
    from src.validators import make_grounding_check

    check = make_grounding_check(_solar_chunks())
    generic_bid = (
        "Bid Draft - [Project Title]. Scope: design & engineering per IEC 61850. "
        "Delivery: 20 weeks. Warranty: 24 months. Pricing: USD 102,000 incl. 20% VAT. "
        "Compliance: ISO 9001, OSHA, GDPR. " + "x" * 200
    )
    try:
        check(generic_bid)
    except ValueError as e:
        assert "not grounded" in str(e)
        return
    raise AssertionError("generic template should be rejected")


def test_grounding_accepts_rfp_based_bid():
    from src.validators import make_grounding_check

    check = make_grounding_check(_solar_chunks())
    grounded = (
        "Bid for Falcon Logistics rooftop solar PV, 10th of Ramadan City. "
        "Scope: 150 kWp installation. Delivery: commissioning within 16 weeks. "
        "Warranty: 5-year workmanship. Pricing: EGP 3,450,000 + 14% VAT. "
        "Compliance: all RFP-2026-005 requirements met. " + "x" * 200
    )
    assert check(grounded) is True


class FakeMemory:
    def __init__(self, code):
        self._code = code

    def get_succinct_steps(self):
        return [{"type": "action", "code": self._code}]


def test_delegation_rejects_direct_answer():
    from src.validators import check_delegation

    try:
        check_delegation("bid", FakeMemory("final_answer('a generic bid')"))
    except ValueError as e:
        assert "delegating" in str(e)
        return
    raise AssertionError("should have raised")


def test_delegation_accepts_worker_calls():
    from src.validators import check_delegation

    mem = FakeMemory(
        "facts = rfp_reader_agent(task='extract requirements')\n"
        "table = pricing_agent(task='price the BOQ')\n"
        "final_answer(bid)"
    )
    assert check_delegation("bid", mem) is True


# --- pricing -----------------------------------------------------------------

def test_calc_bid_table_vat():
    from src.tools.pricing_tools import calc_bid_table

    fn = getattr(calc_bid_table, "func", None) or getattr(calc_bid_table, "__wrapped__", None)
    if fn is None:
        return
    items = json.dumps([{"item": "Laptop i7", "qty": 50, "unit_price_egp": 25000}])
    csv_text = fn(items)
    assert "TOTAL" in csv_text and "VAT" in csv_text
    assert "1425000.0" in csv_text  # 1,250,000 + 14% VAT
