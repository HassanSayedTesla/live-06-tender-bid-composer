"""FAISS + HF embeddings. Ch2.2 pattern, HF-only with local fallback."""
from __future__ import annotations


def get_embedder(model_id: str = "BAAI/bge-base-en-v1.5"):
    """Prefer HF Inference endpoint, fall back to local sentence-transformers."""
    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=model_id, task="feature-extraction"
        )
    except Exception:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_id)


def build_faiss(chunks, embedder=None, model_id: str = "BAAI/bge-base-en-v1.5"):
    from langchain_community.vectorstores import FAISS

    embedder = embedder or get_embedder(model_id)
    return FAISS.from_documents(chunks, embedder)


def similarity_context(vector_store, query: str, k: int = 6) -> str:
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs) or "Nothing found."
