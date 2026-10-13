"""FAISS + embeddings. Local-first (zero credits), HF endpoint optional."""
from __future__ import annotations


def get_embedder(
    model_id: str = "BAAI/bge-base-en-v1.5", mode: str | None = None
):
    """Local sentence-transformers by default; HF endpoint if mode='endpoint'.

    Local mode needs no Inference credits (fixes HTTP 402 on free tier).
    """
    from src.config import EMBEDDING_MODE, LOCAL_EMBEDDING_MODEL_ID

    mode = (mode or EMBEDDING_MODE).strip().lower()
    if mode == "endpoint":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=model_id, task="feature-extraction"
        )
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL_ID)


def build_faiss(chunks, embedder=None, model_id: str = "BAAI/bge-base-en-v1.5"):
    from langchain_community.vectorstores import FAISS

    embedder = embedder or get_embedder(model_id)
    return FAISS.from_documents(chunks, embedder)


def similarity_context(vector_store, query: str, k: int = 6) -> str:
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs) or "Nothing found."
