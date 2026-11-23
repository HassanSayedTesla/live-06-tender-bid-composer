"""FAISS + embeddings. Local-first via fastembed (ONNX, no torch), HF endpoint optional."""
from __future__ import annotations

import os


class LocalEmbedder:
    """fastembed ONNX embeddings — no torch/transformers, zero credits."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class EndpointEmbedder:
    """HF Inference feature-extraction endpoint (needs HF_TOKEN + credits)."""

    def __init__(self, model_id: str):
        from huggingface_hub import InferenceClient

        self._client = InferenceClient(model=model_id, token=os.getenv("HF_TOKEN") or None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in self._client.feature_extraction(t)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedder(model_id: str = "BAAI/bge-base-en-v1.5", mode: str | None = None):
    """Local fastembed by default; HF endpoint if mode='endpoint'.

    Local mode needs no Inference credits (fixes HTTP 402 on free tier).
    """
    from src.config import EMBEDDING_MODE, LOCAL_EMBEDDING_MODEL_ID

    mode = (mode or EMBEDDING_MODE).strip().lower()
    if mode == "endpoint":
        return EndpointEmbedder(model_id)
    return LocalEmbedder(LOCAL_EMBEDDING_MODEL_ID)


class FaissStore:
    """Cosine-similarity FAISS index over Documents (langchain-free)."""

    def __init__(self, index, documents: list, embedder):
        self.index = index
        self.documents = documents
        self.embedder = embedder

    @classmethod
    def from_documents(cls, documents: list, embedder) -> "FaissStore":
        import faiss
        import numpy as np

        vectors = np.asarray(
            embedder.embed_documents([d.page_content for d in documents]),
            dtype="float32",
        )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, documents, embedder)

    def similarity_search(self, query: str, k: int = 6) -> list:
        import faiss
        import numpy as np

        q = np.asarray([self.embedder.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        _, ids = self.index.search(q, min(k, len(self.documents)))
        return [self.documents[i] for i in ids[0] if i >= 0]


def build_faiss(chunks, embedder=None, model_id: str = "BAAI/bge-base-en-v1.5"):
    embedder = embedder or get_embedder(model_id)
    return FaissStore.from_documents(chunks, embedder)


def similarity_context(vector_store, query: str, k: int = 6) -> str:
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs) or "Nothing found."
