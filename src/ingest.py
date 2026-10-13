"""Format-aware ingest: PDF (PyPDF) + TXT/MD. Ch2.1 pattern."""
from __future__ import annotations

from pathlib import Path


def load_documents(source_dir: str | Path):
    """Load PDF + TXT/MD documents from a directory.

    Returns list of langchain Document objects with metadata['source'].
    Lazy-imports langchain so unit tests stay light.
    """
    from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader

    source = Path(source_dir)
    docs = []
    if not source.exists():
        return docs

    pdfs = list(source.glob("*.pdf"))
    if pdfs:
        loader = PyPDFDirectoryLoader(str(source), mode="single")
        docs.extend(loader.load())

    for pattern in ("*.txt", "*.md"):
        for f in source.glob(pattern):
            try:
                docs.extend(TextLoader(str(f), encoding="utf-8").load())
            except Exception:
                continue
    return docs


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """RecursiveCharacterTextSplitter chunking. Ch2.1."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
