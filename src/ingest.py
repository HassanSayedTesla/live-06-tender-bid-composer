"""Format-aware ingest: PDF (pypdf) + TXT/MD. Ch2.1 pattern, no langchain."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """Minimal langchain-compatible document (page_content + metadata)."""

    page_content: str
    metadata: dict = field(default_factory=dict)


def load_documents(source_dir: str | Path) -> list[Document]:
    """Load PDF + TXT/MD documents from a directory.

    PDFs are loaded per page so citations can carry page numbers.
    """
    source = Path(source_dir)
    docs: list[Document] = []
    if not source.exists():
        return docs

    for f in sorted(source.glob("*.pdf")):
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(f))
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(
                        Document(text, {"source": f.name, "page": i + 1})
                    )
        except Exception:
            continue

    for pattern in ("*.txt", "*.md"):
        for f in sorted(source.glob(pattern)):
            try:
                text = f.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text:
                docs.append(Document(text, {"source": f.name}))
    return docs


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursive character splitting (same idea as langchain's splitter)."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    sep = separators[0]
    rest = separators[1:] or [""]
    parts = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    cur = ""
    for part in parts:
        piece = part + sep if sep else part
        if len(cur) + len(piece) <= chunk_size:
            cur += piece
            continue
        if cur.strip():
            chunks.append(cur)
        if len(piece) > chunk_size:
            chunks.extend(_split_text(piece, chunk_size, rest))
            cur = ""
        else:
            cur = piece
    if cur.strip():
        chunks.append(cur)
    return chunks


def split_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Chunk documents, keeping metadata and adding overlap context."""
    out: list[Document] = []
    for doc in documents:
        chunks = _split_text(doc.page_content, chunk_size, _SEPARATORS)
        for i, chunk in enumerate(chunks):
            if chunk_overlap > 0 and i > 0:
                chunk = chunks[i - 1][-chunk_overlap:] + chunk
            out.append(Document(chunk, dict(doc.metadata)))
    return out
