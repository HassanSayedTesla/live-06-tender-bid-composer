"""Class-based stateful RAG tool. Ch2.3 / Ch2.4 pattern."""
from __future__ import annotations

from smolagents import Tool


class RfpSearchTool(Tool):
    name = "rfp_search"
    description = (
        "Search RFP documentation for requirements, delivery terms, "
        "warranty, eligibility, and compliance information."
    )
    inputs = {
        "query": {"type": "string", "description": "Natural language RFP query"}
    }
    output_type = "string"

    def __init__(self, vectorstore, k: int = 6):
        super().__init__()
        self.vectorstore = vectorstore
        self.k = k

    def forward(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query, k=self.k)
        return "\n\n".join(d.page_content for d in docs) or "Nothing found."
