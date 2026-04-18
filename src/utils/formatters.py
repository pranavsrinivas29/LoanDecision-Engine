from __future__ import annotations

import json
import os


def format_dict_pretty(data) -> str:
    """
    Pretty-print dict-like data as JSON string.
    """
    return json.dumps(data, indent=2, default=str)


def format_retrieved_context(retrieved_docs) -> str:
    """
    Convert retrieved docs into a readable grounding string.
    """
    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        content = doc.page_content.strip()
        parts.append(f"[Document {i}: {source}]\n{content}")
    return "\n\n".join(parts)