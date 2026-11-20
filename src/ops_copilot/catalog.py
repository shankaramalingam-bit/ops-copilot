"""Read-only loaders for the document catalog, simulated dashboard data and eval results."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .chunk import parse_heading
from .ingest import load_docs

_HEADING = re.compile(r"^(#{2,3})\s+(.*)$", re.M)


def load_catalog(docs_dir: str | Path) -> list[dict[str, Any]]:
    """One entry per document: front matter plus numbered ## / ### headings. No LLM."""
    out: list[dict[str, Any]] = []
    for doc in load_docs(docs_dir):
        sections = []
        for m in _HEADING.finditer(doc.text):
            parsed = parse_heading(m.group(2))
            if parsed:
                sections.append({"section": parsed[0], "title": parsed[1]})
        out.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "owner_role": doc.metadata.get("owner_role", ""),
                "version": doc.metadata.get("version", ""),
                "effective_date": doc.metadata.get("effective_date", ""),
                "sections": sections,
            }
        )
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dashboard(sim_dir: str | Path) -> dict[str, Any]:
    """Combine data/sim/{kpis,receipts,purchase_orders,suppliers}.json (read-only)."""
    d = Path(sim_dir)
    result: dict[str, Any] = {}
    for key in ("kpis", "receipts", "purchase_orders", "suppliers"):
        data = _read_json(d / f"{key}.json")
        if key != "kpis" and isinstance(data, dict) and key in data:
            data = data[key]
        result[key] = data
    return result


def load_eval_latest(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.is_file():
        return None
    return _read_json(p)
