"""The REAL corpus (data/docs) through ingest + clean + chunk."""

from __future__ import annotations

from pathlib import Path

import pytest

from ops_copilot.chunk import CHUNK_CHARS, NO_SECTION, chunk_doc
from ops_copilot.ingest import load_docs

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
DOCS = load_docs(DOCS_DIR)
FORBIDDEN = ("CONFIDENTIAL", "Document control:", "Uncontrolled when printed", "<!--")


def test_positive_control_ten_docs_with_raw_boilerplate():
    # A scan over an empty/clean-by-construction set proves nothing: the raw files DO contain these.
    assert len(DOCS) == 10
    raw = "\n".join(p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.glob("*.md")))
    for marker in FORBIDDEN:
        assert marker in raw, marker


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_cleaned_doc_has_no_boilerplate(doc):
    for n, line in enumerate(doc.text.split("\n"), 1):
        for marker in FORBIDDEN:
            assert marker not in line, f"{doc.doc_id}:{n} contains {marker!r}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_doc_chunks(doc):
    chunks = chunk_doc(doc)
    assert len(chunks) >= 3
    numbered = {c.section for c in chunks if c.section != NO_SECTION}
    assert len(numbered) >= 3
    assert all(c.doc_id == doc.doc_id for c in chunks)
    assert all(len(c.text) <= CHUNK_CHARS == 1600 for c in chunks)
    for c in chunks:
        for marker in FORBIDDEN:
            assert marker not in c.text
