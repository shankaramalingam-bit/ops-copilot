"""Header-aware chunking: split on #/##/### then by size, with a context prefix."""

from __future__ import annotations

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .types import Chunk, Doc

CHUNK_CHARS = 1600  # ~400 tokens; the limit includes the context prefix
OVERLAP_CHARS = 200
NO_SECTION = "0"  # text before the first numbered heading (preamble)

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")


def parse_heading(heading: str) -> tuple[str, str] | None:
    m = _NUM.match(heading.strip())
    return (m.group(1), m.group(2).strip()) if m else None


def chunk_doc(doc: Doc, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    sections = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=True).split_text(doc.text)
    counters: dict[str, int] = {}
    out: list[Chunk] = []
    for sec in sections:
        section, section_title = NO_SECTION, ""
        for key in ("h2", "h3"):  # deepest numbered heading wins
            parsed = parse_heading(sec.metadata.get(key, ""))
            if parsed:
                section, section_title = parsed
        prefix = f"{doc.doc_id} {doc.title} > {section} {section_title}".rstrip() + "\n"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(size - len(prefix), overlap + 1), chunk_overlap=overlap
        )
        for body in splitter.split_text(sec.page_content):
            if not body.strip():
                continue
            n = counters.get(section, 0)
            counters[section] = n + 1
            out.append(
                Chunk(
                    id=f"{doc.doc_id}#{section}#{n}",
                    text=prefix + body,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    section_title=section_title,
                )
            )
    return out


def chunk_docs(docs: list[Doc]) -> list[Chunk]:
    return [c for d in docs for c in chunk_doc(d)]
