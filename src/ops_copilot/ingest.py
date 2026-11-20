"""Load markdown documents and clean them (front matter, boilerplate, markup)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .types import Doc

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG = re.compile(r"</?[A-Za-z][^>\n]*>")
_BANNER = re.compile(r"^[ \t]*CONFIDENTIAL[ \t]*[—–-][^\n]*$", re.M | re.I)
_FOOTER = re.compile(
    r"^[ \t*_>]*Document control:[^\n]*$"
    r"|^(?=[^\n]*\bPage\s+\d+\s+of\s+\d+\b)(?=[^\n]*Uncontrolled when printed)[^\n]*$",
    re.M | re.I,
)
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITAL_STAR = re.compile(r"(?<![\*\w])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\*\w])")
_ITAL_US = re.compile(r"(?<![_\w])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![_\w])")
_CODE = re.compile(r"`([^`\n]+)`")


def split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    data = yaml.safe_load(m.group(1)) or {}
    meta = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    return meta, raw[m.end() :]


def clean(text: str) -> str:
    """Strip boilerplate and markup. Headings (#, ##, ###) are kept for chunking."""
    m = _FRONT.match(text)
    if m:
        text = text[m.end() :]
    text = _COMMENT.sub("", text)
    text = _BANNER.sub("", text)
    text = _FOOTER.sub("", text)
    text = _TAG.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _BOLD.sub(r"\2", text)
    text = _ITAL_STAR.sub(r"\1", text)
    text = _ITAL_US.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.replace("\r\n", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def load_docs(docs_dir: str | Path) -> list[Doc]:
    docs: list[Doc] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, _ = split_front_matter(raw)
        text = clean(raw)
        h1 = re.search(r"^#\s+(\S+)\s+(.*)$", text, re.M)
        doc_id = meta.get("doc_id") or (h1.group(1) if h1 else path.stem)
        title = meta.get("title") or (h1.group(2).strip() if h1 else doc_id)
        docs.append(Doc(doc_id=doc_id, title=title, text=text, metadata=meta))
    return docs
