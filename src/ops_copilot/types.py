"""Shared result types. AskResult is exactly the API contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class CitationOut(TypedDict):
    doc_id: str
    section: str
    valid: bool


class ChunkOut(TypedDict):
    id: str
    text: str
    bm25_rank: int | None
    dense_rank: int | None
    rrf: float
    used: bool


class TimingsOut(TypedDict):
    embed: float
    bm25: float
    dense: float
    rrf: float
    gate: float
    llm: float


class AskResult(TypedDict):
    answer: str
    citations: list[CitationOut]
    refused: bool
    refusal_reason: str | None
    confidence: float
    chunks: list[ChunkOut]
    timings_ms: TimingsOut


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<section>#<n>"
    text: str  # prefixed text: what is embedded, indexed and shown to the model
    doc_id: str
    title: str
    section: str
    section_title: str


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with its rank fields."""

    chunk: Chunk
    bm25_rank: int | None
    dense_rank: int | None
    dense_score: float
    rrf: float
