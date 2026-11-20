"""BM25, dense and hybrid (RRF) retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .index import Index
from .types import Hit

MODES = ("bm25", "dense", "hybrid")


def rrf_merge(lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: score(id) = sum over lists of 1 / (k + rank), 1-based rank."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def bm25_search(index: Index, query: str, n: int) -> list[tuple[str, float]]:
    """Top-n chunk ids by BM25 score; zero-score chunks are not candidates."""
    scores = index.bm25_scores(query)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [(index.chunks[i].id, scores[i]) for i in order[:n] if scores[i] > 0]


def dense_search(index: Index, query_vec: list[float], n: int) -> list[tuple[str, float]]:
    """Top-n chunk ids by cosine similarity."""
    return index.dense_all(query_vec)[:n]


@dataclass
class Retrieval:
    hits: list[Hit]
    confidence: float  # max dense cosine similarity among the dense candidates
    timings_ms: dict[str, float] = field(default_factory=dict)


def retrieve(
    index: Index,
    query: str,
    mode: str = "hybrid",
    top_k: int = 4,
    rrf_k: int = 60,
    candidates: int = 10,
) -> Retrieval:
    """Run both retrievers (the gate needs dense cosine in every mode); `mode` picks the ordering.

    bm25/dense_rank are None when the chunk is outside that retriever's top `candidates`.
    dense_score is the true cosine similarity for every chunk.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t0 = time.perf_counter()
    qvec = index.embeddings.embed_query(query)
    t1 = time.perf_counter()
    bm = bm25_search(index, query, candidates)
    t2 = time.perf_counter()
    full = index.dense_all(qvec)
    dense = full[:candidates]
    t3 = time.perf_counter()
    bm_rank = {cid: r for r, (cid, _) in enumerate(bm, start=1)}
    de_rank = {cid: r for r, (cid, _) in enumerate(dense, start=1)}
    sim = dict(full)
    rrf = rrf_merge([[c for c, _ in bm], [c for c, _ in dense]], k=rrf_k)
    t4 = time.perf_counter()

    if mode == "bm25":
        order = [c for c, _ in bm]
    elif mode == "dense":
        order = [c for c, _ in dense]
    else:
        order = sorted(rrf, key=lambda c: (-rrf[c], de_rank.get(c, 10**9), c))
    hits = [
        Hit(index.by_id[c], bm_rank.get(c), de_rank.get(c), sim[c], rrf.get(c, 0.0))
        for c in order[:top_k]
    ]
    return Retrieval(
        hits=hits,
        confidence=dense[0][1] if dense else 0.0,
        timings_ms={
            "embed": (t1 - t0) * 1000.0,
            "bm25": (t2 - t1) * 1000.0,
            "dense": (t3 - t2) * 1000.0,
            "rrf": (t4 - t3) * 1000.0,
        },
    )
