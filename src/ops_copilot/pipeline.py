"""End to end: retrieve -> gate -> grounded answer -> validated citations."""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from . import gate
from .answer import REFUSAL, build_messages, finalize
from .chunk import chunk_docs
from .config import Settings
from .index import Index
from .ingest import load_docs
from .llm import get_chat_model, get_embeddings
from .retrieve import retrieve
from .types import AskResult, ChunkOut, Hit, TimingsOut


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        chat_model: BaseChatModel | None = None,
        docs_dir: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings or get_embeddings(settings)
        self.chat_model = chat_model or get_chat_model(settings)
        chunks = chunk_docs(load_docs(docs_dir or settings.docs_dir))
        self.index = Index(chunks, self.embeddings, settings.chroma_dir, settings.embed_model)

    @staticmethod
    def _chunks_out(hits: list[Hit], used: list[bool]) -> list[ChunkOut]:
        return [
            {
                "id": h.chunk.id,
                "text": h.chunk.text,
                "bm25_rank": h.bm25_rank,
                "dense_rank": h.dense_rank,
                "rrf": h.rrf,
                "used": u,
            }
            for h, u in zip(hits, used, strict=True)
        ]

    def _refusal(
        self, reason: str, confidence: float, hits: list[Hit], timings: TimingsOut
    ) -> AskResult:
        """Every refusal: REFUSAL text, no citations, chunks (if any) all unused."""
        return {
            "answer": REFUSAL,
            "citations": [],
            "refused": True,
            "refusal_reason": reason,
            "confidence": confidence,
            "chunks": self._chunks_out(hits, [False] * len(hits)),
            "timings_ms": timings,
        }

    def ask(self, question: str, mode: str = "hybrid") -> AskResult:
        s = self.settings
        if not question.strip():
            t = time.perf_counter()
            decision = gate.check(0.0, s.gate_threshold)
            zero: TimingsOut = {
                "embed": 0.0,
                "bm25": 0.0,
                "dense": 0.0,
                "rrf": 0.0,
                "gate": (time.perf_counter() - t) * 1000.0,
                "llm": 0.0,
            }
            return self._refusal("empty_question", decision.confidence, [], zero)
        r = retrieve(self.index, question, mode, s.top_k, s.rrf_k, s.candidates)
        t = time.perf_counter()
        decision = gate.check(r.confidence, s.gate_threshold)
        gate_ms = (time.perf_counter() - t) * 1000.0
        timings: TimingsOut = {
            "embed": r.timings_ms["embed"],
            "bm25": r.timings_ms["bm25"],
            "dense": r.timings_ms["dense"],
            "rrf": r.timings_ms["rrf"],
            "gate": gate_ms,
            "llm": 0.0,
        }

        if not decision.passed:
            return self._refusal("below_gate", r.confidence, r.hits, timings)
        if not r.hits:
            return self._refusal("no_context", r.confidence, r.hits, timings)

        t = time.perf_counter()
        msg = self.chat_model.invoke(build_messages(question, r.hits))
        timings["llm"] = (time.perf_counter() - t) * 1000.0
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        f = finalize(content, r.hits)
        return {
            "answer": f.answer,
            "citations": f.citations,
            "refused": f.refused,
            "refusal_reason": f.refusal_reason,
            "confidence": r.confidence,
            "chunks": self._chunks_out(r.hits, f.used),
            "timings_ms": timings,
        }
