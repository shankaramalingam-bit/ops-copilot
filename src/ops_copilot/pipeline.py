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

    def ask(self, question: str, mode: str = "hybrid") -> AskResult:
        s = self.settings
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
            return {
                "answer": REFUSAL,
                "citations": [],
                "refused": True,
                "refusal_reason": "below_gate",
                "confidence": r.confidence,
                "chunks": self._chunks_out(r.hits, [False] * len(r.hits)),
                "timings_ms": timings,
            }

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
