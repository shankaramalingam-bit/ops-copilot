"""Chroma (cosine, persisted) + BM25 indexes over the chunks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from .types import Chunk

COLLECTION = "meridian_docs"
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def corpus_hash(chunks: list[Chunk], embed_model: str) -> str:
    h = hashlib.sha256(embed_model.encode())
    for c in chunks:
        h.update(b"\0" + c.id.encode() + b"\0" + c.text.encode())
    return h.hexdigest()


class Index:
    """Holds the chunks, a BM25 index and a persisted Chroma collection."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: Embeddings,
        chroma_dir: str | Path,
        embed_model: str = "",
    ) -> None:
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
        path = Path(chroma_dir)
        path.mkdir(parents=True, exist_ok=True)
        # _env_file=None: Chroma must never implicitly read a .env from the working directory.
        chroma_settings = ChromaSettings(
            anonymized_telemetry=False,
            _env_file=None,  # type: ignore[call-arg]
        )
        self._client = chromadb.PersistentClient(path=str(path), settings=chroma_settings)
        self._hash_file = path / "corpus.sha256"
        self.rebuilt = self._sync(corpus_hash(chunks, embed_model))

    def _sync(self, digest: str) -> bool:
        """Rebuild the collection when the corpus content hash changed. True if rebuilt."""
        existing = {c if isinstance(c, str) else c.name for c in self._client.list_collections()}
        stored = self._hash_file.read_text().strip() if self._hash_file.exists() else ""
        if COLLECTION in existing and stored == digest:
            self._col = self._client.get_collection(COLLECTION)
            return False
        if COLLECTION in existing:
            self._client.delete_collection(COLLECTION)
        self._col = self._client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        if self.chunks:
            vectors = self.embeddings.embed_documents([c.text for c in self.chunks])
            self._col.add(
                ids=[c.id for c in self.chunks],
                embeddings=vectors,  # type: ignore[arg-type]
                documents=[c.text for c in self.chunks],
                metadatas=[{"doc_id": c.doc_id, "section": c.section} for c in self.chunks],
            )
        self._hash_file.write_text(digest)
        return True

    def bm25_scores(self, query: str) -> list[float]:
        if self.bm25 is None:
            return []
        return [float(s) for s in self.bm25.get_scores(tokenize(query))]

    def dense_all(self, query_vec: list[float]) -> list[tuple[str, float]]:
        """All chunks ranked by cosine similarity (1 - cosine distance), best first."""
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[query_vec],  # type: ignore[arg-type]
            n_results=n,
            include=["distances"],
        )
        ids = res["ids"][0]
        dists = (res["distances"] or [[]])[0]
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists, strict=True)]
