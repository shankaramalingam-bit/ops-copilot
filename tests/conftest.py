"""Shared pytest configuration. Tests must run offline: fakes only, no Ollama/Groq/telemetry."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ops_copilot.config import Settings
from ops_copilot.pipeline import Pipeline

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "docs"


class HashedBowEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings, L2-normalised. No network."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class CountingChat(FakeListChatModel):
    """Fake chat model that counts calls."""

    calls: int = 0

    def _call(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        object.__setattr__(self, "calls", self.calls + 1)
        return super()._call(*args, **kwargs)


@pytest.fixture
def embeddings() -> HashedBowEmbeddings:
    return HashedBowEmbeddings()


@pytest.fixture
def make_pipeline(tmp_path: Path, embeddings: HashedBowEmbeddings):
    def _make(response: str = "x", threshold: float = 1e-9, **kw) -> tuple[Pipeline, CountingChat]:
        chat = CountingChat(responses=[response])
        settings = Settings(
            gate_threshold=threshold,
            chroma_dir=str(tmp_path / "chroma"),
            docs_dir=str(FIXTURE_DOCS),
            **kw,
        )
        return Pipeline(settings, embeddings, chat, FIXTURE_DOCS), chat

    return _make
