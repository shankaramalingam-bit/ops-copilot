"""Settings, read from os.environ only. Nothing here loads a .env file on import."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

PROVIDERS = ("ollama", "groq")


class ConfigError(ValueError):
    """Invalid configuration. Messages never include environment values."""


def load_env_file() -> None:
    """Load a .env file into os.environ. Entry points call this; library code never does."""
    from dotenv import load_dotenv

    load_dotenv()


@dataclass
class Settings:
    llm_provider: str = "ollama"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: str | None = field(default=None, repr=False)
    groq_model: str = "llama-3.1-8b-instant"
    embed_model: str = "nomic-embed-text"
    top_k: int = 4
    rrf_k: int = 60
    # Placeholder: the real value is calibrated later on a held-out set, NOT the eval set.
    gate_threshold: float = 0.55
    candidates: int = 10
    docs_dir: str = "data/docs"
    chroma_dir: str = "data/cache/chroma"

    def __post_init__(self) -> None:
        self.llm_provider = self.llm_provider.strip().lower()
        if self.llm_provider not in PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER must be one of {PROVIDERS}")
        if self.groq_api_key is not None and not self.groq_api_key.strip():
            self.groq_api_key = None
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ConfigError("LLM_PROVIDER=groq requires GROQ_API_KEY to be set")
        if self.top_k < 1 or self.rrf_k < 1 or self.candidates < 1:
            raise ConfigError("top_k, rrf_k and candidates must be >= 1")

    def __repr__(self) -> str:
        key = "***" if self.groq_api_key else "None"
        parts = [
            f"{f}={getattr(self, f)!r}" for f in self.__dataclass_fields__ if f != "groq_api_key"
        ]
        return f"Settings({', '.join(parts)}, groq_api_key={key})"

    __str__ = __repr__

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        d = cls()  # defaults

        def get(name: str, default: str) -> str:
            v = e.get(name)
            return v if v else default

        try:
            return cls(
                llm_provider=get("LLM_PROVIDER", d.llm_provider),
                ollama_model=get("OLLAMA_MODEL", d.ollama_model),
                ollama_base_url=get("OLLAMA_BASE_URL", d.ollama_base_url),
                groq_api_key=e.get("GROQ_API_KEY") or None,
                groq_model=get("GROQ_MODEL", d.groq_model),
                embed_model=get("EMBED_MODEL", d.embed_model),
                top_k=int(get("TOP_K", str(d.top_k))),
                rrf_k=int(get("RRF_K", str(d.rrf_k))),
                gate_threshold=float(get("GATE_THRESHOLD", str(d.gate_threshold))),
                candidates=int(get("CANDIDATES", str(d.candidates))),
                docs_dir=get("DOCS_DIR", d.docs_dir),
                chroma_dir=get("CHROMA_DIR", d.chroma_dir),
            )
        except ValueError as exc:
            if isinstance(exc, ConfigError):
                raise
            # int()/float() errors would echo the offending value; do not.
            raise ConfigError("a numeric setting has a non-numeric value") from None
