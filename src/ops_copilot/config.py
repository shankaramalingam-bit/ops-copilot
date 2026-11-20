"""Settings, read from os.environ only. Nothing here loads a .env file on import."""

from __future__ import annotations

import math
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


class Secret:
    """Holds a secret string. repr/str are masked; read it only via .get().

    Slots (no __dict__) and copy/pickle never expose the value, so dataclasses.asdict, vars(),
    json default=str and logging of a Settings object cannot leak it.
    """

    __slots__ = ("_v",)

    def __init__(self, value: str) -> None:
        self._v = value

    def get(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._v)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._v == self._v

    __hash__ = None  # type: ignore[assignment]

    def __copy__(self) -> Secret:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Secret:
        return self

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("Secret cannot be pickled")


@dataclass
class Settings:
    llm_provider: str = "ollama"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: Secret | str | None = field(default=None, repr=False)
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
        key = self.groq_api_key
        if isinstance(key, str):
            key = Secret(key) if key.strip() else None
        elif isinstance(key, Secret) and not key.get().strip():
            key = None
        self.groq_api_key = key
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ConfigError("LLM_PROVIDER=groq requires GROQ_API_KEY to be set")
        if not (math.isfinite(self.gate_threshold) and 0.0 <= self.gate_threshold <= 1.0):
            raise ConfigError("gate_threshold must be a finite number in [0, 1]")
        if not 1 <= self.top_k <= 20:
            raise ConfigError("top_k must be in 1..20")
        if not self.top_k <= self.candidates <= 100:
            raise ConfigError("candidates must be in top_k..100")
        if not 1 <= self.rrf_k <= 1000:
            raise ConfigError("rrf_k must be in 1..1000")

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
