"""Chat model and embeddings factories (injectable; imports are lazy)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        return ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=SecretStr(settings.groq_api_key or ""),
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)


def get_embeddings(settings: Settings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
