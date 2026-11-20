"""Chat model and embeddings factories (injectable; imports are lazy)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .config import Secret, Settings

LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 1


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        key = settings.groq_api_key

        return ChatGroq(
            model=settings.groq_model,
            temperature=0,
            timeout=LLM_TIMEOUT_S,
            max_retries=LLM_MAX_RETRIES,
            api_key=SecretStr(key.get() if isinstance(key, Secret) else (key or "")),
        )
    from langchain_ollama import ChatOllama

    # ChatOllama has no retry option; the timeout is passed to the underlying httpx client.
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        client_kwargs={"timeout": LLM_TIMEOUT_S},
    )


def get_embeddings(settings: Settings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
