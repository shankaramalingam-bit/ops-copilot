"""Chat model and embeddings factories (injectable; imports are lazy)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .config import Secret, Settings

# Worst case per /api/ask must stay below api.ASK_DEADLINE_S (90 s):
#   groq:   embed 25 + 30 x (1 + 1 retry) = 85;   ollama: embed 25 + 60 = 85.
GROQ_TIMEOUT_S = 30
GROQ_MAX_RETRIES = 1
OLLAMA_TIMEOUT_S = 60
EMBED_TIMEOUT_S = 25
MAX_OUTPUT_TOKENS = 512  # output cap: answers are short; bounds cost and finalize() input


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        key = settings.groq_api_key

        return ChatGroq(
            model=settings.groq_model,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=GROQ_TIMEOUT_S,
            max_retries=GROQ_MAX_RETRIES,
            api_key=SecretStr(key.get() if isinstance(key, Secret) else (key or "")),
        )
    from langchain_ollama import ChatOllama

    # ChatOllama has no retry option; the timeout is passed to the underlying httpx client.
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=MAX_OUTPUT_TOKENS,
        client_kwargs={"timeout": OLLAMA_TIMEOUT_S},
    )


def get_embeddings(settings: Settings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": EMBED_TIMEOUT_S},
    )
