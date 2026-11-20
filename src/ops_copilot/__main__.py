"""Entry point: `uv run python -m ops_copilot`.

Binds 127.0.0.1:8000 by default (OPS_COPILOT_HOST / OPS_COPILOT_PORT override, e.g. for docker).
There is no authentication: do not expose this beyond localhost until the owner decides on auth.
"""

from __future__ import annotations

import os

import uvicorn

from .api import create_app
from .config import Settings, load_env_file
from .llm import get_chat_model, get_embeddings
from .pipeline import Pipeline


def main() -> None:
    load_env_file()  # only entry points may do this
    settings = Settings.from_env()

    def build() -> Pipeline:
        return Pipeline(settings, get_embeddings(settings), get_chat_model(settings))

    app = create_app(pipeline_factory=build)  # lazy: starts even if Ollama is down
    host = os.environ.get("OPS_COPILOT_HOST") or "127.0.0.1"
    port = int(os.environ.get("OPS_COPILOT_PORT") or "8000")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
