"""Entry point: `uv run python -m ops_copilot`.

Binds 127.0.0.1:8000 by default (OPS_COPILOT_HOST / OPS_COPILOT_PORT override). There is no
authentication, so a non-loopback host is refused (exit 2) unless OPS_COPILOT_ALLOW_NONLOCAL=1
(e.g. inside docker, where the published port is itself bound to loopback on the host).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys
from collections.abc import Mapping

import uvicorn

from .api import create_app
from .config import Settings, load_env_file
from .llm import get_chat_model, get_embeddings
from .pipeline import Pipeline

log = logging.getLogger("ops_copilot")


def _is_loopback(host: str) -> bool:
    h = host.strip().strip("[]")
    if h.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def bind_problem(host: str, env: Mapping[str, str]) -> str | None:
    """None if binding `host` is allowed, else the refusal message. Pure: no side effects."""
    if _is_loopback(host) or env.get("OPS_COPILOT_ALLOW_NONLOCAL") == "1":
        return None
    return (
        f"refusing to bind non-loopback host {host!r}: this service has no authentication. "
        "Set OPS_COPILOT_ALLOW_NONLOCAL=1 to override (and put auth in front of it)."
    )


def main() -> None:
    load_env_file()  # only entry points may do this
    host = os.environ.get("OPS_COPILOT_HOST") or "127.0.0.1"
    problem = bind_problem(host, os.environ)
    if problem:
        print(problem, file=sys.stderr)
        sys.exit(2)
    if not _is_loopback(host):
        logging.basicConfig(level=logging.WARNING)
        log.warning("binding %s: there is NO authentication on this service", host)
    settings = Settings.from_env()

    def build() -> Pipeline:
        return Pipeline(settings, get_embeddings(settings), get_chat_model(settings))

    app = create_app(pipeline_factory=build)  # lazy: starts even if Ollama is down
    port = int(os.environ.get("OPS_COPILOT_PORT") or "8000")
    uvicorn.run(app, host=host, port=port, log_level="info", server_header=False)


if __name__ == "__main__":
    main()
