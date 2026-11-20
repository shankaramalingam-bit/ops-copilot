"""FastAPI app: /api/ask, /api/docs, /api/dashboard, /api/eval/latest, static web/.

Binds to localhost by default (see __main__). There is NO authentication: exposing this beyond
localhost requires an auth decision from the owner first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from .catalog import load_catalog, load_dashboard, load_eval_latest
from .config import MAX_QUESTION_CHARS
from .pipeline import Pipeline
from .types import AskResult

log = logging.getLogger("ops_copilot.api")

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BODY_BYTES = 16 * 1024
ASK_DEADLINE_S = 90.0
QUEUE_WAIT_S = 5.0
MAX_CONCURRENT_ASKS = 2

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
)


class AskRequest(BaseModel):
    """The only client-controlled inputs. Anything else is rejected (extra=forbid)."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


def _quiet_loggers() -> None:
    for name in ("groq", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _err(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


async def _read_capped(request: Request) -> bytes | None:
    """Read the body, giving up (None) as soon as it exceeds MAX_BODY_BYTES."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_BODY_BYTES:
                return None
        except ValueError:
            return None
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > MAX_BODY_BYTES:
            return None
    return bytes(buf)


def create_app(
    pipeline: Pipeline | None = None,
    *,
    pipeline_factory: Callable[[], Pipeline] | None = None,
    docs_dir: str | Path | None = None,
    sim_dir: str | Path | None = None,
    eval_path: str | Path | None = None,
    web_dir: str | Path | None = None,
    deadline_s: float = ASK_DEADLINE_S,
    max_concurrent: int = MAX_CONCURRENT_ASKS,
    queue_wait_s: float = QUEUE_WAIT_S,
) -> FastAPI:
    _quiet_loggers()
    docs_p = Path(docs_dir) if docs_dir else REPO_ROOT / "data" / "docs"
    sim_p = Path(sim_dir) if sim_dir else REPO_ROOT / "data" / "sim"
    eval_p = Path(eval_path) if eval_path else REPO_ROOT / "eval" / "latest.json"
    web_p = Path(web_dir) if web_dir else REPO_ROOT / "web"

    app = FastAPI(debug=False, title="Ops Copilot", docs_url=None, redoc_url=None, openapi_url=None)
    state: dict[str, Pipeline | None] = {"pipeline": pipeline}
    build_lock = threading.Lock()
    sem = asyncio.Semaphore(max_concurrent)

    def get_pipeline() -> Pipeline:
        """Lazy build (blocking; run in a thread). A failed build is not cached."""
        with build_lock:
            p = state["pipeline"]
            if p is None:
                if pipeline_factory is None:
                    raise RuntimeError("no pipeline configured")
                p = pipeline_factory()
                state["pipeline"] = p
            return p

    def run_ask(question: str, mode: str) -> AskResult | None:
        try:
            p = get_pipeline()
        except Exception as exc:
            log.warning("pipeline unavailable: %s", type(exc).__name__)
            return None
        return p.ask(question, mode)

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if not request.url.path.startswith("/api/"):
            response.headers["Content-Security-Policy"] = CSP
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, __: RequestValidationError) -> JSONResponse:
        return _err(422, "invalid request")  # never echo the input

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        return _err(exc.status_code, exc.detail if isinstance(exc.detail, str) else "error")

    @app.post("/api/ask")
    async def ask(request: Request) -> JSONResponse:
        body = await _read_capped(request)
        if body is None:
            return _err(413, "request body too large")
        try:
            req = AskRequest.model_validate(json.loads(body))
        except (ValueError, ValidationError):  # JSONDecodeError/UnicodeDecodeError are ValueError
            return _err(422, "invalid request")

        try:
            await asyncio.wait_for(sem.acquire(), timeout=queue_wait_s)
        except TimeoutError:
            return _err(503, "server busy")
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(run_ask, req.question, req.mode), timeout=deadline_s
            )
        except TimeoutError:
            log.warning("ask timed out")
            return _err(504, "upstream timeout")
        except Exception as exc:
            log.warning("ask failed: %s", type(exc).__name__)  # class only, never the message
            return _err(502, "upstream model error")
        finally:
            sem.release()
        if result is None:
            return _err(503, "model backend unavailable")
        return JSONResponse(result)

    @app.get("/api/docs")
    async def docs() -> list[dict[str, Any]]:
        return await run_in_threadpool(load_catalog, docs_p)

    @app.get("/api/dashboard")
    async def dashboard() -> dict[str, Any]:
        return await run_in_threadpool(load_dashboard, sim_p)

    @app.get("/api/eval/latest")
    async def eval_latest() -> Any:
        data = await run_in_threadpool(load_eval_latest, eval_p)
        if data is None:
            return _err(404, "no eval run yet")
        return JSONResponse(data)

    if (web_p / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_p, html=True), name="web")
    return app
