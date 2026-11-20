"""FastAPI app: /api/ask, /api/docs, /api/dashboard, /api/eval/latest, static web/.

Binds to localhost by default (see __main__). There is NO authentication: exposing this beyond
localhost requires an auth decision from the owner first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
SECURITY_HEADERS: dict[bytes, bytes] = {
    b"content-security-policy": CSP.encode(),
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
}
# refusal_reason is a bounded enum: the API never forwards any other string to the browser.
REFUSAL_REASONS = frozenset(
    {
        "below_gate",
        "model_refused",
        "no_valid_citations",
        "invalid_citation",
        "invalid_characters",
        "empty_question",
        "no_context",
        "question_too_long",
    }
)
REFUSAL_FALLBACK = "refused"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")


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


def _safe(obj: Any) -> Any:
    """Map non-finite floats to None (JSON has no NaN/Infinity; never 500 on them)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_safe(v) for v in obj]
    return obj


def _bound_reason(result: AskResult) -> AskResult:
    """Unknown refusal_reason -> "refused" (None stays None)."""
    reason = result.get("refusal_reason")
    if reason is None or (isinstance(reason, str) and reason in REFUSAL_REASONS):
        return result
    return {**result, "refusal_reason": REFUSAL_FALLBACK}  # type: ignore[typeddict-item]


def _bare_host(host: str) -> str:
    return host.strip().lower().strip("[]")


def allowed_hosts_from_env(extra: Iterable[str] = ()) -> frozenset[str]:
    names = [*LOOPBACK_HOSTS, *extra]
    names += os.environ.get("OPS_COPILOT_ALLOWED_HOSTS", "").split(",")
    return frozenset(_bare_host(n) for n in names if n.strip())


def _host_of_header(value: str) -> str:
    """Host header -> bare lowercase hostname (port stripped; IPv6 brackets handled)."""
    v = value.strip().lower()
    if v.startswith("["):
        return v[1 : v.find("]")] if "]" in v else ""
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def _origin_allowed(origin: str, hosts: frozenset[str]) -> bool:
    try:
        u = urlsplit(origin)
        _ = u.port  # raises ValueError on a malformed port
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    if u.path or u.query or u.fragment or u.username or u.password:
        return False
    return u.hostname.lower() in hosts


class HardeningMiddleware:
    """Pure ASGI: Host allow-list (DNS rebinding), same-origin check, generic 500, and the
    security headers on EVERY response (including its own 4xx/5xx and static files)."""

    def __init__(self, app: ASGIApp, hosts: frozenset[str]) -> None:
        self.app = app
        self.hosts = hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def send_wrapped(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                kept = [
                    h for h in message.get("headers", []) if h[0].lower() not in SECURITY_HEADERS
                ]
                headers = [*kept, *SECURITY_HEADERS.items()]
                message = {**message, "headers": headers}
            await send(message)

        async def reply(status: int, detail: str) -> None:
            await _err(status, detail)(scope, receive, send_wrapped)

        hdrs = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        if _host_of_header(hdrs.get("host", "")) not in self.hosts:
            await reply(400, "invalid host")
            return
        origin = hdrs.get("origin")
        if origin is not None and not _origin_allowed(origin, self.hosts):
            await reply(403, "forbidden")
            return
        try:
            await self.app(scope, receive, send_wrapped)
        except Exception as exc:
            log.error("unhandled error: %s", type(exc).__name__)  # class only, never the message
            if started:
                raise
            await reply(500, "internal error")


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
    allowed_hosts: Iterable[str] = (),
) -> FastAPI:
    _quiet_loggers()
    docs_p = Path(docs_dir) if docs_dir else REPO_ROOT / "data" / "docs"
    sim_p = Path(sim_dir) if sim_dir else REPO_ROOT / "data" / "sim"
    eval_p = Path(eval_path) if eval_path else REPO_ROOT / "eval" / "latest.json"
    web_p = Path(web_dir) if web_dir else REPO_ROOT / "web"
    if not docs_p.is_dir():
        # Fail loudly at start: serving an empty catalog would hide a misconfigured deployment.
        log.error("docs directory not found: %s", docs_p)
        raise FileNotFoundError(f"docs directory not found: {docs_p}")

    app = FastAPI(debug=False, title="Ops Copilot", docs_url=None, redoc_url=None, openapi_url=None)
    state: dict[str, Pipeline | None] = {"pipeline": pipeline}
    build_lock = threading.Lock()
    # Slots are released ONLY by the worker thread when it actually finishes, never on the
    # timeout path: a timed-out call keeps running and must keep counting against the limit.
    slots = threading.BoundedSemaphore(max_concurrent)
    ask_pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="ask")

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

    def ask_worker(question: str, mode: str) -> AskResult | None:
        try:
            return run_ask(question, mode)
        finally:
            slots.release()

    app.add_middleware(HardeningMiddleware, hosts=allowed_hosts_from_env(allowed_hosts))

    @app.post("/api/ask")
    async def ask(request: Request) -> JSONResponse:
        ctype = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            return _err(415, "unsupported media type")
        body = await _read_capped(request)
        if body is None:
            return _err(413, "request body too large")
        try:
            req = AskRequest.model_validate(json.loads(body))
        except (ValueError, ValidationError):  # JSONDecodeError/UnicodeDecodeError are ValueError
            return _err(422, "invalid request")

        give_up = time.monotonic() + queue_wait_s
        while not slots.acquire(blocking=False):
            if time.monotonic() >= give_up:
                return _err(503, "server busy")
            await asyncio.sleep(0.01)
        try:
            fut = asyncio.get_running_loop().run_in_executor(
                ask_pool, ask_worker, req.question, req.mode
            )
        except BaseException:
            slots.release()  # never reached the worker
            raise
        # asyncio.wait does not cancel on timeout (wait_for would): the worker keeps its slot.
        done, _ = await asyncio.wait({fut}, timeout=deadline_s)
        if not done:
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # mark retrieved
            log.warning("ask timed out")
            return _err(504, "upstream timeout")
        try:
            result = fut.result()
        except Exception as exc:
            log.warning("ask failed: %s", type(exc).__name__)  # class only, never the message
            return _err(502, "upstream model error")
        if result is None:
            return _err(503, "model backend unavailable")
        return JSONResponse(_safe(_bound_reason(result)))

    @app.get("/api/docs")
    async def docs() -> list[dict[str, Any]]:
        return await run_in_threadpool(load_catalog, docs_p)

    @app.get("/api/dashboard")
    async def dashboard() -> dict[str, Any]:
        return _safe(await run_in_threadpool(load_dashboard, sim_p))

    @app.get("/api/eval/latest")
    async def eval_latest() -> Any:
        data = await run_in_threadpool(load_eval_latest, eval_p)
        if data is None:
            return _err(404, "no eval run yet")
        return JSONResponse(_safe(data))

    if (web_p / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_p, html=True), name="web")
    return app
