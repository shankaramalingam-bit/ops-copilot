"""Unit d.1: API hardening (slot-safe concurrency, host/CT/origin, bind policy, headers, F6/F7)."""

from __future__ import annotations

import json
import logging
import math
import threading
import time

import pytest
from fastapi.testclient import TestClient

from ops_copilot import __main__ as entry
from ops_copilot import pipeline as pipeline_mod
from ops_copilot.api import ASK_DEADLINE_S, CSP, create_app

Q = "Who approves a standard purchase order?"
CANARY = "CANARY-d1-9b1e47c2a05f3d68-qqzx"
SEC_HEADERS = {
    "content-security-policy": CSP,
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}


class Echo:
    def ask(self, question, mode="hybrid"):
        return {"answer": "ok", "confidence": 0.5}


def mk(pipe=None, **kw) -> TestClient:
    return TestClient(create_app(pipe or Echo(), **kw), base_url="http://localhost")  # type: ignore[arg-type]


def jpost(c: TestClient, **kw):
    return c.post("/api/ask", json={"question": Q}, **kw)


def assert_sec(r) -> None:
    for k, v in SEC_HEADERS.items():
        assert r.headers.get(k) == v, (k, r.status_code, dict(r.headers))


# ------------------------------------------------------------------ F1


def test_slot_released_only_when_worker_finishes():
    active, peak, lock = [0], [0], threading.Lock()

    class Slow:
        def ask(self, question, mode="hybrid"):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(1.5)
            with lock:
                active[0] -= 1
            return {"answer": "late"}

    app = create_app(Slow(), max_concurrent=2, deadline_s=0.2, queue_wait_s=0.05)  # type: ignore[arg-type]
    codes: list[int] = []
    docs_ms: list[float] = []
    stop = threading.Event()

    def one() -> None:
        with TestClient(app, base_url="http://localhost") as c:
            codes.append(jpost(c).status_code)

    def poll_docs() -> None:
        with TestClient(app, base_url="http://localhost") as c:
            while not stop.is_set():
                t = time.perf_counter()
                assert c.get("/api/docs").status_code == 200
                docs_ms.append((time.perf_counter() - t) * 1000)
                time.sleep(0.05)

    poller = threading.Thread(target=poll_docs)
    poller.start()
    for _ in range(5):  # 5 waves of 2, each wave after the previous one's deadline expired
        wave = [threading.Thread(target=one) for _ in range(2)]
        for t in wave:
            t.start()
        for t in wave:
            t.join()
        time.sleep(0.05)
    stop.set()
    poller.join()
    time.sleep(1.7)  # let the stragglers finish before reading the peak
    assert len(codes) == 10
    assert peak[0] <= 2, peak
    assert set(codes) <= {503, 504}
    assert docs_ms and max(docs_ms) < 1000  # GET endpoints are not starved


def test_deadline_exceeds_inner_timeouts():
    from ops_copilot import llm

    assert ASK_DEADLINE_S == 90
    assert llm.GROQ_TIMEOUT_S == 30 and llm.GROQ_MAX_RETRIES == 1
    assert llm.OLLAMA_TIMEOUT_S == 60
    assert llm.EMBED_TIMEOUT_S <= 30
    groq_worst = llm.EMBED_TIMEOUT_S + llm.GROQ_TIMEOUT_S * (1 + llm.GROQ_MAX_RETRIES)
    ollama_worst = llm.EMBED_TIMEOUT_S + llm.OLLAMA_TIMEOUT_S
    assert groq_worst < ASK_DEADLINE_S and ollama_worst < ASK_DEADLINE_S


def test_llm_clients_carry_timeouts():
    from ops_copilot.config import Settings
    from ops_copilot.llm import get_chat_model, get_embeddings

    emb = get_embeddings(Settings())
    assert emb.client_kwargs.get("timeout") is not None  # type: ignore[attr-defined]
    chat = get_chat_model(Settings())
    assert chat.client_kwargs["timeout"] == 60  # type: ignore[attr-defined]


# ------------------------------------------------------------------ F2


def test_bad_host_400():
    c = TestClient(create_app(Echo()), base_url="http://attacker.example")  # type: ignore[arg-type]
    r = c.post("/api/ask", json={"question": Q})
    assert r.status_code == 400
    assert_sec(r)


@pytest.mark.parametrize(
    "base",
    ["http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000", "http://localhost"],
)
def test_loopback_hosts_ok(base):
    c = TestClient(create_app(Echo()), base_url=base)  # type: ignore[arg-type]
    assert jpost(c).status_code == 200


def test_allowed_hosts_env(monkeypatch):
    monkeypatch.setenv("OPS_COPILOT_ALLOWED_HOSTS", "copilot.internal, other.local")
    c = TestClient(create_app(Echo()), base_url="http://copilot.internal:8000")  # type: ignore[arg-type]
    assert jpost(c).status_code == 200


@pytest.mark.parametrize(
    "kw",
    [
        {"content": b'{"question": "x"}', "headers": {"content-type": "text/plain"}},
        {"content": b'{"question": "x"}', "headers": {"content-type": "text/plain;charset=utf-8"}},
        {"data": {"question": "x"}},  # application/x-www-form-urlencoded
        {"content": b'{"question": "x"}'},  # no content-type
        {"content": b'{"question": "x"}', "headers": {"content-type": "application/jsonx"}},
    ],
)
def test_content_type_415(kw):
    r = mk().post("/api/ask", **kw)
    assert r.status_code == 415 and r.json() == {"detail": "unsupported media type"}
    assert_sec(r)


def test_content_type_json_with_charset_ok():
    r = mk().post(
        "/api/ask",
        content=b'{"question": "x"}',
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert r.status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "https://localhost.evil.example",
        "null",
        "http://",
        "ftp://localhost",
        "http://localhost.evil.example:8000",
    ],
)
def test_evil_origin_403(origin):
    r = jpost(mk(), headers={"origin": origin})
    assert r.status_code == 403 and r.json() == {"detail": "forbidden"}
    assert_sec(r)


@pytest.mark.parametrize(
    "origin",
    ["http://localhost", "http://localhost:8000", "https://127.0.0.1:9", "http://[::1]:8000"],
)
def test_same_origin_ok(origin):
    assert jpost(mk(), headers={"origin": origin}).status_code == 200


def test_no_origin_ok():
    assert jpost(mk()).status_code == 200


def test_origin_checked_on_get_too():
    assert mk().get("/api/docs", headers={"origin": "http://evil.example"}).status_code == 403


# ------------------------------------------------------------------ F3


def test_bind_decision():
    assert entry.bind_problem("127.0.0.1", {}) is None
    assert entry.bind_problem("localhost", {}) is None
    assert entry.bind_problem("::1", {}) is None
    assert entry.bind_problem("127.5.5.5", {}) is None
    for h in ("0.0.0.0", "::", "192.168.1.5", "example.com", ""):
        msg = entry.bind_problem(h, {})
        assert msg and "OPS_COPILOT_ALLOW_NONLOCAL" in msg
        assert entry.bind_problem(h, {"OPS_COPILOT_ALLOW_NONLOCAL": "0"})
        assert entry.bind_problem(h, {"OPS_COPILOT_ALLOW_NONLOCAL": "1"}) is None


def test_main_refuses_nonlocal(monkeypatch, capsys):
    monkeypatch.setenv("OPS_COPILOT_HOST", "0.0.0.0")
    monkeypatch.delenv("OPS_COPILOT_ALLOW_NONLOCAL", raising=False)
    monkeypatch.setattr(entry, "load_env_file", lambda: None)
    called = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: called.append(k))
    with pytest.raises(SystemExit) as ei:
        entry.main()
    assert ei.value.code == 2 and not called
    assert "OPS_COPILOT_ALLOW_NONLOCAL" in capsys.readouterr().err


def test_main_nonlocal_allowed_warns_and_hides_server_header(monkeypatch, caplog):
    monkeypatch.setenv("OPS_COPILOT_HOST", "0.0.0.0")
    monkeypatch.setenv("OPS_COPILOT_ALLOW_NONLOCAL", "1")
    monkeypatch.setattr(entry, "load_env_file", lambda: None)
    called = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: called.append(k))
    with caplog.at_level(logging.WARNING):
        entry.main()
    assert called and called[0]["server_header"] is False
    assert any(
        r.levelno == logging.WARNING and "auth" in r.getMessage().lower() for r in caplog.records
    )


def test_main_loopback_server_header_off(monkeypatch):
    monkeypatch.delenv("OPS_COPILOT_HOST", raising=False)
    monkeypatch.setattr(entry, "load_env_file", lambda: None)
    called = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: called.append(k))
    entry.main()
    assert called[0]["server_header"] is False


# ------------------------------------------------------------------ F4 / F5


def test_security_headers_on_every_response(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<title>x</title>")
    (web / "app.js").write_text("1")
    c = mk(web_dir=web)
    for r in (
        c.get("/api/docs"),
        c.get("/api/"),
        c.get("/api"),
        c.get("/"),
        c.get("/app.js"),
        c.get("/nope"),
        c.get("/api/ask"),  # 405
        jpost(c),
        c.post("/api/ask", json={}),  # 422
    ):
        assert_sec(r)


def test_corrupt_data_gives_generic_500_with_headers(tmp_path, caplog):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "kpis.json").write_text("{ not json " + CANARY)
    ev = tmp_path / "latest.json"
    ev.write_text("][ " + CANARY)
    c = TestClient(
        create_app(Echo(), sim_dir=sim, eval_path=ev),  # type: ignore[arg-type]
        base_url="http://localhost",
        raise_server_exceptions=False,
    )
    caplog.set_level(logging.DEBUG)
    for path in ("/api/dashboard", "/api/eval/latest"):
        r = c.get(path)
        assert r.status_code == 500
        assert r.json() == {"detail": "internal error"}
        assert CANARY not in r.text
        assert_sec(r)
    assert CANARY not in caplog.text


def test_nan_confidence_serialises_as_null():
    class Nan:
        def ask(self, question, mode="hybrid"):
            return {"answer": "a", "confidence": math.nan, "chunks": [{"rrf": math.inf}]}

    r = mk(Nan()).post("/api/ask", json={"question": Q})
    assert r.status_code == 200
    d = json.loads(r.text)
    assert d["confidence"] is None and d["chunks"][0]["rrf"] is None


# ------------------------------------------------------------------ F6 / F7


def test_lazy_build_failure_log_does_not_leak(caplog):
    def factory():
        raise ConnectionError(f"ollama down {CANARY}")

    caplog.set_level(logging.DEBUG)
    c = TestClient(create_app(None, pipeline_factory=factory), base_url="http://localhost")
    r = jpost(c)
    assert r.status_code == 503
    assert any("pipeline unavailable" in rec.getMessage() for rec in caplog.records)  # control
    assert CANARY not in caplog.text and CANARY not in r.text
    for rec in caplog.records:
        assert rec.exc_info is None and CANARY not in str(rec.args)


def test_pipeline_passes_normalised_question_to_retrieve(make_pipeline, monkeypatch):
    pipe, _ = make_pipeline("x")
    seen = []
    orig = pipeline_mod.retrieve

    def spy(index, query, *a, **k):
        seen.append(query)
        return orig(index, query, *a, **k)

    monkeypatch.setattr(pipeline_mod, "retrieve", spy)
    pipe.ask("Who approves\x00 a stan​dard purchase order?")
    assert seen == ["Who approves a standard purchase order?"]
