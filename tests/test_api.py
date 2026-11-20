"""Unit d: API contract, validation, no-leakage, catalog/dashboard/eval, static. Offline."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ops_copilot.answer import REFUSAL
from ops_copilot.api import CSP, MAX_BODY_BYTES, create_app
from ops_copilot.config import Secret, Settings

from .conftest import FIXTURE_DOCS, CountingChat, HashedBowEmbeddings

ROOT = Path(__file__).resolve().parent.parent
Q = "Who approves a standard purchase order?"
GOOD = "A standard PO is approved per the matrix [PR-001 §2.1]."
CANARY = "CANARY-7f3a9c1e5b2d4086-zzqx"

TOP_KEYS = {
    "answer",
    "citations",
    "refused",
    "refusal_reason",
    "confidence",
    "chunks",
    "timings_ms",
}
TIMING_KEYS = {"embed", "bm25", "dense", "rrf", "gate", "llm"}
CHUNK_KEYS = {"id", "text", "bm25_rank", "dense_rank", "rrf", "used"}


@pytest.fixture(autouse=True)
def _allow_testserver_host(monkeypatch):
    """d.1: Host is allow-listed; TestClient's default host is 'testserver' (change 1)."""
    monkeypatch.setenv("OPS_COPILOT_ALLOWED_HOSTS", "testserver")


JSON_CT = {"content-type": "application/json"}


@pytest.fixture
def client(make_pipeline, tmp_path):
    pipe, _ = make_pipeline(GOOD)
    return TestClient(create_app(pipe, eval_path=tmp_path / "none.json"))


def post(c: TestClient, body, **kw):
    return c.post("/api/ask", json=body, **kw)


# ---------------------------------------------------------------- contract


def test_ask_contract_shape(client):
    r = post(client, {"question": Q})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    d = r.json()
    assert set(d) == TOP_KEYS
    assert isinstance(d["answer"], str) and isinstance(d["refused"], bool)
    assert isinstance(d["confidence"], float)
    assert set(d["timings_ms"]) == TIMING_KEYS
    assert all(isinstance(v, float) for v in d["timings_ms"].values())
    assert d["chunks"] and all(set(c) == CHUNK_KEYS for c in d["chunks"])
    assert d["refused"] is False
    assert d["citations"] == [{"doc_id": "PR-001", "section": "2.1", "valid": True}]
    assert any(c["used"] for c in d["chunks"])


def test_ask_passes_pipeline_result_unchanged(make_pipeline, tmp_path):
    pipe, _ = make_pipeline(GOOD)
    c = TestClient(create_app(pipe))
    got = post(c, {"question": Q}).json()
    want = pipe.ask(Q, "hybrid")
    for k in TOP_KEYS - {"timings_ms"}:
        assert got[k] == want[k]


def test_refused_path(make_pipeline):
    pipe, _ = make_pipeline("An uncited answer.")
    d = post(TestClient(create_app(pipe)), {"question": Q}).json()
    assert d["refused"] is True
    assert d["citations"] == []
    assert d["answer"] == REFUSAL
    assert d["refusal_reason"]
    assert not any(c["used"] for c in d["chunks"])


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_modes_pass_through(make_pipeline, mode):
    pipe, _ = make_pipeline(GOOD)
    seen = []
    orig = pipe.ask
    pipe.ask = lambda q, m="hybrid": (seen.append(m), orig(q, m))[1]  # type: ignore[method-assign]
    assert post(TestClient(create_app(pipe)), {"question": Q, "mode": mode}).status_code == 200
    assert seen == [mode]


def test_default_mode_is_hybrid(make_pipeline):
    pipe, _ = make_pipeline(GOOD)
    seen = []
    orig = pipe.ask
    pipe.ask = lambda q, m="hybrid": (seen.append(m), orig(q, m))[1]  # type: ignore[method-assign]
    post(TestClient(create_app(pipe)), {"question": Q})
    assert seen == ["hybrid"]


def test_question_is_stripped(make_pipeline):
    pipe, _ = make_pipeline(GOOD)
    seen = []
    orig = pipe.ask
    pipe.ask = lambda q, m="hybrid": (seen.append(q), orig(q, m))[1]  # type: ignore[method-assign]
    post(TestClient(create_app(pipe)), {"question": f"  {Q} \n"})
    assert seen == [Q]


def test_nosniff_header_everywhere(client):
    for r in (post(client, {"question": Q}), client.get("/api/docs"), post(client, {})):
        assert r.headers["x-content-type-options"] == "nosniff"


# -------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   \n\t "},
        {"question": "x" * 2001},
        {"question": "  " + "x" * 2001 + "  "},
        {"question": Q, "extra": 1},
        {"question": Q, "mode": "semantic"},
        {"question": Q, "mode": 1},
        {"question": 123},
        {"question": ["a"]},
        {"question": None},
        {},
        [],
        "just a string",
        {"question": Q, "top_k": 50},
        {"question": Q, "candidates": 99},
        {"question": Q, "rrf_k": 1},
        {"question": Q, "gate_threshold": 0.0},
        {"question": Q, "nonce": "abc"},
        {"question": Q, "provider": "groq"},
        {"question": Q, "base_url": "http://evil"},
        {"question": Q, "docs_dir": "/etc"},
    ],
)
def test_invalid_bodies_422(client, body):
    r = post(client, body)
    assert r.status_code == 422
    assert r.json() == {"detail": "invalid request"}


def test_boundary_lengths_accepted(client):
    assert post(client, {"question": "x" * 2000}).status_code == 200
    assert post(client, {"question": "  " + "x" * 2000 + "  "}).status_code == 200


def test_non_json_422(client):
    assert client.post("/api/ask", content=b"not json", headers=JSON_CT).status_code == 422
    r = client.post("/api/ask", content=b"\xff\xfe", headers=JSON_CT)
    assert r.status_code == 422
    assert client.post("/api/ask", content=b"", headers=JSON_CT).status_code == 422


def test_body_cap_413_before_parsing(client):
    big = b'{"question": "' + b"x" * (20 * 1024) + b'"}'
    assert len(big) > MAX_BODY_BYTES
    assert client.post("/api/ask", content=big, headers=JSON_CT).status_code == 413
    # not even valid JSON: must be 413 (size checked before parsing), not 422
    assert client.post("/api/ask", content=b"z" * (20 * 1024), headers=JSON_CT).status_code == 413
    # chunked (no content-length) is capped too
    r = client.post("/api/ask", content=iter([b"z" * 9000, b"z" * 9000]), headers=JSON_CT)
    assert r.status_code == 413


def test_validation_error_does_not_echo_input(client):
    secret_q = {"question": "ECHO-ME-" + "x" * 2100}
    assert "ECHO-ME" not in post(client, secret_q).text
    assert "ECHO-ME" not in post(client, {"question": "ECHO-ME", "mode": "bad"}).text


def test_openapi_and_docs_disabled(client):
    for p in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(p).status_code == 404


# ------------------------------------------------------------ error mapping


class Boom:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def ask(self, question, mode="hybrid"):
        raise self.exc


def test_pipeline_exception_502(caplog):
    caplog.set_level(logging.DEBUG)
    c = TestClient(create_app(Boom(RuntimeError(f"auth failed key={CANARY} q=SECRETQ"))))  # type: ignore[arg-type]
    r = post(c, {"question": "SECRETQ please"})
    assert r.status_code == 502
    assert r.json() == {"detail": "upstream model error"}
    assert CANARY not in r.text and "SECRETQ" not in r.text
    assert CANARY not in caplog.text and "SECRETQ" not in caplog.text


def test_timeout_504():
    class Slow:
        def ask(self, question, mode="hybrid"):
            time.sleep(0.5)

    c = TestClient(create_app(Slow(), deadline_s=0.05))  # type: ignore[arg-type]
    r = post(c, {"question": Q})
    assert r.status_code == 504
    assert r.json() == {"detail": "upstream timeout"}


def test_concurrency_limit_and_busy_503():
    gate, active, peak = threading.Event(), [0], [0]
    lock = threading.Lock()

    class Blocking:
        def ask(self, question, mode="hybrid"):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            gate.wait(2)
            with lock:
                active[0] -= 1
            raise RuntimeError("done")

    app = create_app(Blocking(), max_concurrent=2, queue_wait_s=0.2)  # type: ignore[arg-type]
    codes: list[int] = []

    def one() -> None:
        with TestClient(app) as c:
            codes.append(post(c, {"question": Q}).status_code)

    ts = [threading.Thread(target=one) for _ in range(3)]
    for t in ts:
        t.start()
    time.sleep(0.8)
    gate.set()
    for t in ts:
        t.join()
    assert peak[0] <= 2
    assert sorted(codes) == [502, 502, 503]


def test_lazy_pipeline_unavailable_503_then_recovers(make_pipeline):
    pipe, _ = make_pipeline(GOOD)
    calls = [0]

    def factory():
        calls[0] += 1
        if calls[0] == 1:
            raise ConnectionError(f"ollama down {CANARY}")
        return pipe

    c = TestClient(create_app(pipeline_factory=factory))
    # app starts and non-model endpoints work while the backend is down
    assert c.get("/api/docs").status_code == 200
    assert calls[0] == 0
    r = post(c, {"question": Q})
    assert r.status_code == 503 and r.json() == {"detail": "model backend unavailable"}
    assert CANARY not in r.text
    assert post(c, {"question": Q}).status_code == 200
    assert post(c, {"question": Q}).status_code == 200
    assert calls[0] == 2  # built once successfully, then cached


# --------------------------------------------------------------- no leakage


def test_no_canary_in_any_response_or_log(tmp_path, embeddings, caplog):
    caplog.set_level(logging.DEBUG)
    settings = Settings(
        groq_api_key=Secret(CANARY),
        gate_threshold=0.05,
        chroma_dir=str(tmp_path / "chroma"),
        docs_dir=str(FIXTURE_DOCS),
    )
    from ops_copilot.pipeline import Pipeline

    pipe = Pipeline(settings, embeddings, CountingChat(responses=[GOOD]), FIXTURE_DOCS)
    ev = tmp_path / "latest.json"
    ev.write_text(json.dumps({"ok": True}))
    c = TestClient(create_app(pipe, eval_path=ev))
    bodies = [
        post(c, {"question": Q}),
        post(c, {"question": Q, "mode": "bm25"}),
        post(c, {"question": ""}),
        post(c, {"question": Q, "top_k": 1}),
        c.post("/api/ask", content=b"x" * 20000, headers=JSON_CT),
        c.get("/api/docs"),
        c.get("/api/dashboard"),
        c.get("/api/eval/latest"),
        c.get("/nope"),
        c.get("/api/ask"),
    ]
    # positive control: the canary really is the configured key
    assert settings.groq_api_key is not None and settings.groq_api_key.get() == CANARY
    for r in bodies:
        assert CANARY not in r.text
        assert CANARY not in json.dumps(dict(r.headers))
    assert CANARY not in caplog.text
    assert "Settings" not in "".join(r.text for r in bodies)


def test_canary_exception_logged_never(caplog):
    caplog.set_level(logging.DEBUG)
    c = TestClient(create_app(Boom(ValueError(CANARY))))  # type: ignore[arg-type]
    assert CANARY not in post(c, {"question": Q}).text
    assert CANARY not in caplog.text
    assert caplog.records  # positive control: something was logged


def test_third_party_loggers_quiet():
    create_app(None)
    for n in ("groq", "httpx", "httpcore"):
        assert logging.getLogger(n).level >= logging.WARNING


# ------------------------------------------------- docs / dashboard / eval

EXPECTED_DOCS = {
    "CS-001", "FIN-001", "PR-001", "PR-002", "QA-001",
    "WH-001", "WH-002", "WH-003", "WH-004", "WH-005",
}  # fmt: skip


def test_docs_catalog():
    r = TestClient(create_app(None)).get("/api/docs")
    assert r.status_code == 200
    docs = r.json()
    assert len(docs) == 10
    assert {d["doc_id"] for d in docs} == EXPECTED_DOCS
    for d in docs:
        assert set(d) == {
            "doc_id", "title", "owner_role", "version", "effective_date", "sections", "summary"
        }  # fmt: skip
        assert d["title"] and d["owner_role"] and d["version"] and d["effective_date"]
        assert len(d["sections"]) >= 3
        for s in d["sections"]:
            assert set(s) == {"section", "title"}
            assert s["section"].replace(".", "").isdigit() and s["title"]
    pr1 = next(d for d in docs if d["doc_id"] == "PR-001")
    assert pr1["title"] == "PO Approval Matrix"
    assert pr1["owner_role"] == "Director of Operations"
    assert pr1["sections"][0] == {"section": "1", "title": "Purpose"}
    assert {"section": "2.1", "title": "Approval bands"} in pr1["sections"]


def test_dashboard():
    r = TestClient(create_app(None)).get("/api/dashboard")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"kpis", "receipts", "purchase_orders", "suppliers"}
    assert isinstance(d["receipts"], list) and isinstance(d["suppliers"], list)
    po = next(p for p in d["purchase_orders"] if p["po"] == "PO-2291")
    assert po["total_usd"] == 40000
    assert po["status"] == "awaiting Director of Operations"
    assert d["kpis"]["open_pos"] == 12


def test_dashboard_read_only(client):
    assert client.post("/api/dashboard", json={}).status_code == 405
    assert client.put("/api/dashboard", json={}).status_code == 405


def test_eval_latest_404_then_json(tmp_path):
    p = tmp_path / "latest.json"
    c = TestClient(create_app(None, eval_path=p))
    r = c.get("/api/eval/latest")
    assert r.status_code == 404 and r.json() == {"detail": "no eval run yet"}
    p.write_text(json.dumps({"faithfulness": 0.93, "rows": [1, 2]}))
    r = c.get("/api/eval/latest")
    assert r.status_code == 200 and r.json() == {"faithfulness": 0.93, "rows": [1, 2]}


# ------------------------------------------------------------------ static


def test_static_served_with_csp(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>hi</title>")
    (web / "app.js").write_text("1")
    c = TestClient(create_app(None, web_dir=web))
    r = c.get("/")
    assert r.status_code == 200 and "<title>hi</title>" in r.text
    assert r.headers["content-security-policy"] == CSP
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in CSP and "default-src 'self'" in CSP
    js = c.get("/app.js")
    assert js.status_code == 200 and js.headers["content-security-policy"] == CSP
    assert c.get("/api/docs").status_code == 200  # API still wins over the static mount
    assert c.get("/..%2f..%2fpyproject.toml").status_code in (400, 404)


def test_no_static_when_web_missing(tmp_path):
    c = TestClient(create_app(None, web_dir=tmp_path / "absent"))
    assert c.get("/").status_code == 404


def test_default_web_dir_placeholder_exists():
    assert (ROOT / "web").is_dir()


# ---------------------------------------------------------------- misc


def test_embeddings_fixture_is_offline():
    assert HashedBowEmbeddings().embed_query("a")
