"""Unit d.2: literal CSP, tested defaults, no dead handlers, catalog failure mode, doc summaries,
bounded refusal reasons."""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ops_copilot import __main__ as entry
from ops_copilot import api as api_mod
from ops_copilot.api import create_app
from ops_copilot.ingest import load_docs

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ops_copilot"
Q = "Who approves a standard purchase order?"

EXPECTED_CSP = {
    "default-src": "'self'",
    "script-src": "'self'",
    "style-src": "'self'",
    "img-src": "'self' data:",
    "connect-src": "'self'",
    "object-src": "'none'",
    "base-uri": "'none'",
    "form-action": "'self'",
    "frame-ancestors": "'none'",
}
DOCUMENTED_REASONS = {
    "below_gate",
    "model_refused",
    "no_valid_citations",
    "invalid_citation",
    "invalid_characters",
    "empty_question",
    "no_context",
    "question_too_long",
}


class Echo:
    def ask(self, question, mode="hybrid"):
        return {"answer": "ok", "refused": False, "refusal_reason": None}


def mk(pipe=None, **kw) -> TestClient:
    return TestClient(create_app(pipe or Echo(), **kw), base_url="http://localhost")  # type: ignore[arg-type]


# ------------------------------------------------------------------ 1. CSP (literal)


def _parse_csp(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in value.split(";"):
        part = part.strip()
        if part:
            name, _, rest = part.partition(" ")
            assert name not in out, f"duplicate directive {name}"
            out[name] = rest
    return out


def test_csp_literal_directives(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html>")
    c = TestClient(create_app(Echo(), web_dir=web), base_url="http://localhost")  # type: ignore[arg-type]
    for r in (c.get("/"), c.get("/api/docs"), c.get("/nope"), c.post("/api/ask", content="x")):
        value = r.headers["content-security-policy"]
        assert _parse_csp(value) == EXPECTED_CSP, value
        for bad in ("unsafe-inline", "unsafe-eval", "*"):
            assert bad not in value
    assert (
        c.get("/").headers["content-security-policy"]
        == "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    )


# ------------------------------------------------------------------ 2. defaults


@pytest.mark.parametrize("env", [None, ""])
def test_main_default_bind_is_loopback_8000(monkeypatch, env):
    if env is None:
        monkeypatch.delenv("OPS_COPILOT_HOST", raising=False)
    else:
        monkeypatch.setenv("OPS_COPILOT_HOST", env)
    monkeypatch.delenv("OPS_COPILOT_PORT", raising=False)
    monkeypatch.delenv("OPS_COPILOT_ALLOW_NONLOCAL", raising=False)
    monkeypatch.setattr(entry, "load_env_file", lambda: None)
    called: list[dict] = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: called.append(k))
    entry.main()
    assert called[0]["host"] == "127.0.0.1"
    assert called[0]["port"] == 8000


def test_default_deadline_is_90_and_passed_to_wait(monkeypatch):
    assert api_mod.ASK_DEADLINE_S == 90
    assert inspect.signature(create_app).parameters["deadline_s"].default == 90
    seen: list[float | None] = []
    real_wait = asyncio.wait

    async def spy(fs, **kw):
        seen.append(kw.get("timeout"))
        return await real_wait(fs, **kw)

    monkeypatch.setattr(api_mod.asyncio, "wait", spy)
    assert mk().post("/api/ask", json={"question": Q}).status_code == 200
    assert seen == [90]


def test_default_concurrency_limit_is_2():
    assert api_mod.MAX_CONCURRENT_ASKS == 2
    assert inspect.signature(create_app).parameters["max_concurrent"].default == 2
    active, peak, lock = [0], [0], threading.Lock()
    release = threading.Event()

    class Hold:
        def ask(self, question, mode="hybrid"):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            release.wait(5)
            with lock:
                active[0] -= 1
            return {"answer": "ok"}

    c = mk(Hold(), queue_wait_s=0.3)
    codes: list[int] = []

    def go():
        codes.append(c.post("/api/ask", json={"question": Q}).status_code)

    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(1.0)  # two hold slots; the third gave up after queue_wait_s
    assert sorted(codes) == [503]
    release.set()
    for t in threads:
        t.join(10)
    assert sorted(codes) == [200, 200, 503]
    assert peak[0] == 2


# ------------------------------------------------------------------ 3. no dead handlers


def test_no_unreachable_exception_handlers():
    src = (SRC / "api.py").read_text()
    assert "RequestValidationError" not in src
    assert "exception_handler" not in src


def test_framework_errors_keep_detail_shape(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("x")
    c = TestClient(create_app(Echo(), web_dir=web), base_url="http://localhost")  # type: ignore[arg-type]
    r404 = c.get("/missing.js")
    assert r404.status_code == 404 and set(r404.json()) == {"detail"}
    assert isinstance(r404.json()["detail"], str)
    r405 = c.post("/api/dashboard", json={})
    assert r405.status_code == 405 and isinstance(r405.json()["detail"], str)
    # Pydantic failures never echo the input back.
    r422 = c.post("/api/ask", json={"question": "SECRET-echo-canary", "mode": 5})
    assert r422.status_code == 422 and r422.json() == {"detail": "invalid request"}
    assert "SECRET-echo-canary" not in r422.text


# ------------------------------------------------------------------ 4. missing catalog fails loudly


def test_missing_docs_dir_fails_at_start(tmp_path, caplog):
    missing = tmp_path / "does-not-exist"
    with caplog.at_level(logging.ERROR, logger="ops_copilot.api"):
        with pytest.raises(FileNotFoundError):
            create_app(Echo(), docs_dir=missing)  # type: ignore[arg-type]
    assert any(r.levelno == logging.ERROR and "docs" in r.getMessage() for r in caplog.records)


def test_docs_path_that_is_a_file_fails_at_start(tmp_path):
    f = tmp_path / "f.md"
    f.write_text("x")
    with pytest.raises(FileNotFoundError):
        create_app(Echo(), docs_dir=f)  # type: ignore[arg-type]


def test_existing_empty_docs_dir_still_starts(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    assert mk(docs_dir=d).get("/api/docs").json() == []


# ------------------------------------------------------------------ 5. doc summaries


def test_ten_summaries_nonempty_substring_of_cleaned_text():
    entries = mk().get("/api/docs").json()
    assert len(entries) == 10
    by_id = {d.doc_id: d for d in load_docs(ROOT / "data" / "docs")}
    for e in entries:
        s = e["summary"]
        assert isinstance(s, str) and s.strip(), e["doc_id"]
        assert len(s) <= 200
        assert s in by_id[e["doc_id"]].text, e["doc_id"]
        assert s.endswith((".", "!", "?")) and "\n" not in s
    # first sentence of 1.1 Objective, not the whole paragraph
    pr1 = next(e for e in entries if e["doc_id"] == "PR-001")["summary"]
    assert pr1.startswith("This document sets the approval limits")
    assert "It applies" not in pr1


def test_summary_is_additive_only():
    e = mk().get("/api/docs").json()[0]
    assert set(e) == {
        "doc_id",
        "title",
        "owner_role",
        "version",
        "effective_date",
        "sections",
        "summary",
    }


def test_summary_truncated_to_200_on_word_boundary(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    long_sentence = " ".join(["alpha"] * 80) + "."
    (d / "X-001.md").write_text(
        "---\ndoc_id: X-001\ntitle: T\n---\n# X-001 T\n\n## 1. Purpose\n\n"
        f"{long_sentence} Second.\n"
    )
    (d / "X-002.md").write_text(
        "---\ndoc_id: X-002\ntitle: T\n---\n# X-002 T\n\n## 2. Other\n\nHi.\n"
    )
    out = {e["doc_id"]: e for e in mk(docs_dir=d).get("/api/docs").json()}
    s = out["X-001"]["summary"]
    assert 0 < len(s) <= 200 and long_sentence.startswith(s) and not s.endswith(" ")
    assert out["X-002"]["summary"] == ""  # no section 1: empty, never invented


# ------------------------------------------------------------------ 6. bounded refusal_reason


def _emitted_reasons() -> set[str]:
    found: set[str] = set()
    for fname in ("answer.py", "pipeline.py"):
        tree = ast.parse((SRC / fname).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name in ("refuse", "_refusal") and node.args:
                    a = node.args[0]
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        found.add(a.value)
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "reason" for t in node.targets)
                and isinstance(node.value, ast.IfExp)
            ):
                for part in (node.value.body, node.value.orelse):
                    assert isinstance(part, ast.Constant)
                    found.add(part.value)
    return found


def test_pipeline_emits_only_documented_reasons():
    assert _emitted_reasons() == DOCUMENTED_REASONS
    assert api_mod.REFUSAL_REASONS == DOCUMENTED_REASONS


def test_pipeline_behaviour_reasons_documented(make_pipeline):
    seen: set[str | None] = set()
    p, _ = make_pipeline(response="x")
    seen.add(p.ask("   ", "hybrid")["refusal_reason"])
    seen.add(p.ask("a" * 5000, "hybrid")["refusal_reason"])
    p2, _ = make_pipeline(response="x", threshold=0.99)
    seen.add(p2.ask("zzzz qqqq xxxx", "hybrid")["refusal_reason"])
    p3, _ = make_pipeline(response="Totally uncited answer.")
    seen.add(p3.ask(Q, "hybrid")["refusal_reason"])
    assert None not in seen and seen <= DOCUMENTED_REASONS and len(seen) >= 3


@pytest.mark.parametrize("reason", sorted(DOCUMENTED_REASONS))
def test_api_passes_documented_reasons_through(reason):
    class P:
        def ask(self, question, mode="hybrid"):
            return {"answer": "x", "refused": True, "refusal_reason": reason}

    assert mk(P()).post("/api/ask", json={"question": Q}).json()["refusal_reason"] == reason


@pytest.mark.parametrize(
    "bad", ["<script>alert(1)</script>", "internal: boom", "", "BELOW_GATE", 7]
)
def test_api_maps_unknown_reason_to_refused(bad):
    class P:
        def ask(self, question, mode="hybrid"):
            return {"answer": "x", "refused": True, "refusal_reason": bad}

    body = mk(P()).post("/api/ask", json={"question": Q}).json()
    assert body["refusal_reason"] == "refused"


def test_api_null_reason_stays_null():
    assert mk().post("/api/ask", json={"question": Q}).json()["refusal_reason"] is None
