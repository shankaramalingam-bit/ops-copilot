"""Unit H-sec: strict Host parsing, allow-list env filtering, duplicate Host/Origin rejection.

Raw ASGI (no sockets), so header lists can carry duplicates and malformed values verbatim.
"""

from __future__ import annotations

import asyncio

import pytest

from ops_copilot.api import allowed_hosts_from_env, create_app


class Echo:
    def ask(self, question, mode="hybrid"):
        return {"answer": "ok", "confidence": 0.5}


def call(app, headers: list[tuple[str, str]], path: str = "/api/docs") -> int:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1),
        "headers": [(k.lower().encode("latin-1"), v.encode("utf-8")) for k, v in headers],
    }
    status: list[int] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            status.append(msg["status"])

    asyncio.run(app(scope, receive, send))
    return status[0]


def mkapp(tmp_path):
    (tmp_path / "docs").mkdir(exist_ok=True)
    return create_app(Echo(), docs_dir=tmp_path / "docs", web_dir=tmp_path)  # type: ignore[arg-type]


@pytest.fixture
def app(tmp_path):
    return mkapp(tmp_path)


def test_positive_control_loopback_hosts_pass(app):
    for h in ("localhost", "localhost:8000", "127.0.0.1:1", "[::1]", "[::1]:65535", "LOCALHOST"):
        assert call(app, [("host", h)]) == 200, h


def test_empty_brackets_env_entry_adds_nothing(monkeypatch):
    monkeypatch.setenv("OPS_COPILOT_ALLOWED_HOSTS", "[],  ,[ ],[]")
    assert "" not in allowed_hosts_from_env()


def test_empty_brackets_extra_arg_adds_nothing():
    assert "" not in allowed_hosts_from_env(["[]", "  "])


def test_empty_host_header_rejected_with_empty_bracket_entry(monkeypatch, tmp_path):
    monkeypatch.setenv("OPS_COPILOT_ALLOWED_HOSTS", "[]")
    app = mkapp(tmp_path)
    for h in ("", "[]"):
        assert call(app, [("host", h)]) == 400, h


@pytest.mark.parametrize(
    "host",
    [
        "[::1]evil.example",
        "[::1]x",
        "[::1]:",
        "[::1]:abc",
        "[::1",
        "localhost:abc",
        "localhost:",
        "localhost:99999",
        "localhost:65536",
        "localhost:123456",
        "localhost:-1",
        "localhost:+80",
        "localhost: 80",
        "localhost:8٣",  # non-ASCII digit
        "localhost:80:80",
        "localhost:80x",
        "localhost evil.example",
    ],
)
def test_malformed_host_rejected(app, host):
    assert call(app, [("host", host)]) == 400


def test_port_boundaries_accepted(app):
    assert call(app, [("host", "localhost:0")]) == 200
    assert call(app, [("host", "localhost:65535")]) == 200
    assert call(app, [("host", "localhost:00080")]) == 200  # 5 digits


def test_duplicate_host_rejected(app):
    assert call(app, [("host", "localhost"), ("host", "localhost")]) == 400
    assert call(app, [("host", "localhost"), ("host", "evil.example")]) == 400
    assert call(app, [("host", "evil.example"), ("host", "localhost")]) == 400


def test_missing_host_rejected(app):
    assert call(app, []) == 400


def test_duplicate_origin_rejected(app):
    ok = "http://localhost"
    assert call(app, [("host", "localhost"), ("origin", ok)]) == 200  # positive control
    assert call(app, [("host", "localhost"), ("origin", ok), ("origin", ok)]) == 403
    evil = "http://evil.example"
    assert call(app, [("host", "localhost"), ("origin", ok), ("origin", evil)]) == 403
    assert call(app, [("host", "localhost"), ("origin", evil), ("origin", ok)]) == 403
