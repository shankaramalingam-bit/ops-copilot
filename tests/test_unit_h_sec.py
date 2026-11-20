"""Unit H-sec: strict Host parsing, allow-list env filtering, duplicate Host/Origin rejection.

Raw ASGI (no sockets), so header lists can carry duplicates and malformed values verbatim.
"""

from __future__ import annotations

import asyncio

import pytest

from ops_copilot.api import allowed_hosts_from_env, create_app
from ops_copilot.config import ConfigError


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
    assert call(app, [("host", "localhost:1")]) == 200
    assert call(app, [("host", "localhost:8000")]) == 200
    assert call(app, [("host", "localhost:65535")]) == 200


@pytest.mark.parametrize(
    "host", ["localhost:0", "localhost:00000", "LOCALHOST:08000", "localhost:00080", "localhost:01"]
)
def test_port_zero_and_leading_zeros_rejected(app, host):
    assert call(app, [("host", "localhost:1")]) == 200  # positive control
    assert call(app, [("host", host)]) == 400


@pytest.mark.parametrize(
    "host", ["localhost\x1f", "localhost\x0b", "localhost\x85", "\xa0localhost", "localhost\x0c"]
)
def test_non_ows_whitespace_rejected(app, host):
    assert call(app, [("host", "localhost")]) == 200  # positive control
    assert call(app, [("host", host)]) == 400


@pytest.mark.parametrize("host", ["\tlocalhost", "localhost:8000 ", " localhost\t"])
def test_ows_space_tab_accepted(app, host):
    assert call(app, [("host", host)]) == 200


LOOP = frozenset({"127.0.0.1", "localhost", "::1"})


def test_bracket_space_entry_adds_nothing(monkeypatch):
    for v in ("[ ]", "[],[ ], "):
        monkeypatch.setenv("OPS_COPILOT_ALLOWED_HOSTS", v)
        assert allowed_hosts_from_env() == LOOP, v
    monkeypatch.delenv("OPS_COPILOT_ALLOWED_HOSTS")
    assert allowed_hosts_from_env(["[ ]"]) == LOOP


@pytest.mark.parametrize("entry", ["example.com:8443", "[::1]:8443", "ok.example,example.com:80"])
def test_allowed_hosts_entry_with_port_rejected(monkeypatch, entry):
    monkeypatch.setenv("OPS_COPILOT_ALLOWED_HOSTS", entry)
    with pytest.raises(ConfigError) as ei:
        allowed_hosts_from_env()
    assert str(ei.value) == "OPS_COPILOT_ALLOWED_HOSTS entries must be hostnames without ports"
    monkeypatch.delenv("OPS_COPILOT_ALLOWED_HOSTS")
    with pytest.raises(ConfigError):
        allowed_hosts_from_env(["example.com:8443"])


def test_ipv6_allowed_hosts_entry_still_ok(monkeypatch):
    monkeypatch.setenv("OPS_COPILOT_ALLOWED_HOSTS", "[2001:db8::1],Example.COM")
    assert {"2001:db8::1", "example.com"} <= allowed_hosts_from_env()


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
