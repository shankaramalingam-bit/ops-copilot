"""Unit e: static analysis + serving checks for the web portal. Offline, no browser."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ops_copilot.api import create_app

WEB = Path(__file__).resolve().parent.parent / "web"
JS_FILES = sorted(WEB.glob("*.js"))
HTML_FILES = sorted(WEB.glob("*.html"))
CSS_FILES = sorted(WEB.glob("*.css"))

# Dynamic-HTML / code-execution sinks. The allowlist is intentionally empty (zero occurrences).
FORBIDDEN = [
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval(",
    "new Function",
    'setTimeout("',
    "setTimeout('",
    'setInterval("',
    "setInterval('",
    "createContextualFragment",
    "DOMParser",
    "srcdoc",
]
ALLOWLIST: dict[str, set[str]] = {}

EXPECTED_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    # d.1: the Host allow-list refuses TestClient's default "testserver"; localhost is allowed.
    return TestClient(create_app(web_dir=WEB), base_url="http://localhost")


def test_web_files_exist() -> None:
    assert (WEB / "index.html").is_file()
    assert (WEB / "app.js").is_file()
    assert (WEB / "app.css").is_file()
    assert JS_FILES and HTML_FILES and CSS_FILES  # positive control for the scans below


@pytest.mark.parametrize("path", JS_FILES + HTML_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("needle", FORBIDDEN)
def test_no_forbidden_sinks(path: Path, needle: str) -> None:
    if needle in ALLOWLIST.get(path.name, set()):
        pytest.skip("allowlisted")
    assert needle not in path.read_text(encoding="utf-8"), f"{needle!r} in {path.name}"


def test_scan_would_catch_a_violation() -> None:
    """Positive control: the scan logic flags a known-bad snippet."""
    bad = "el.innerHTML = x; eval(y)"
    assert [n for n in FORBIDDEN if n in bad] == ["innerHTML", "eval("]


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
def test_js_no_assigned_event_handlers_or_style_strings(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    assert not re.search(r"\.on[a-z]+\s*=[^=]", src), "use addEventListener"
    assert not re.search(r"\.style\.cssText|setAttribute\(\s*['\"]style", src)
    assert not re.search(r"setAttribute\(\s*['\"]on", src)


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
def test_js_fetch_same_origin_only(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    assert not re.search(r"https?://|//cdn|XMLHttpRequest|WebSocket|sendBeacon|importScripts", src)
    for m in re.finditer(r"""(?:fetch|request)\(\s*(["'])(.*?)\1""", src):
        assert m.group(2).startswith("/api/"), m.group(2)


def test_js_fetch_positive_control() -> None:
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert re.findall(r"""request\(\s*"(/api/[a-z/]+)"\)""", js), "no fetch call sites matched"


@pytest.mark.parametrize("path", HTML_FILES, ids=lambda p: p.name)
def test_html_no_inline_script_or_handlers_or_external_urls(path: Path) -> None:
    html = path.read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S | re.I)
    assert scripts, "expected script tags"
    for attrs, body in scripts:
        assert body.strip() == "", "inline <script> body"
        assert re.search(r'\bsrc="/[A-Za-z0-9_.-]+\.js"', attrs), attrs
    for tag in re.findall(r"<[a-zA-Z][^>]*>", html):
        assert not re.search(r"\son[a-z]+\s*=", tag, re.I), f"inline handler in {tag}"
        assert not re.search(r"\sstyle\s*=", tag, re.I), f"inline style attribute in {tag}"
        assert "javascript:" not in tag.lower()
    assert not re.search(r"https?://", html, re.I), "external URL in HTML"
    assert not re.search(r"<style\b", html, re.I)


def test_inline_handler_regex_has_positive_control() -> None:
    assert re.search(r"\son[a-z]+\s*=", '<button onclick="x()">', re.I)


def test_css_has_no_external_urls() -> None:
    for p in CSS_FILES:
        css = p.read_text(encoding="utf-8")
        assert not re.search(r"https?://|@import", css), p.name


def test_assets_are_same_origin_relative() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]*)"', html)
    assert refs, "no asset references found"
    for ref in refs:
        assert ref.startswith("/") and not ref.startswith("//"), ref
        assert (WEB / ref.lstrip("/")).is_file(), f"{ref} missing in web/"


def test_index_served_with_csp(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    assert csp == EXPECTED_CSP  # literal d.2 policy, deliberately not imported from src/
    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "Meridian Ops Portal" in r.text


@pytest.mark.parametrize(
    ("path", "types"),
    [
        ("/app.js", ("text/javascript", "application/javascript")),
        ("/helpers.js", ("text/javascript", "application/javascript")),
        ("/app.css", ("text/css",)),
    ],
)
def test_assets_content_types(client: TestClient, path: str, types: tuple[str, ...]) -> None:
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] in types
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in r.headers


PAGE_IDS = ["dash", "recv", "po", "sup", "kb", "console"]
TAB_IDS = ["insp", "trace", "eval"]


def test_pages_tabs_and_nav_exist() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for pid in PAGE_IDS:
        assert re.search(rf'<section class="page[^"]*" id="{pid}"', html), pid
        assert f'data-p="{pid}"' in html, f"nav for {pid}"
    for tid in TAB_IDS:
        assert f'id="{tid}"' in html, tid
        assert f'data-t="{tid}"' in html, f"tab button {tid}"
    for needed in [
        "Retrieval inspector",
        "Run trace",
        "Evals",
        "Answers only; actions come in Phase 2.",
    ]:
        assert needed in html
    assert 'id="themebtn"' in html and 'id="msgs"' in html and 'id="chips"' in html


def test_theme_persisted_and_no_architecture_diagram() -> None:
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "localStorage" in js
    html = (WEB / "index.html").read_text(encoding="utf-8").lower()
    assert "architecture" not in html and "<svg" not in html


def test_js_uses_validated_citations_and_sanitisers() -> None:
    js = (WEB / "app.js").read_text(encoding="utf-8")
    helpers = (WEB / "helpers.js").read_text(encoding="utf-8")
    assert "validCitations(res.citations)" in js
    assert "valid !== true" in helpers
    assert r"/^[A-Z]{2,3}-\d{3}$/" in helpers
    assert r"/^\d+(\.\d+)*$/" in helpers
    # ids built from API values go through getElementById, not selector strings
    assert not re.search(r"querySelector(All)?\([^)]*\+", js)
    # answer text is never regex-scanned for citations
    assert not re.search(r"\.answer\s*\.(match|replace|matchAll|exec)|\.match\(.*answer", js)
    # refusal: chips are only built in the non-refused branch
    assert "if (!refused) {" in js


# ------------------------------------------------- portal against the hardened API


@pytest.mark.parametrize("path", ["/", "/app.js", "/helpers.js", "/app.css"])
def test_portal_assets_carry_all_security_headers(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["content-security-policy"] == EXPECTED_CSP
    assert "unsafe-inline" not in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_portal_host_allowlist_positive_control() -> None:
    """The default TestClient host is refused, so the localhost client above is meaningful."""
    assert TestClient(create_app(web_dir=WEB)).get("/").status_code == 400


def test_portal_post_ask_json_from_portal_origin(make_pipeline) -> None:
    pipe, _ = make_pipeline("A standard PO is approved per the matrix [PR-001 §2.1].")
    c = TestClient(create_app(pipe, web_dir=WEB), base_url="http://localhost")
    body = '{"question": "Who approves a standard purchase order?"}'
    ok = c.post(
        "/api/ask",
        content=body,
        headers={"content-type": "application/json", "origin": "http://localhost:8000"},
    )
    assert ok.status_code == 200, ok.text
    assert "answer" in ok.json()
    bad = c.post(
        "/api/ask",
        content=body,
        headers={"content-type": "text/plain", "origin": "http://localhost:8000"},
    )
    assert bad.status_code == 415
