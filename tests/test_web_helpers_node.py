"""Unit e: node-based test of the pure helpers in web/helpers.js. Skips cleanly without node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
HELPERS = Path(__file__).resolve().parent.parent / "web" / "helpers.js"

pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed; helper tests skipped")


def run_js(expr: str) -> object:
    script = (
        f"const H=require({json.dumps(str(HELPERS))});process.stdout.write(JSON.stringify({expr}));"
    )
    out = subprocess.run(
        [NODE or "node", "-e", script], capture_output=True, text=True, check=True, timeout=30
    )
    return json.loads(out.stdout)


def test_valid_citations_filters_valid_true_only() -> None:
    cites = [
        {"doc_id": "WH-003", "section": "2.1", "valid": True},
        {"doc_id": "WH-003", "section": "9", "valid": False},
        {"doc_id": "PR-001", "section": "2", "valid": "true"},
        {"doc_id": "WH-003", "section": "2.1", "valid": True},
    ]
    got = run_js(f"H.validCitations({json.dumps(cites)}).map(c=>c.label)")
    assert got == ["WH-003 §2.1"]


def test_valid_citations_rejects_malformed_ids_even_if_valid() -> None:
    cites = [
        {"doc_id": '"><img src=x>', "section": "1", "valid": True},
        {"doc_id": "WH-003", "section": "1;alert(1)", "valid": True},
        {"doc_id": "wh-003", "section": "1", "valid": True},
        {"doc_id": "FIN-001", "section": "3.2.1", "valid": True},
    ]
    got = run_js(f"H.validCitations({json.dumps(cites)}).map(c=>c.label)")
    assert got == ["FIN-001 §3.2.1"]


def test_valid_citations_non_array() -> None:
    assert run_js("H.validCitations(null)") == []
    assert run_js("H.validCitations('WH-003 [WH-003 sec 1]')") == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("WH-001", "WH-001"),
        ("FIN-001", "FIN-001"),
        ("QA-001", "QA-001"),
        ("W-001", None),
        ("WHXX-001", None),
        ("WH-0011", None),
        ("WH-001\n", None),
        ("wh-001", None),
        ("WH-001' ] x", None),
        (5, None),
    ],
)
def test_sanitize_doc_id(value: object, expected: str | None) -> None:
    assert run_js(f"H.sanitizeDocId({json.dumps(value)})") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "1"),
        ("2.1", "2.1"),
        ("3.2.1", "3.2.1"),
        ("1.", None),
        (".1", None),
        ("a", None),
        ("1 2", None),
        ("1\n", None),
    ],
)
def test_sanitize_section(value: str, expected: str | None) -> None:
    assert run_js(f"H.sanitizeSection({json.dumps(value)})") == expected


def test_anchor_ids_split_lines_and_judge() -> None:
    assert run_js("H.sectionAnchorId('WH-003','2.1')") == "sec-WH-003-2.1"
    assert run_js("H.sectionAnchorId('bad','2.1')") is None
    assert run_js("H.splitLines('a\\r\\nb\\nc')") == ["a", "b", "c"]
    assert run_js("[H.judge('faithfulness',0.9),H.judge('faithfulness',0.89)]") == ["pass", "fail"]
    assert run_js("[H.judge('p95s',7.9),H.judge('p95s',8),H.judge('mrr',null)]") == [
        "pass",
        "fail",
        "none",
    ]
    assert run_js("H.clampPct(150)") == 100
