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


# ---- unit H-app E1: section depth cap, chip label cap, refusal constant, eval summary

LATEST = Path(__file__).resolve().parent.parent / "eval" / "latest.json"
SERVER_REFUSAL = "I couldn't find this in Meridian's documents."


def test_sanitize_section_depth_capped_at_five_levels() -> None:
    assert run_js("H.sanitizeSection('1.2.3.4.5')") == "1.2.3.4.5"
    assert run_js("H.sanitizeSection('1.2.3.4.5.6')") is None
    assert run_js("H.sectionAnchorId('WH-003','1.2.3.4.5.6')") is None


def test_chip_label_length_capped_at_40() -> None:
    long_sec = "1." + ".".join(["123456789012"] * 4)  # valid depth (5), very long digits
    cites = [{"doc_id": "WH-003", "section": long_sec, "valid": True}]
    labels = run_js(f"H.validCitations({json.dumps(cites)}).map(c=>c.label)")
    assert isinstance(labels, list) and len(labels) == 1
    assert len(labels[0]) <= 40 and labels[0].startswith("WH-003 §1.")
    # a normal label is untouched
    short = [{"doc_id": "WH-003", "section": "2.1", "valid": True}]
    assert run_js(f"H.validCitations({json.dumps(short)}).map(c=>c.label)") == ["WH-003 §2.1"]
    # too deep: no chip at all
    deep = [{"doc_id": "WH-003", "section": "1.2.3.4.5.6", "valid": True}]
    assert run_js(f"H.validCitations({json.dumps(deep)})") == []


def test_client_refusal_constant_is_identical_to_server_text() -> None:
    assert run_js("H.REFUSAL_TEXT") == SERVER_REFUSAL
    from ops_copilot.answer import REFUSAL

    assert run_js("H.REFUSAL_TEXT") == REFUSAL


def test_app_renders_refusal_constant_not_server_answer() -> None:
    js = (HELPERS.parent / "app.js").read_text(encoding="utf-8")
    assert "H.REFUSAL_TEXT" in js
    assert "multiline(refused ? H.REFUSAL_TEXT : res.answer)" in js


def _js_summary(path: Path) -> dict:
    expr = f"H.evalSummary(JSON.parse(require('fs').readFileSync({json.dumps(str(path))},'utf8')))"
    s = run_js(expr)
    assert isinstance(s, dict)
    return s


def _verdict(ok: bool) -> str:
    return "pass" if ok else "fail"


def _assert_summary_matches(s: dict, d: dict) -> None:
    """Expected values computed in Python from the file's own metrics and pass_bars."""
    m, bars, k = d["metrics"], d["pass_bars"], s["kpis"]
    for key, mkey, bkey in (
        ("faithfulness", "faithfulness", "faithfulness"),
        ("hit4", "hit_at_4", "hit_at_4"),
        ("mrr", "mrr", "mrr"),
    ):
        assert k[key]["value"] == pytest.approx(m[mkey])
        assert k[key]["text"] == f"{m[mkey]:.2f}"
        assert k[key]["verdict"] == _verdict(m[mkey] >= bars[bkey])
    assert k["refusal"]["text"] == f"{m['refusal_correct']} / {m['refusal_total']}"
    ref_ok = m["refusal_correct"] >= bars["refusal_correct"]
    ref_all = m["refusal_correct"] >= m["refusal_total"]
    assert k["refusal"]["verdict"] == _verdict(ref_ok and ref_all)
    p95_s = m["p95_latency_ms"] / 1000
    assert k["p95"]["seconds"] == pytest.approx(p95_s)
    assert k["p95"]["text"] == f"{p95_s:.1f} s"
    assert k["p95"]["verdict"] == _verdict(m["p95_latency_ms"] < bars["p95_latency_ms"])
    assert [r["id"] for r in s["rows"]] == [q["id"] for q in d["questions"]]


def test_eval_summary_reads_real_latest_json() -> None:
    d = json.loads(LATEST.read_text(encoding="utf-8"))
    s = _js_summary(LATEST)
    _assert_summary_matches(s, d)
    assert len(s["rows"]) == 15 and s["rows"][0]["id"] == "Q1"


def test_eval_summary_matches_committed_file_values() -> None:
    """JS output equals values derived in Python from the committed file (no hard-coded run)."""
    d = json.loads(LATEST.read_text(encoding="utf-8"))
    assert len(d["questions"]) == 15
    for key in ("faithfulness", "hit_at_4", "mrr", "refusal_correct", "p95_latency_ms"):
        assert key in d["metrics"] and key in d["pass_bars"]
    s = _js_summary(LATEST)
    _assert_summary_matches(s, d)
    m = d["metrics"]
    assert s["kpis"]["p95"]["seconds"] == pytest.approx(m["p95_latency_ms"] / 1000)
    assert s["kpis"]["refusal"]["text"] == f"{m['refusal_correct']} / {m['refusal_total']}"


def test_eval_summary_fixture_with_known_numbers(tmp_path: Path) -> None:
    """Hand-written metrics with exact expected outputs: JS logic checked against known numbers."""
    fixture: dict = {
        "metrics": {
            "faithfulness": 0.9,
            "hit_at_4": 0.8,
            "mrr": 0.7,
            "refusal_correct": 2,
            "refusal_total": 2,
            "p95_latency_ms": 7999.0,
        },
        "questions": [{"id": "A1"}, {"id": "A2"}, {"id": "A3"}],
    }
    f = tmp_path / "latest.json"
    f.write_text(json.dumps(fixture), encoding="utf-8")
    s = _js_summary(f)
    k = s["kpis"]
    assert (k["faithfulness"]["text"], k["faithfulness"]["verdict"]) == ("0.90", "pass")
    assert (k["hit4"]["text"], k["hit4"]["verdict"]) == ("0.80", "fail")
    assert (k["mrr"]["text"], k["mrr"]["verdict"]) == ("0.70", "pass")
    assert (k["refusal"]["text"], k["refusal"]["verdict"]) == ("2 / 2", "pass")
    assert k["p95"]["seconds"] == pytest.approx(7.999)
    assert (k["p95"]["text"], k["p95"]["verdict"]) == ("8.0 s", "pass")
    assert [r["id"] for r in s["rows"]] == ["A1", "A2", "A3"]

    fixture["metrics"].update(
        faithfulness=0.381, hit_at_4=1.0, mrr=1.0, refusal_correct=1, p95_latency_ms=13808.5
    )
    f.write_text(json.dumps(fixture), encoding="utf-8")
    k = _js_summary(f)["kpis"]
    assert (k["faithfulness"]["text"], k["faithfulness"]["verdict"]) == ("0.38", "fail")
    assert (k["hit4"]["text"], k["hit4"]["verdict"]) == ("1.00", "pass")
    assert (k["refusal"]["text"], k["refusal"]["verdict"]) == ("1 / 2", "fail")
    assert (k["p95"]["text"], k["p95"]["verdict"]) == ("13.8 s", "fail")


def test_eval_summary_edge_cases() -> None:
    assert run_js("H.evalSummary({}).kpis.refusal.verdict") == "none"
    assert run_js("H.evalSummary({}).rows") == []
    assert (
        run_js("H.evalSummary({metrics:{refusal_correct:1,refusal_total:2}}).kpis.refusal.verdict")
        == "fail"
    )
    assert run_js("H.evalSummary({metrics:{p95_latency_ms:7999}}).kpis.p95.verdict") == "pass"
    assert run_js("H.evalSummary({metrics:{p95_latency_ms:8000}}).kpis.p95.verdict") == "fail"
