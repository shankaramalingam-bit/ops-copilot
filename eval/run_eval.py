"""RAG evaluation over the 15 owner-approved questions (eval/questions.yaml).

Builds the real Pipeline (OllamaEmbeddings + ChatOllama, temperature 0), asks every question in
hybrid mode, computes the metrics in code and writes eval/results/<timestamp>.json (full
records) and eval/latest.json (the shape the UI reads).

Usage (needs Ollama): LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1:8b python eval/run_eval.py

Per-question `passed` (see eval/report.md):
  unanswerable: refused.
  answerable:   answered (not refused) AND hit@4 AND citation accurate AND faithfulness == 1.0
                (every claim judged supported); multi/ambiguous additionally need ALL expected
                docs among the valid citations; ambiguous also needs a labelled "If you mean"
                reading.

Faithfulness limitation: the judge is the same small local model (llama3.1:8b) that wrote the
answer, so it shares the generator's blind spots and is noisy. Treat it as a smoke signal.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
K = 4
PASS_BARS: dict[str, float | int] = {
    "faithfulness": 0.90,
    "hit_at_4": 0.85,
    "mrr": 0.70,
    "refusal_correct": 2,
    "p95_latency_ms": 8000,
}
LATEST_KEYS = [
    "run_at",
    "model",
    "embed_model",
    "gate_threshold",
    "metrics",
    "pass_bars",
    "questions",
]
METRIC_KEYS = [
    "faithfulness",
    "hit_at_4",
    "mrr",
    "citation_accuracy",
    "refusal_correct",
    "refusal_total",
    "false_refusals",
    "answer_rate",
    "p95_latency_ms",
]
QUESTION_KEYS = [
    "id",
    "type",
    "question",
    "expected_docs",
    "retrieved_docs",
    "hit",
    "rr",
    "refused",
    "refusal_reason",
    "citations",
    "faithfulness",
    "latency_ms",
    "passed",
    "notes",
]

JUDGE_PROMPT = (
    "You are a strict fact checker. Below is CONTEXT and a CLAIM.\n"
    "Is this claim fully supported by the context? Answer YES or NO.\n\n"
    "CONTEXT:\n{context}\n\nCLAIM: {claim}\n\nAnswer (YES or NO):"
)

# ---------------------------------------------------------------- pure metric functions


def retrieval_hit(retrieved_docs: list[str], expected_docs: list[str], k: int = K) -> bool:
    """Any expected doc among the top-k retrieved chunks."""
    return any(d in expected_docs for d in retrieved_docs[:k])


def reciprocal_rank(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    """1/rank (1-based) of the first chunk from an expected doc, else 0."""
    for i, d in enumerate(retrieved_docs, start=1):
        if d in expected_docs:
            return 1.0 / i
    return 0.0


def coverage(retrieved_docs: list[str], expected_docs: list[str], k: int = K) -> bool:
    """All expected docs among the top-k chunks."""
    return bool(expected_docs) and set(expected_docs) <= set(retrieved_docs[:k])


def citation_accurate(citations: list[dict[str, Any]], expected_docs: list[str]) -> bool:
    """Valid citations are a subset of the expected docs and include at least one of them."""
    docs = {c["doc_id"] for c in citations if c.get("valid")}
    return bool(docs) and docs <= set(expected_docs)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(s)))
    return s[rank - 1]


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence claims. Citations stay attached to the sentence they follow.
    Bare labels ("If you mean X:") and fragments without a verb-like length are dropped."""
    text = re.sub(r"([.!?])((?:\s*\[[^\[\]]*\])*)(?:\s+|$)", r"\1\2\n", answer.strip())
    claims = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip()
        body = re.sub(r"\[[^\[\]]*\]", "", line).strip()
        if len(body.split()) < 3:
            continue
        if body.endswith(":") and "[" not in line:
            continue
        claims.append(line)
    return claims


def claim_citations(claim: str) -> list[tuple[str, str]]:
    return [
        (m.group(1), m.group(2))
        for m in re.finditer(r"\[([A-Z]{2,3}-\d{3}) §\s*(\d+(?:\.\d+)*)\]", claim)
    ]


def faithfulness_score(verdicts: list[bool]) -> float | None:
    return sum(verdicts) / len(verdicts) if verdicts else None


def aggregate_faithfulness(per_question: list[list[bool]]) -> float:
    """Micro average: supported claims / total claims over all answered questions."""
    total = sum(len(v) for v in per_question)
    return sum(sum(v) for v in per_question) / total if total else 0.0


def question_passed(rec: dict[str, Any]) -> bool:
    if rec["type"] == "unanswerable":
        return bool(rec["refused"])
    if rec["refused"] or not rec["hit"]:
        return False
    exp = rec["expected_docs"]
    if not citation_accurate(rec["citations"], exp):
        return False
    if rec["faithfulness"] is None or rec["faithfulness"] < 1.0:
        return False
    if rec["type"] in ("multi", "ambiguous"):
        cited = {c["doc_id"] for c in rec["citations"] if c["valid"]}
        if not set(exp) <= cited:
            return False
    if rec["type"] == "ambiguous" and "if you mean" not in rec["answer"].lower():
        return False
    return True


def compute_metrics(recs: list[dict[str, Any]], verdicts: list[list[bool]]) -> dict[str, Any]:
    ans = [r for r in recs if r["type"] != "unanswerable"]
    unans = [r for r in recs if r["type"] == "unanswerable"]
    answered = [r for r in recs if not r["refused"]]
    return {
        "faithfulness": aggregate_faithfulness(verdicts),
        "hit_at_4": sum(bool(r["hit"]) for r in ans) / len(ans) if ans else 0.0,
        "mrr": sum(r["rr"] for r in ans) / len(ans) if ans else 0.0,
        "citation_accuracy": (
            sum(citation_accurate(r["citations"], r["expected_docs"]) for r in answered)
            / len(answered)
            if answered
            else 0.0
        ),
        "refusal_correct": sum(bool(r["refused"]) for r in unans),
        "refusal_total": len(unans),
        "false_refusals": sum(bool(r["refused"]) for r in ans),
        "answer_rate": sum(not r["refused"] for r in ans) / len(ans) if ans else 0.0,
        "p95_latency_ms": percentile([r["latency_ms"] for r in recs], 95),
    }


def kpi_status(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "faithfulness": metrics["faithfulness"] >= PASS_BARS["faithfulness"],
        "hit_at_4": metrics["hit_at_4"] >= PASS_BARS["hit_at_4"],
        "mrr": metrics["mrr"] >= PASS_BARS["mrr"],
        "refusal_correct": metrics["refusal_correct"] >= PASS_BARS["refusal_correct"],
        "p95_latency_ms": metrics["p95_latency_ms"] < PASS_BARS["p95_latency_ms"],
    }


def build_latest(
    run_at: str,
    model: str,
    embed_model: str,
    gate_threshold: float,
    metrics: dict[str, Any],
    recs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_at": run_at,
        "model": model,
        "embed_model": embed_model,
        "gate_threshold": gate_threshold,
        "metrics": {k: metrics[k] for k in METRIC_KEYS},
        "pass_bars": dict(PASS_BARS),
        "questions": [{k: r[k] for k in QUESTION_KEYS} for r in recs],
    }


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    data = yaml.safe_load((path or EVAL_DIR / "questions.yaml").read_text())
    return list(data["questions"])


# ---------------------------------------------------------------- live run


def _judge(chat: Any, context: str, claim: str) -> tuple[bool, str]:
    from langchain_core.messages import HumanMessage

    msg = chat.invoke([HumanMessage(content=JUDGE_PROMPT.format(context=context, claim=claim))])
    raw = msg.content if isinstance(msg.content, str) else str(msg.content)
    first = raw.strip().upper().lstrip("*\"' ")
    return first.startswith("YES"), raw.strip()[:200]


def _notes(rec: dict[str, Any], cov: bool | None) -> str:
    t, why = rec["type"], rec["refusal_reason"]
    if t == "unanswerable":
        if rec["refused"]:
            return f"correctly refused ({why})"
        return "ANSWERED an unanswerable question"
    parts = []
    if rec["refused"]:
        parts.append(f"false refusal: {why}")
        if not rec["hit"]:
            parts.append("retrieval miss")
    else:
        if not rec["hit"]:
            parts.append("retrieval miss")
        if not citation_accurate(rec["citations"], rec["expected_docs"]):
            parts.append("citation not within expected docs")
        if rec["faithfulness"] is not None and rec["faithfulness"] < 1.0:
            parts.append("unfaithful claim(s)")
        if t in ("multi", "ambiguous"):
            cited = {c["doc_id"] for c in rec["citations"] if c["valid"]}
            if not set(rec["expected_docs"]) <= cited:
                parts.append("not all expected docs cited")
        if t == "ambiguous" and "if you mean" not in rec["answer"].lower():
            parts.append("no labelled readings")
    if cov is False:
        parts.append("not all expected docs in top-4")
    return "; ".join(parts) if parts else "ok"


def main() -> int:
    from ops_copilot.config import Settings
    from ops_copilot.llm import get_chat_model
    from ops_copilot.pipeline import Pipeline

    s = Settings.from_env()
    qs = load_questions()
    pipe = Pipeline(s)
    judge = get_chat_model(s)  # same model, temperature 0
    recs: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    verdicts: list[list[bool]] = []

    for q in qs:
        t0 = time.perf_counter()
        res = pipe.ask(q["question"], "hybrid")
        latency = (time.perf_counter() - t0) * 1000.0
        chunks = res["chunks"]
        retrieved = [c["id"].split("#")[0] for c in chunks]
        exp = q["expected_docs"]
        answerable = q["type"] != "unanswerable"
        hit = retrieval_hit(retrieved, exp) if answerable else None
        rr = reciprocal_rank(retrieved, exp) if answerable else None
        cov = coverage(retrieved, exp) if answerable and len(exp) > 1 else None

        judged: list[dict[str, Any]] = []
        faith: float | None = None
        if not res["refused"]:
            used_text = {c["id"]: c["text"] for c in chunks if c["used"]}
            vs: list[bool] = []
            for claim in split_claims(res["answer"]):
                cited = claim_citations(claim)
                ctx_ids = [
                    cid
                    for cid in used_text
                    if any(
                        cid.split("#")[0] == d
                        and (cid.split("#")[1] == sec or cid.split("#")[1].startswith(sec + "."))
                        for d, sec in cited
                    )
                ] or list(used_text)
                ctx = "\n\n".join(used_text[c] for c in ctx_ids)
                ok, raw = _judge(judge, ctx, claim)
                vs.append(ok)
                judged.append(
                    {"claim": claim, "supported": ok, "judge_raw": raw, "chunks": ctx_ids}
                )
            verdicts.append(vs)
            faith = faithfulness_score(vs)

        rec: dict[str, Any] = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected_docs": exp,
            "retrieved_docs": retrieved,
            "hit": hit,
            "rr": rr,
            "refused": res["refused"],
            "refusal_reason": res["refusal_reason"],
            "citations": res["citations"],
            "faithfulness": faith,
            "latency_ms": round(latency, 1),
            "answer": res["answer"],
        }
        rec["passed"] = question_passed(rec)
        rec["notes"] = _notes(rec, cov)
        recs.append(rec)
        full.append(
            {
                **rec,
                "expected_sections": q["expected_sections"],
                "expected_answer_points": q["expected_answer_points"],
                "coverage_at_4": cov,
                "confidence": res["confidence"],
                "chunk_ids": [c["id"] for c in chunks],
                "chunk_ranks": [
                    {
                        "id": c["id"],
                        "bm25_rank": c["bm25_rank"],
                        "dense_rank": c["dense_rank"],
                        "rrf": c["rrf"],
                        "used": c["used"],
                    }
                    for c in chunks
                ],
                "timings_ms": res["timings_ms"],
                "judge": judged,
            }
        )
        print(
            f"{q['id']:<4}{q['type']:<13} refused={res['refused']!s:<5} "
            f"reason={res['refusal_reason']!s:<20} hit={hit!s:<5} faith={faith} "
            f"{latency:7.0f}ms passed={rec['passed']}",
            flush=True,
        )

    metrics = compute_metrics(recs, verdicts)
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    latest = build_latest(run_at, s.ollama_model, s.embed_model, s.gate_threshold, metrics, recs)
    (EVAL_DIR / "results").mkdir(exist_ok=True)
    stamp = run_at.replace(":", "").replace("-", "").replace("+0000", "Z").replace("+00:00", "Z")
    detail = {**latest, "questions": full, "kpi_pass": kpi_status(metrics)}
    (EVAL_DIR / "results" / f"{stamp}.json").write_text(json.dumps(detail, indent=2) + "\n")
    (EVAL_DIR / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")

    print("\n=== summary ===")
    print(f"run_at={run_at} model={s.ollama_model} embed={s.embed_model} gate={s.gate_threshold}")
    for k in METRIC_KEYS:
        print(f"  {k:<18} {metrics[k]}")
    for k, ok in kpi_status(metrics).items():
        print(f"  KPI {k:<16} {'PASS' if ok else 'FAIL'} (bar {PASS_BARS[k]})")
    print(f"  questions passed: {sum(r['passed'] for r in recs)}/{len(recs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
