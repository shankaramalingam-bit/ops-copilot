"""Gate-threshold calibration on the held-out set (eval/calibration.yaml). Retrieval only, no LLM.

Records the max dense cosine per question, then picks the threshold on a 0.01 grid in
[0.05, 1.00] that maximises balanced accuracy (answerable passes when score >= threshold,
unanswerable is refused when score < threshold). Ties go to the LOWER threshold (fewer false
refusals). Writes eval/results/calibration.json.

Usage (needs Ollama): LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1:8b python eval/calibrate.py
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
MIN_T, MAX_T = 0.05, 1.0


def balanced_accuracy(pos: list[float], neg: list[float], t: float) -> float:
    """pos = scores of answerable questions (should pass), neg = unanswerable (should refuse)."""
    tpr = sum(s >= t for s in pos) / len(pos)
    tnr = sum(s < t for s in neg) / len(neg)
    return (tpr + tnr) / 2


def choose_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Return (threshold, balanced_accuracy) on a 0.01 grid; ties -> lowest threshold."""
    if not pos or not neg:
        raise ValueError("need both answerable and unanswerable scores")
    best_t, best_ba = MIN_T, -1.0
    for i in range(int(round(MIN_T * 100)), int(round(MAX_T * 100)) + 1):
        t = i / 100
        ba = balanced_accuracy(pos, neg, t)
        if ba > best_ba + 1e-12:
            best_t, best_ba = t, ba
    return round(best_t, 2), best_ba


def distribution(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    return {
        "n": len(s),
        "min": s[0],
        "p25": statistics.quantiles(s, n=4)[0] if len(s) > 1 else s[0],
        "median": statistics.median(s),
        "p75": statistics.quantiles(s, n=4)[2] if len(s) > 1 else s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


def main() -> int:
    from ops_copilot.config import Settings
    from ops_copilot.pipeline import Pipeline
    from ops_copilot.retrieve import retrieve

    cal: dict[str, Any] = yaml.safe_load((EVAL_DIR / "calibration.yaml").read_text())
    s = Settings.from_env()
    pipe = Pipeline(s)
    rows: list[dict[str, Any]] = []
    for label, key in (("answerable", "answerable"), ("unanswerable", "unanswerable")):
        for item in cal[key]:
            r = retrieve(pipe.index, item["question"], "dense", s.top_k, s.rrf_k, s.candidates)
            top = r.hits[0].chunk.id if r.hits else None
            rows.append(
                {
                    "id": item["id"],
                    "label": label,
                    "question": item["question"],
                    "source": item.get("source"),
                    "max_cosine": r.confidence,
                    "top_chunk": top,
                }
            )
    pos = [r["max_cosine"] for r in rows if r["label"] == "answerable"]
    neg = [r["max_cosine"] for r in rows if r["label"] == "unanswerable"]
    t, ba = choose_threshold(pos, neg)
    grid = {f"{i / 100:.2f}": balanced_accuracy(pos, neg, i / 100) for i in range(5, 101)}
    best_grid = sorted(k for k, v in grid.items() if abs(v - ba) < 1e-12)
    out = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": s.embed_model,
        "chosen_threshold": t,
        "balanced_accuracy": ba,
        "tpr": sum(x >= t for x in pos) / len(pos),
        "tnr": sum(x < t for x in neg) / len(neg),
        "tied_optimal_thresholds": [best_grid[0], best_grid[-1]],
        "distribution": {"answerable": distribution(pos), "unanswerable": distribution(neg)},
        "questions": rows,
    }
    (EVAL_DIR / "results").mkdir(exist_ok=True)
    (EVAL_DIR / "results" / "calibration.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"embed_model={s.embed_model} questions: {len(pos)} answerable, {len(neg)} unanswerable")
    for r in sorted(rows, key=lambda r: r["max_cosine"]):
        print(f"  {r['label']:<12} {r['id']:<4} {r['max_cosine']:.4f}  {r['question']}")
    for name in ("answerable", "unanswerable"):
        d = out["distribution"][name]
        print(
            f"{name}: n={d['n']} min={d['min']:.3f} p25={d['p25']:.3f} median={d['median']:.3f} "
            f"p75={d['p75']:.3f} max={d['max']:.3f} mean={d['mean']:.3f}"
        )
    print(f"tied optimal thresholds span {best_grid[0]}..{best_grid[-1]}")
    print(
        f"chosen threshold={t:.2f} balanced_accuracy={ba:.3f} TPR={out['tpr']:.3f} "
        f"TNR={out['tnr']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
