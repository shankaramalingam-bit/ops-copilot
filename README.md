# Ops Copilot

Ops Copilot answers questions about Meridian Supply Co.'s (fictional) warehouse and procurement SOPs, with citations, and refuses when the documents do not cover the question. Stack: Python 3.12, LangChain, Chroma, BM25, Ollama (or Groq), FastAPI, uv.

## Status

**Phase 1: working prototype, eval bars not all met.** The official 15-question eval (2026-11-20, code `6e2718f`, nonce-hardened judge prompt) passes 3 of 5 KPIs and **fails 2**: faithfulness (0.381 vs 0.90, FAIL) and p95 latency (13.8 s vs 8 s, FAIL). Retrieval (hit@4 1.0, MRR 1.0) and refusal (2/2) pass; 7 of 15 questions pass; answer quality on multi-document and ambiguous questions is weak. Nothing was loosened to pass. Details: [eval/report.md](eval/report.md).

## Setup

```bash
uv sync
ollama pull nomic-embed-text
ollama pull llama3.1:8b
cp .env.example .env    # then set LLM_PROVIDER=ollama and OLLAMA_MODEL=llama3.1:8b
```

Set `GROQ_API_KEY=` in `.env` only if you use `LLM_PROVIDER=groq`. Never commit `.env` (it is git-ignored).

With Docker (from the repository root; `--env-file .env` is required because compose's project directory is `deploy/`, so otherwise it reads `deploy/.env` and falls back to the defaults):

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml exec ollama ollama pull nomic-embed-text
docker compose -f deploy/docker-compose.yml exec ollama ollama pull llama3.1:8b
```

## Run the portal

```bash
uv run python -m ops_copilot
```

Then open http://localhost:8000 (or `make run`). Needs Ollama running locally. Docker: see [deploy/README.md](deploy/README.md).

## Notebook, eval, tests

```bash
make notebook                                   # executes notebooks/01_rag_walkthrough.ipynb (needs Ollama)
uv run python eval/run_eval.py                  # 15-question eval -> eval/latest.json
uv run python eval/calibrate.py                 # gate-threshold calibration on the held-out set
uv run pytest -q                                # offline tests
for g in scripts/gate_*.sh; do bash "$g"; done  # all gates (lint, typecheck, tests, secrets)
```

The eval sends corpus text to the LLM, so it refuses to run unless the provider is local Ollama on a loopback host. To run it against a remote provider (for example `LLM_PROVIDER=groq`) or a non-loopback Ollama, set `EVAL_ALLOW_REMOTE=1` deliberately.

## Architecture

```mermaid
flowchart LR
    D[Documents] --> I[Ingest and clean]
    I --> C[Chunk by heading]
    C --> V[(Chroma dense index)]
    C --> B[BM25 index]
    V --> H[Hybrid RRF k=60, top 4]
    B --> H
    H --> G{Confidence gate 0.61}
    G -- below --> R[Refuse, LLM not called]
    G -- pass --> P[Grounded prompt]
    P --> L[LLM]
    L --> X[Citation validator]
    X --> A[FastAPI]
    R --> A
    A --> W[Portal]
```

## Security notes

- **No authentication.** The API and portal are open to whoever can reach them.
- **Localhost only by default**: the server binds 127.0.0.1. Binding elsewhere requires `OPS_COPILOT_ALLOW_NONLOCAL=1`, and the API checks the `Host` and `Origin` headers against `OPS_COPILOT_ALLOWED_HOSTS` and requires a JSON content type. The Docker setup publishes only `127.0.0.1:8000`.
- **Plain-text rendering**: the portal renders model output as text, never HTML.
- **Citations are validated**: every `[DOC §section]` must match a retrieved chunk, otherwise the answer is replaced by a refusal.
- **Never print resolved secrets**: `docker compose config` and `docker inspect` print `GROQ_API_KEY` in clear text when it is set. Never paste their output into chat, issues or logs.
- No secrets in the repo; `.env` is git-ignored and excluded from the Docker build context.

## Latest eval

Official run 2026-11-20T09:57:08Z on commit `6e2718f`, `llama3.1:8b` (generator and judge, hardened judge prompt), `nomic-embed-text`, hybrid top-4, gate 0.61. Source: `eval/latest.json`; full report with per-question results and failure analysis: [eval/report.md](eval/report.md).

| KPI | Result | Bar | Status |
|---|---|---|---|
| Faithfulness | 0.381 (8/21 claims) | >= 0.90 | **FAIL** |
| hit@4 | 1.000 | >= 0.85 | PASS |
| MRR | 1.000 | >= 0.70 | PASS |
| Refusal correct | 2/2 | 2/2 | PASS |
| p95 latency | 13,808.5 ms | < 8,000 ms | **FAIL** |

Questions passed: 7 of 15. Also: citation accuracy 0.917, answer rate 0.923, false refusals 1 (Q10, model refused). The faithfulness drop from the earlier 0.632 is explained in [eval/report.md](eval/report.md): same-model judge, changed judge prompt and run-to-run nondeterminism, not separable from the records.

## Known limitations

- **Official run on the hardened judge prompt**: the figures above come from the single official run with the nonce-hardened judge prompt; there is no second run to show the spread.
- **Same-model judge**: the faithfulness judge is the same 8B model as the generator; in the official run it gave false negatives on claims that match the corpus word for word (for example Q1, Q3) and a lenient pass on a garbled answer (Q6). Treat faithfulness as a smoke signal.
- **8B latency**: a local 8B model takes about 3 to 14 s per answered question; multi-document questions set the p95.
- **Validator false refusals**: the citation validator can refuse a correct answer, for example when it mentions a token like `PO-2291` or `COVID-19`, uses a citation form outside the grammar (ranges, "Section n"), uses non-ASCII symbols, or cites a section that was not retrieved. The full list is in [eval/report.md](eval/report.md).
- Multi-document and ambiguous questions are often answered incompletely; the gate threshold margin is thin (one unanswerable question cannot be separated from the answerable ones).
- Single eval run; local Ollama at temperature 0 is not guaranteed deterministic.
