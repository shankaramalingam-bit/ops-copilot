# Ops Copilot — Phase 1 eval report

**Run:** 2026-11-20T09:06:01+00:00 · generator and judge `llama3.1:8b` (Ollama, temperature 0) · embeddings `nomic-embed-text` · hybrid retrieval (BM25 + dense, RRF k=60, top-4) · gate threshold 0.61 (calibrated on a held-out set).
Raw data: `eval/results/20261120T090601Z.json`. Single run; local Ollama at temperature 0 is not guaranteed deterministic.

## KPIs vs pass bars

| KPI | Result | Bar | Status |
|---|---|---|---|
| Faithfulness | 0.632 (12/19 claims) | ≥ 0.90 | **FAIL** |
| hit@4 | 1.000 | ≥ 0.85 | PASS |
| MRR | 1.000 | ≥ 0.70 | PASS |
| Refusal correct | 2/2 | 2/2 | PASS |
| p95 latency | 15,877 ms | < 8,000 ms | **FAIL** |

Supporting: citation accuracy 0.917 (11/12 answered), answer rate 0.923 (12/13 answerable), false refusals 1, questions passed 7/15.

## Definition of `passed`
- Unanswerable: refused.
- Answerable: answered, hit@4, valid citations ⊆ expected docs with ≥1 expected doc cited, and per-question faithfulness = 1.0.
- Multi-doc and ambiguous additionally require every expected doc among the valid citations; ambiguous additionally requires labelled "If you mean …" readings.

## Per-question results

| Q | Type | Refused (reason) | hit@4 | Faithfulness | Latency (ms) | Passed |
|---|---|---|---|---|---|---|
| Q1 | direct | no | yes | 1.0 | 10,769 | yes |
| Q2 | direct | no | yes | 1.0 | 9,563 | yes |
| Q3 | direct | no | yes | 1.0 | 4,148 | yes |
| Q4 | direct | no | yes | 0.0 | 4,140 | no |
| Q5 | direct | no | yes | 1.0 | 4,008 | yes |
| Q6 | direct | no | yes | 0.0 | 4,618 | no |
| Q7 | direct | no | yes | 0.5 | 6,813 | no |
| Q8 | direct | no | yes | 1.0 | 4,761 | yes |
| Q9 | multi | no | yes | 1.0 | 10,852 | no |
| Q10 | multi | yes (invalid_citation) | yes | – | 13,147 | no |
| Q11 | multi | no | yes | 0.25 | 15,877 | no |
| Q12 | ambiguous | no | yes | 0.0 | 4,008 | no |
| Q13 | ambiguous | no | yes | 1.0 | 4,709 | no |
| Q14 | unanswerable | yes (below_gate) | – | – | 65 | yes |
| Q15 | unanswerable | yes (below_gate) | – | – | 22 | yes |

## Failure analysis (what failed and why)
- **Retrieval misses (hit@4):** none — every answerable question's first chunk came from an expected doc. Multi-doc *coverage* within top-4 was partial: Q9 (QA-001 absent), Q10 (WH-005 absent; all four chunks WH-002), Q13 (WH-003 absent). Expected for multi-doc questions at top-k=4; not tuned.
- **Gate refusals:** none on answerable questions (lowest answerable confidence 0.663, Q13). Q14 (0.548) and Q15 (0.530) correctly refused without calling the LLM.
- **Validator refusal:** Q10 (`invalid_citation`). The raw model text is not retained by the pipeline, so the exact trigger is unknown.
- **Model refusals:** none.
- **Q4** — answer correct against PR-002 §2.2 but the judge said NO: judge false negative.
- **Q6** — genuine error: garbled the tolerance ("up to $50, whichever is lower than ±2% of the PO price").
- **Q7** — gave two "If you mean" readings on a direct question; the first put exactly $3,000 in the wrong band (the right approver was named).
- **Q9** — judge scored 4/4 supported, but the answer is a set of near-empty conditional paragraphs, cites WH-001 §4.2 (not expected), and omits the 24-hour notice and the NCR: judge false positive; caught by citation accuracy.
- **Q11** — genuine errors: computed $31,500 then said the CFO approves it as "above $100,000" (wrong band; $25,000–$100,000 → Director of Operations); cited PR-001 §5.3 for a claim it does not make; never said 5% exceeds the price tolerance.
- **Q12** — one reading only (WH-005 $5,000); no labelled readings, PO matrix not mentioned.
- **Q13** — only the QA-001 reading (1 business day); no labelled readings; faithful but incomplete.
- **Latency:** p95 is set by the multi-doc questions (Q11 15.9 s, Q10 13.1 s, Q9 10.9 s). The LLM step is 95–100% of each answered question's time; Q1 includes the first model load.
- **Judge limitation:** the judge is the same small local model as the generator and showed both a false negative (Q4) and a false positive (Q9). Each claim is checked only against the chunks it cites (or all used chunks if it cites none). Treat faithfulness as a smoke signal, not a precise measure.

## Gate threshold calibration (held-out, not the 15)
16 answerable and 10 unanswerable in-domain questions (`eval/calibration.yaml`), retrieval only.
- answerable: min 0.631 · p25 0.716 · median 0.767 · p75 0.826 · max 0.855
- unanswerable: min 0.501 · p25 0.526 · median 0.577 · p75 0.605 · max 0.664
- Chosen threshold **0.61** (balanced accuracy 0.950; TPR 1.000; TNR 0.900); ties 0.61–0.63 broken toward the lower value. The margin is thin: U3 (EDI formats, 0.664) cannot be separated by the gate; lowest answerable is C10 (0.631).

## Known validator false-refusal classes
- Any 2–4 letter + hyphen + digits token that is not a validly cited uppercase doc ID (PO-2291, RCV-5104, COVID-19, lowercase wh-003, Tier-2).
- Citation forms outside the grammar: ranges, parentheses, "Section n", a cited child of a retrieved section.
- Characters outside the allowlist (arrows, comparison signs, other non-ASCII symbols).
- Square/angle brackets or braces used for anything other than a canonical citation.
- A cited section not among the retrieved chunks.
The validator was not loosened for this eval; the prompt change reduces but does not eliminate this class (Q10).
