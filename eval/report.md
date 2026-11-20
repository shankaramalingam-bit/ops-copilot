# Ops Copilot — Phase 1 eval (official run)

**Run:** 2026-11-20T09:57:08Z · code commit `6e2718f` · generator and judge `llama3.1:8b` (Ollama, temperature 0) · embeddings `nomic-embed-text` · hybrid retrieval (BM25 + dense, RRF k=60, top-4) · gate threshold 0.61 (calibrated on a held-out set) · judge prompt: nonce-hardened (per-call random-token tags around context and claim).
Raw data: `eval/results/20261120T095708Z.json` (`eval/latest.json` is the same run without the answers, judge detail and timings). Single run; local Ollama at temperature 0 is not guaranteed deterministic. Nothing was tuned for this run.

## KPIs vs pass bars

| KPI | Result | Bar | Status |
|---|---|---|---|
| Faithfulness | 0.381 (8/21 claims) | ≥ 0.90 | **FAIL** |
| hit@4 | 1.000 | ≥ 0.85 | PASS |
| MRR | 1.000 | ≥ 0.70 | PASS |
| Refusal correct | 2/2 | 2/2 | PASS |
| p95 latency | 13,808.5 ms | < 8,000 ms | **FAIL** |

Notes: p95 uses the nearest-rank method (rank ceil(0.95 × 15) = 15, i.e. the maximum, Q11 = 13,808.5 ms). Linear interpolation over the same 15 recorded latencies gives 11,442.5 ms, which also fails the 8,000 ms bar. hit@4/MRR are doc-level (an expected document appears in the top-4 / its rank), not section-level.

Supporting: citation accuracy 0.917 (11/12 answered), answer rate 0.923 (12/13 answerable), false refusals 1 (Q10, `model_refused`), questions passed 7/15 (Q2, Q4, Q5, Q6, Q8, Q14, Q15).

## Definition of `passed`
- Unanswerable: refused.
- Answerable: answered, hit@4, valid citations ⊆ expected docs with ≥1 expected doc cited, and per-question faithfulness = 1.0.
- Multi-doc and ambiguous additionally require every expected doc among the valid citations; ambiguous additionally requires labelled "If you mean …" readings.

## Per-question results

Prior = the earlier run of 2026-11-20T09:06:01Z (judge prompt before hardening), shown in brackets.

| Q | Type | Refused (reason) | hit@4 | Faithfulness (prior) | Latency (ms) | Passed (prior) |
|---|---|---|---|---|---|---|
| Q1 | direct | no | yes | 0.0 (1.0) | 3,362 | no (yes) |
| Q2 | direct | no | yes | 1.0 (1.0) | 3,650 | yes (yes) |
| Q3 | direct | no | yes | 0.0 (1.0) | 3,798 | no (yes) |
| Q4 | direct | no | yes | 1.0 (0.0) | 3,923 | yes (no) |
| Q5 | direct | no | yes | 1.0 (1.0) | 4,211 | yes (yes) |
| Q6 | direct | no | yes | 1.0 (0.0) | 4,427 | yes (no) |
| Q7 | direct | no | yes | 0.5 (0.5) | 6,545 | no (no) |
| Q8 | direct | no | yes | 1.0 (1.0) | 4,418 | yes (yes) |
| Q9 | multi | no | yes | 0.25 (1.0) | 10,423 | no (no) |
| Q10 | multi | yes (model_refused; prior invalid_citation) | yes | – | 10,429 | no (no) |
| Q11 | multi | no | yes | 0.0 (0.25) | 13,809 | no (no) |
| Q12 | ambiguous | no | yes | 0.0 (0.0) | 6,624 | no (no) |
| Q13 | ambiguous | no | yes | 1.0 (1.0) | 3,740 | no (no) |
| Q14 | unanswerable | yes (below_gate) | – | – | 65 | yes (yes) |
| Q15 | unanswerable | yes (below_gate) | – | – | 18 | yes (yes) |

## Why faithfulness fell from 0.632 to 0.381

Faithfulness is claim-level: 12/19 claims judged supported before, 8/21 now. What the records show:

- **Same answer text, different verdict.** For Q2, Q3, Q7, Q8, Q9 and Q11 the recorded answer text is identical in both runs. Q3 went 1.0 → 0.0, Q9 1.0 → 0.25 and Q11 0.25 → 0.0 on identical text; Q2, Q7 and Q8 are unchanged. Q1 is the same sentence apart from the position of a full stop and also went 1.0 → 0.0. So a large part of the drop is the judge scoring the same text differently.
- **Three things changed and cannot be separated from these records:** (1) the judge prompt was changed (nonce-hardened), (2) the judge is the same 8B model as the generator, (3) run-to-run nondeterminism at temperature 0: six of the 13 answered questions have different answer text between the runs (Q1, Q4, Q5, Q6, Q12, Q13), including Q4 and Q6 going 0.0 → 1.0 and Q12 gaining labelled readings. There is one run per judge prompt, so no attribution to a single cause is claimed.
- **The new verdicts contradict the corpus in several places** (below), so 0.381 understates answer quality there. It also should not be read as "the earlier 0.632 was the true number": Q6 and Q9 contain wording problems that the judge passed or partly passed.

## Failure analysis (per question, from the recorded answers and judge verdicts)

- **Retrieval misses (hit@4):** none; every answerable question had an expected doc in the top-4. Multi-doc *coverage* within the top-4 was incomplete for Q9, Q10 and Q13. Not tuned.
- **Gate refusals:** none on answerable questions (lowest answerable confidence 0.663, Q13). Q14 (0.548) and Q15 (0.530) were correctly refused without calling the LLM.
- **Validator refusals:** none this run (the earlier run's Q10 `invalid_citation` did not recur).
- **Q1** — "A-class items are counted twelve times a year [WH-002 §2.2]" matches the corpus sentence in WH-002 §2.2. The judge said NO: judge false negative (the earlier run judged the same sentence YES).
- **Q3** — "valid for 30 days from the date of issue [WH-004 §2.2]" is the corpus sentence word for word. Judge said NO: false negative; the same answer scored 1.0 earlier.
- **Q4** — correct ($2M per occurrence, PR-002 §2.2), judged YES. The earlier 0.0 was on an answer that carried an extra citation, `[PR-002 §1.2]`, so the two answers differ.
- **Q2, Q5, Q8** — correct and judged YES in both runs.
- **Q6** — judged YES and passed, but the wording is garbled: "up to $50, whichever is lower, or within ±2% of the PO price" (expected: within 2% or $50, whichever is lower). Read this pass as lenient: a judge false positive on wording.
- **Q7** — two labelled readings. Reading 1 (exactly $3,000 → Inventory Control Manager, WH-005 §2.1) matches the WH-005 band table ($500 to $5,000); the judge said NO, a false negative. Reading 2 (more than $5,000 → Director of Operations plus Finance, §2.2) is supported but irrelevant to a $3,000 question. The earlier report called reading 1 "the wrong band"; that reading of the identical answer is withdrawn. The question fails on faithfulness 0.5.
- **Q9** — faithfulness 0.25 (1/4); the answer is four conditional "If you mean …" paragraphs that do not state the steps. Retrieved chunks were WH-003 §1.2, WH-003 §1.1, WH-001 §4.2 and PR-002 §3.2: none of the expected sections (WH-003 §2.x, QA-001), so coverage is false. It cites WH-001 §4.2, a valid chunk but not an expected doc; this is the one citation-accuracy miss. The verdicts are inconsistent with the corpus: readings matching the WH-001 §4.2 rows for carton damage and wet packaging were judged NO, while "hold the delivery and notify the buyer" (the wrong-label row, not a damage case) was judged YES. Missing from the answer: the 24-hour Procurement notice and the NCR within 1 business day.
- **Q10** — `model_refused`: the model itself produced the refusal text after the gate passed it (confidence 0.744). All four retrieved chunks were WH-002 (§4.2, §4.1, §3.1, §4.3); none from WH-005, which holds the approver bands. The records do not retain model reasoning, so no cause beyond that is claimed. The earlier run gave `invalid_citation` (a validator refusal). This is the single false refusal.
- **Q11** — faithfulness 0.0 (0/4). Claim 1 (Director of Operations for $25,000 to $100,000, PR-001 §2.1) and claims 3 and 4 (invoice held, buyer resolves, PO change approved under PR-001; FIN-001 §3.1, PR-001 §5.3) match the corpus text but were judged NO: false negatives. Claim 2 is a genuine error: for $31,500 it says the CFO approves as "above $100,000", but $31,500 is in the $25,000 to $100,000 band. The answer also never says 5% exceeds the price tolerance (2% or $50, whichever is lower), an expected point. Both expected docs were retrieved and cited.
- **Q12** — two labelled readings, but the purchase-order reading says the limit "is not explicitly stated" although PR-001 holds the PO bands (the retrieved PR-001 chunks were §5.1 and §1.1, not §2.1); the inventory reading gives "$5,000", a band boundary rather than the full answer (WH-005 §2.2). The judge said NO to all 3 claims; the first is an absence statement that a chunk-based judge cannot support. Here 0.0 is not clearly a judge error.
- **Q13** — only the QA-001 reading (1 business day); no labelled readings; WH-003 (24 hours to Procurement) was not in the top-4 (QA-001 §2.1, PR-001 §4.2, WH-004 §2.2, QA-001 §1.2). Faithful (1.0) but incomplete.
- **Latency:** p95 is set by the multi-doc questions (Q11 13.8 s, Q9 10.4 s, Q10 10.4 s). The LLM step is nearly all of each answered question's time (Q11: 13,720 of 13,809 ms). The earlier Q1 included the first model load and this run's did not, so 15,877 ms vs 13,808.5 ms is not a like-for-like improvement claim.
- **Judge limitation:** the judge is the same small local model as the generator. This run shows false negatives on correct claims (Q1, Q3, Q7 reading 1, Q11 claims 1, 3, 4) and a lenient pass on a garbled answer (Q6). Each claim is checked only against the chunks it cites (or all used chunks if it cites none). Faithfulness is a smoke signal, not a precise measure; the FAIL stands as the recorded result.

## Gate threshold calibration (held-out, not the 15)
16 answerable and 10 unanswerable in-domain questions (`eval/calibration.yaml`), retrieval only. Unchanged by this run.
- answerable: min 0.631 · p25 0.716 · median 0.767 · p75 0.826 · max 0.855
- unanswerable: min 0.501 · p25 0.526 · median 0.577 · p75 0.605 · max 0.664
- Chosen threshold **0.61** (balanced accuracy 0.950; TPR 1.000; TNR 0.900); ties 0.61–0.63 broken toward the lower value. The margin is thin: U3 (EDI formats, 0.664) cannot be separated by the gate; lowest answerable is C10 (0.631).

## Known validator false-refusal classes
- Any 2–4 letter + hyphen + digits token that is not a validly cited uppercase doc ID (PO-2291, RCV-5104, COVID-19, lowercase wh-003, Tier-2).
- Citation forms outside the grammar: ranges, parentheses, "Section n", a cited child of a retrieved section.
- Characters outside the allowlist (arrows, comparison signs, other non-ASCII symbols).
- Square/angle brackets or braces used for anything other than a canonical citation.
- A cited section not among the retrieved chunks.
The validator was not loosened for this eval. It refused no answer in this run; the earlier run's one validator refusal (Q10) became a model refusal.
