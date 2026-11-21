# Ops Copilot — Phase 1 eval report

Two official runs of the same 15-question eval are reported. **Iteration 1 (baseline)** is the local `llama3.1:8b` run. **Iteration 2** is the Groq `openai/gpt-oss-120b` run, which is the current result (`eval/latest.json`, the Console Evals tab, the README and the notebook all show it). Nothing was tuned or loosened for either run.

| | Iteration 1 (baseline) | Iteration 2 |
|---|---|---|
| Run | 2026-11-20T09:57:08Z | 2026-11-20T22:57:27Z |
| Code commit | `6e2718f` | `00d4363` |
| Generator | `llama3.1:8b` (Ollama, local, temperature 0) | `openai/gpt-oss-120b` (Groq, temperature 0) |
| Judge | `llama3.1:8b` (same model) | `openai/gpt-oss-120b` (same model) |
| Embeddings / retrieval | `nomic-embed-text`, hybrid BM25 + dense, RRF k=60, top-4 | same |
| Gate threshold / judge prompt | 0.61 (calibrated on a held-out set) / nonce-hardened | same / same |
| Pacing | none | `EVAL_PACE_S=10` |
| Raw data | `eval/results/20261120T095708Z.json` | `eval/results/20261120T225727Z.json` |

Code differences between the two commits are eval-harness changes only (F3/F3.1/F4: `.env` loading, `EVAL_JUDGE_MODEL` and the judge-size guard, rate-limit backoff and `EVAL_PACE_S`). No change to prompts, the 0.61 threshold, the questions, retrieval or scoring. `rate_limit_retries` and `judge_rate_limit_retries` are 0 for every Iteration 2 question.

The eval now refuses an 8B judge (and any judge smaller than the generator), so the Iteration 1 configuration cannot be re-run as it stands.

---

## Before / after

### The five KPIs

| KPI | Bar | Iteration 1 (baseline) | Iteration 2 |
|---|---|---|---|
| Faithfulness | ≥ 0.90 | 0.381 (8/21 claims), **FAIL** | 0.867 (13/15 claims), **FAIL** |
| hit@4 | ≥ 0.85 | 1.000, PASS | 1.000, PASS |
| MRR | ≥ 0.70 | 1.000, PASS | 1.000, PASS |
| Refusal correct | 2/2 | 2/2, PASS | 2/2, PASS |
| p95 latency | < 8,000 ms | 13,808.5 ms, **FAIL** | 1,790.5 ms, PASS |

Iteration 1 passed 3 of 5 KPIs; Iteration 2 passes 4 of 5. Faithfulness still fails: 0.867 vs 0.90.

p95 uses the nearest-rank method (rank ceil(0.95 × 15) = 15, the maximum): Iteration 1 Q11 = 13,808.5 ms; Iteration 2 Q11 = 1,790.5 ms (a refused answer). hit@4/MRR are doc-level (an expected document appears in the top-4 / its rank), not section-level.

### Secondary metrics

| Metric | Iteration 1 (baseline) | Iteration 2 |
|---|---|---|
| Citation accuracy | 0.917 (11/12 answered) | 0.909 (10/11 answered) |
| False refusals | 1 (Q10, `model_refused`) | 2 (Q10 `invalid_citation`, Q11 `no_valid_citations`) |
| Answer rate | 0.923 (12/13 answerable) | 0.846 (11/13 answerable) |
| Questions passed | 7/15 (Q2, Q4, Q5, Q6, Q8, Q14, Q15) | 11/15 (Q1–Q8, Q12, Q14, Q15) |

Faithfulness rose while answer rate and citation accuracy slipped. The two false refusals (Q10, Q11) took two multi-document questions out of the faithfulness sample, so 0.867 is measured on 15 claims from 11 answered questions and is not directly comparable to the 21 claims of Iteration 1.

### What the change can and cannot be attributed to

Model, judge model, hardware (local vs remote API) and run-to-run nondeterminism all changed together; there is one run each. The records support "the Iteration 2 configuration scores higher on faithfulness and is faster", not "the pipeline improved". The retrieval stack, gate, prompts and questions are unchanged: hit@4, MRR and the per-question retrieval confidences are identical in the two runs.

### Per-question changes

Latency in ms; faith = per-question faithfulness; `-` = nothing was judged (refused).

| Q | Type | Iteration 1: passed / faith / ms / refusal | Iteration 2: passed / faith / ms / refusal | Change |
|---|---|---|---|---|
| Q1 | direct | no / 0.0 / 3,362.2 | yes / 1.0 / 1,599.2 | fail → pass |
| Q2 | direct | yes / 1.0 / 3,649.6 | yes / 1.0 / 854.5 | unchanged |
| Q3 | direct | no / 0.0 / 3,797.6 | yes / 1.0 / 641.4 | fail → pass |
| Q4 | direct | yes / 1.0 / 3,922.5 | yes / 1.0 / 971.3 | unchanged (now also cites PR-002 §2.1) |
| Q5 | direct | yes / 1.0 / 4,211.0 | yes / 1.0 / 784.7 | unchanged |
| Q6 | direct | yes / 1.0 / 4,426.7 | yes / 1.0 / 1,005.0 | unchanged (wording now matches FIN-001 §2.2) |
| Q7 | direct | no / 0.5 / 6,545.2 | yes / 1.0 / 1,121.6 | fail → pass (single reading, WH-005 §2.1) |
| Q8 | direct | yes / 1.0 / 4,418.1 | yes / 1.0 / 1,114.1 | unchanged (now also cites CS-001 §4.3) |
| Q9 | multi | no / 0.25 / 10,423.2 | no / 0.5 / 1,774.3 | still fail; faith 0.25 → 0.5 |
| Q10 | multi | no / - / 10,428.5 / `model_refused` | no / - / 1,667.1 / `invalid_citation` | still a false refusal; reason changed |
| Q11 | multi | no / 0.0 / 13,808.5 | no / - / 1,790.5 / `no_valid_citations` | still fail; unfaithful answer → false refusal |
| Q12 | ambiguous | no / 0.0 / 6,624.3 | yes / 1.0 / 1,485.2 | fail → pass |
| Q13 | ambiguous | no / 1.0 / 3,739.5 | no / 1.0 / 1,078.6 | unchanged (still fail) |
| Q14 | unanswerable | yes / - / 65.0 / `below_gate` | yes / - / 54.8 / `below_gate` | unchanged |
| Q15 | unanswerable | yes / - / 18.2 / `below_gate` | yes / - / 60.9 / `below_gate` | unchanged |

Four questions went fail → pass (Q1, Q3, Q7, Q12) and none went pass → fail. Seven were passes in both runs (Q2, Q4, Q5, Q6, Q8, Q14, Q15) and four failed in both (Q9, Q10, Q11, Q13). That gives 7 + 4 = 11 passed.

---

## Iteration 2: failure analysis

From the records in `eval/results/20261120T225727Z.json` (answers, judge verdicts, notes) and the corpus text in `data/docs/`.

- **Retrieval:** no hit@4 misses. Multi-doc coverage in the top-4 is incomplete for Q9, Q10 and Q13, as in Iteration 1.
- **Gate refusals:** none on answerable questions (lowest answerable confidence 0.663, Q13). Q14 (0.548) and Q15 (0.530) were correctly refused without calling the LLM.
- **Validator refusals:** two, Q10 (`invalid_citation`) and Q11 (`no_valid_citations`). Both are false refusals.
- **Q1–Q8, Q12:** answered, faithfulness 1.0, passed. Q12's purchase-order reading says the monetary thresholds "are not detailed in the excerpts provided", and the judge accepted that absence claim as supported by PR-001 §1.1. The retrieved PR-001 chunks were §5.1 and §1.1, not §2.1 where the PO bands are, so this pass is on an incomplete answer.

### Q9: faithfulness 0.5 (2 of 4 claims), not passed

Answer: record the damage on the bill of lading; photograph the damaged units; follow the OS&D policy; "these actions are required by the inspection step for outer-carton damage and by the OS&D policy's objective". Cited WH-001 §4.2 and WH-003 §1.1; the judge was given `WH-003#1.1#0` and `WH-001#4.2#0`.

The judge rejected two claims. Checked against the chunk text:

- **Claim 2, "Take a photograph of the damaged units or packaging as evidence": judge error (false negative).** WH-001 §4.2, a cited chunk, has the row "Outer carton damage | Note on bill of lading, photograph, see WH-003". The claim is supported by the cited text.
- **Claim 4, "These actions are required by the inspection step for outer-carton damage and by the OS&D policy's objective": not shown to be a judge error.** The outer-carton row exists in WH-001 §4.2, and WH-003 §1.1 is titled "Objective" (preserve the right to recover cost from the carrier or supplier). But the claim is a vague summary, and "required by ... the policy's objective" is not something WH-003 §1.1 says. The NO is defensible.

The question fails for reasons independent of the judge. The answer omits the Procurement notice within 24 hours (WH-003 §2.3), quarantine in the OS&D lane, and the NCR within 1 business day (QA-001 §2.1): three of the four expected points. None of WH-003 §2.x or QA-001 was in the top-4 (retrieved: WH-003 §1.2, WH-003 §1.1, WH-001 §4.2, PR-002 §3.2). It also cites WH-001, which is not an expected doc; that is the one citation-accuracy miss. Notes: "citation not within expected docs; unfaithful claim(s); not all expected docs cited; not all expected docs in top-4".

### Q10: false refusal, `invalid_citation`

The gate passed (confidence 0.744), the model produced an answer, and the citation validator refused it. The record keeps only the standard refusal text, with no citations and no model output. Retrieved chunks were all WH-002 (§4.2, §4.1, §3.1, §4.3), none from WH-005, which holds the approver bands. So even an accepted answer could only cover the recount rules, not who approves $7,000. The official run's raw output was not recorded; see the post-run diagnostic below.

### Q11: false refusal, `no_valid_citations`

The gate passed (confidence 0.750) and both expected docs were retrieved (PR-001 §2.1, PR-001 §4.1, FIN-001 §3.1, PR-001 §5.3; coverage true). The validator found no valid citation in the model output. The official run's raw output was not recorded, so the exact cause is unknown. A single post-run re-ask was answered, so this refusal is nondeterministic (diagnostic below). In Iteration 1 the same question was answered but unfaithfully (see Iteration 1).

### Q13: passed = False despite faithfulness 1.0

Answer: "You must raise the non-conformance report within one business day of discovering the problem. [QA-001 §2.1]". The claim is supported by QA-001 §2.1 ("within 1 business day"). The notes field reads "not all expected docs cited; no labelled readings; not all expected docs in top-4". The question ("How long do I have to report a problem?") is ambiguous, with two expected readings: Procurement within 24 hours of receipt for OS&D exceptions (WH-003 §2.3), and an NCR within 1 business day (QA-001 §2.1). The answer gives only the QA-001 reading, with no "If you mean ..." labels, and WH-003 was not in the top-4 (retrieved: QA-001 §2.1, PR-001 §4.2, WH-004 §2.2, QA-001 §1.2). So the failure is retrieval coverage plus the missing labelled readings plus the second expected doc not cited, not only missing labels. It is faithful but incomplete, as in Iteration 1.

### Post-run diagnostic (NOT part of the official run)

After the official run, the orchestrator re-asked Q10 and Q11 once each through the same pipeline and model and captured the raw model output. These are single ad hoc observations. They are not in `eval/results/` and change no figure above.

- Groq returns gpt-oss reasoning in a separate `reasoning_content` field (`additional_kwargs`), not in the message content, so reasoning text did not reach the citation parser.
- **Q10 re-ask:** refused again with `invalid_citation`. The raw content cited [WH-002 §4.1], [WH-002 §4.2] and [WH-002 §4.3], and also contained a bare doc mention "WH‑005" written with a U+2011 non-breaking hyphen and no §. That bare mention is the likely trigger of the residual-citation check (an inference, not verified).
- **Q11 re-ask:** the answer was accepted, not refused. So the official run's Q11 refusal is nondeterministic. The official run's raw output was not recorded, so its exact cause is unknown. One possibility is that reasoning tokens consumed the output token budget; this is unverified.

No parser or validator change was made or proposed. Possible next steps, all owner decisions (the owner chose option 1; options 2–5 are in the Phase 2 backlog under Next steps):

1. Keep the validator as is and accept the two false refusals as the recorded result.
2. Record the raw model output (and `reasoning_content` and finish reason) for refused answers in future runs, so causes are known rather than inferred.
3. Decide whether the validator should treat a bare doc ID with no section differently. Per the governance policy, measure how many existing records such a rule would change before adding it.
4. Review the output token budget for reasoning models.
5. Repeat the run to measure spread.

### Caveats

- **Self-judging.** The judge is the same model as the generator (`openai/gpt-oss-120b`), so it shares the generator's blind spots. Faithfulness is a smoke signal. The one verified judge error in Iteration 2 is Q9 claim 2.
- **Hidden retries.** The Groq SDK's own `max_retries=1` can retry a 429 inside a timed call, invisibly to the harness. All Iteration 2 latencies are under 1.8 s, so there is no evidence of hidden retries.
- **Latency is not like-for-like.** Iteration 2 latency is dominated by the network and Groq, not local hardware. The p95 change from 13.8 s to 1.8 s reflects a move from a local 8B model to a hosted 120B model, not a pipeline improvement.
- **Data left the machine.** The corpus and questions were sent to Groq with the owner's approval (`EVAL_ALLOW_REMOTE=1`).
- **Nondeterminism.** Temperature is 0, but outputs are still nondeterministic (the Q11 re-ask). One run per iteration; no spread is available.
- **Model choice.** The owner chose `openai/gpt-oss-120b`; `llama-3.3-70b-versatile` was not available on the owner's key.

## Next steps

**Decision (owner):** Iteration 2 is accepted as the recorded Phase 1 result. Faithfulness (0.867) stays below the 0.90 bar and is recorded as a FAIL. No further changes are made to Phase 1.

### Phase 2 backlog

From the Iteration 2 analysis:

1. **Record raw output for refused answers.** Store the raw model output, `reasoning_content` and finish reason for refused answers in future eval runs, so causes such as the Q11 `no_valid_citations` refusal are known rather than inferred.
2. **Bare doc IDs in answers.** Decide whether the citation validator should treat a bare doc ID with no section (the Q10 "WH‑005" case) differently. Per the governance policy, first measure how many existing records such a rule would change.
3. **Output token budget for reasoning models.** Review whether reasoning tokens can exhaust the output budget and leave the answer empty or cut off (unverified for Q11).
4. **Repeat runs.** Run the eval several times per configuration to measure spread, since there is one run per iteration and outputs are nondeterministic.

Low-priority follow-ups:

5. **`eval/calibrate.py` does not load `.env`.** Unlike `eval/run_eval.py`, it reads only the shell environment.
6. **`EMBED_MODEL` is printed without `repr`.** In `eval/run_eval.py` and `eval/calibrate.py`, a value containing a newline could fake console lines. It cannot expose the API key.
7. **Notebook table alignment.** In the notebook's eval output, the Q10 and Q11 rows ("yes (invalid_citation)", "yes (no_valid_citations)") overflow the refused column. Cosmetic only.

---

## Iteration 1 (baseline)

**Run:** 2026-11-20T09:57:08Z · code commit `6e2718f` · generator and judge `llama3.1:8b` (Ollama, temperature 0) · embeddings `nomic-embed-text` · hybrid retrieval (BM25 + dense, RRF k=60, top-4) · gate threshold 0.61 (calibrated on a held-out set) · judge prompt: nonce-hardened (per-call random-token tags around context and claim).
Raw data: `eval/results/20261120T095708Z.json`. Single run; local Ollama at temperature 0 is not guaranteed deterministic. Nothing was tuned for this run.

### KPIs vs pass bars

| KPI | Result | Bar | Status |
|---|---|---|---|
| Faithfulness | 0.381 (8/21 claims) | ≥ 0.90 | **FAIL** |
| hit@4 | 1.000 | ≥ 0.85 | PASS |
| MRR | 1.000 | ≥ 0.70 | PASS |
| Refusal correct | 2/2 | 2/2 | PASS |
| p95 latency | 13,808.5 ms | < 8,000 ms | **FAIL** |

Notes: p95 uses the nearest-rank method (rank ceil(0.95 × 15) = 15, i.e. the maximum, Q11 = 13,808.5 ms). Linear interpolation over the same 15 recorded latencies gives 11,442.5 ms, which also fails the 8,000 ms bar.

Supporting: citation accuracy 0.917 (11/12 answered), answer rate 0.923 (12/13 answerable), false refusals 1 (Q10, `model_refused`), questions passed 7/15 (Q2, Q4, Q5, Q6, Q8, Q14, Q15).

### Definition of `passed` (both iterations)
- Unanswerable: refused.
- Answerable: answered, hit@4, valid citations ⊆ expected docs with ≥1 expected doc cited, and per-question faithfulness = 1.0.
- Multi-doc and ambiguous additionally require every expected doc among the valid citations; ambiguous additionally requires labelled "If you mean …" readings.

### Per-question results (Iteration 1)

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

### Why faithfulness fell from 0.632 to 0.381 (within Iteration 1)

Faithfulness is claim-level: 12/19 claims judged supported before, 8/21 now. What the records show:

- **Same answer text, different verdict.** For Q2, Q3, Q7, Q8, Q9 and Q11 the recorded answer text is identical in both runs. Q3 went 1.0 → 0.0, Q9 1.0 → 0.25 and Q11 0.25 → 0.0 on identical text; Q2, Q7 and Q8 are unchanged. Q1 is the same sentence apart from the position of a full stop and also went 1.0 → 0.0. So a large part of the drop is the judge scoring the same text differently.
- **Three things changed and cannot be separated from these records:** (1) the judge prompt was changed (nonce-hardened), (2) the judge is the same 8B model as the generator, (3) run-to-run nondeterminism at temperature 0: six of the 13 answered questions have different answer text between the runs (Q1, Q4, Q5, Q6, Q12, Q13), including Q4 and Q6 going 0.0 → 1.0 and Q12 gaining labelled readings. There is one run per judge prompt, so no attribution to a single cause is claimed.
- **The new verdicts contradict the corpus in several places** (below), so 0.381 understates answer quality there. It also should not be read as "the earlier 0.632 was the true number": Q6 and Q9 contain wording problems that the judge passed or partly passed.

### Failure analysis (Iteration 1, per question, from the recorded answers and judge verdicts)

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
- **Q11** — faithfulness 0.0 (0/4). Claim 1 (Director of Operations for $25,000 to $100,000, PR-001 §2.1) and claim 3 (invoice held, buyer resolves; FIN-001 §3.1) match the corpus text but were judged NO: false negatives. Claim 4 ("Any change to the PO is approved under PR-001 [PR-001 §5.3]") is a mis-citation, not a judge false negative: its content comes from FIN-001 §3.1, while PR-001 §5.3 is "Segregation of duties" and says nothing about approving PO changes, so the judge, which checks a claim only against the chunk it cites (PR-001#5.3#0), was right to reject it. Claim 2 is a genuine error: for $31,500 it says the CFO approves as "above $100,000", but $31,500 is in the $25,000 to $100,000 band. The answer also never says 5% exceeds the price tolerance (2% or $50, whichever is lower), an expected point. Both expected docs were retrieved and cited.
- **Q12** — two labelled readings, but the purchase-order reading says the limit "is not explicitly stated" although PR-001 holds the PO bands (the retrieved PR-001 chunks were §5.1 and §1.1, not §2.1); the inventory reading gives "$5,000", a band boundary rather than the full answer (WH-005 §2.2). The judge said NO to all 3 claims; the first is an absence statement that a chunk-based judge cannot support. Here 0.0 is not clearly a judge error.
- **Q13** — only the QA-001 reading (1 business day); no labelled readings; WH-003 (24 hours to Procurement) was not in the top-4 (QA-001 §2.1, PR-001 §4.2, WH-004 §2.2, QA-001 §1.2). Faithful (1.0) but incomplete.
- **Latency:** p95 is set by the multi-doc questions (Q11 13.8 s, Q9 10.4 s, Q10 10.4 s). The LLM step is nearly all of each answered question's time (Q11: 13,720 of 13,809 ms). The earlier Q1 included the first model load and this run's did not, so 15,877 ms vs 13,808.5 ms is not a like-for-like improvement claim.
- **Judge limitation:** the judge is the same small local model as the generator. This run shows false negatives on correct claims (Q1, Q3, Q7 reading 1, Q11 claims 1 and 3) and a lenient pass on a garbled answer (Q6). Each claim is checked only against the chunks it cites (or all used chunks if it cites none). Faithfulness is a smoke signal, not a precise measure; the FAIL stands as the recorded result.

---

## Gate threshold calibration (held-out, not the 15; shared by both iterations)
16 answerable and 10 unanswerable in-domain questions (`eval/calibration.yaml`), retrieval only. Unchanged by either run.
- answerable: min 0.631 · p25 0.716 · median 0.767 · p75 0.826 · max 0.855
- unanswerable: min 0.501 · p25 0.526 · median 0.577 · p75 0.605 · max 0.664
- Chosen threshold **0.61** (balanced accuracy 0.950; TPR 1.000; TNR 0.900); ties 0.61–0.63 broken toward the lower value. The margin is thin: U3 (EDI formats, 0.664) cannot be separated by the gate; lowest answerable is C10 (0.631).

## Known validator false-refusal classes
- Any 2–4 letter + hyphen + digits token that is not a validly cited uppercase doc ID (PO-2291, RCV-5104, COVID-19, lowercase wh-003, Tier-2).
- Citation forms outside the grammar: ranges, parentheses, "Section n", a cited child of a retrieved section.
- Characters outside the allowlist (arrows, comparison signs, other non-ASCII symbols).
- Square/angle brackets or braces used for anything other than a canonical citation.
- A cited section not among the retrieved chunks.
The validator was not loosened for either run. It refused no answer in Iteration 1 (the earlier run's one validator refusal, Q10, became a model refusal); it refused Q10 and Q11 in Iteration 2 (see above).
