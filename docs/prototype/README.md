# Phase 1 prototype (Week 2 RAG): Meridian Ops Portal, the build target

Open `index.html` in a browser. Screenshots: dashboard.png, copilot.png, console.png, trace.png.

## What it is
Meridian Supply Co.'s (fictional) **operations management portal**, with **Ops Copilot** built in as a side panel.
Dashboard figures are simulated. Copilot answers only from the 10 SOP and policy documents.

## Screens
1. **Dashboard:** five KPI tiles (open POs, receipts today, open discrepancies, supplier quality, key-account fill rate),
   inbound-receipt exceptions, supplier quality bars, and POs awaiting approval.
   Every exception row has a **✦ Ask** button that opens Copilot with a question about that row.
2. **Receiving / Purchase orders / Suppliers:** simple list views of the same simulated data.
3. **SOPs & policies:** 10 document cards (codes RCV, CC, OSD, RMA, POA, SUP, NCR, 3WM, ADJ, SLA).
   Clicking a citation chip in Copilot jumps to its document card and highlights it.
4. **Ops Copilot panel (right):** chat with citation chips, the exact refusal text
   "I couldn't find this in Meridian's documents.", a confidence pill, a "View in Console" link,
   sample chips, and a scope note ("answers only; actions come in Phase 2").
5. **Console:** three tabs.
   - *Retrieval inspector:* BM25 / dense / hybrid toggle, the confidence-gate banner, and a chunk table (BM25 rank, dense rank, RRF score, used).
   - *Run trace:* time per step (embed, BM25, dense, RRF, gate, LLM). The LLM step shows 0 ms when the question is refused.
   - *Evals:* 5 KPIs (faithfulness ≥0.90, hit@4 ≥0.85, MRR ≥0.70, refusal 2/2, p95 <8 s) and the 15-question table.

The architecture diagram is NOT in the app; it goes in the separate presentation.

## Pipeline the code must follow
docs → clean → chunk (by heading, ~400 tokens) → embed (nomic-embed-text → Chroma) + BM25 →
hybrid retrieve (RRF k=60, top k=4) → confidence gate (fail → refuse WITHOUT calling the LLM) →
grounded prompt (context only, cite [DOC §section]) → answer + citations. Eval: 15 questions.

Numbers, document text and the 0.55 threshold are illustrative; the real values come from the corpus and the eval.
