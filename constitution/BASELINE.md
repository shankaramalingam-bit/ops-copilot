# Functional baseline — Ops Copilot (version 0.1, 2026-11-20)

## Purpose
Ops Copilot helps Meridian Supply Co. (fictional) warehouse, procurement and quality staff answer SOP and policy questions from 10 internal documents, with cited answers, "not in our docs" refusals, and ≥90% faithfulness on a 15-question eval. Phase 1 = course Week 1–2 concepts only (chunking, embeddings, vector store, hybrid retrieval, grounded prompt with citations + refusal, RAG eval). No agents.

## Users and roles
Warehouse, procurement and quality staff. No auth in Phase 1.

## Core flows
1. Ask a question in the Ops Copilot panel of the "Meridian Ops Portal" and get a cited answer.
2. An out-of-corpus question gets exactly "I couldn't find this in Meridian's documents."
3. An ambiguous question gets an answer for each plausible reading, labelled ("If you mean …"), each with its own citations; no clarifying turn.
4. Clicking a citation chip jumps to the SOP card.
5. Console: retrieval inspector (BM25 / dense / hybrid + gate banner), run trace, evals.
6. Dashboard exception rows have "Ask", which pre-fills the Copilot.

## Corpus (10 documents; fictional names and numbers only)
WH-001 Receiving & Putaway; WH-002 Cycle Count; WH-003 OS&D; WH-004 Returns (RMA); PR-001 PO Approval Matrix; PR-002 Supplier Onboarding; QA-001 Supplier Quality/NCR; FIN-001 3-Way Match; WH-005 Inventory Adjustment Approvals; CS-001 Key-Account Service Levels.

## Explicitly out of scope
Agents/tools, auth, real company data, multi-turn clarification, pushing/deploying by agents.

## Non-negotiables (security, privacy, data ownership)
- No secrets in repo or logs.
- .env is never read or committed by agents.
- The refusal string is exact.
- Citations must refer to retrieved sections.
- Tests run offline.

## Acceptance: how the owner will know a flow works
- Faithfulness ≥ 0.90, hit@4 ≥ 0.85, MRR ≥ 0.70.
- Refusal 2/2 on unanswerable questions.
- p95 latency < 8 s.
- Gates green: tests, lint, typecheck, secrets.
