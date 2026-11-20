# Phase 1 prototype (Week 2 RAG): build target

Open `index.html` in a browser. It has three tabs:

1. **Ask:** the chat screen to build in Streamlit (`app.py`).
   - Left: the knowledge base (10 docs). Docs used in the last answer are highlighted.
   - Centre: chat. Answers carry citation chips (`OSD §2.1`). A refusal uses the exact text
     "I couldn't find this in Meridian's documents." and shows the score vs the threshold.
   - Right: the **retrieval inspector**. It has a toggle between Keyword (BM25), Meaning (dense) and Hybrid (RRF),
     a confidence-gate banner, and for each chunk: BM25 rank, dense rank, RRF score, and cited / not used.
2. **How it works:** the pipeline the code and the notebook must follow:
   docs → clean → chunk (by heading, ~400 tokens) → embed (nomic-embed-text → Chroma) + BM25 index →
   hybrid retrieve (RRF, k=60, top k=4) → **confidence gate** (fail → refuse without calling the LLM) →
   grounded prompt (context only, cite §) → answer + citations → 15-question eval.
3. **Eval:** the report layout. KPIs: faithfulness ≥0.90, hit@4 ≥0.85, MRR ≥0.70, refusal 2/2, p95 <8 s,
   plus the 15 questions (8 direct, 3 multi-doc, 2 ambiguous, 2 unanswerable).

The numbers, document text and threshold (0.55) in the prototype are illustrative. The real values come from the corpus
and from tuning on the eval set. Doc codes: RCV, CC, OSD, RMA, POA, SUP, NCR, 3WM, ADJ, SLA.
