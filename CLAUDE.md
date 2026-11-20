# Ops Copilot

Ops Copilot: AI assistant for Meridian Supply Co. (fictional). Phase 1: RAG over simulated warehouse and procurement docs. Stack: Python 3.12 + uv, LangChain, Chroma + BM25 hybrid retrieval, Ollama (nomic-embed-text; LLM via LLM_PROVIDER=ollama|groq), FastAPI + static web frontend.

## Standing rules
1. A failed gate is never weakened by whatever is trying to pass it. Fix the cause or stop.
2. Every claim of state ("tests pass", "zero occurrences", "deployed") is accompanied by the command and its output. A measurement whose positive control is empty is not evidence.
3. Only the owner pushes to the deploy branch and only CI deploys. No agent uses the owner's credentials.
4. No secrets in chat, commits, logs or documentation.
5. The orchestrator never edits application source. The implementer works in a worktree. No agent reviews its own work.
6. Requirements come from constitution/BASELINE.md. A gap is a question to the owner, never a guess.

## How work flows
BASELINE.md → BUILD_GRAPH.md (architect) → unit dispatched to implementer → gates → regression-guardian (+ security-review for auth/PII/infra) → done → owner pushes → CI deploys.

## Governance policy
New rules are added only in response to an incident, and only after measuring how many existing items the rule would change. Record the incident next to the rule.
