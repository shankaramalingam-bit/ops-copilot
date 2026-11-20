.PHONY: run test gates eval notebook

run:
	uv run python -m ops_copilot

test:
	uv run pytest -q

gates:
	for g in scripts/gate_*.sh; do bash "$$g" || exit 1; done

eval:
	uv run python eval/run_eval.py

# Needs Ollama running locally; executes the notebook in place with outputs saved.
notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_rag_walkthrough.ipynb
