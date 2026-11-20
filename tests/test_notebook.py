"""The demo notebook ships executed, clean and free of local paths. Offline: only reads the file."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parent.parent / "notebooks" / "01_rag_walkthrough.ipynb"
REQUIRED = [
    "cleaning",
    "chunking",
    "embeddings",
    "bm25 vs dense vs hybrid",
    "grounded answer",
    "refusal",
    "eval summary",
]


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _output_text(cell: dict) -> str:
    parts: list[str] = []
    for o in cell.get("outputs", []):
        if "text" in o:
            parts.append("".join(o["text"]))
        for v in o.get("data", {}).values():
            parts.append("".join(v) if isinstance(v, list) else str(v))
        parts.append(o.get("ename", "") + o.get("evalue", "") + "".join(o.get("traceback", [])))
    return "\n".join(parts)


def test_valid_notebook(nb: dict) -> None:
    assert nb["nbformat"] == 4
    assert nb["cells"]


def test_required_sections(nb: dict) -> None:
    headings = " | ".join(
        ln.lower()
        for c in nb["cells"]
        if c["cell_type"] == "markdown"
        for ln in "".join(c["source"]).splitlines()
        if ln.startswith("#")
    )
    for word in REQUIRED:
        assert word in headings, f"missing section: {word}"


def test_all_code_cells_executed(nb: dict) -> None:
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    assert all(c["execution_count"] is not None for c in code)
    assert all(c["outputs"] for c in code[1:]), "every cell after setup should print something"


def test_no_errors_in_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        assert all(o.get("output_type") != "error" for o in c["outputs"])
        assert "Traceback" not in _output_text(c)


def test_no_local_paths_or_emails(nb: dict) -> None:
    text = "\n".join(_output_text(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "/Users/" not in text and "/home/" not in text and "/private/" not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)


def test_does_not_load_env_file(nb: dict) -> None:
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "load_env_file" not in src and "load_dotenv" not in src
    assert "LLM_PROVIDER" in src and "OLLAMA_MODEL" in src


def test_shows_failing_kpis_and_refusal(nb: dict) -> None:
    text = "\n".join(_output_text(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "FAIL" in text and "PASS" in text
    assert "refused: True" in text


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_settings_are_not_printed(nb: dict) -> None:
    src = "\n".join("".join(c["source"]) for c in _code_cells(nb))
    assert "print(settings)" not in src
    text = "\n".join(_output_text(c) for c in _code_cells(nb))
    assert "Settings(" not in text and "ollama_base_url" not in text


def test_saved_refusal_cell_shows_llm_zero_ms(nb: dict) -> None:
    cells = [
        c for c in _code_cells(nb) if "forklift operators be recertified" in "".join(c["source"])
    ]
    assert len(cells) == 1
    out = _output_text(cells[0])
    assert "refused: True" in out and "reason: below_gate" in out
    assert re.search(r"'llm': 0\.0\b", out), out


def test_no_url_in_any_output(nb: dict) -> None:
    for c in _code_cells(nb):
        assert not re.search(r"https?://", _output_text(c)), "".join(c["source"])[:60]
