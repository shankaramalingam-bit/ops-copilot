"""Unit c.3: re-review fixes (M1, M2, L1-L4) and regression-review test gaps."""

from __future__ import annotations

import dataclasses
import pickle
import time

import pytest

from ops_copilot import config
from ops_copilot.answer import (
    REFUSAL,
    finalize,
    is_refusal,
    neutralise_question,
    normalise_text,
)
from ops_copilot.config import ConfigError, Settings
from ops_copilot.types import Chunk, Hit

from .conftest import FIXTURE_DOCS, CountingChat

Q = "What happens to damaged cartons?"


def hit(doc: str, section: str) -> Hit:
    c = Chunk(f"{doc}#{section}#0", f"{doc} T > {section} X\nbody", doc, "T", section, "X")
    return Hit(c, 1, 1, 0.9, 0.03)


HITS = [hit("WH-003", "2.1"), hit("WH-003", "2.2"), hit("PR-001", "2"), hit("FIN-001", "3.2")]
VALID = "Do X [WH-003 §2.1]."


# ---- M1 -------------------------------------------------------------------------------------
PROBES = [
    "[ZZ-999]",
    "[ZZ-999, §1]",
    "[ZZ-999 section 1]",
    "[Doc ZZ-999 §1]",
    "[See ZZ-999 §1]",
    "[ZZ-999:1]",
    "[ZZ-999 #1]",
    "(see ZZ-999 §1)",
    "ZZ-999 §1",
    "[zz-999 §1]",
    "[ZZZZ-999 §1]",
    "[ZZ-999 ¶1]",
    "［ZZ-999 §1］",
    "⟦ZZ-999 §1⟧",
    "ZZ‐999",  # U+2010 hyphen
    "[ＺＺ-999 §1]",  # fullwidth letters
    "[ZZ-999​ §1]",  # ZWSP inside bracket
    "[ZZ-999§1",  # unclosed
    "see ZZ-999",
    "[Z­Z-999 §1]",  # soft hyphen
]


@pytest.mark.parametrize("probe", PROBES)
def test_m1_probe_with_valid_citation_is_refused(probe):
    f = finalize(f"{VALID} {probe}", HITS)
    # c.4: "¶" and "⟦⟧" are outside the character allowlist: refused earlier, as invalid_characters
    expected = "invalid_characters" if any(c in probe for c in "¶⟦⟧") else "invalid_citation"
    assert f.refused and f.refusal_reason == expected, probe
    assert f.answer == REFUSAL and f.citations == [] and f.used == [False] * len(HITS)


def test_m1_probe_list_nonempty_and_not_vacuous():
    # positive control: the same valid text without a probe is accepted
    assert len(PROBES) >= 20
    assert not finalize(VALID, HITS).refused


@pytest.mark.parametrize(
    "raw",
    [
        "Any PO change is approved under PR-001 [PR-001 §2].",
        "Per WH-003: do X [WH-003 §2.1]. Also WH-003 says Y [WH-003 §2.2].",  # c.4: uppercase only
        "Do X ［WH-003 §2.1］.",  # fullwidth brackets normalise to canonical
        "Do X [WH-003 §2.1] and Y [PR-001 §2].  See PR-001 and WH-003.",
        "Do​ X [WH-003 §2.1].",  # ZWSP elsewhere is harmless
    ],
)
def test_m1_legit_answers_accepted_and_sanitised_text_returned(raw):
    f = finalize(raw, HITS)
    assert not f.refused and f.refusal_reason is None, raw
    # c.4: what was validated is what is returned (sanitised), not the raw text
    assert f.answer == normalise_text(raw).strip()
    assert f.citations and all(c["valid"] for c in f.citations)


def test_m1_bare_doc_of_uncited_doc_refused():
    f = finalize("Do X [WH-003 §2.1]. Also see PR-001.", HITS)  # PR-001 retrieved, not cited
    assert f.refused and f.refusal_reason == "invalid_citation"


def test_m1_section_word_in_non_citation_bracket_refused():
    assert finalize("Do X [WH-003 §2.1] [Section 4].", HITS).refusal_reason == "invalid_citation"
    # c.4: "¶" is outside the character allowlist
    assert finalize("Do X [WH-003 §2.1] (¶3).", HITS).refusal_reason == "invalid_characters"
    assert finalize("Do X [WH-003 §2.1] (section 3).", HITS).refusal_reason == "invalid_citation"


def test_m1_no_valid_citations_has_empty_citations():
    f = finalize("Nothing cited here.", HITS)
    assert f.refusal_reason == "no_valid_citations" and f.citations == []


def test_m1_no_valid_citations_via_pipeline_asserts_empty(make_pipeline):
    p, _ = make_pipeline(response="uncited answer")
    r = p.ask(Q)
    assert r["refusal_reason"] == "no_valid_citations" and r["citations"] == []
    assert r["chunks"] and all(not c["used"] for c in r["chunks"])


# ---- M2 -------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    ["[" + "§" * 32000, "[" * 32000, "[A" * 16000, "[" + "AB-1" * 8000, "[§" * 16000],
)
def test_m2_neutralise_is_linear(q):
    t = time.perf_counter()
    neutralise_question(q)
    assert time.perf_counter() - t < 0.2


def test_m2_neutralise_still_defangs():
    assert neutralise_question("see [WH-003 §2] now") == "see (WH-003 §2) now"
    assert neutralise_question("a [plain] b") == "a [plain] b"
    assert neutralise_question("<x> [ＷＨ-003 §2]") == "&lt;x&gt; (WH-003 §2)"


def test_m2_max_question_chars_constant():
    assert config.MAX_QUESTION_CHARS == 2000


def test_m2_over_length_refused_without_calls(tmp_path, embeddings):
    from ops_copilot.pipeline import Pipeline

    from .test_review_fixes import CountingEmb

    emb = CountingEmb(embeddings)
    chat = CountingChat(responses=["x"])
    s = Settings(gate_threshold=0.1, chroma_dir=str(tmp_path / "c"), docs_dir=str(FIXTURE_DOCS))
    p = Pipeline(s, emb, chat, FIXTURE_DOCS)  # type: ignore[arg-type]
    r = p.ask("a" * (config.MAX_QUESTION_CHARS + 1))
    assert r["refused"] and r["refusal_reason"] == "question_too_long" and r["answer"] == REFUSAL
    assert emb.n == 0 and chat.calls == 0 and r["citations"] == [] and r["chunks"] == []
    assert r["confidence"] == 0.0
    ok = p.ask("a" * config.MAX_QUESTION_CHARS)  # boundary is not "too long"
    assert ok["refusal_reason"] != "question_too_long"


# ---- L1 -------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "I couldnʼt find this in Meridianʼs documents.",
        "I couldn’t find this in Meridian’s documents.",
        "I could​n't find this in Meridian's documents.",
        "I couldn't fi­nd this in Meridian's documents.",
        "I couldn`t find this in Meridian`s documents.",
        "I couldn′t find this in Meridian′s documents.",
        "Ｉ couldn't find this in Meridian's documents.",  # fullwidth I
    ],
)
def test_l1_refusal_detected_after_normalisation(text):
    assert is_refusal(text)
    f = finalize(f"{text} [WH-003 §2.1]", HITS)
    assert f.refused and f.refusal_reason == "model_refused"


def test_l1_non_refusal_not_detected():
    assert not is_refusal("I found this in Meridian's documents.")


# ---- L2 -------------------------------------------------------------------------------------
def test_l2_gate_threshold_must_be_positive():
    with pytest.raises(ConfigError):
        Settings(gate_threshold=0.0)
    with pytest.raises(ConfigError):
        Settings(gate_threshold=1e-9)
    Settings(gate_threshold=0.05)  # c.4 L7: floor is 0.05
    Settings(gate_threshold=1.0)
    with pytest.raises(ConfigError):
        Settings(gate_threshold=1.0000001)


@pytest.mark.parametrize("q", ["​", " ​‍﻿ ", "­", "​\n\t"])
def test_l2_cf_only_question_is_empty(make_pipeline, q):
    p, chat = make_pipeline(response="x")
    r = p.ask(q)
    assert r["refusal_reason"] == "empty_question" and chat.calls == 0
    assert r["confidence"] == 0.0 and r["citations"] == [] and r["chunks"] == []


def test_l2_empty_question_confidence_zero(make_pipeline):
    p, _ = make_pipeline(response="x")
    assert p.ask("   ")["confidence"] == 0.0


# ---- L3 -------------------------------------------------------------------------------------
def test_l3_chroma_client_settings_pass_env_file_none(monkeypatch, tmp_path, embeddings):
    """Chroma's module-level Settings() may read ./.env at import time; we cannot stop that.
    What we control: the client Settings we construct passes _env_file=None."""
    import ops_copilot.index as index_mod
    from ops_copilot.chunk import chunk_docs
    from ops_copilot.ingest import load_docs

    seen: list[dict] = []
    real = index_mod.ChromaSettings

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(index_mod, "ChromaSettings", spy)
    # No .env file is created anywhere: this only checks the kwargs we pass to Chroma.
    idx = index_mod.Index(chunk_docs(load_docs(FIXTURE_DOCS)), embeddings, tmp_path / "chroma", "m")
    assert seen and seen[0].get("_env_file", "missing") is None
    assert idx._client.get_settings().allow_reset is False


# ---- L4 / config gaps -----------------------------------------------------------------------
def test_l4_settings_frozen():
    s = Settings(llm_provider="groq", groq_api_key="fake-test-key-3")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.groq_api_key = "plain"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.top_k = 3  # type: ignore[misc]


def test_l4_chat_model_has_timeout_and_retries_groq():
    from ops_copilot.llm import GROQ_MAX_RETRIES, GROQ_TIMEOUT_S, get_chat_model

    m = get_chat_model(Settings(llm_provider="groq", groq_api_key="fake-test-key-4"))
    assert m.request_timeout == GROQ_TIMEOUT_S == 30  # type: ignore[attr-defined]  # d.1: 60 -> 30
    assert m.max_retries == GROQ_MAX_RETRIES == 1  # type: ignore[attr-defined]


def test_l4_chat_model_has_timeout_ollama():
    from ops_copilot.llm import OLLAMA_TIMEOUT_S, get_chat_model

    m = get_chat_model(Settings())
    assert m.client_kwargs.get("timeout") == OLLAMA_TIMEOUT_S  # type: ignore[attr-defined]


def test_iii_groq_chat_model_receives_raw_key_offline():
    from ops_copilot.llm import get_chat_model

    key = "fake-test-key-5"
    m = get_chat_model(Settings(llm_provider="groq", groq_api_key=key))
    assert m.groq_api_key is not None and m.groq_api_key.get_secret_value() == key  # type: ignore[attr-defined]


def test_iv_top_k_boundary_with_large_candidates():
    assert Settings(top_k=20, candidates=100).top_k == 20
    with pytest.raises(ConfigError):
        Settings(top_k=21, candidates=100)


def test_iv_rrf_k_boundary():
    assert Settings(rrf_k=1).rrf_k == 1
    with pytest.raises(ConfigError):
        Settings(rrf_k=0)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_iv_blank_groq_key_is_none(blank):
    assert Settings(groq_api_key=blank).groq_api_key is None
    assert Settings.from_env({"GROQ_API_KEY": blank}).groq_api_key is None


def test_iv_settings_with_key_cannot_be_pickled():
    s = Settings(llm_provider="groq", groq_api_key="fake-test-key-6")
    with pytest.raises(TypeError):
        pickle.dumps(s)
