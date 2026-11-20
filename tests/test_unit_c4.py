"""Unit c.4: allowlist answer validation (sanitise, character allowlist, residual check)."""

from __future__ import annotations

import time

import pytest

from ops_copilot import llm
from ops_copilot.answer import REFUSAL, finalize, is_refusal, normalise_text
from ops_copilot.config import MAX_QUESTION_CHARS, ConfigError, Settings
from ops_copilot.types import Chunk, Hit

from .conftest import FIXTURE_DOCS, CountingChat


def hit(doc: str, section: str) -> Hit:
    c = Chunk(f"{doc}#{section}#0", f"{doc} T > {section} X\nbody", doc, "T", section, "X")
    return Hit(c, 1, 1, 0.9, 0.03)


HITS = [
    hit("WH-003", "2.1"),
    hit("WH-003", "2.2"),
    hit("WH-001", "5.1"),
    hit("WH-002", "4.2"),
    hit("WH-005", "2.2"),
    hit("PR-001", "2.1"),
    hit("PR-002", "2.1"),
    hit("FIN-001", "4.1"),
    hit("FIN-001", "2.2"),
]
VALID = "Do X [WH-003 §2.1]."

# ---- (a) forgery probes: each appended to a valid answer must be refused ---------------------
PROBES = [
    "[WH-003:9]",
    "[WH-003 #9]",
    "[WH-003 sec. 9]",
    "[WH-003 s. 9]",
    "[WH-003 para 9]",
    "[WH-003 clause 9]",
    "WH-003 §9",
    "see WH-003 section 9",
    "{WH-003 §9}",
    "<WH-003 §9>",
    "【WH-003 §9】",
    "⟦WH-003 §9⟧",
    "〔WH-003 §9〕",
    "«WH-003 §9»",
    "&#91;WH-003 §9&#93;",
    "&#91;WH-003 &sect;9&#93;",
    "see §9.9",
    "xxxxZZ-999 §1",
    "_ZZ-999 §1",
    "1ZZ-999",
    "ZZ-9999 §1",
    "**ZZ**-999 §1",
    "[WН-003]",  # Cyrillic Н
    "WН-003 §9",
    "ВН-003 §9",
    "[WH-003 §9‮.1]",  # RTL override inside the section
    "[WH-003 §9\U000e0031]",  # tag character
    "§②.①",
    "[WH-003 §2.1](http://evil)",
    "![](http://x/?q=1)",
    # extra variants of the same class
    "WH-003:9",
    "WH-003#9",
    "wh-003",
    "Section 9",
    "PO-2291",
]


@pytest.mark.parametrize("probe", PROBES)
def test_probe_with_valid_citation_is_refused(probe):
    f = finalize(f"{VALID} {probe}", HITS)
    assert f.refused and f.answer == REFUSAL, probe
    assert f.refusal_reason in {"invalid_citation", "invalid_characters"}, probe
    assert f.citations == [] and f.used == [False] * len(HITS)


def test_probe_list_not_vacuous():
    assert len(PROBES) >= 30
    f = finalize(VALID, HITS)  # positive control: the unadorned answer is accepted
    assert not f.refused and f.citations


def test_disallowed_characters_reason():
    for probe in ["【WH-003 §9】", "⟦x⟧", "〔x〕", "«x»", "ВН-003", "你好", "é", "x y"]:
        f = finalize(f"{VALID} {probe}", HITS)
        assert f.refusal_reason == "invalid_characters", probe


@pytest.mark.parametrize(
    "text",
    [
        "I couldn't find this in Meridian's documents",  # missing final period
        "I couldn’t find this in Meridian’s documents.",
        "I couldn‛t find this in Meridian‛s documents.",  # U+201B
        "I couldn´t find this in Meridian´s documents.",  # U+00B4
        "I **couldn't find this** in Meridian's documents.",  # bold fragment
        "_I couldn't find this in Meridian's documents._",
        "`I couldn't find this in Meridian's documents.`",
        "I couldn`t find this in Meridian`s documents.",
        "I could́n't find this in Meridian's documents.",  # combining mark
        "I cоldn't find this in Meridian's documents.",  # Cyrillic о
        "Sorry. I couldn't find this in Meridian's documents. [WH-003 §2.1]",
    ],
)
def test_refusal_variants_are_refused(text):
    f = finalize(f"{text} {VALID}", HITS)
    assert f.refused and f.answer == REFUSAL and f.citations == []
    assert f.refusal_reason in {"model_refused", "invalid_characters"}


def test_refusal_detection_tolerances_are_model_refused():
    for t in [
        "I couldn't find this in Meridian's documents",
        "**I couldn't find this in Meridian's documents.**",
        "I couldn´t find this in Meridian´s documents.",
    ]:
        assert is_refusal(t) and finalize(t, HITS).refusal_reason == "model_refused", t


# ---- (b) legitimate answers must pass ---------------------------------------------------------
LEGIT = [
    "Put away within 4 hours of receipt [WH-001 §5.1].",
    "Per WH-003, note the damage on the bill of lading before signing [WH-003 §2.1] and "
    "photograph it [WH-003 §2.2].",
    "If you mean PO approvals: a $40,000 PO needs the Director of Operations [PR-001 §2.1].\n"
    "If you mean inventory adjustments: above $5,000 needs the Director of Operations plus "
    "Finance [WH-005 §2.2].",
    "- Recount by a different counter [WH-002 §4.2]",
    "**Net 45** [FIN-001 §4.1]",
    "±2% or $50, whichever is lower [FIN-001 §2.2]",
    "Suppliers provide a W-9 [PR-002 §2.1].",
    "Do X [WH-003 §2.1, §2.2].",
    "It’s the “standard” rule – see below [WH-003 §2.1].",
    "Don‘t exceed 5 – 10 units — ever [WH-003 §2.2].",
    "Temp is 4° C, cost £5 or €6 × 2 • fine [WH-001 §5.1].",
]


@pytest.mark.parametrize("raw", LEGIT)
def test_legit_answers_accepted(raw):
    f = finalize(raw, HITS)
    assert not f.refused and f.refusal_reason is None, raw
    assert f.citations and all(c["valid"] for c in f.citations)
    assert f.answer == normalise_text(raw).strip()
    assert any(f.used)


def test_legit_plain_ascii_returned_verbatim():
    for raw in LEGIT[:7]:
        assert finalize(raw, HITS).answer == raw.strip()


def test_curly_quotes_folded_in_returned_answer():
    f = finalize("It’s the “standard” rule – see [WH-003 §2.1].", HITS)
    assert f.answer == 'It\'s the "standard" rule - see [WH-003 §2.1].'


def test_returned_text_has_no_format_characters():
    f = finalize("Do​ X ‮[WH-003 §2.1]‬.", HITS)
    assert not f.refused and f.answer == "Do X [WH-003 §2.1]."


def test_po_four_digit_is_refused_documented():
    # documented false refusal: 4-digit simulated ids are not in the corpus
    assert finalize("See PO-2291 [WH-003 §2.1].", HITS).refusal_reason == "invalid_citation"


# ---- timing -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "adv",
    [
        "[" * 100_000,
        "[WH-003 §2.1] " * 7_000,
        "ab-" * 33_000,
        "Do X [WH-003 §2.1]. " + "é" * 100_000,
        "​" * 100_000,
        "&#" * 50_000,
        "[§" * 50_000,
        "AB-1" * 25_000,
        "a" * 100_000,
        # these reach the residual check (a valid citation is present)
        VALID + "[" * 100_000,
        VALID + "[§" * 50_000,
        VALID + "AB-1" * 25_000,
        VALID + "ab-" * 33_000,
        VALID + "&#" * 50_000,
        VALID + "section " * 12_000,
        VALID + "ZZ-999: " * 12_000,
        VALID + "word " * 20_000,
    ],
)
def test_finalize_is_fast_on_adversarial_input(adv):
    t = time.perf_counter()
    finalize(adv, HITS)
    assert time.perf_counter() - t < 0.2


# ---- config / pipeline ------------------------------------------------------------------------
def test_gate_threshold_floor():
    for bad in (0.0, 0.049, 1e-9, 1.01, float("nan")):
        with pytest.raises(ConfigError):
            Settings(gate_threshold=bad)
    Settings(gate_threshold=0.05)
    Settings(gate_threshold=1.0)


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://h", "localhost:11434", "javascript:alert(1)", "http://", ""],
)
def test_ollama_base_url_rejected(url):
    with pytest.raises(ConfigError) as e:
        Settings(ollama_base_url=url)
    assert url not in str(e.value) or url == ""


@pytest.mark.parametrize("url", ["http://localhost:11434", "https://ollama.example.com/"])
def test_ollama_base_url_accepted(url):
    Settings(ollama_base_url=url)


def test_ollama_base_url_from_env_rejected():
    with pytest.raises(ConfigError):
        Settings.from_env({"OLLAMA_BASE_URL": "file:///x"})


def test_output_cap_ollama():
    pytest.importorskip("langchain_ollama")
    m = llm.get_chat_model(Settings())
    assert llm.MAX_OUTPUT_TOKENS == 512 and m.num_predict == 512  # type: ignore[attr-defined]


def test_output_cap_groq():
    pytest.importorskip("langchain_groq")
    m = llm.get_chat_model(Settings(llm_provider="groq", groq_api_key="k"))
    assert m.max_tokens == 512  # type: ignore[attr-defined]


def test_normalised_question_length_capped(tmp_path, embeddings):
    from ops_copilot.pipeline import Pipeline

    s = Settings(gate_threshold=0.1, chroma_dir=str(tmp_path / "c"), docs_dir=str(FIXTURE_DOCS))
    chat = CountingChat(responses=["x"])
    p = Pipeline(s, embeddings, chat, FIXTURE_DOCS)  # type: ignore[arg-type]
    q = "ﷺ" * 200  # 200 chars raw, 18 each after NFKC = 3600
    assert len(q) <= MAX_QUESTION_CHARS and len(normalise_text(q)) > MAX_QUESTION_CHARS
    r = p.ask(q)
    assert r["refusal_reason"] == "question_too_long" and chat.calls == 0


def test_pipeline_returns_invalid_characters(make_pipeline):
    p, _ = make_pipeline(response="Do X [WH-003 §2.1] 【WH-003 §9】")
    r = p.ask("What happens to damaged cartons?")
    assert r["refusal_reason"] == "invalid_characters"
    assert r["answer"] == REFUSAL and r["citations"] == []
