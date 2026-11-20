"""Unit c.5: pipeline/answer-validation test gaps found by the c.4 mutation review (tests only)."""

from __future__ import annotations

import pytest

from ops_copilot.answer import REFUSAL, finalize, residual_is_clean
from ops_copilot.config import ConfigError, Settings

from .test_unit_c4 import HITS, VALID


def refused(extra: str) -> bool:
    f = finalize(f"{VALID}{extra}", HITS)
    return f.refused and f.answer == REFUSAL and f.citations == []


@pytest.mark.parametrize("ch", list("<>{}[]§¶"))
def test_each_residual_char_individually_refused(ch):
    assert refused(f" a {ch} b")
    assert refused(f" {ch}x")


def test_residual_char_examples_from_review():
    assert refused(" a > b")
    assert refused(" {x}")


@pytest.mark.parametrize("word", ["sec", "sec.", "para", "paragraph", "clause", "Section", "SEC."])
@pytest.mark.parametrize("sep", [" ", ""])
def test_each_residual_word_with_number_refused(word, sep):
    assert refused(f" see {word}{sep}9")


@pytest.mark.parametrize("word", ["sec", "para", "clause", "paragraph"])
def test_residual_word_without_number_not_refused(word):
    f = finalize(f"{VALID} the {word} is fine", HITS)
    assert not f.refused


@pytest.mark.parametrize(
    "doc", ["ZZ-12", "ZZ-1234", "ABCDE-123", "WH-003x", "xWH-003", "WH-0031", "zz-123", "AB-1"]
)
def test_doc_id_like_variants_refused(doc):
    assert refused(f" {doc}")


def test_doc_id_exact_and_valid_is_allowed():
    assert not finalize(f"{VALID} also WH-003.", HITS).refused


def test_citation_removed_as_space_not_empty_string():
    # Removing the citation as "" would glue "Z" + "Z-999" into the doc-like "ZZ-999" (refused).
    # It is replaced by a space, so no new doc-id-like token is formed.
    assert not finalize("Z[WH-003 §2.1]Z-999", HITS).refused
    # Two letters on one side still form a doc-like token with the digits and are refused.
    f = finalize("W[WH-003 §2.1]HH-003", HITS)
    assert f.refused and f.refusal_reason == "invalid_citation"


def test_tab_allowed_in_allowlist():
    f = finalize(f"{VALID}\tindented", HITS)
    assert not f.refused and "\t" in f.answer


def test_nul_control_char_stripped():
    f = finalize(f"{VALID}\x00 more", HITS)
    assert not f.refused and "\x00" not in f.answer


def test_nul_cannot_split_a_forgery():
    assert refused(" ZZ\x00-999 §1")


def test_base_url_valueerror_path():
    with pytest.raises(ConfigError):
        Settings(ollama_base_url="http://[::1")


# ---- residual_is_clean directly (the allowlist would otherwise mask ¶ and similar) -------------
@pytest.mark.parametrize("ch", list("<>{}[]§¶"))
def test_residual_is_clean_rejects_each_char(ch):
    assert not residual_is_clean(f"a {ch} b", {"WH-003"})


@pytest.mark.parametrize("ref", ["WH-003:9", "WH-003#9", "WH-003 : 9", "WH-003 # 9"])
def test_residual_is_clean_rejects_doc_ref_forms(ref):
    assert not residual_is_clean(f"see {ref}", {"WH-003"})


def test_residual_is_clean_accepts_plain_text():
    assert residual_is_clean("Put away within 24 hours of receipt.", {"WH-003"})
