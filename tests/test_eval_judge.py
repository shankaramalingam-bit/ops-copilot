"""Offline tests for the eval's .env loading and judge-model selection/size rule (unit F3)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
run_eval = importlib.import_module("run_eval")

CANARY = "gsk_CANARY_f3_do_not_print_0123456789"


@pytest.mark.parametrize(
    ("name", "size"),
    [
        ("llama-3.3-70b-versatile", 70.0),
        ("llama-3.1-8b-instant", 8.0),
        ("llama3.1:8b", 8.0),
        ("Llama3.1:8B", 8.0),
        ("llama3.1:8b-instruct-q4_0", 8.0),
        ("qwen3:0.6b", 0.6),
        ("gemma2:27b", 27.0),
        ("llama-3.1-405b", 405.0),
        ("llama3.1:8.0b", 8.0),
        ("mixtral-8x7b-32768", None),
        ("mixtral:8x22b", None),
        ("some-model", None),
        ("llama3.1:latest", None),
        ("llama-8b-vs-70b", None),
        ("", None),
    ],
)
def test_parse_model_size_b(name, size):
    assert run_eval.parse_model_size_b(name) == size


@pytest.mark.parametrize(
    ("generator", "judge", "refused"),
    [
        ("llama-3.3-70b-versatile", "llama-3.3-70b-versatile", False),  # default: judge = gen
        ("llama-3.3-70b-versatile", "llama-3.1-405b", False),  # larger judge
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", True),  # 8B
        ("llama-3.3-70b-versatile", "LLAMA3.1:8B", True),  # 8B, other spelling/case
        ("llama-3.3-70b-versatile", "llama3.1:8.0b", True),
        ("llama-3.3-70b-versatile", "gemma2:27b", True),  # smaller than generator
        ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", True),  # unparseable judge
        ("llama-3.3-70b-versatile", "mystery-model", True),
        ("mystery-model", "llama-3.3-70b-versatile", True),  # unparseable generator
        ("llama3.1:8b", "llama3.1:8b", True),  # old local config: 8B judge refused
        ("llama3.1:8b", "llama-3.3-70b-versatile", False),  # bigger judge for small generator
    ],
)
def test_check_judge_model(generator, judge, refused):
    msg = run_eval.check_judge_model(generator, judge)
    assert (msg is not None) is refused
    if msg:
        assert msg.startswith("refusing")


# ---- main()


class _Stop(Exception):
    pass


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Patch everything main() touches; never reads a real .env or the network."""
    import ops_copilot.config as config
    import ops_copilot.llm as llm
    import ops_copilot.pipeline as pipeline

    calls: list[Any] = []
    built: list[Any] = []

    def fake_load() -> None:
        calls.append("load_env_file")

    real_from_env = config.Settings.from_env.__func__  # type: ignore[attr-defined]

    def fake_from_env(cls, env=None):
        calls.append("from_env")
        return real_from_env(cls, env)

    class FakePipeline:
        def __init__(self, settings):
            self.settings = settings

        def ask(self, q, mode):  # pragma: no cover - no questions are loaded
            raise _Stop

    def fake_chat(settings):
        built.append(settings)
        return object()

    monkeypatch.setattr(config, "load_env_file", fake_load)
    monkeypatch.setattr(config.Settings, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(pipeline, "Pipeline", FakePipeline)
    monkeypatch.setattr(llm, "get_chat_model", fake_chat)
    monkeypatch.setattr(run_eval, "load_questions", lambda path=None: [])
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    for k in ("EVAL_JUDGE_MODEL", "GROQ_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", CANARY)
    monkeypatch.setenv("EVAL_ALLOW_REMOTE", "1")
    return calls, built, tmp_path


def test_main_loads_env_file_before_settings(harness, monkeypatch):
    calls, _, _ = harness
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    assert run_eval.main() == 0
    assert calls[:2] == ["load_env_file", "from_env"]


def test_judge_uses_eval_judge_model_and_generator_uses_groq_model(harness, monkeypatch, capsys):
    _, built, tmp_path = harness
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "  llama-3.1-405b  ")
    assert run_eval.main() == 0
    assert [b.groq_model for b in built] == ["llama-3.1-405b"]  # only the judge is built here
    assert built[0].llm_provider == "groq"
    out = capsys.readouterr().out
    assert "generator model='llama-3.3-70b-versatile' judge model='llama-3.1-405b'" in out
    detail = json.loads(next((tmp_path / "results").glob("*.json")).read_text())
    assert detail["provider"] == "groq"
    assert detail["generator_model"] == "llama-3.3-70b-versatile"
    assert detail["judge_model"] == "llama-3.1-405b"
    assert detail["model"] == "llama-3.3-70b-versatile"
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert list(latest) == run_eval.LATEST_KEYS  # latest.json schema unchanged


def test_judge_defaults_to_generator_and_blank_is_unset(harness, monkeypatch, capsys):
    _, built, _ = harness
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "   ")
    assert run_eval.main() == 0
    assert built[0].groq_model == "llama-3.3-70b-versatile"
    assert "judge model='llama-3.3-70b-versatile'" in capsys.readouterr().out


def test_ollama_provider_judge_gets_ollama_model(harness, monkeypatch):
    _, built, _ = harness
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "llama3.3:70b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    assert run_eval.main() == 0
    assert built[0].ollama_model == "llama3.3:70b"
    assert built[0].llm_provider == "ollama"


@pytest.mark.parametrize(
    ("gen", "judge"),
    [
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),  # 8B judge
        ("llama-3.3-70b-versatile", "gemma2:27b"),  # smaller
        ("llama-3.3-70b-versatile", "mixtral-8x7b-32768"),  # unparseable
        ("llama-3.1-8b-instant", ""),  # default judge = 8B generator
    ],
)
def test_main_refuses_and_never_prints_key(harness, monkeypatch, capsys, gen, judge):
    _, built, tmp_path = harness
    monkeypatch.setenv("GROQ_MODEL", gen)
    monkeypatch.setenv("EVAL_JUDGE_MODEL", judge)
    assert run_eval.main() == 2
    cap = capsys.readouterr()
    assert "refusing" in cap.err
    assert built == []  # no model constructed
    assert not (tmp_path / "latest.json").exists()
    assert CANARY not in cap.out + cap.err


def test_canary_key_never_printed_on_success(harness, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    assert run_eval.main() == 0
    cap = capsys.readouterr()
    assert cap.out and CANARY not in cap.out + cap.err
    detail = next((harness[2] / "results").glob("*.json")).read_text()
    assert CANARY not in detail
    assert CANARY not in (harness[2] / "latest.json").read_text()
    assert "judge='llama-3.3-70b-versatile'" in cap.out  # summary line uses repr


def test_canary_key_never_printed_on_remote_guard_refusal(harness, monkeypatch, capsys):
    monkeypatch.delenv("EVAL_ALLOW_REMOTE")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    assert run_eval.main() == 2
    cap = capsys.readouterr()
    assert CANARY not in cap.out + cap.err


def test_allow_remote_from_dotenv_is_ignored(harness, monkeypatch, capsys):
    """A .env (simulated by load_env_file setting os.environ) must not enable remote."""
    import os

    import ops_copilot.config as config

    monkeypatch.delenv("EVAL_ALLOW_REMOTE")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def dotenv_load() -> None:
        os.environ["EVAL_ALLOW_REMOTE"] = "1"
        os.environ["LLM_PROVIDER"] = "groq"

    monkeypatch.setattr(config, "load_env_file", dotenv_load)
    assert run_eval.main() == 2
    assert "refusing" in capsys.readouterr().err
    monkeypatch.delenv("EVAL_ALLOW_REMOTE", raising=False)


def test_allow_remote_from_real_environment_is_honoured(harness, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EVAL_ALLOW_REMOTE", "1")
    assert run_eval.main() == 0


# ---- rate-limit tolerance (unit F4)

MSG_CANARY = "org_CANARY_f4_secret_org_id_987"


class FakeRateLimit(Exception):
    def __init__(self, msg: str = f"429 too many requests {MSG_CANARY}") -> None:
        super().__init__(msg)
        self.status_code = 429


class RateLimitError(Exception):  # matched by class name alone
    pass


@pytest.fixture
def clock(monkeypatch):
    """Fake clock + sleep: sleeping advances the clock; records every sleep."""
    state: dict[str, Any] = {"now": 0.0, "sleeps": [], "events": []}

    def fake_sleep(s):
        state["sleeps"].append(s)
        state["events"].append(("sleep", s))
        state["now"] += s

    monkeypatch.setattr(run_eval, "_sleep", fake_sleep)
    monkeypatch.setattr(run_eval, "_clock", lambda: state["now"])
    return state


class _FlakyChat:
    def __init__(self, n_fail, exc=FakeRateLimit, answer="YES"):
        self.n_fail, self.exc, self.answer, self.calls = n_fail, exc, answer, 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise self.exc()
        answer = self.answer

        class M:
            content = answer

        return M()


def test_is_rate_limit():
    assert run_eval.is_rate_limit(FakeRateLimit())
    assert run_eval.is_rate_limit(RateLimitError())
    assert not run_eval.is_rate_limit(ValueError("x"))
    e = Exception()
    e.status_code = 500  # type: ignore[attr-defined]
    assert not run_eval.is_rate_limit(e)


def test_judge_retries_and_counts(clock):
    chat = _FlakyChat(3)
    ok, raw, retries = run_eval.judge_with_retry(chat, "ctx", "claim")
    assert (ok, raw, retries) == (True, "YES", 3)
    assert chat.calls == 4
    assert clock["sleeps"] == [10, 20, 40]


def test_judge_class_name_only_match(clock):
    chat = _FlakyChat(1, exc=RateLimitError)
    assert run_eval.judge_with_retry(chat, "c", "x")[2] == 1


def test_judge_max_attempts_reraises(clock):
    chat = _FlakyChat(99)
    with pytest.raises(FakeRateLimit):
        run_eval.judge_with_retry(chat, "c", "x")
    assert chat.calls == run_eval.MAX_ATTEMPTS == 6
    assert clock["sleeps"] == [10, 20, 40, 60, 60]


def test_non_429_propagates_immediately(clock):
    chat = _FlakyChat(5, exc=lambda: ValueError("boom"))
    with pytest.raises(ValueError):
        run_eval.judge_with_retry(chat, "c", "x")
    assert chat.calls == 1
    assert clock["sleeps"] == []

    class Pipe:
        calls = 0

        def ask(self, q, mode):
            Pipe.calls += 1
            raise KeyError("nope")

    with pytest.raises(KeyError):
        run_eval.timed_ask(Pipe(), "q")
    assert Pipe.calls == 1


def test_generator_retry_latency_excludes_failures_and_backoff(clock):
    class Pipe:
        calls = 0

        def ask(self, q, mode):
            Pipe.calls += 1
            clock["now"] += 5.0  # time spent in a failed or successful attempt
            if Pipe.calls <= 2:
                raise FakeRateLimit()
            clock["now"] += 0.25  # extra cost of the successful attempt only
            return {"answer": "a"}

    res, latency_ms, retries = run_eval.timed_ask(Pipe(), "q")
    assert res == {"answer": "a"}
    assert retries == 2
    assert Pipe.calls == 3
    assert clock["sleeps"] == [10, 20]
    assert latency_ms == pytest.approx(5250.0)


def test_generator_max_attempts_reraises(clock):
    class Pipe:
        calls = 0

        def ask(self, q, mode):
            Pipe.calls += 1
            raise FakeRateLimit()

    with pytest.raises(FakeRateLimit):
        run_eval.timed_ask(Pipe(), "q")
    assert Pipe.calls == 6


@pytest.mark.parametrize("bad", ["abc", "-1", "nan", "inf", "-0.5", "1e999x"])
def test_parse_pace_invalid(bad):
    with pytest.raises(ValueError) as ei:
        run_eval.parse_pace({"EVAL_PACE_S": bad})
    assert "EVAL_PACE_S" in str(ei.value)
    assert bad not in str(ei.value)


def test_parse_pace_valid():
    assert run_eval.parse_pace({}) == 0.0
    assert run_eval.parse_pace({"EVAL_PACE_S": ""}) == 0.0
    assert run_eval.parse_pace({"EVAL_PACE_S": " 2.5 "}) == 2.5
    assert run_eval.parse_pace({"EVAL_PACE_S": "0"}) == 0.0


def test_main_invalid_pace_exits_2_without_echo(harness, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EVAL_PACE_S", "bogus-value-xyz")
    assert run_eval.main() == 2
    cap = capsys.readouterr()
    assert "EVAL_PACE_S" in cap.err
    assert "bogus-value-xyz" not in cap.out + cap.err


def test_pacing_sleeps_outside_timed_region(clock):
    class Pipe:
        def ask(self, q, mode):
            clock["events"].append(("ask_start", clock["now"]))
            clock["now"] += 1.5
            return {"answer": "a"}

    _, latency_ms, retries = run_eval.timed_ask(Pipe(), "q", pace_s=7.0)
    assert retries == 0
    assert clock["events"][0] == ("sleep", 7.0)
    assert clock["events"][1][0] == "ask_start"
    assert latency_ms == pytest.approx(1500.0)

    clock["events"].clear()
    chat = _FlakyChat(0)
    run_eval.judge_with_retry(chat, "c", "x", pace_s=3.0)
    assert clock["events"] == [("sleep", 3.0)]
    assert chat.calls == 1


def test_backoff_print_has_no_exception_message(clock, capsys):
    run_eval.judge_with_retry(_FlakyChat(2), "c", "x")
    out = capsys.readouterr()
    text = out.out + out.err
    assert MSG_CANARY not in text
    assert "429" not in text
    assert "rate limited (FakeRateLimit); waiting 10s" in text
    assert "waiting 20s" in text


def test_main_records_retries_and_pace(harness, monkeypatch, clock):
    import ops_copilot.llm as llm
    import ops_copilot.pipeline as pipeline

    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EVAL_PACE_S", "1.5")
    q = {
        "id": "Q1",
        "type": "factual",
        "question": "what?",
        "expected_docs": ["SOP-001"],
        "expected_sections": ["1"],
        "expected_answer_points": ["p"],
    }
    monkeypatch.setattr(run_eval, "load_questions", lambda path=None: [q])

    class Pipe:
        calls = 0

        def __init__(self, settings):
            pass

        def ask(self, question, mode):
            Pipe.calls += 1
            clock["now"] += 2.0
            if Pipe.calls == 1:
                raise FakeRateLimit()
            return {
                "answer": "The dock opens at six every morning [SOP-001 § 1].",
                "refused": False,
                "refusal_reason": None,
                "citations": [{"doc_id": "SOP-001", "valid": True}],
                "confidence": 1.0,
                "timings_ms": {},
                "chunks": [
                    {
                        "id": "SOP-001#1",
                        "text": "dock opens at six",
                        "used": True,
                        "bm25_rank": 1,
                        "dense_rank": 1,
                        "rrf": 0.1,
                    }
                ],
            }

    chat = _FlakyChat(2)
    monkeypatch.setattr(pipeline, "Pipeline", Pipe)
    monkeypatch.setattr(llm, "get_chat_model", lambda settings: chat)
    _, _, tmp_path = harness
    assert run_eval.main() == 0
    detail = json.loads(next((tmp_path / "results").glob("*.json")).read_text())
    rec = detail["questions"][0]
    assert detail["pace_s"] == 1.5
    assert rec["rate_limit_retries"] == 1
    assert rec["judge_rate_limit_retries"] == 2
    assert rec["latency_ms"] == pytest.approx(2000.0)
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert list(latest) == run_eval.LATEST_KEYS
    assert "rate_limit_retries" not in latest["questions"][0]
