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
