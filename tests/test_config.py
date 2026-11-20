import pytest

from ops_copilot.config import ConfigError, Settings

FAKE_KEY = "fake-test-key-1"


def test_defaults_from_empty_env():
    s = Settings.from_env({})
    assert s.llm_provider == "ollama" and s.ollama_model == "llama3.1:8b"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.groq_model == "llama-3.1-8b-instant" and s.embed_model == "nomic-embed-text"
    assert (s.top_k, s.rrf_k, s.gate_threshold) == (4, 60, 0.55)
    assert s.docs_dir == "data/docs" and s.chroma_dir == "data/cache/chroma"


def test_reads_environment_overrides():
    s = Settings.from_env({"TOP_K": "6", "GATE_THRESHOLD": "0.3", "LLM_PROVIDER": "OLLAMA"})
    assert (s.top_k, s.gate_threshold, s.llm_provider) == (6, 0.3, "ollama")


def test_groq_without_key_errors_without_echoing_env():
    env = {"LLM_PROVIDER": "groq", "OLLAMA_BASE_URL": "http://secret-host.invalid"}
    with pytest.raises(ConfigError) as ei:
        Settings.from_env(env)
    msg = str(ei.value)
    assert "GROQ_API_KEY" in msg
    assert "secret-host" not in msg


def test_groq_with_key_ok_and_repr_masks_key():
    s = Settings.from_env({"LLM_PROVIDER": "groq", "GROQ_API_KEY": FAKE_KEY})
    assert s.groq_api_key == FAKE_KEY
    for text in (repr(s), str(s), f"{s}"):
        assert FAKE_KEY not in text and "groq_api_key=***" in text


def test_invalid_provider_rejected_without_echo():
    with pytest.raises(ConfigError) as ei:
        Settings.from_env({"LLM_PROVIDER": "bogus-provider"})
    assert "bogus-provider" not in str(ei.value)


def test_non_numeric_setting_rejected_without_echo():
    with pytest.raises(ConfigError) as ei:
        Settings.from_env({"TOP_K": "abc-value"})
    assert "abc-value" not in str(ei.value)


def test_import_does_not_load_dotenv(monkeypatch):
    import ops_copilot.config as cfg

    assert not hasattr(cfg, "load_dotenv")
    assert callable(cfg.load_env_file)
