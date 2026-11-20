"""Deploy artefacts stay localhost-only and secret-free. Offline: only reads files."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


def _compose() -> dict:
    return yaml.safe_load((DEPLOY / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_services_and_only_loopback_ports() -> None:
    services = _compose()["services"]
    assert {"app", "ollama"} <= set(services)
    published = [p for s in services.values() for p in s.get("ports", [])]
    assert published == ["127.0.0.1:8000:8000"]
    assert "0.0.0.0" not in (DEPLOY / "docker-compose.yml").read_text().replace(
        "OPS_COPILOT_HOST: 0.0.0.0", ""
    ).replace("OPS_COPILOT_HOST=0.0.0.0", "")


def test_compose_app_env_and_volume() -> None:
    c = _compose()
    env = c["services"]["app"]["environment"]
    env = dict(e.split("=", 1) for e in env) if isinstance(env, list) else env
    assert env["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert str(env["OPS_COPILOT_ALLOW_NONLOCAL"]) == "1"
    assert "localhost" in env["OPS_COPILOT_ALLOWED_HOSTS"]
    assert c["services"]["ollama"]["image"].startswith("ollama/ollama")
    assert c["volumes"], "named volume for models"
    assert "ports" not in c["services"]["ollama"], "ollama must not be published"


def test_compose_has_no_secret_values() -> None:
    text = (DEPLOY / "docker-compose.yml").read_text()
    assert not re.search(r"(?i)(api_key|secret|token|password)\s*[:=]\s*\S", text)


def test_dockerfile_non_root_and_no_env_copy() -> None:
    lines = (DEPLOY / "Dockerfile").read_text().splitlines()
    users = [ln.split()[1] for ln in lines if ln.upper().startswith("USER ")]
    assert users and users[-1] not in ("root", "0")
    for ln in lines:
        if ln.upper().startswith(("COPY", "ADD")):
            assert ".env" not in ln and "data/cache" not in ln
    assert any(ln.startswith("FROM python:3.12-slim") for ln in lines)


def test_dockerignore_excludes_secrets_and_caches() -> None:
    # Context is the repo root; BuildKit reads <Dockerfile>.dockerignore, so both copies must match.
    for name in (".dockerignore", "Dockerfile.dockerignore"):
        entries = {ln.strip() for ln in (DEPLOY / name).read_text().splitlines() if ln.strip()}
        for needed in (".env*", ".venv", "data/cache", ".git"):
            assert needed in entries, (name, needed)
    assert (DEPLOY / ".dockerignore").read_text() == (
        DEPLOY / "Dockerfile.dockerignore"
    ).read_text()


def test_readme_has_start_command_and_mermaid() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "uv run python -m ops_copilot" in readme
    assert "```mermaid" in readme
    assert "http://localhost:8000" in readme
    assert "eval/report.md" in readme
    assert not re.search(r"(?i)api_key\s*=\s*[^\s`]", readme)
    deploy_readme = (DEPLOY / "README.md").read_text(encoding="utf-8")
    assert "docker compose" in deploy_readme and "127.0.0.1" in deploy_readme
    assert "no authentication" in deploy_readme.lower()


def test_makefile_targets() -> None:
    mk = (ROOT / "Makefile").read_text()
    for t in ("run", "test", "gates", "eval", "notebook"):
        assert re.search(rf"^{t}:", mk, re.M), t
