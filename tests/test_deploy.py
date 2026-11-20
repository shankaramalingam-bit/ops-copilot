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
    # a `${VAR:-}` interpolation is a reference, not a value
    text = re.sub(r"\$\{[^}]*\}", "REF", text)
    assert not re.search(r"(?i)(api_key|secret|token|password)\s*[:=]\s*(?!REF\b)\S", text)


def test_dockerfile_non_root_and_no_env_copy() -> None:
    lines = (DEPLOY / "Dockerfile").read_text().splitlines()
    users = [ln.split()[1] for ln in lines if ln.upper().startswith("USER ")]
    assert users and users[-1] not in ("root", "0")
    for ln in lines:
        if ln.upper().startswith(("COPY", "ADD")):
            assert ".env" not in ln and "data/cache" not in ln
    assert any(ln.startswith("FROM python:3.12.14-slim") for ln in lines)


def test_dockerignore_excludes_secrets_and_caches() -> None:
    # Context is the repo root; BuildKit reads <Dockerfile>.dockerignore, else the root
    # .dockerignore. All three copies must be identical.
    paths = [ROOT / ".dockerignore", DEPLOY / ".dockerignore", DEPLOY / "Dockerfile.dockerignore"]
    for path in paths:
        assert path.is_file(), path
        entries = {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}
        for needed in ("**/.env*", ".venv", "data/cache", ".git"):
            assert needed in entries, (path, needed)
        assert not any(e.startswith("!") for e in entries), (path, "no re-includes")
        assert ".env*" not in entries, "root-only pattern would miss nested .env files"
    texts = {p.read_text() for p in paths}
    assert len(texts) == 1, "root, deploy/.dockerignore and Dockerfile.dockerignore must match"


def _image_refs() -> list[str]:
    refs = []
    for ln in (DEPLOY / "Dockerfile").read_text().splitlines():
        m = re.match(r"FROM\s+(?:--\S+\s+)?(\S+)", ln, re.I)
        if m and m.group(1).lower() != "scratch":
            refs.append(m.group(1))
        m = re.search(r"COPY\s+--from=(\S+)", ln, re.I)
        if m and not re.fullmatch(r"\d+|[a-z_][\w-]*", m.group(1)):
            refs.append(m.group(1))
    refs += [s["image"] for s in _compose()["services"].values() if "image" in s]
    return refs


def test_images_are_pinned_to_explicit_non_latest_tags() -> None:
    refs = _image_refs()
    assert any(r.startswith("python:3.12.14-slim") for r in refs)
    assert any(r.startswith("ghcr.io/astral-sh/uv:0.12.5") for r in refs)
    assert any(r.startswith("ollama/ollama:") for r in refs)
    for ref in refs:
        last = ref.split("@")[0].rsplit("/", 1)[-1]
        tag = last.split(":", 1)[1] if ":" in last else ""
        assert tag, f"untagged image: {ref}"
        assert tag != "latest", f"floating tag: {ref}"
        assert re.match(r"\d+\.\d+", tag), f"tag is not a version: {ref}"


def test_compose_has_no_env_file_and_explicit_environment() -> None:
    c = _compose()
    assert "env_file" not in (DEPLOY / "docker-compose.yml").read_text()
    for name, svc in c["services"].items():
        assert "env_file" not in svc, name
    env = c["services"]["app"]["environment"]
    env = dict(e.split("=", 1) for e in env) if isinstance(env, list) else env
    assert {"LLM_PROVIDER", "OLLAMA_MODEL", "GROQ_API_KEY"} <= set(env)
    assert env["GROQ_API_KEY"] == "${GROQ_API_KEY:-}"  # reference only, empty default
    assert env["LLM_PROVIDER"].startswith("${LLM_PROVIDER")
    assert env["OLLAMA_MODEL"].startswith("${OLLAMA_MODEL")
    # no other key is passed through from the host
    for k, v in env.items():
        if "${" in str(v):
            assert k in {"LLM_PROVIDER", "OLLAMA_MODEL", "GROQ_API_KEY"}, k


def test_healthchecks_and_service_healthy_dependency() -> None:
    c = _compose()["services"]
    app_hc = c["app"]["healthcheck"]["test"]
    joined = " ".join(app_hc)
    assert "http://127.0.0.1:8000/api/docs" in joined
    assert "urllib" in joined and "localhost" in joined  # Host header for the allow-list
    assert c["ollama"]["healthcheck"]["test"] == ["CMD", "ollama", "list"]
    for svc in ("app", "ollama"):
        hc = c[svc]["healthcheck"]
        assert {"interval", "timeout", "retries"} <= set(hc), svc
    assert c["app"]["depends_on"] == {"ollama": {"condition": "service_healthy"}}


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
