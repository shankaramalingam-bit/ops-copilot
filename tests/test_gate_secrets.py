"""Tests for scripts/gate_secrets.sh. Offline: every case runs against a throwaway git repo.

Every fake credential is built at runtime so no key-shaped string exists in this source file.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gate_secrets.sh"

ENV_EXAMPLE = ".env" + ".example"  # protected path name, assembled to keep tooling away


def rep(ch: str, n: int) -> str:
    return ch * n


# (id, runtime-built line). Every one must be flagged by the gate.
FAKES = {
    "groq": lambda: "gsk_" + rep("x", 24),
    "sk": lambda: "sk-" + rep("x", 24),
    "aws": lambda: "AKIA" + rep("A", 16),
    "pem": lambda: "-----BEGIN RSA PRIVATE" + " KEY-----",
    "ghp": lambda: "ghp_" + rep("a", 36),
    "gho": lambda: "gho_" + rep("a", 36),
    "ghs": lambda: "ghs_" + rep("a", 36),
    "github_pat": lambda: "github_pat_" + rep("a", 30),
    "slack": lambda: "xoxb-" + rep("1", 12) + "-" + rep("a", 12),
    "google": lambda: "AIza" + rep("a", 35),
    "hf": lambda: "hf_" + rep("a", 32),
    "lsv2": lambda: "lsv2_pt_" + rep("a", 24),
    "bearer": lambda: "Authorization" + ": Bearer " + rep("a", 24),
    "url": lambda: "https" + "://" + "u" + ":" + "p" + "@" + "host.example",
    "json_password": lambda: '{"pass' + 'word": "ab.cd/ef+gh=ij-klmnop"}',
    "json_token_dash": lambda: '{"to' + 'ken": "' + "a-b" * 6 + '"}',
    "plain_apikey": lambda: "api_" + "key = " + rep("a", 20),
}


def run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=repo, capture_output=True, text=True, check=False
    )


def git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    (r / "README.md").write_text("hello\n")
    (r / "blob.bin").write_bytes(bytes(range(256)) * 8)
    (r / ENV_EXAMPLE).write_text("API_" + "KEY=\nDEBUG=1\nPASS" + "WORD=\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    return r


def assert_red(res: subprocess.CompletedProcess[str], needle: str, secret: str = "") -> None:
    assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
    assert needle in res.stderr, res.stderr
    if secret:
        assert secret not in res.stdout + res.stderr, "matched text leaked"


def test_clean_repo_green(repo: Path) -> None:
    res = run(repo)
    assert res.returncode == 0, res.stderr
    assert "gate_secrets: OK" in res.stdout
    hist = run(repo, "--history")
    assert hist.returncode == 0, hist.stderr


def test_self_test_passes() -> None:
    res = subprocess.run(["bash", str(SCRIPT), "--self-test"], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PASS" in res.stdout


@pytest.mark.parametrize("kind", sorted(FAKES))
def test_each_pattern_flagged_untracked(repo: Path, kind: str) -> None:
    secret = FAKES[kind]()
    (repo / "notes.txt").write_text(f"line one\nvalue {secret}\n")
    res = run(repo)
    assert_red(res, "notes.txt:2", secret)


def test_pattern_negative_control(repo: Path) -> None:
    """Same file shape without any secret is green (proves the red above is the pattern)."""
    (repo / "notes.txt").write_text("line one\nvalue nothing-here\n")
    assert run(repo).returncode == 0


def test_json_value_charset(repo: Path) -> None:
    (repo / "c.json").write_text('{"pass' + 'word": "' + "Ab.1/2+3=4-" + "xyzxyz" + '"}\n')
    assert_red(run(repo), "c.json:1")


def test_short_generic_value_not_flagged(repo: Path) -> None:
    (repo / "c.json").write_text('{"pass' + 'word": "short"}\n')
    assert run(repo).returncode == 0


def test_staged_then_cleaned_on_disk_fails(repo: Path) -> None:
    secret = FAKES["ghp"]()
    (repo / "leak.txt").write_text(secret + "\n")
    git(repo, "add", "leak.txt")
    (repo / "leak.txt").write_text("clean now\n")
    res = run(repo)
    assert_red(res, "leak.txt:1", secret)


def test_staged_deleted_on_disk_fails(repo: Path) -> None:
    secret = FAKES["json_password"]()
    (repo / "leak.json").write_text(secret + "\n")
    git(repo, "add", "leak.json")
    (repo / "leak.json").unlink()
    assert_red(run(repo), "leak.json:1", secret)


@pytest.mark.skipif(
    os.geteuid() == 0, reason="running as root: chmod 000 does not make files unreadable"
)
def test_unreadable_tracked_file_fails_closed(repo: Path) -> None:
    f = repo / "locked.txt"
    f.write_text("harmless\n")
    git(repo, "add", "locked.txt")
    f.chmod(0)
    try:
        res = run(repo)
    finally:
        f.chmod(0o644)
    assert_red(res, "locked.txt")
    assert "unreadable" in res.stderr


def test_env_paths_skipped_not_read(repo: Path) -> None:
    """A secret inside an untracked .env.local is never read (skipped), so the gate stays green."""
    (repo / (".env" + ".local")).write_text(FAKES["groq"]() + "\n")
    res = run(repo)
    assert res.returncode == 0, res.stderr


def test_tracked_env_file_fails(repo: Path) -> None:
    (repo / ".env").write_text("A=1\n")
    git(repo, "add", "-f", ".env")
    assert_red(run(repo), ".env")


def test_symlink_not_followed(repo: Path) -> None:
    outside = repo.parent / "outside.txt"
    outside.write_text(FAKES["groq"]() + "\n")
    os.symlink(outside, repo / "link.txt")
    git(repo, "add", "link.txt")
    res = run(repo)
    assert res.returncode == 0, res.stderr
    # positive control: the same content as a regular file is flagged
    (repo / "link.txt").unlink()
    shutil.copy(outside, repo / "link.txt")
    assert_red(run(repo), "link.txt:1")


def test_symlink_to_env_fails(repo: Path) -> None:
    os.symlink(".env", repo / "innocent.txt")
    git(repo, "add", "innocent.txt")
    res = run(repo)
    assert_red(res, "innocent.txt")
    assert "symlink" in res.stderr


def test_untracked_symlink_to_env_fails(repo: Path) -> None:
    os.symlink("sub/.env.production", repo / "innocent.txt")
    assert_red(run(repo), "innocent.txt")


def test_symlink_to_env_example_allowed(repo: Path) -> None:
    os.symlink(ENV_EXAMPLE, repo / "example-link")
    assert run(repo).returncode == 0


def test_binary_vendor_pattern_flagged(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / "img.png").write_bytes(b"\x89PNG\x00\x00" + secret.encode() + b"\x00\x01\n")
    assert_red(run(repo), "img.png", secret)


def test_binary_generic_pattern_not_flagged(repo: Path) -> None:
    secret = FAKES["plain_apikey"]()
    (repo / "img.png").write_bytes(b"\x89PNG\x00\x00" + secret.encode() + b"\x00\x01\n")
    assert run(repo).returncode == 0


def test_history_catches_removed_secret(repo: Path) -> None:
    secret = FAKES["hf"]()
    (repo / "old.txt").write_text(secret + "\n")
    git(repo, "add", "old.txt")
    git(repo, "commit", "-qm", "oops")
    git(repo, "rm", "-q", "old.txt")
    git(repo, "commit", "-qm", "remove")
    assert run(repo).returncode == 0, "normal mode only sees current tree"
    res = run(repo, "--history")
    assert_red(res, ":old.txt", secret)
    assert "commit" not in res.stdout.lower()


def test_history_ignores_env_paths(repo: Path) -> None:
    (repo / ENV_EXAMPLE).write_text("API_" + "KEY=" + FAKES["groq"]() + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "env")
    assert run(repo, "--history").returncode == 0


def test_env_example_unreadable_prints_skip_line(repo: Path) -> None:
    """Directory in place of the file stands in for 'unreadable': cat fails, skip line printed."""
    (repo / ENV_EXAMPLE).unlink()
    (repo / ENV_EXAMPLE).mkdir()
    res = run(repo)
    assert res.returncode == 0, res.stderr
    assert "skipped content scan of " + ENV_EXAMPLE in res.stdout


def test_env_example_secret_named_value_fails_name_only(repo: Path) -> None:
    value = "v" + "alue123"
    (repo / ENV_EXAMPLE).write_text("OPENAI_API_" + "KEY=" + value + "\nDEBUG=1\n")
    res = run(repo)
    assert_red(res, "OPENAI_API_KEY", value)
    assert "DEBUG" not in res.stderr


@pytest.mark.parametrize("name", ["GROQ_API_KEY", "LANGSMITH_TOKEN", "APP_SECRET", "DB_PASSWORD"])
def test_env_example_each_keyword(repo: Path, name: str) -> None:
    (repo / ENV_EXAMPLE).write_text(f"{name}=abc\n")
    assert_red(run(repo), name, "abc")


def test_env_example_empty_and_nonsecret_ok(repo: Path) -> None:
    (repo / ENV_EXAMPLE).write_text(
        'GROQ_API_KEY=\nAPP_SECRET=""\nDB_PASSWORD= # fill me\nMODEL=llama3\nLOG_LEVEL=debug\n'
    )
    res = run(repo)
    assert res.returncode == 0, res.stderr
    assert "checked" in res.stdout


def test_gate_script_and_tests_hold_no_literals() -> None:
    """Running the gate on this repo must never flag the script or this test file."""
    res = subprocess.run(
        ["bash", str(SCRIPT)], cwd=SCRIPT.parent.parent, capture_output=True, text=True
    )
    assert "gate_secrets.sh" not in res.stderr
    assert "test_gate_secrets.py" not in res.stderr
