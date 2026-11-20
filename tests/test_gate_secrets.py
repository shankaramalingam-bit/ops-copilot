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
    "stripe_sk": lambda: "sk_" + "live_" + rep("a", 24),
    "stripe_rk": lambda: "rk_" + "live_" + rep("a", 24),
    "aws_asia": lambda: "ASIA" + rep("A", 16),
    "pgp": lambda: "-----BEGIN PGP PRIVATE" + " KEY BLOCK-----",
    "slack_app": lambda: "xapp-" + rep("1", 12) + "-" + rep("a", 12),
    "bearer": lambda: "Authorization" + ": Bearer " + rep("a", 24),
    "url": lambda: "https" + "://" + "u" + ":" + "p" + "@" + "host.example",
    "json_password": lambda: '{"pass' + 'word": "ab.cd/ef+gh=ij-klmnop"}',
    "json_token_dash": lambda: '{"to' + 'ken": "' + "a-b" * 6 + '"}',
    "plain_apikey": lambda: "api_" + "key = " + rep("a", 20),
}


def base_env(ci: bool = False) -> dict[str, str]:
    """Hermetic env. CI markers are removed unless ci=True (the gate's mode depends on them)."""
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS")}
    if ci:
        env["CI"] = "1"
    return {**env, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}


def run(repo: Path, *args: str, ci: bool = False) -> subprocess.CompletedProcess[str]:
    # hermetic: a user-level global gitignore (e.g. ".env*") must not hide test files
    env = base_env(ci)
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=repo, env=env, capture_output=True, text=True, check=False
    )


def git(repo: Path, *args: str, input: str | None = None) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    res = subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True, input=input
    )
    return res.stdout.strip()


def track(repo: Path, path: str, content: str | bytes) -> None:
    """Put `path` in the index WITHOUT writing it to disk (so no .env* file ever exists)."""
    data = content if isinstance(content, bytes) else content.encode()
    blob = (
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=repo,
            input=data,
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")


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


def test_env_local_content_never_printed(repo: Path) -> None:
    """A tracked .env.local (path-only logic, never on disk) fails on its PATH; content not read."""
    secret = FAKES["groq"]()
    track(repo, ".env" + ".local", secret + "\n")
    res = run(repo)
    assert_red(res, ".env.local", secret)
    assert "possible secrets" not in res.stderr


def test_tracked_env_file_fails(repo: Path) -> None:
    track(repo, ".env", "A=1\n")
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


def test_history_scans_env_example(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / ENV_EXAMPLE).write_text("NOTE=" + secret + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "env")
    assert_red(run(repo, "--history"), ENV_EXAMPLE, secret)


def test_env_example_unreadable_prints_skip_line(repo: Path) -> None:
    """Directory in place of the file stands in for 'unreadable': cat fails, skip line printed."""
    (repo / ENV_EXAMPLE).unlink()
    (repo / ENV_EXAMPLE).mkdir()
    res = run(repo)
    assert res.returncode == 0, res.stderr
    assert "skipped content scan of " + ENV_EXAMPLE in res.stdout


def test_h2_nested_env_example_unreadable_names_path(repo: Path) -> None:
    track(repo, "config/" + ENV_EXAMPLE, "A=1\n")
    git(repo, "commit", "-qm", "nested")
    (repo / "config").mkdir(exist_ok=True)
    (repo / "config" / ENV_EXAMPLE).mkdir()
    res = run(repo)
    assert res.returncode == 0, res.stderr
    assert "skipped content scan of config/" + ENV_EXAMPLE in res.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_h2_env_example_chmod0_names_path(repo: Path) -> None:
    f = repo / ENV_EXAMPLE
    f.chmod(0)
    try:
        res = run(repo)
    finally:
        f.chmod(0o644)
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


# ---------------------------------------------------------------- round 2 (unit a.2)
def assert_no_leak(res: subprocess.CompletedProcess[str], secret: str) -> None:
    assert secret not in res.stdout + res.stderr, "secret text leaked"


# ---- H1: history must see binaries
def test_h1_history_binary_added_then_deleted(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / "img.png").write_bytes(b"\x89PNG\x00\x00" + secret.encode() + b"\x00\x01\n")
    git(repo, "add", "img.png")
    git(repo, "commit", "-qm", "add bin")
    git(repo, "rm", "-q", "img.png")
    git(repo, "commit", "-qm", "rm bin")
    assert run(repo).returncode == 0
    res = run(repo, "--history")
    assert_red(res, ":img.png", secret)


# ---- H2: exact .env exclusion
@pytest.mark.parametrize("name", [".envrc", ".env-notes.md", ".environment", "sub/.envrc"])
def test_h2_lookalike_paths_scanned_untracked(repo: Path, name: str) -> None:
    secret = FAKES["ghp"]()
    p = repo / name
    p.parent.mkdir(exist_ok=True)
    p.write_text("v " + secret + "\n")
    assert_red(run(repo), name + ":1", secret)


def test_h2_file_under_env_dir_scanned_index(repo: Path) -> None:
    secret = FAKES["ghp"]()
    track(repo, ".env/notes.txt", "v " + secret + "\n")
    assert_red(run(repo), ".env/notes.txt:1", secret)


def test_h2_lookalike_scanned_in_history(repo: Path) -> None:
    secret = FAKES["ghp"]()
    (repo / ".envrc").write_text(secret + "\n")
    git(repo, "add", ".envrc")
    git(repo, "commit", "-qm", "x")
    git(repo, "rm", "-q", ".envrc")
    git(repo, "commit", "-qm", "y")
    assert_red(run(repo, "--history"), ":.envrc", secret)


def test_h2_env_example_vendor_pattern_root(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / ENV_EXAMPLE).write_text("NOTE=" + secret + "\n")
    assert_red(run(repo), ENV_EXAMPLE + ":1", secret)


def test_h2_env_example_vendor_pattern_nested(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / "config").mkdir()
    (repo / "config" / ENV_EXAMPLE).write_text("NOTE=" + secret + "\n")
    assert_red(run(repo), "config/" + ENV_EXAMPLE + ":1", secret)


def test_h2_env_example_vendor_pattern_nested_tracked(repo: Path) -> None:
    secret = FAKES["groq"]()
    track(repo, "a/b/" + ENV_EXAMPLE, "NOTE=" + secret + "\n")
    (repo / "a" / "b").mkdir(parents=True)  # readable on disk (clean): the index copy is scanned
    (repo / "a" / "b" / ENV_EXAMPLE).write_text("A=1\n")
    assert_red(run(repo), "a/b/" + ENV_EXAMPLE + ":1", secret)


def test_h2_env_example_nested_secret_var_nonempty(repo: Path) -> None:
    value = "v" + "alue123"
    (repo / "config").mkdir()
    (repo / "config" / ENV_EXAMPLE).write_text("OPENAI_API_" + "KEY=" + value + "\n")
    res = run(repo)
    assert_red(res, "OPENAI_API_KEY", value)
    assert "config/" + ENV_EXAMPLE in res.stderr


def test_h2_env_example_url_pattern(repo: Path) -> None:
    (repo / ENV_EXAMPLE).write_text("DB=" + FAKES["url"]() + "\n")
    assert_red(run(repo), ENV_EXAMPLE + ":1", "u:p@")


def test_h2_tracked_env_exact_definition(repo: Path) -> None:
    """Lookalikes are NOT tracked-env failures; real .env.* names are."""
    track(repo, ".envrc", "A=1\n")
    track(repo, "x/.env-notes.md", "A=1\n")
    assert run(repo).returncode == 0
    track(repo, "x/.env.staging", "A=1\n")
    assert_red(run(repo), "x/.env.staging")


# ---- M3: never print secret text via filenames
def test_m3_filename_with_key_untracked(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / (secret + ".txt")).write_text("clean\n")
    res = run(repo)
    assert_red(res, "<redacted-path>", secret)
    assert_no_leak(res, secret)


def test_m3_filename_with_key_and_content_hit(repo: Path) -> None:
    secret = FAKES["ghp"]()
    (repo / (secret + ".txt")).write_text("v " + secret + "\n")
    res = run(repo)
    assert_red(res, "<redacted-path>:1", secret)
    assert_no_leak(res, secret)


def test_m3_filename_with_key_in_history(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / (secret + ".txt")).write_text("clean\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "named")
    git(repo, "rm", "-q", secret + ".txt")
    git(repo, "commit", "-qm", "unnamed")
    res = run(repo, "--history")
    assert_red(res, "<redacted-path>", secret)
    assert_no_leak(res, secret)


def test_m3_tracked_index_only_filename(repo: Path) -> None:
    secret = FAKES["ghp"]()
    track(repo, secret + ".txt", "clean\n")
    res = run(repo)
    assert_red(res, "<redacted-path>", secret)
    assert_no_leak(res, secret)


# ---- M4: strict .env.example parsing
def test_m4_reviewer_leak_case(repo: Path) -> None:
    secret = FAKES["sk"]() + "=="
    (repo / ENV_EXAMPLE).write_text("OPENAI_API_" + "KEY " + secret + "\n")
    res = run(repo)
    assert_red(res, ENV_EXAMPLE + ":1: malformed line", secret)
    assert_no_leak(res, secret)
    assert_no_leak(res, FAKES["sk"]())


def test_m4_yaml_style_malformed(repo: Path) -> None:
    (repo / ENV_EXAMPLE).write_text("A=1\nOPENAI_API_" + "KEY: abc\n")
    res = run(repo)
    assert_red(res, ENV_EXAMPLE + ":2: malformed line", "abc")


@pytest.mark.parametrize("line", ["1BAD=x", "BAD NAME=x", "BAD-NAME=x", "just words", "=x", " : "])
def test_m4_malformed_variants(repo: Path, line: str) -> None:
    (repo / ENV_EXAMPLE).write_text("# c\n\n" + line + "\n")
    res = run(repo)
    assert_red(res, ENV_EXAMPLE + ":3: malformed line")


def test_m4_quoted_empty_with_comment_is_empty(repo: Path) -> None:
    (repo / ENV_EXAMPLE).write_text(
        'A_API_KEY="" # fill me\nB_TOKEN=\'\'  # x\nexport C_SECRET=\nD_PASSWORD="" \n'
    )
    res = run(repo)
    assert res.returncode == 0, res.stderr


def test_m4_quoted_then_text_is_nonempty(repo: Path) -> None:
    (repo / ENV_EXAMPLE).write_text('A_API_KEY="" oops\n')
    assert_red(run(repo), "A_API_KEY")


def test_m4_keyshaped_name_redacted(repo: Path) -> None:
    secret = FAKES["groq"]()
    (repo / ENV_EXAMPLE).write_text(secret + "=1\n")
    res = run(repo)
    assert res.returncode == 1
    assert_no_leak(res, secret)


def test_m4_index_copy_checked_when_disk_differs(repo: Path) -> None:
    secret = "v" + "alue123"
    track(repo, ENV_EXAMPLE, "OPENAI_API_" + "KEY=" + secret + "\n")  # disk copy stays clean
    assert_red(run(repo), "OPENAI_API_KEY", secret)


# ---- M5: history coverage
def test_m5_commit_message(repo: Path) -> None:
    secret = FAKES["ghp"]()
    git(repo, "commit", "-q", "--allow-empty", "-m", "oops " + secret)
    res = run(repo, "--history")
    assert_red(res, "commit message", secret)
    assert_no_leak(res, secret)


def test_m5_annotated_tag_message(repo: Path) -> None:
    secret = FAKES["ghp"]()
    git(repo, "tag", "-a", "v1", "-m", "release " + secret)
    res = run(repo, "--history")
    assert_red(res, "tag message", secret)
    assert_no_leak(res, secret)


def test_m5_merge_commit_resolution(repo: Path) -> None:
    secret = FAKES["ghp"]()
    base = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "x.txt").write_text("x\n")
    git(repo, "add", "x.txt")
    git(repo, "commit", "-qm", "side")
    git(repo, "checkout", "-q", base)
    (repo / "y.txt").write_text("y\n")
    git(repo, "add", "y.txt")
    git(repo, "commit", "-qm", "main")
    git(repo, "merge", "-q", "--no-commit", "--no-ff", "side")
    (repo / "x.txt").write_text("x\n" + secret + "\n")
    git(repo, "add", "x.txt")
    git(repo, "commit", "-qm", "merge")
    git(repo, "rm", "-q", "x.txt")
    git(repo, "commit", "-qm", "gone")
    res = run(repo, "--history")
    assert_red(res, ":x.txt", secret)


def test_m5_blob_reachable_only_from_tag(repo: Path) -> None:
    secret = FAKES["ghp"]()
    blob = git(repo, "hash-object", "-w", "--stdin", input="k " + secret + "\n")
    git(repo, "tag", "blobtag", blob)
    res = run(repo, "--history")
    assert_red(res, "tag-only blob", secret)
    assert_no_leak(res, secret)


def test_m5_negative_controls(repo: Path) -> None:
    git(repo, "tag", "-a", "v1", "-m", "release notes")
    blob = git(repo, "hash-object", "-w", "--stdin", input="harmless\n")
    git(repo, "tag", "blobtag", blob)
    git(repo, "commit", "-q", "--allow-empty", "-m", "fine")
    res = run(repo, "--history")
    assert res.returncode == 0, res.stderr


# ---- L6: env path ever in history
def test_l6_env_path_in_history_fails_path_only(repo: Path) -> None:
    secret = FAKES["groq"]()
    track(repo, ".env" + ".production", secret + "\n")
    git(repo, "commit", "-qm", "oops")
    git(repo, "rm", "-q", "--cached", ".env" + ".production")
    git(repo, "commit", "-qm", "undo")
    res = run(repo, "--history")
    assert_red(res, ".env.production", secret)
    assert "possible secrets" not in res.stderr


def test_l6_nested_env_path_in_history(repo: Path) -> None:
    track(repo, "svc/.env", "A=1\n")
    git(repo, "commit", "-qm", "oops")
    git(repo, "rm", "-q", "--cached", "svc/.env")
    git(repo, "commit", "-qm", "undo")
    assert_red(run(repo, "--history"), "svc/.env")


def test_l6_env_example_in_history_ok(repo: Path) -> None:
    (repo / "sub").mkdir()
    (repo / "sub" / ENV_EXAMPLE).write_text("A=1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "ex")
    assert run(repo, "--history").returncode == 0


# ---- F-A: fail closed when enumeration errors
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_fa_unreadable_untracked_dir_fails_closed(repo: Path) -> None:
    d = repo / "locked"
    d.mkdir()
    (d / "f.txt").write_text("hi\n")
    d.chmod(0)
    try:
        res = run(repo)
    finally:
        d.chmod(0o755)
    assert_red(res, "cannot enumerate files (fail closed)")


# ---- new patterns / self-test
def test_self_test_covers_new_patterns() -> None:
    res = subprocess.run(["bash", str(SCRIPT), "--self-test"], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    for kind in ["stripe", "stripe-rk", "asia", "pgp", "xapp"]:
        assert f"text {kind} flagged" in res.stdout, kind
        assert f"binary {kind} flagged" in res.stdout, kind
    for case in ["is_env_path .envrc", "is_env_path .env.example", "is_env_path .env/x"]:
        assert case in res.stdout, case


@pytest.mark.parametrize("kind", ["stripe_sk", "stripe_rk", "aws_asia", "pgp", "slack_app"])
def test_new_patterns_in_history_binary(repo: Path, kind: str) -> None:
    secret = FAKES[kind]()
    (repo / "b.bin").write_bytes(b"\x00\x01" + secret.encode() + b"\x00")
    git(repo, "add", "b.bin")
    git(repo, "commit", "-qm", "b")
    git(repo, "rm", "-q", "b.bin")
    git(repo, "commit", "-qm", "c")
    assert_red(run(repo, "--history"), ":b.bin", secret)


# ---- a.2.1: never read .env* content via git objects where the filesystem denies it
SKIP = "(protected path; checked in CI)"


def plant(repo: Path, rel: str = ENV_EXAMPLE, kind: str = "groq") -> tuple[str, str]:
    """Commit a planted key at `rel`; return (secret, blob sha)."""
    secret = FAKES[kind]()
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("NOTE=" + secret + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "plant")
    return secret, git(repo, "rev-parse", "HEAD:" + rel)


def deny(repo: Path, rel: str, how: str) -> None:
    f = repo / rel
    if how == "chmod":
        f.chmod(0)
    else:
        f.unlink()
        f.mkdir()


def undeny(repo: Path, rel: str, how: str) -> None:
    f = repo / rel
    if how == "chmod":
        f.chmod(0o644)


def git_shim(tmp: Path) -> tuple[dict[str, str], Path]:
    """PATH shim that logs every git argv (one line per call) then execs the real git."""
    real = shutil.which("git")
    log = tmp / "git_calls.log"
    bindir = tmp / "shim"
    bindir.mkdir()
    shim = bindir / "git"
    shim.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >>"{log}"\nexec "{real}" "$@"\n')
    shim.chmod(0o755)
    return {"PATH": f"{bindir}:{os.environ['PATH']}"}, log


def run_env(
    repo: Path, extra: dict[str, str], *args: str, ci: bool = False
) -> subprocess.CompletedProcess[str]:
    env = {**base_env(ci), **extra}
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=repo, env=env, capture_output=True, text=True, check=False
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
@pytest.mark.parametrize("how", ["chmod", "dir"])
def test_a21_unreadable_example_index_not_read(repo: Path, tmp_path: Path, how: str) -> None:
    secret, blob = plant(repo)
    deny(repo, ENV_EXAMPLE, how)
    extra, log = git_shim(tmp_path)
    try:
        res = run_env(repo, extra)
    finally:
        undeny(repo, ENV_EXAMPLE, how)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert f"skipped content scan of {ENV_EXAMPLE} {SKIP}" in res.stdout
    assert secret not in res.stdout + res.stderr
    calls = log.read_text()
    assert "ls-files" in calls, "positive control: the shim sees the gate's git calls"
    assert blob not in calls, "gate asked git for the protected blob"
    assert "checked (strict format" not in res.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
@pytest.mark.parametrize("how", ["chmod", "dir"])
def test_a21_unreadable_example_history_not_read(repo: Path, tmp_path: Path, how: str) -> None:
    secret, blob = plant(repo)
    (repo / ENV_EXAMPLE).write_text("A=1\n")
    git(repo, "commit", "-qam", "clean")
    deny(repo, ENV_EXAMPLE, how)
    extra, log = git_shim(tmp_path)
    try:
        res = run_env(repo, extra, "--history")
    finally:
        undeny(repo, ENV_EXAMPLE, how)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert f"skipped content scan of {ENV_EXAMPLE} {SKIP}" in res.stdout
    assert secret not in res.stdout + res.stderr
    logs = [ln for ln in log.read_text().splitlines() if ln.startswith("log ") and " -p " in ln]
    assert logs, "positive control: the content-reading git log was run"
    assert all("exclude" in ln for ln in logs), logs
    assert blob not in log.read_text()


def test_a21_readable_example_still_scanned(repo: Path) -> None:
    """CI behaviour: readable file -> planted key flagged in worktree, index and history."""
    secret, _ = plant(repo)
    res = run(repo)
    assert_red(res, ENV_EXAMPLE + ":1", secret)
    assert "skipped content scan" not in res.stdout
    (repo / ENV_EXAMPLE).write_text("A=1\n")
    git(repo, "commit", "-qam", "clean")
    assert_red(run(repo, "--history"), ENV_EXAMPLE, secret)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_a21_mixed_only_unreadable_path_excluded(repo: Path) -> None:
    """One nested unreadable example is skipped; the readable root one is still scanned."""
    s_nested, _ = plant(repo, "svc/" + ENV_EXAMPLE, "groq")
    s_root = FAKES["hf"]()
    (repo / ENV_EXAMPLE).write_text("NOTE=" + s_root + "\n")
    deny(repo, "svc/" + ENV_EXAMPLE, "chmod")
    try:
        res = run(repo)
    finally:
        undeny(repo, "svc/" + ENV_EXAMPLE, "chmod")
    assert_red(res, ENV_EXAMPLE + ":1", s_root)
    assert "svc/" + ENV_EXAMPLE + ":1" not in res.stderr
    assert f"skipped content scan of svc/{ENV_EXAMPLE} {SKIP}" in res.stdout
    assert s_nested not in res.stdout + res.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_a21_other_checks_still_run_when_skipping(repo: Path) -> None:
    plant(repo)
    deny(repo, ENV_EXAMPLE, "chmod")
    track(repo, ".env" + ".local", "X=1\n")  # tracked env path: path-only failure persists
    other = FAKES["ghp"]()
    (repo / "other.txt").write_text(other + "\n")
    try:
        res = run(repo)
    finally:
        undeny(repo, ENV_EXAMPLE, "chmod")
    assert_red(res, "other.txt:1", other)
    assert ".env" + ".local" in res.stderr
    assert f"skipped content scan of {ENV_EXAMPLE} {SKIP}" in res.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_a21_tag_only_blob_at_protected_path_skipped(repo: Path) -> None:
    secret = FAKES["groq"]()
    blob = (
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=repo,
            input=("NOTE=" + secret + "\n").encode(),
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    tree = git(repo, "mktree", input=f"100644 blob {blob}\t{ENV_EXAMPLE}\n")
    git(repo, "tag", "tonly", tree)
    res = run(repo, "--history")
    assert_red(res, "tag-only blob", secret)  # readable: scanned (positive control)
    deny(repo, ENV_EXAMPLE, "chmod")
    try:
        res = run(repo, "--history")
    finally:
        undeny(repo, ENV_EXAMPLE, "chmod")
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert f"skipped content scan of {ENV_EXAMPLE} {SKIP}" in res.stdout


# ---- a.3: fail closed in CI; symlink/gitlink; no .env inflation; tag/tree coverage
CI_MSG = "protected path in CI"
SYM_MSG = "symlink or gitlink"


def loose(repo: Path, sha: str) -> Path:
    return repo / ".git" / "objects" / sha[:2] / sha[2:]


def blob_sha(repo: Path, text: str) -> str:
    return git(repo, "hash-object", "-w", "--stdin", input=text)


def tag_tree(repo: Path, entries: str, tagname: str = "tonly") -> str:
    tree = git(repo, "mktree", input=entries)
    git(repo, "tag", tagname, tree)
    return tree


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
@pytest.mark.parametrize("how", ["chmod", "dir"])
@pytest.mark.parametrize("mode", ["", "--history"])
@pytest.mark.parametrize("marker", ["CI", "GITHUB_ACTIONS"])
def test_a3_h1_protected_example_fails_in_ci(repo: Path, how: str, mode: str, marker: str) -> None:
    plant(repo)
    deny(repo, ENV_EXAMPLE, how)
    args = (mode,) if mode else ()
    try:
        res = run_env(repo, {marker: "true"}, *args)
    finally:
        undeny(repo, ENV_EXAMPLE, how)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert CI_MSG in res.stderr and "cannot verify" in res.stderr, res.stderr
    assert ENV_EXAMPLE in res.stderr
    assert "skipped content scan" not in res.stdout


@pytest.mark.parametrize("mode", ["", "--history"])
def test_a3_ci_readable_example_still_green(repo: Path, mode: str) -> None:
    args = (mode,) if mode else ()
    res = run(repo, *args, ci=True)
    assert res.returncode == 0, (res.stdout, res.stderr)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_a3_sandbox_still_skips_without_ci(repo: Path) -> None:
    plant(repo)
    deny(repo, ENV_EXAMPLE, "chmod")
    try:
        res = run(repo)
        hist = run(repo, "--history")
    finally:
        undeny(repo, ENV_EXAMPLE, "chmod")
    assert res.returncode == 0 and hist.returncode == 0, (res.stderr, hist.stderr)
    assert f"skipped content scan of {ENV_EXAMPLE} {SKIP}" in res.stdout
    assert f"skipped content scan of {ENV_EXAMPLE} {SKIP}" in hist.stdout


@pytest.mark.parametrize("ci", [False, True])
@pytest.mark.parametrize("mode", ["", "--history"])
def test_a3_symlink_example_after_secret_commit_fails(repo: Path, ci: bool, mode: str) -> None:
    secret, _ = plant(repo)
    (repo / ENV_EXAMPLE).unlink()
    os.symlink("README.md", repo / ENV_EXAMPLE)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "swap for symlink")
    args = (mode,) if mode else ()
    res = run(repo, *args, ci=ci)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert SYM_MSG in res.stderr, res.stderr
    assert secret not in res.stdout + res.stderr


def test_a3_working_tree_symlink_example_fails_even_if_index_regular(repo: Path) -> None:
    (repo / ENV_EXAMPLE).unlink()
    os.symlink("README.md", repo / ENV_EXAMPLE)  # not staged: index still has the regular file
    for args in ((), ("--history",)):
        res = run(repo, *args)
        assert res.returncode == 1, (args, res.stdout, res.stderr)
        assert SYM_MSG in res.stderr


@pytest.mark.parametrize("mode_bits", ["160000", "120000"])
@pytest.mark.parametrize("ci", [False, True])
@pytest.mark.parametrize("mode", ["", "--history"])
def test_a3_index_gitlink_or_symlink_example_fails(
    repo: Path, mode_bits: str, ci: bool, mode: str
) -> None:
    if mode_bits == "160000":
        sha = git(repo, "rev-parse", "HEAD")
    else:
        sha = blob_sha(repo, "README.md")
    git(repo, "update-index", "--add", "--cacheinfo", f"{mode_bits},{sha},sub/{ENV_EXAMPLE}")
    args = (mode,) if mode else ()
    res = run(repo, *args, ci=ci)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert SYM_MSG in res.stderr, res.stderr
    assert f"sub/{ENV_EXAMPLE}" in res.stderr


def test_a3_symlink_example_leaves_other_link_names_alone(repo: Path) -> None:
    """Negative control: only a link NAMED .env.example is rejected."""
    os.symlink(ENV_EXAMPLE, repo / "example-link")
    assert run(repo).returncode == 0
    assert run(repo, "--history").returncode == 0


def test_a3_ci_workflow_runs_secrets_history_before_gate_loop() -> None:
    text = (SCRIPT.parent.parent / ".github" / "workflows" / "ci.yml").read_text()
    hist = text.index("gate_secrets.sh --history")
    loop = text.index("scripts/gate_*.sh")
    sync = text.index("uv sync")
    assert hist < loop and hist < sync, "history scan must precede any repo-code-running step"
    assert text.count("gate_secrets.sh --history") == 1


def log_p_calls(log: Path) -> list[str]:
    return [ln for ln in log.read_text().splitlines() if ln.startswith("log ") and " -p " in ln]


def test_a3_m4_history_log_excludes_env_paths(repo: Path, tmp_path: Path) -> None:
    extra, log = git_shim(tmp_path)
    res = run_env(repo, extra, "--history")
    assert res.returncode == 0, (res.stdout, res.stderr)
    calls = log_p_calls(log)
    assert calls, "positive control: content-reading git log was run"
    base = ".en" + "v"
    main = [c for c in calls if f":(exclude,glob)**/{base}" in c]
    assert main, calls
    for c in main:
        assert f":(exclude,glob)**/{base}.*" in c, c
    # the .env.example content pass is separate and does not carry the .env excludes
    ex_pass = [c for c in calls if f"**/{ENV_EXAMPLE}" in c and "exclude,glob" not in c]
    assert ex_pass, calls


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_a3_m4_unreadable_env_object_does_not_break_history(repo: Path) -> None:
    secret = FAKES["groq"]()
    paths = (".en" + "v", "svc/.en" + "v.local")
    shas = []
    for path in paths:
        sha = blob_sha(repo, "K=" + secret + "\n")
        shas.append(sha)
        git(repo, "update-index", "--add", "--cacheinfo", f"100644,{sha},{path}")
    git(repo, "commit", "-qm", "oops")
    git(repo, "rm", "-q", "--cached", *paths)
    git(repo, "commit", "-qm", "undo")
    for sha in shas:
        loose(repo, sha).chmod(0)
    try:
        res = run(repo, "--history")
    finally:
        for sha in shas:
            loose(repo, sha).chmod(0o444)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "git log -p failed" not in res.stderr, res.stderr
    assert "unreadable" not in res.stderr, res.stderr
    assert "svc/.en" + "v.local" in res.stderr  # path-only L record
    assert secret not in res.stdout + res.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_a3_l5_protected_blob_not_touched_by_cat_file(repo: Path, tmp_path: Path) -> None:
    secret = FAKES["groq"]()
    blob = blob_sha(repo, "NOTE=" + secret + "\n")
    tag_tree(repo, f"100644 blob {blob}\t{ENV_EXAMPLE}\n")
    deny(repo, ENV_EXAMPLE, "chmod")
    extra, log = git_shim(tmp_path)
    try:
        res = run_env(repo, extra, "--history")
    finally:
        undeny(repo, ENV_EXAMPLE, "chmod")
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert log.exists(), "positive control: shim saw calls"
    assert blob not in log.read_text(), "cat-file touched the protected blob (even -t)"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 000")
def test_a3_l5_unreadable_object_fails_closed(repo: Path) -> None:
    blob = blob_sha(repo, "harmless\n")
    tag_tree(repo, f"100644 blob {blob}\tnotes.txt\n")
    loose(repo, blob).chmod(0)
    try:
        res = run(repo, "--history")
    finally:
        loose(repo, blob).chmod(0o444)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "cannot read object" in res.stderr, res.stderr


def mk_tag_of(repo: Path, target: str, ttype: str, name: str, message: str) -> str:
    body = (
        f"object {target}\ntype {ttype}\ntag {name}\n"
        f"tagger t <t@example.invalid> 1700000000 +0000\n\n{message}\n"
    )
    return git(repo, "mktag", input=body)


def test_a3_l6_nested_annotated_tag_message_scanned(repo: Path) -> None:
    secret = FAKES["ghp"]()
    head = git(repo, "rev-parse", "HEAD")
    inner = mk_tag_of(repo, head, "commit", "inner", "release " + secret)
    outer = mk_tag_of(repo, inner, "tag", "outer", "clean message")
    git(repo, "update-ref", "refs/tags/outer", outer)
    res = run(repo, "--history")
    assert_red(res, "tag message", secret)
    assert_no_leak(res, secret)


def test_a3_l6_nested_tag_negative_control(repo: Path) -> None:
    head = git(repo, "rev-parse", "HEAD")
    inner = mk_tag_of(repo, head, "commit", "inner", "fine")
    outer = mk_tag_of(repo, inner, "tag", "outer", "also fine")
    git(repo, "update-ref", "refs/tags/outer", outer)
    assert run(repo, "--history").returncode == 0


def test_a3_l7_env_path_in_tag_only_tree_fails(repo: Path) -> None:
    blob = blob_sha(repo, "A=1\n")
    tag_tree(repo, f"100644 blob {blob}\t.en" + "v\n")
    res = run(repo, "--history")
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert ".en" + "v" in res.stderr and "history" in res.stderr


def test_a3_l7_nested_env_local_in_tag_only_tree_fails(repo: Path) -> None:
    blob = blob_sha(repo, "A=1\n")
    sub = git(repo, "mktree", input=f"100644 blob {blob}\t.en" + "v.local\n")
    tag_tree(repo, f"040000 tree {sub}\tsvc\n")
    assert_red(run(repo, "--history"), "svc/.en" + "v.local")


def test_a3_l7_env_named_directory_is_not_an_env_file(repo: Path) -> None:
    blob = blob_sha(repo, "A=1\n")
    sub = git(repo, "mktree", input=f"100644 blob {blob}\tx\n")
    tag_tree(repo, f"040000 tree {sub}\t.en" + "v\n")
    res = run(repo, "--history")
    assert res.returncode == 0, (res.stdout, res.stderr)


def test_a3_l8_generic_pattern_path_redacted(repo: Path) -> None:
    name = "api_" + "key=" + "a" * 20 + ".txt"
    (repo / name).write_text(FAKES["groq"]() + "\n")
    res = run(repo)
    assert res.returncode == 1
    assert "<redacted-path>" in res.stderr
    assert "a" * 20 not in res.stderr


def test_a3_l8_commit_message_generic_pattern(repo: Path) -> None:
    secret = FAKES["json_password"]()
    git(repo, "commit", "-q", "--allow-empty", "-m", "oops " + secret)
    res = run(repo, "--history")
    assert_red(res, "commit message", "ab.cd/ef+gh=ij-klmnop")


def test_a3_l8_tag_message_generic_pattern(repo: Path) -> None:
    secret = FAKES["plain_apikey"]()
    git(repo, "tag", "-a", "v1", "-m", "release " + secret)
    res = run(repo, "--history")
    assert_red(res, "tag message", "a" * 20)


# ---------------------------------------------------------------- unit a.3.1
# R1: the generic pattern (and bearer/url) is case-insensitive; vendor prefixes keep exact case.
UPPER_GENERIC = {
    "db_password": lambda: "DB_PASS" + "WORD=" + "hunter2" * 3,
    "api_key_colon": lambda: "API_" + "KEY: " + "abcdefghijklmnopqrstuv",
    "mixed_secret": lambda: "My_Sec" + 'ret = "' + "Ab1" * 6 + '"',
    "upper_token_json": lambda: '{"TO' + 'KEN": "' + "a-b" * 6 + '"}',
    "upper_bearer": lambda: "AUTHORIZATION" + ": BEARER " + rep("a", 24),
}


@pytest.mark.parametrize("kind", sorted(UPPER_GENERIC))
def test_a31_r1_upper_case_generic_untracked(repo: Path, kind: str) -> None:
    secret = UPPER_GENERIC[kind]()
    (repo / "notes.txt").write_text(f"one\n{secret}\n")
    assert_red(run(repo), "notes.txt:2", secret)


@pytest.mark.parametrize("kind", sorted(UPPER_GENERIC))
def test_a31_r1_upper_case_generic_index_only(repo: Path, kind: str) -> None:
    secret = UPPER_GENERIC[kind]()
    (repo / "leak.txt").write_text(secret + "\n")
    git(repo, "add", "leak.txt")
    (repo / "leak.txt").write_text("clean now\n")
    assert_red(run(repo), "leak.txt:1", secret)


@pytest.mark.parametrize("kind", sorted(UPPER_GENERIC))
def test_a31_r1_upper_case_generic_commit_message(repo: Path, kind: str) -> None:
    secret = UPPER_GENERIC[kind]()
    git(repo, "commit", "-q", "--allow-empty", "-m", "oops " + secret)
    assert_red(run(repo, "--history"), "commit message", secret)


def test_a31_r1_upper_case_generic_tag_message(repo: Path) -> None:
    secret = UPPER_GENERIC["db_password"]()
    git(repo, "tag", "-a", "v1", "-m", "release " + secret)
    assert_red(run(repo, "--history"), "tag message", "hunter2")


def test_a31_r1_upper_case_generic_path_redacted(repo: Path) -> None:
    name = "API_" + "KEY=" + "a" * 20 + ".txt"
    (repo / name).write_text(FAKES["groq"]() + "\n")
    res = run(repo)
    assert res.returncode == 1
    assert "<redacted-path>" in res.stderr
    assert "a" * 20 not in res.stderr


def test_a31_r1_upper_case_generic_in_env_example(repo: Path) -> None:
    (repo / ENV_EXAMPLE).write_text("# DB_PASS" + "WORD=" + "hunter2" * 3 + "\nDEBUG=1\n")
    assert_red(run(repo), ENV_EXAMPLE + ":1", "hunter2")


def test_a31_r1_upper_case_short_value_still_clean(repo: Path) -> None:
    (repo / "c.txt").write_text("DB_PASS" + "WORD=short\n")
    assert run(repo).returncode == 0


def test_a31_r1_vendor_prefix_keeps_exact_case(repo: Path) -> None:
    """Vendor prefixes are case-specific: an upper-cased lookalike is not a vendor key."""
    (repo / "c.txt").write_text(("GSK_" + rep("x", 24)) + "\n")
    assert run(repo).returncode == 0


# R2: skip blobs equal to the working-tree file, never touch protected paths
def test_a31_r2_perf_500_files(tmp_path: Path) -> None:
    import time

    r = tmp_path / "big"
    r.mkdir()
    git(r, "init", "-q")
    for i in range(500):
        d = r / f"d{i % 20}"
        d.mkdir(exist_ok=True)
        (d / f"f{i}.txt").write_text(f"file {i}\nplain content\n" * 5)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "big")
    t0 = time.monotonic()
    res = run(r)
    dt = time.monotonic() - t0
    assert res.returncode == 0, res.stderr
    assert dt < 8, f"gate took {dt:.1f}s on 500 files"  # target <= 6 s; margin for CI noise


def test_a31_r2_blob_equal_to_worktree_hit_reported_once(repo: Path) -> None:
    secret = FAKES["ghp"]()
    (repo / "same.txt").write_text("x\n" + secret + "\n")
    git(repo, "add", "same.txt")
    res = run(repo)
    assert_red(res, "same.txt:2", secret)
    assert res.stderr.count("same.txt:2") == 1


def test_a31_r2_staged_differs_from_worktree_still_scanned_among_many(repo: Path) -> None:
    for i in range(5):
        (repo / f"ok{i}.txt").write_text(f"fine {i}\n")
    secret = FAKES["json_password"]()
    (repo / "mid.json").write_text(secret + "\n")
    git(repo, "add", "-A")
    (repo / "mid.json").write_text("{}\n")
    res = run(repo)
    assert_red(res, "mid.json:1", secret)
    assert "ok0.txt" not in res.stderr


def test_a31_r2_protected_path_never_hashed_or_read(repo: Path, tmp_path: Path) -> None:
    """Shim `git`: no hash-object / cat-file call is ever aimed at a .env.local path or its blob."""
    protected = ".env" + ".local"
    track(repo, protected, FAKES["groq"]() + "\n")
    blob = git(repo, "ls-files", "-s", "--", protected).split()[1]
    log = tmp_path / "calls.log"
    shim = tmp_path / "shim"
    shim.mkdir()
    real_git = shutil.which("git")
    (shim / "git").write_text(f'#!/bin/bash\necho "git $*" >> "{log}"\nexec "{real_git}" "$@"\n')
    (shim / "git").chmod(0o755)
    env = base_env()
    env["PATH"] = f"{shim}:{env['PATH']}"
    res = subprocess.run(
        ["bash", str(SCRIPT)], cwd=repo, env=env, capture_output=True, text=True, check=False
    )
    assert res.returncode == 1
    calls = log.read_text()
    assert blob not in calls
    assert protected not in calls.replace("ls-files", "")
    assert FAKES["groq"]() not in res.stdout + res.stderr


# R3: the env-path classification (strings only: an untracked .env.* file cannot be created here)
def test_a31_r3_env_path_names_classified_path_only() -> None:
    script = (
        f'eval "$(sed -n "/^ENV_RE=/p;/^EX_NAME=/p;/^is_env_path()/,/^}}/p" "{SCRIPT}")"; '
        'for p in .env.local sub/.env.production .env; do is_env_path "$p" && echo "env:$p"; done; '
        'for p in .envrc sub/.env.example x.env; do is_env_path "$p" || echo "plain:$p"; done'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert out.split() == [
        "env:.env.local",
        "env:sub/.env.production",
        "env:.env",
        "plain:.envrc",
        "plain:sub/.env.example",
        "plain:x.env",
    ]


# ---------------------------------------------------------------- unit H-sec (S1, S2)
def hs_shim(tmp: Path) -> tuple[dict[str, str], Path, Path]:
    """PATH shim: logs every git argv; for hash-object also tees the stdin (the path list)."""
    real = shutil.which("git")
    bindir = tmp / "hshim"
    bindir.mkdir()
    calls, stdin_log = tmp / "hcalls.log", tmp / "hstdin.log"
    (bindir / "git").write_text(
        "#!/bin/bash\n"
        f'echo "git $*" >> "{calls}"\n'
        'if [ "$1" = hash-object ]; then\n'
        f'  tee -a "{stdin_log}" | "{real}" "$@"\n'
        "  exit ${PIPESTATUS[1]}\n"
        "fi\n"
        f'exec "{real}" "$@"\n'
    )
    (bindir / "git").chmod(0o755)
    return {"PATH": f"{bindir}:{os.environ['PATH']}"}, calls, stdin_log


def decoy_setup(repo: Path, index_name: str, decoy: str) -> str:
    """Index holds a secret blob under `index_name`; the working tree has a CLEAN file of that
    exact name plus an ignored `decoy` carrying the same secret (what hash-object would open
    if it mangled the name). Returns the secret."""
    secret = FAKES["ghp"]() + "\n"
    track(repo, index_name, secret)
    (repo / index_name).write_text("clean\n")
    (repo / decoy).write_text(secret)
    (repo / ".gitignore").write_text(decoy + "\n")
    return secret.strip()


def test_hsec_s1_quoted_index_name_scans_index_blob(repo: Path, tmp_path: Path) -> None:
    secret = decoy_setup(repo, '"x.txt"', "x.txt")
    extra, calls, stdin_log = hs_shim(tmp_path)
    res = run_env(repo, extra)
    assert_red(res, "x.txt", secret)
    assert not stdin_log.exists() or '"' not in stdin_log.read_text()


def test_hsec_s1_cr_suffixed_name_scans_index_blob(repo: Path, tmp_path: Path) -> None:
    secret = decoy_setup(repo, "y.txt\r", "y.txt")
    extra, calls, stdin_log = hs_shim(tmp_path)
    res = run_env(repo, extra)
    assert_red(res, "y.txt", secret)
    assert not stdin_log.exists() or "\r" not in stdin_log.read_text()


@pytest.mark.parametrize("name", ["b\\s.txt", " lead.txt", "trail.txt ", "café.txt"])
def test_hsec_s1_unusual_names_never_hashed(repo: Path, tmp_path: Path, name: str) -> None:
    track(repo, name, "plain\n")
    (repo / name).write_text("plain\n")
    track(repo, "ok-plain.txt", "plain two\n")
    (repo / "ok-plain.txt").write_text("plain two\n")
    extra, calls, stdin_log = hs_shim(tmp_path)
    res = run_env(repo, extra)
    assert res.returncode == 0, (res.stdout, res.stderr)
    logged = stdin_log.read_text().splitlines()
    assert "ok-plain.txt" in logged, "positive control: hash-object saw the plain path"
    assert name not in logged


def test_hsec_s1_hash_object_never_given_quote_path(repo: Path, tmp_path: Path) -> None:
    """A name that C-unquotes to the protected .env.local must never reach hash-object."""
    protected = ".env" + ".local"
    track(repo, '"' + protected + '"', "plain\n")
    track(repo, "ok-plain.txt", "plain two\n")
    (repo / "ok-plain.txt").write_text("plain two\n")
    extra, calls, stdin_log = hs_shim(tmp_path)
    res = run_env(repo, extra)
    assert res.returncode == 0, (res.stdout, res.stderr)
    seen = stdin_log.read_text()
    assert "ok-plain.txt" in seen, "positive control"
    assert '"' not in seen and protected not in seen


def test_hsec_s1_identical_clean_blob_is_skipped(repo: Path, tmp_path: Path) -> None:
    """Functional guard for the skip itself (not only the perf test): the blob of a file that
    equals its index copy is never cat-file'd; one that differs on disk is (positive control)."""
    (repo / "same.txt").write_text("identical unique content\n")
    (repo / "diff.txt").write_text("staged version\n")
    git(repo, "add", "-A")
    (repo / "diff.txt").write_text("edited on disk\n")
    same = git(repo, "ls-files", "-s", "--", "same.txt").split()[1]
    diff = git(repo, "ls-files", "-s", "--", "diff.txt").split()[1]
    extra, calls, _ = hs_shim(tmp_path)
    res = run_env(repo, extra)
    assert res.returncode == 0, (res.stdout, res.stderr)
    log = calls.read_text()
    assert f"cat-file blob {diff}" in log, "positive control: differing blob is read"
    assert f"cat-file blob {same}" not in log, "identical blob was not skipped"


def test_hsec_s2_history_record_prints_unquoted_path(repo: Path) -> None:
    secret = FAKES["ghp"]()
    (repo / "café.txt").write_text(secret + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "add")
    git(repo, "rm", "-q", "-f", "--cached", "café.txt")
    (repo / "café.txt").unlink()
    git(repo, "commit", "-qm", "rm")
    res = run(repo, "--history")
    assert res.returncode == 1
    assert "café.txt" in res.stderr, res.stderr
    assert "diff --git" not in res.stderr and "\\303" not in res.stderr and 'b/"' not in res.stderr
    assert secret not in res.stdout + res.stderr


def test_hsec_s2_history_record_quote_in_name(repo: Path) -> None:
    secret = FAKES["ghp"]()
    (repo / 'a"b.txt').write_text(secret + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "add")
    res = run(repo, "--history")
    assert res.returncode == 1
    assert 'a"b.txt' in res.stderr, res.stderr
    assert "diff --git" not in res.stderr and '\\"' not in res.stderr
    assert secret not in res.stdout + res.stderr
