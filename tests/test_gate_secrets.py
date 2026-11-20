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


def run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # hermetic: a user-level global gitignore (e.g. ".env*") must not hide test files
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
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


def run_env(repo: Path, extra: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        **extra,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
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
