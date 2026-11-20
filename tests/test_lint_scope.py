"""The ruff scope must exclude agent worktrees but never tracked repo content. Offline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _tracked(*patterns: str) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", *patterns], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return {line for line in out.splitlines() if line}


def test_worktrees_ignored_and_excluded() -> None:
    assert ".claude/worktrees/" in (ROOT / ".gitignore").read_text().splitlines()
    cfg = (ROOT / "pyproject.toml").read_text()
    assert "extend-exclude" in cfg and ".claude/worktrees" in cfg


def test_every_tracked_python_file_is_in_ruff_scope() -> None:
    tracked = _tracked("*.py")
    assert tracked, "positive control: no tracked .py files found"
    listed = _ruff("check", "--show-files", ".")
    assert listed.returncode == 0, listed.stderr
    in_scope = {str(Path(p).resolve().relative_to(ROOT)) for p in listed.stdout.splitlines() if p}
    assert not any(p.startswith(".claude/worktrees/") for p in in_scope)
    missing = tracked - in_scope
    assert not missing, f"tracked files excluded from ruff: {sorted(missing)}"


def test_format_gate_still_catches_repo_files() -> None:
    probe = ROOT / "tests" / "_tmp_lint_probe.py"
    try:
        probe.write_text("x  =  {'a':1}\n", encoding="utf-8")
        res = _ruff("format", "--check", "tests/_tmp_lint_probe.py")
        assert res.returncode != 0, "negative control failed: bad file was not flagged"
        assert "_tmp_lint_probe.py" in res.stdout + res.stderr
    finally:
        probe.unlink(missing_ok=True)
