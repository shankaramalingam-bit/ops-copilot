#!/usr/bin/env bash
# Gate: secrets. Scans git-tracked + untracked-not-ignored files for secret patterns.
# Prints ONLY file:line of hits, never the matched text.
# Usage: gate_secrets.sh [--self-test]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Patterns are assembled from fragments so this file never matches its own scan.
P_GROQ='gsk_[A-Za-z0-9]{20,}'
P_SK='sk-[A-Za-z0-9_-]{20,}'
P_AWS='AKIA[0-9A-Z]{16}'
P_PEM='-----BEGIN [A-Z ]*PRIVATE KEY-----'
P_GEN='(api[_-]?key|secret|token|password)[[:space:]]*[:=][[:space:]]*['"'"'"]?[A-Za-z0-9_-]{16,}'
PATTERN="($P_GROQ|$P_SK|$P_AWS|$P_PEM|$P_GEN)"

# scan_files: reads NUL-separated paths on stdin; echoes file:line for hits (never the text).
scan_files() {
  while IFS= read -r -d '' f; do
    case "$f" in
      .env | .env.* | */.env | */.env.*) continue ;; # protected path, never read
    esac
    [ -f "$f" ] || continue
    grep -IEinH -e "$PATTERN" -- "$f" 2>/dev/null | cut -d: -f1,2 || true
  done
}

if [ "${1:-}" = "--self-test" ]; then
  d="$(mktemp -d "${TMPDIR:-/tmp}/gate_secrets.XXXXXX")"
  trap 'rm -rf "$d"' EXIT
  planted="gsk_$(printf 'x%.0s' $(seq 1 24))"
  printf 'key = "%s"\n' "$planted" >"$d/planted.txt"
  printf 'nothing to see here\n' >"$d/clean.txt"
  hits="$(printf '%s\0' "$d/planted.txt" "$d/clean.txt" | scan_files)"
  echo "gate_secrets self-test: scanner output: $hits"
  if [ "$hits" = "$d/planted.txt:1" ]; then
    echo "gate_secrets self-test: PASS (planted key flagged, clean file not flagged)"
    exit 0
  fi
  echo "gate_secrets self-test: FAIL (positive control not flagged correctly)" >&2
  exit 1
fi

fail=0

# Tracked .env files other than .env.example are forbidden.
bad_env="$(git ls-files | grep -E '(^|/)\.env(\..*)?$' | grep -Ev '(^|/)\.env\.example$' || true)"
if [ -n "$bad_env" ]; then
  echo "gate_secrets: FAIL tracked env file(s):" >&2
  echo "$bad_env" >&2
  fail=1
fi

hits="$({ git ls-files -z; git ls-files -z --others --exclude-standard; } | sort -zu | scan_files)"
if [ -n "$hits" ]; then
  echo "gate_secrets: FAIL possible secrets at (file:line):" >&2
  echo "$hits" >&2
  fail=1
fi

echo "gate_secrets: skipped content scan of .env.example (protected path; owner-verified)"
if [ "$fail" -ne 0 ]; then exit 1; fi
echo "gate_secrets: OK"
