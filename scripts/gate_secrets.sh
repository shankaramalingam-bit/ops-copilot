#!/usr/bin/env bash
# Gate: secrets. Scans the working tree (tracked + untracked-not-ignored) AND the git index
# for secret patterns. Prints ONLY file:line (or commit:file, or a variable name), never
# matched text.
# Usage: gate_secrets.sh [--self-test | --history]
#   (none)       scan working tree + index, check tracked .env files, check .env.example
#   --history    scan every commit's diff (git log -p --all) with the vendor patterns
#   --self-test  prove each pattern fires on a runtime-built fake and not on clean input
# Binary files are scanned with the vendor-prefix patterns only (generic pattern is noisy).
# .env* paths are never read (protected); they are skipped explicitly.
# Symlinks are never followed. A symlink to a .env* target (other than .env.example) fails.
# Read errors fail CLOSED.
set -euo pipefail
export LC_ALL=C # byte semantics: binary files must not break cut/sed
cd "$(git rev-parse --show-toplevel)"

# Patterns are assembled from fragments so this file never matches its own scan.
P_GROQ='gsk_[A-Za-z0-9]{20,}'
P_SK='sk-[A-Za-z0-9_-]{20,}'
P_AWS='AKIA[0-9A-Z]{16}'
P_PEM='-----BEGIN [A-Z ]*PRIVATE KEY-----'
P_GH='gh[pos]_[A-Za-z0-9]{20,}'
P_GHPAT='github_pat_[A-Za-z0-9_]{20,}'
P_SLACK='xox[abprs]-[A-Za-z0-9-]{10,}'
P_GOOG='AIza[0-9A-Za-z_-]{35}'
P_HF='hf_[A-Za-z0-9]{30,}'
P_LS='lsv2_[A-Za-z0-9_]{20,}'
# Vendor-prefix patterns: safe on binary files and in history.
VENDOR="($P_GROQ|$P_SK|$P_AWS|$P_PEM|$P_GH|$P_GHPAT|$P_SLACK|$P_GOOG|$P_HF|$P_LS)"
# Text-only patterns.
P_BEARER='Authorization:[[:space:]]*Bearer[[:space:]]+[A-Za-z0-9._~+/=-]{20,}'
P_URL='://[^/[:space:]:@]+:[^/[:space:]@]+@'
P_GEN='(api[_-]?key|secret|token|password)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"']?[A-Za-z0-9_./+=-]{16,}'
PATTERN="($VENDOR|$P_BEARER|$P_URL|$P_GEN)"

# Pathspec excluding protected .env* paths (never read, never scanned).
EXCL=(':(exclude,glob).env*' ':(exclude,glob)**/.env*')

# is_env_target <link-target>: true if its basename is .env or .env.* but not .env.example.
is_env_target() {
  local t="$1" base
  while [ "${t%/}" != "$t" ]; do t="${t%/}"; done
  base="${t##*/}"
  case "$base" in
    .env.example) return 1 ;;
    .env | .env.*) return 0 ;;
  esac
  return 1
}

# scan_files: reads NUL-separated paths on stdin. Emits tagged records, never matched text:
#   H|file:line   possible secret      E|file   unreadable (fail closed)
#   S|file        symlink to a .env* target
scan_files() {
  local f rc out t
  while IFS= read -r -d '' f; do
    case "$f" in
      .env | .env.* | */.env | */.env.*) continue ;; # protected path, never read
    esac
    if [ -L "$f" ]; then
      t="$(readlink "$f" || true)"
      if is_env_target "$t"; then echo "S|$f"; fi
      continue # never follow symlinks
    fi
    [ -e "$f" ] || continue # tracked but deleted on disk: index scan still covers it
    [ -f "$f" ] || continue # directories (submodules), sockets, etc.
    if [ ! -r "$f" ]; then echo "E|$f"; continue; fi
    rc=0
    out="$(grep -IEinH -e "$PATTERN" -- "$f" 2>/dev/null | cut -d: -f1,2)" || rc=$?
    [ "$rc" -le 1 ] || { echo "E|$f"; continue; }
    [ -z "$out" ] || printf 'H|%s\n' "$out"
    rc=0
    out="$(grep -aEnH -e "$VENDOR" -- "$f" 2>/dev/null | cut -d: -f1,2)" || rc=$?
    [ "$rc" -le 1 ] || { echo "E|$f"; continue; }
    [ -z "$out" ] || printf 'H|%s\n' "$out"
  done
}

# index_scan: scans staged content. Emits the same tagged records as scan_files.
index_scan() {
  local rc out rec meta path mode sha t
  rc=0
  out="$(git grep --cached -I -nE -e "$PATTERN" -- . "${EXCL[@]}" 2>/dev/null | cut -d: -f1,2)" || rc=$?
  [ "$rc" -le 1 ] || echo "E|(index: text scan error)"
  [ -z "$out" ] || printf 'H|%s\n' "$out"
  out=""
  rc=0
  out="$(git grep --cached -a -nE -e "$VENDOR" -- . "${EXCL[@]}" 2>/dev/null | cut -d: -f1,2)" || rc=$?
  [ "$rc" -le 1 ] || echo "E|(index: binary scan error)"
  [ -z "$out" ] || printf 'H|%s\n' "$out"
  # Tracked symlinks whose target is a .env* file (read from the blob, never followed).
  while IFS= read -r -d '' rec; do
    meta="${rec%%$'\t'*}"
    path="${rec#*$'\t'}"
    mode="${meta%% *}"
    [ "$mode" = "120000" ] || continue
    sha="$(printf '%s' "$meta" | awk '{print $2}')"
    t="$(git cat-file blob "$sha" 2>/dev/null || true)"
    if is_env_target "$t"; then echo "S|$path"; fi
  done < <(git ls-files -s -z)
}

# history_scan: vendor patterns over every added line in git log -p --all. Prints commit:file.
history_scan() {
  git rev-parse --verify -q HEAD >/dev/null 2>&1 || return 0
  git log -p --all --no-color --no-ext-diff --format='commit %H' -- . "${EXCL[@]}" |
    grep -aE -e "^(commit [0-9a-f]+|diff --git .*|\\+.*$VENDOR.*)$" |
    awk '
      /^commit [0-9a-f]+$/ { c = substr($2, 1, 12); next }
      /^diff --git / { f = $0; sub(/^diff --git a\/.* b\//, "", f); next }
      /^\+\+\+ / { next }
      /^\+/ { k = c ":" f; if (!(k in seen)) { seen[k] = 1; print k } }
    '
}

# fake <kind>: runtime-built fake credential (nothing key-shaped exists in source).
rep() { printf "%${2}s" '' | tr ' ' "$1"; }
fake() {
  local dash=----- colon=: sl=//
  case "$1" in
    groq) printf 'gsk_%s' "$(rep x 24)" ;;
    sk) printf 'sk-%s' "$(rep x 24)" ;;
    aws) printf 'AKIA%s' "$(rep A 16)" ;;
    pem) printf '%sBEGIN RSA PRIVATE KEY%s' "$dash" "$dash" ;;
    ghp) printf 'ghp_%s' "$(rep a 36)" ;;
    gho) printf 'gho_%s' "$(rep a 36)" ;;
    ghs) printf 'ghs_%s' "$(rep a 36)" ;;
    ghpat) printf 'github_pat_%s' "$(rep a 30)" ;;
    slack) printf 'xoxb-%s-%s' "$(rep 1 12)" "$(rep a 12)" ;;
    google) printf 'AIza%s' "$(rep a 35)" ;;
    hf) printf 'hf_%s' "$(rep a 32)" ;;
    lsv2) printf 'lsv2_pt_%s' "$(rep a 24)" ;;
    bearer) printf 'Authorization%s Bearer %s' "$colon" "$(rep a 24)" ;;
    url) printf 'https%s%su%sp%sh.example' "$colon" "$sl" "$colon" "@" ;;
    gen-json) printf '{"%s": "ab.cd/ef+gh=ij-klmnop"}' "pass""word" ;;
    gen-plain) printf 'api_key = "%s"' "$(rep a 20)" ;;
  esac
}

KINDS="groq sk aws pem ghp gho ghs ghpat slack google hf lsv2"
TEXT_KINDS="bearer url gen-json gen-plain"

if [ "${1:-}" = "--self-test" ]; then
  d="$(mktemp -d "${TMPDIR:-/tmp}/gate_secrets.XXXXXX")"
  trap 'rm -rf "$d"' EXIT
  ok=1
  expect() { # expect <label> <want> <got>
    if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1 (want '$2' got '$3')" >&2; ok=0; fi
  }
  for k in $KINDS $TEXT_KINDS; do
    printf 'x = %s\n' "$(fake "$k")" >"$d/$k.txt"
    hits="$(printf '%s\0' "$d/$k.txt" | scan_files | sort -u)"
    expect "text $k flagged" "H|$d/$k.txt:1" "$hits"
  done
  printf 'nothing to see here\n' >"$d/clean.txt"
  expect "clean file not flagged" "" "$(printf '%s\0' "$d/clean.txt" | scan_files)"
  for k in $KINDS; do
    { printf '\211PNG\0\0\0'; fake "$k"; printf '\0\0\n'; } >"$d/$k.bin"
    hits="$(printf '%s\0' "$d/$k.bin" | scan_files | sort -u)"
    expect "binary $k flagged" "H|$d/$k.bin:1" "$hits"
  done
  { printf '\211PNG\0\0\0'; fake gen-plain; printf '\0\n'; } >"$d/generic.bin"
  expect "binary generic not flagged" "" "$(printf '%s\0' "$d/generic.bin" | scan_files)"
  ln -s ../.env "$d/link-env"
  expect "symlink to .env flagged" "S|$d/link-env" "$(printf '%s\0' "$d/link-env" | scan_files)"
  ln -s ../.env.example "$d/link-ex"
  expect "symlink to .env.example allowed" "" "$(printf '%s\0' "$d/link-ex" | scan_files)"
  ln -s "$d/planted.txt" "$d/link-plant"
  expect "symlink not followed" "" "$(printf '%s\0' "$d/link-plant" | scan_files)"
  if [ "$ok" -eq 1 ]; then
    echo "gate_secrets self-test: PASS (every pattern flagged, clean/generic-binary/symlink not flagged)"
    exit 0
  fi
  echo "gate_secrets self-test: FAIL (positive/negative control mismatch)" >&2
  exit 1
fi

if [ "${1:-}" = "--history" ]; then
  hits="$(history_scan)"
  if [ -n "$hits" ]; then
    echo "gate_secrets --history: FAIL possible secrets at (commit:file):" >&2
    echo "$hits" >&2
    exit 1
  fi
  echo "gate_secrets --history: OK"
  exit 0
fi

if [ -n "${1:-}" ]; then
  echo "usage: gate_secrets.sh [--self-test | --history]" >&2
  exit 2
fi

fail=0

# Tracked .env files other than .env.example are forbidden.
bad_env="$(git ls-files | grep -E '(^|/)\.env(\..*)?$' | grep -Ev '(^|/)\.env\.example$' || true)"
if [ -n "$bad_env" ]; then
  echo "gate_secrets: FAIL tracked env file(s):" >&2
  echo "$bad_env" >&2
  fail=1
fi

records="$({ { git ls-files -z; git ls-files -z --others --exclude-standard; } | sort -zu | scan_files; index_scan; } | sort -u)"
hits="$(printf '%s\n' "$records" | sed -n 's/^H|//p')"
errs="$(printf '%s\n' "$records" | sed -n 's/^E|//p')"
syms="$(printf '%s\n' "$records" | sed -n 's/^S|//p')"
if [ -n "$hits" ]; then
  echo "gate_secrets: FAIL possible secrets at (file:line):" >&2
  echo "$hits" >&2
  fail=1
fi
if [ -n "$errs" ]; then
  echo "gate_secrets: FAIL unreadable file(s) (fail closed):" >&2
  echo "$errs" >&2
  fail=1
fi
if [ -n "$syms" ]; then
  echo "gate_secrets: FAIL symlink(s) to .env* target:" >&2
  echo "$syms" >&2
  fail=1
fi

# .env.example: when readable (CI), secret-named variables must have empty values.
# Only variable NAMES are ever printed. When unreadable (agent sandbox), say so explicitly.
if [ -L .env.example ]; then
  echo "gate_secrets: skipped content scan of .env.example (symlink, not followed)"
elif [ -e .env.example ] && content="$(cat .env.example 2>/dev/null)"; then
  bad_vars=""
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in '' | '#'*) continue ;; esac
    line="${line#export }"
    case "$line" in *=*) ;; *) continue ;; esac
    name="${line%%=*}"
    val="${line#*=}"
    name="$(printf '%s' "$name" | tr -d '[:space:]')"
    up="$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"
    case "$up" in *KEY* | *TOKEN* | *SECRET* | *PASSWORD*) ;; *) continue ;; esac
    val="${val#"${val%%[![:space:]]*}"}"
    case "$val" in '#'*) val="" ;; esac
    val="$(printf '%s' "$val" | tr -d "\"'[:space:]")"
    if [ -n "$val" ]; then bad_vars="$bad_vars$name"$'\n'; fi
  done <<<"$content"
  if [ -n "$bad_vars" ]; then
    echo "gate_secrets: FAIL .env.example has non-empty value for variable(s) (names only):" >&2
    printf '%s' "$bad_vars" >&2
    fail=1
  else
    echo "gate_secrets: .env.example checked (secret-named variables are empty)"
  fi
else
  echo "gate_secrets: skipped content scan of .env.example (protected path; owner-verified)"
fi

if [ "$fail" -ne 0 ]; then exit 1; fi
echo "gate_secrets: OK"
