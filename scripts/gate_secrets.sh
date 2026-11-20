#!/usr/bin/env bash
# Gate: secrets. Scans the working tree (tracked + untracked-not-ignored) AND the git index
# for secret patterns. Prints ONLY file:line (or commit:file, or a variable name), never
# matched text. A path that itself matches a vendor pattern is printed as <redacted-path>.
# Usage: gate_secrets.sh [--self-test | --history]
#   (none)       scan working tree + index, check tracked .env files, check every .env.example
#   --history    scan history: every commit's diff (git log -p --text --all, merges against
#                the first parent) with `.env` / `.env.*` paths EXCLUDED by pathspec (their content
#                is never asked of git) and `.env.example` scanned in its own pass, commit
#                messages, every annotated tag message (nested tags peeled), blobs reachable only
#                from tags, plus a path-only check for .env* files that ever existed (commits AND
#                trees reachable only from tags)
#   --self-test  prove each pattern fires on a runtime-built fake and not on clean input
# Binary files are scanned with the vendor-prefix patterns only (generic pattern is noisy).
#
# .env handling (ONE definition, is_env_path / ENV_RE): a path whose basename is exactly
# `.env` or matches `.env.*`, and is not `.env.example`, is a protected env file: its content
# is never read (working tree, index, history: git log -p excludes them by pathspec) and, if
# tracked or ever committed, it FAILS.
# `.envrc`, `.env-notes.md`, `.environment`, files under a `.env/` dir are scanned normally.
# EVERY `.env.example` (any depth) is content-scanned (all patterns) and must be well formed:
# lines are NAME=VALUE (NAME matches [A-Za-z_][A-Za-z0-9_]*, optional `export `), values of
# KEY/TOKEN/SECRET/PASSWORD-named variables must be empty.
# Protected-path rule (a.2.1, a.3): `.env*` content is read ONLY where the filesystem permits it.
# env_readable(path) = regular file in the working tree, not a symlink, and a 1-byte read works.
#   * CI (CI or GITHUB_ACTIONS set): ANY .env.example that is not readable FAILS ("protected path
#     in CI - cannot verify"), never skipped. Readable ones are fully scanned (working tree, index
#     blob, history, tag-only blobs, strict format check).
#   * Everywhere: a `.env.example` whose index mode is 120000 (symlink) or 160000 (gitlink), or
#     that is a symlink in the working tree, FAILS (path-only checks).
#   * Agent sandbox (CI unset): for each .env.example that is NOT readable the content is read from NO
#     source: it is skipped by exact path in the index scan, excluded from `git log -p`
#     (:(exclude,literal)<path>), skipped in the tag-only blob scan (blobs named by that path; no
#     cat-file, not even -t, is run on them) and in the strict-format check. `skipped content scan of <path> (protected path; checked in CI)`
#     is printed. Path-only checks (tracked .env, history paths, symlinks) run everywhere.
# Symlinks are never followed. A symlink to a .env* target (other than .env.example) fails.
# Read and enumeration errors fail CLOSED (including an object git cannot read).
#
# Known limitations (not detected): base64-encoded or split keys; the content of submodules;
# unreachable objects (dangling blobs, reflog-only commits); generic-pattern false positives on
# identifier-like values (owner decision pending, the generic pattern is deliberately NOT
# relaxed); file names containing newlines may be attributed imprecisely in history output.
set -euo pipefail
export LC_ALL=C # byte semantics: binary files must not break cut/sed
cd "$(git rev-parse --show-toplevel)"
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/gate_secrets.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

# Patterns are assembled from fragments so this file never matches its own scan.
P_GROQ='gsk_[A-Za-z0-9]{20,}'
P_SK='sk-[A-Za-z0-9_-]{20,}'
P_STRIPE='(sk|rk)_live_[0-9A-Za-z]{20,}'
P_AWS='(AKIA|ASIA)[0-9A-Z]{16}'
P_PEM='-----BEGIN [A-Z ]*PRIVATE KEY-----'
P_PGP='-----BEGIN PGP PRIVATE KEY'' BLOCK-----'
P_GH='gh[pos]_[A-Za-z0-9]{20,}'
P_GHPAT='github_pat_[A-Za-z0-9_]{20,}'
P_SLACK='xox[abprs]-[A-Za-z0-9-]{10,}'
P_SLACKAPP='xapp-[A-Za-z0-9-]{10,}'
P_GOOG='AIza[0-9A-Za-z_-]{35}'
P_HF='hf_[A-Za-z0-9]{30,}'
P_LS='lsv2_[A-Za-z0-9_]{20,}'
# Vendor-prefix patterns: safe on binary files, in history, and on path names.
VENDOR="($P_GROQ|$P_SK|$P_STRIPE|$P_AWS|$P_PEM|$P_PGP|$P_GH|$P_GHPAT|$P_SLACK|$P_SLACKAPP|$P_GOOG|$P_HF|$P_LS)"
# Text-only patterns.
P_BEARER='Authorization:[[:space:]]*Bearer[[:space:]]+[A-Za-z0-9._~+/=-]{20,}'
P_URL='://[^/[:space:]:@]+:[^/[:space:]@]+@'
P_GEN='(api[_-]?key|secret|token|password)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"']?[A-Za-z0-9_./+=-]{16,}'
PATTERN="($VENDOR|$P_BEARER|$P_URL|$P_GEN)"
# Case rules (a.3.1): the non-vendor text patterns (bearer, url, generic) are matched
# CASE-INSENSITIVELY (DB_PASSWORD=..., API_KEY: ... are secrets too). Vendor patterns are
# case-specific literal prefixes (gsk_, sk-, AKIA, ghp_, ...) and are matched with exact case,
# so an upper-cased lookalike is not a vendor key. Every grep that applies the generic pattern
# therefore runs with -i on TEXTPAT and, separately, without -i on VENDOR.
TEXTPAT="($P_BEARER|$P_URL|$P_GEN)"

IN_CI=0
if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; then IN_CI=1; fi

# THE definition of a protected env basename (bash and awk both use this regex).
ENV_RE='^[.]env([.].*)?$'
EX_NAME='.env.example'
HDR=$'\001'
RE_ASSIGN='^(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$'

# is_env_path <path>: basename is .env or .env.* but not .env.example.
is_env_path() {
  local b="${1##*/}"
  [[ $b =~ $ENV_RE ]] && [ "$b" != "$EX_NAME" ]
}
is_example() { [ "${1##*/}" = "$EX_NAME" ]; }

# env_readable <path>: working-tree regular file (not a symlink) whose first byte can be read.
env_readable() {
  [ -f "$1" ] && [ ! -L "$1" ] && [ -r "$1" ] && head -c 1 -- "$1" >/dev/null 2>&1
}
# PROT: .env.example paths (exact, repo-relative) that are not readable here; computed once.
PROT=()
compute_prot() {
  local f
  PROT=()
  git ls-files -z >"$TMPD/p_tracked" 2>/dev/null || true
  git ls-files -z --others --exclude-standard >"$TMPD/p_others" 2>/dev/null || true
  while IFS= read -r -d '' f; do
    if is_example "$f" && ! env_readable "$f"; then PROT+=("$f"); fi
  done < <(cat "$TMPD/p_tracked" "$TMPD/p_others" | sort -zu)
}
# is_prot <path>: exact match against PROT.
is_prot() {
  local x
  [ "${#PROT[@]}" -gt 0 ] || return 1
  for x in "${PROT[@]}"; do [ "$x" = "$1" ] && return 0; done
  return 1
}
# print_skips: one stdout line per protected path.
print_skips() {
  local x
  [ "$IN_CI" = 0 ] || return 0 # in CI a protected path is a failure (P record), never a skip
  [ "${#PROT[@]}" -gt 0 ] || return 0
  for x in "${PROT[@]}"; do
    echo "gate_secrets: skipped content scan of $(rp "$x") (protected path; checked in CI)"
  done
}
# prot_records: in CI every protected .env.example path is a failure.
prot_records() {
  local x
  { [ "$IN_CI" = 1 ] && [ "${#PROT[@]}" -gt 0 ]; } || return 0
  for x in "${PROT[@]}"; do echo "P|$(rp "$x")"; done
}
# link_records: Y for any .env.example that is a symlink/gitlink in the index or a symlink on
# disk. Path-only (no content read), runs in every mode. Needs compute_prot's file lists.
link_records() {
  local rec meta path mode sha stage f
  if git ls-files -s -z >"$TMPD/l_idx" 2>"$TMPD/l_idx.err" && [ ! -s "$TMPD/l_idx.err" ]; then
    while IFS= read -r -d '' rec; do
      meta="${rec%%$'\t'*}"
      path="${rec#*$'\t'}"
      read -r mode sha stage <<<"$meta"
      if is_example "$path" && { [ "$mode" = 120000 ] || [ "$mode" = 160000 ]; }; then
        echo "Y|$(rp "$path") (index mode $mode)"
      fi
    done <"$TMPD/l_idx"
  else
    echo "D|index"
  fi
  while IFS= read -r -d '' f; do
    if is_example "$f" && [ -L "$f" ]; then echo "Y|$(rp "$f") (working-tree symlink)"; fi
  done < <(cat "$TMPD/p_tracked" "$TMPD/p_others" | sort -zu)
}
# is_env_target <link-target>: same definition applied to a symlink target.
is_env_target() {
  local t="$1"
  while [ "${t%/}" != "$t" ]; do t="${t%/}"; done
  is_env_path "$t"
}

# rp <string>: print the string, or <redacted-path> if it matches a vendor or generic pattern.
rp() {
  if grep -qaE -e "$VENDOR" <<<"$1" || grep -qaiE -e "$TEXTPAT" <<<"$1"; then
    printf '<redacted-path>'
  else
    printf '%s' "$1"
  fi
}

# scan_lines <file>: prints matching line numbers (never text). rc 2 = read error (fail closed).
scan_lines() {
  local f="$1" rc=0 a b
  # fast negative path (one process for a clean file): a case-insensitive, binary-as-text match of
  # EVERY pattern is a superset of both exact scans below, so no match here means no hits.
  grep -aqiE -e "$VENDOR|$TEXTPAT" -- "$f" 2>/dev/null || rc=$?
  if [ "$rc" -eq 1 ]; then return 0; fi
  [ "$rc" -eq 0 ] || return 2
  rc=0
  a="$(grep -IEin -e "$TEXTPAT" -- "$f" 2>/dev/null | cut -d: -f1)" || rc=$?
  [ "$rc" -le 1 ] || return 2
  rc=0
  b="$(grep -aEn -e "$VENDOR" -- "$f" 2>/dev/null | cut -d: -f1)" || rc=$?
  [ "$rc" -le 1 ] || return 2
  printf '%s\n%s\n' "$a" "$b" | sort -un | sed '/^$/d'
}

# emit_hits <path> <line-numbers>: H records, path redacted if needed.
emit_hits() {
  local p n
  p="$(rp "$1")"
  for n in $2; do echo "H|$p:$n"; done
}

# example_check <label> <file>: strict .env.example parse. Emits M|label:line: malformed line,
# V|label: NAME (non-empty secret-named value), X|label (checked). rc 1 = unreadable.
example_check() {
  local label="$1" file="$2" p n=0 line name val v up content
  content="$(cat -- "$file" 2>/dev/null)" || return 1
  p="$(rp "$label")"
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in '' | '#'*) continue ;; esac
    if [[ ! $line =~ $RE_ASSIGN ]]; then
      echo "M|$p:$n: malformed line"
      continue
    fi
    name="${BASH_REMATCH[2]}"
    val="${BASH_REMATCH[3]}"
    up="$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"
    case "$up" in *KEY* | *TOKEN* | *SECRET* | *PASSWORD*) ;; *) continue ;; esac
    v="${val#"${val%%[![:space:]]*}"}"
    case "$v" in '""'* | "''"*) v="${v:2}"; v="${v#"${v%%[![:space:]]*}"}" ;; esac
    case "$v" in '' | '#'*) continue ;; esac
    echo "V|$p: $(rp "$name")"
  done <<<"$content"
  echo "X|$p"
}

# scan_files: reads NUL-separated working-tree paths on stdin. Emits tagged records:
#   H|file:line  possible secret   E|file  unreadable (fail closed)
#   S|file       symlink to a .env* target   K|file  .env.example not readable here (skipped)
scan_files() {
  local f lines t ex
  while IFS= read -r -d '' f; do
    if is_env_path "$f"; then continue; fi # protected path, never read
    ex=0
    if is_example "$f"; then ex=1; fi
    if is_prot "$f"; then # content never read from a protected path (symlink check is path-only)
      if [ -L "$f" ]; then
        t="$(readlink "$f" || true)"
        if is_env_target "$t"; then echo "S|$(rp "$f")"; fi
      fi
      continue
    fi
    if [ -L "$f" ]; then
      t="$(readlink "$f" || true)"
      if is_env_target "$t"; then echo "S|$(rp "$f")"; elif [ "$ex" = 1 ]; then echo "K|$(rp "$f")"; fi
      continue # never follow symlinks
    fi
    [ -e "$f" ] || continue # tracked but deleted on disk: index scan still covers it
    if [ ! -f "$f" ]; then # directories (submodules), sockets, etc.
      if [ "$ex" = 1 ]; then echo "K|$(rp "$f")"; fi
      continue
    fi
    if [ ! -r "$f" ] || ! lines="$(scan_lines "$f")"; then
      if [ "$ex" = 1 ]; then echo "K|$(rp "$f")"; else echo "E|$(rp "$f")"; fi
      continue
    fi
    emit_hits "$f" "$lines"
    if [ "$ex" = 1 ]; then example_check "$f" "$f" || echo "K|$(rp "$f")"; fi
  done
}

# hash_safe_path <path>: success only for plain printable ASCII without `"`, `\`, CR or LF and
# without leading/trailing spaces. `git hash-object --stdin-paths` C-unquotes a line that starts
# with `"` and strips a trailing CR, so any other name could make it open a different file than
# the one this gate scanned (even a .env*). Such paths are never passed to hash-object: their
# index blob is always scanned.
hash_safe_path() {
  local LC_ALL=C
  case "$1" in
    "" | *[!\ -~]* | *\"* | *\\* | " "* | *" ") return 1 ;;
  esac
  return 0
}

# index_same <idx-file>: writes $TMPD/same = ascending record numbers (1-based, in index order) of
# regular index entries whose blob is byte-identical to the working-tree file (so scan_files has
# already scanned exactly that content). ONE `git hash-object --no-filters --stdin-paths` for all.
# Only ordinary files are hashed: never a .env* path, a protected .env.example, any .env.example,
# a symlink, a file that is not readable, or a path that is not hash_safe_path. Any doubt -> no skip.
index_same() {
  local rec meta path mode sha stage n=0 hn=0
  : >"$TMPD/same"
  : >"$TMPD/hp"
  : >"$TMPD/hi"
  while IFS= read -r -d '' rec; do
    n=$((n + 1))
    meta="${rec%%$'\t'*}"
    path="${rec#*$'\t'}"
    read -r mode sha stage <<<"$meta"
    case "$mode" in 100644 | 100755) ;; *) continue ;; esac
    [ "$stage" = 0 ] || continue
    case "$path" in *$'\n'*) continue ;; esac
    hash_safe_path "$path" || continue # hash-object would C-unquote / strip it: scan the blob
    if is_env_path "$path" || is_example "$path" || is_prot "$path"; then continue; fi
    if [ -f "$path" ] && [ ! -L "$path" ] && [ -r "$path" ]; then
      hn=$((hn + 1))
      printf '%s\n' "$path" >>"$TMPD/hp"
      printf '%s\t%s\n' "$n" "$sha" >>"$TMPD/hi"
    fi
  done <"$1"
  [ "$hn" -gt 0 ] || return 0
  if git hash-object --no-filters --stdin-paths <"$TMPD/hp" >"$TMPD/hout" 2>/dev/null &&
    [ "$(wc -l <"$TMPD/hout")" -eq "$hn" ]; then
    awk -F'\t' 'NR == FNR { h[NR] = $1; next } h[FNR] == $2 { print $1 }' \
      "$TMPD/hout" "$TMPD/hi" >"$TMPD/same"
  fi
}

# index_scan: scans staged blobs (not the working tree). Same records as scan_files. Blobs equal
# to their working-tree file are skipped (index_same); the others are read blob by blob.
index_scan() {
  local rec meta path mode sha stage t lines ex blob="$TMPD/blob" n=0 nxt=0
  if ! git ls-files -s -z >"$TMPD/idx" 2>"$TMPD/idx.err" || [ -s "$TMPD/idx.err" ]; then
    echo "D|index"
    return 0
  fi
  index_same "$TMPD/idx"
  exec 4<"$TMPD/same"
  read -r nxt <&4 || nxt=0
  while IFS= read -r -d '' rec; do
    n=$((n + 1))
    if [ "$n" = "$nxt" ]; then
      read -r nxt <&4 || nxt=0
      continue
    fi
    meta="${rec%%$'\t'*}"
    path="${rec#*$'\t'}"
    read -r mode sha stage <<<"$meta"
    [ "$mode" != "160000" ] || continue # submodule commit pointer
    if [ "$mode" = "120000" ]; then
      t="$(git cat-file blob "$sha" 2>/dev/null </dev/null || true)"
      if is_env_target "$t"; then echo "S|$(rp "$path")"; fi
      continue
    fi
    if is_env_path "$path"; then continue; fi
    if is_prot "$path"; then continue; fi # protected: index blob not read
    ex=0
    if is_example "$path"; then ex=1; fi
    if ! git cat-file blob "$sha" >"$blob" 2>/dev/null </dev/null || ! lines="$(scan_lines "$blob")"; then
      echo "E|$(rp "$path")"
      continue
    fi
    emit_hits "$path" "$lines"
    if [ "$ex" = 1 ]; then example_check "$path" "$blob" || echo "E|$(rp "$path")"; fi
  done <"$TMPD/idx"
  exec 4<&-
}

# name_hits <NUL-file>: N records for list entries whose NAME matches a vendor pattern.
# (BSD grep has no working -z, so NULs become newlines; only the entry number is kept.)
name_hits() {
  tr '\0' '\n' <"$1" | { grep -aEn -e "$VENDOR" || true; } | cut -d: -f1 |
    sed -n 's/^\([0-9][0-9]*\)$/N|<redacted-path> (entry \1 of the sorted file list)/p'
}

# msg_hits <C|G> <file>: ids of records (header line \001<id>) that contain a vendor match.
msg_hits() {
  { { grep -aEn -e "^$HDR[0-9a-f]{40,64}\$|$VENDOR" "$2" || true; }
    { grep -aEin -e "$TEXTPAT" "$2" || true; }; } | sort -t: -k1,1n -s |
    awk -v t="$1" -v h="$HDR" '
      { sub(/^[0-9]+:/, "") }
      substr($0, 1, 1) == h { c = substr($0, 2, 12); next }
      !(c in seen) { seen[c] = 1; print t "|" c }'
}

# history_scan: emits tagged records (commit/tag/blob ids are 12-char prefixes).
history_scan() {
  if [ -z "$(git for-each-ref --count=1 --format=x 2>/dev/null)" ]; then
    git rev-parse --verify -q HEAD >/dev/null 2>&1 || return 0
  fi
  local c f rc sha path line otype blob="$TMPD/hblob"

  # 1. added lines of every commit (binaries as text, merges against the first parent).
  #    Pass A: everything except `.env` and `.env.*` (this also drops .env.example, which git
  #    pathspecs cannot re-include). Their content is never requested from git.
  #    Pass B: `.env.example` only, minus the protected (unreadable) paths.
  local pathspec=() x lopts=(-p --text --all --no-color --no-ext-diff --no-textconv
    --diff-merges=first-parent --full-history)
  local hist_ok=1
  git log "${lopts[@]}" --format='commit %H' -- . ':(exclude,glob)**/.env' \
    ':(exclude,glob)**/.env.*' >"$TMPD/hist" 2>/dev/null || hist_ok=0
  pathspec=(-- ":(glob)**/$EX_NAME")
  for x in "${PROT[@]+"${PROT[@]}"}"; do pathspec+=(":(exclude,literal)$x"); done
  git log "${lopts[@]}" --format='commit %H' "${pathspec[@]}" >>"$TMPD/hist" 2>/dev/null || hist_ok=0
  if [ "$hist_ok" = 1 ]; then
    { grep -aE -e "^(commit [0-9a-f]+|diff --git .*|@@ .*|\\+.*$VENDOR.*)$" "$TMPD/hist" || true; } |
      LC_ALL=C awk -v re="$ENV_RE" '
        /^commit [0-9a-f]+$/ { c = substr($2, 1, 12); next }
        function unq(q,   o, i, c, n, d) {   # undo git C-quoting of a path
          o = ""
          for (i = 1; i <= length(q); i++) {
            c = substr(q, i, 1)
            if (c != "\\") { o = o c; continue }
            i++; c = substr(q, i, 1)
            if (c ~ /[0-7]/) {
              n = c + 0
              while (length(d = substr(q, i + 1, 1)) && d ~ /[0-7]/ && ++m < 3) { n = n * 8 + d; i++ }
              m = 0
              o = o ((n < 32 || n == 127) ? "?" : sprintf("%c", n))
            } else if (c == "\"" || c == "\\") o = o c
            else o = o "?"
          }
          return o
        }
        /^diff --git / { f = $0
                         if (f ~ /^diff --git .* "b\/.*"$/) {
                           t = f; k = 0
                           while ((j = index(t, " \"b/")) > 0) { k += j; t = substr(t, j + 1) }
                           f = unq(substr(f, k + 4, length(f) - k - 4))
                         } else sub(/^diff --git a\/.* b\//, "", f)
                         b = f; sub(/^.*\//, "", b)
                         skip = (b ~ re && b != ".env.example"); hdr = 1; next }
        /^@@ / { hdr = 0; next }
        hdr && /^\+\+\+ / { next }
        /^\+/ { if (!skip) { k = c "|" f; if (!(k in seen)) { seen[k] = 1; print k } } }
      ' | while IFS='|' read -r c f; do echo "H|$c:$(rp "$f")"; done
  else
    echo "E|(history: git log -p failed)"
  fi

  # 2. paths ever added: .env* paths (path only, L) and names matching vendor patterns (N).
  if git log --all -z --name-only --diff-filter=A --no-renames --diff-merges=first-parent \
    --format='%x01%H' >"$TMPD/names" 2>/dev/null; then
    tr '\0' '\n' <"$TMPD/names" | awk -v re="$ENV_RE" '
      substr($0, 1, 1) == "\001" { c = substr($0, 2, 12); next }
      $0 == "" { next }
      { b = $0; sub(/^.*\//, "", b)
        if (b ~ re && b != ".env.example") print "L|" c "|" $0
        print "A|" c "|" $0 }' >"$TMPD/names.out"
    { grep -a '^L|' "$TMPD/names.out" || true; } |
      while IFS='|' read -r _ c f; do echo "L|$c:$(rp "$f")"; done
    { grep -aE -e "^A\\|[0-9a-f]+\\|.*$VENDOR" "$TMPD/names.out" || true; } |
      while IFS='|' read -r _ c _; do echo "N|$c:<redacted-path>"; done
  else
    echo "E|(history: cannot list paths)"
  fi

  # 3. commit messages and annotated tag messages (ids only).
  if git log --all --format='%x01%H%n%B' >"$TMPD/msgs" 2>/dev/null; then
    msg_hits C "$TMPD/msgs"
  else
    echo "E|(history: cannot read commit messages)"
  fi
  # Every ref that points at an annotated tag: peel nested tags (tag -> tag -> ...) and scan
  # the full tag object (headers + message) of each layer.
  if git for-each-ref --format='%(objectname) %(objecttype)' >"$TMPD/refs" 2>/dev/null; then
    : >"$TMPD/tags"
    : >"$TMPD/seentags"
    local tsha ttype depth tobj ttyp
    while read -r tsha ttype; do
      depth=0
      while [ "$ttype" = tag ] && [ "$depth" -lt 200 ]; do
        depth=$((depth + 1))
        if grep -qx -- "$tsha" "$TMPD/seentags"; then break; fi
        echo "$tsha" >>"$TMPD/seentags"
        printf '\001%s\n' "$tsha" >>"$TMPD/tags"
        if ! git cat-file tag "$tsha" >>"$TMPD/tags" 2>/dev/null </dev/null; then
          echo "E|(history: cannot read tag object ${tsha:0:12})"
          break
        fi
        tobj="$(git cat-file tag "$tsha" 2>/dev/null </dev/null | sed -n 's/^object //p;/^$/q' | head -n1)"
        ttyp="$(git cat-file tag "$tsha" 2>/dev/null </dev/null | sed -n 's/^type //p;/^$/q' | head -n1)"
        if [ -z "$tobj" ] || [ -z "$ttyp" ]; then
          echo "E|(history: malformed tag object ${tsha:0:12})"
          break
        fi
        tsha="$tobj"
        ttype="$ttyp"
      done
    done <"$TMPD/refs"
    msg_hits G "$TMPD/tags"
  else
    echo "E|(history: cannot read tags)"
  fi

  # 4. blobs not reachable from any commit (e.g. a tag pointing at a blob or tree).
  if git rev-list --objects --all >"$TMPD/all" 2>/dev/null && git rev-list --all >"$TMPD/commits" 2>/dev/null; then
    : >"$TMPD/reach"
    if [ -s "$TMPD/commits" ]; then
      git rev-list --objects --stdin <"$TMPD/commits" >"$TMPD/reach" 2>/dev/null || echo "E|(history: reachability)"
    fi
    awk 'NR == FNR { r[$1] = 1; next } !($1 in r) { print }' "$TMPD/reach" "$TMPD/all" >"$TMPD/cand"
    # L7: .env / .env.* paths in trees reachable only from tags (path-only; no object is read).
    # A name that has children (`name/...` also listed) is a directory, not an env file.
    awk -v re="$ENV_RE" '
      { sha = $1; p = ""; if (length($0) > length(sha) + 1) p = substr($0, length(sha) + 2)
        if (p == "") next
        n++; P[n] = p; S[n] = sha; d = p
        while (sub(/\/[^\/]*$/, "", d) && d != "") dir[d] = 1 }
      END { for (i = 1; i <= n; i++) { b = P[i]; sub(/^.*\//, "", b)
              if (b ~ re && b != ".env.example" && !(P[i] in dir)) print "L|" substr(S[i], 1, 12) "|" P[i] } }
    ' "$TMPD/cand" >"$TMPD/cand.l"
    while IFS='|' read -r _ c f; do echo "L|$c:$(rp "$f")"; done <"$TMPD/cand.l"
    while IFS= read -r line; do
      sha="${line%% *}"
      path=""
      if [ "$line" != "$sha" ]; then path="${line#* }"; fi
      # protected-path checks come BEFORE any object access (cat-file -t included)
      if [ -n "$path" ] && is_env_path "$path"; then continue; fi
      if [ -n "$path" ] && is_prot "$path"; then continue; fi # protected: blob not read
      if ! otype="$(git cat-file -t "$sha" 2>/dev/null </dev/null)"; then
        echo "E|(cannot read object ${sha:0:12})"
        continue
      fi
      [ "$otype" = blob ] || continue
      if ! git cat-file blob "$sha" >"$blob" 2>/dev/null </dev/null; then echo "E|(blob ${sha:0:12})"; continue; fi
      rc=0
      grep -aqE -e "$VENDOR" "$blob" 2>/dev/null || rc=$?
      if [ "$rc" -eq 0 ]; then echo "B|${sha:0:12}"; elif [ "$rc" -ne 1 ]; then echo "E|(blob ${sha:0:12})"; fi
    done <"$TMPD/cand"
  else
    echo "E|(history: cannot enumerate objects)"
  fi
}

# report <prefix> <records>: prints each failing group to stderr; sets fail=1. K/X go to stdout.
fail=0
report() {
  local pfx="$1" recs="$2" tag head out
  while IFS='|' read -r tag head; do
    out="$(printf '%s\n' "$recs" | sed -n "s/^$tag|//p")"
    [ -n "$out" ] || continue
    echo "$pfx: FAIL $head" >&2
    printf '%s\n' "$out" >&2
    fail=1
  done <<'EOF'
D|cannot enumerate files (fail closed) (git reported an error listing:)
P|protected path in CI — cannot verify (.env.example not readable; path only):
Y|.env.example is a symlink or gitlink (path only):
W|tracked env file(s) (path only):
H|possible secrets at (file:line or commit:file):
E|unreadable/unscannable item(s) (fail closed):
S|symlink(s) to .env* target:
N|file name(s) match a secret pattern (path redacted):
L|.env* path(s) exist in history (commit-or-tag-tree:path; path only, excluded from the content scan):
M|malformed .env.example line(s) (path:line, content never shown):
V|non-empty value for secret-named variable(s) (path: NAME only):
C|commit message(s) match a secret pattern (commit ids only):
G|annotated tag message(s) match a secret pattern (tag object ids only):
B|tag-only blob(s) match a secret pattern (blob ids only):
EOF
}

# fake <kind>: runtime-built fake credential (nothing key-shaped exists in source).
rep() { printf "%${2}s" '' | tr ' ' "$1"; }
fake() {
  local dash=----- colon=: sl=//
  case "$1" in
    groq) printf 'gsk_%s' "$(rep x 24)" ;;
    sk) printf 'sk-%s' "$(rep x 24)" ;;
    stripe) printf 'sk_%s%s' live_ "$(rep a 24)" ;;
    stripe-rk) printf 'rk_%s%s' live_ "$(rep a 24)" ;;
    aws) printf 'AKIA%s' "$(rep A 16)" ;;
    asia) printf 'ASIA%s' "$(rep A 16)" ;;
    pem) printf '%sBEGIN RSA PRIVATE KEY%s' "$dash" "$dash" ;;
    pgp) printf '%sBEGIN PGP PRIVATE KEY BLOCK%s' "$dash" "$dash" ;;
    ghp) printf 'ghp_%s' "$(rep a 36)" ;;
    gho) printf 'gho_%s' "$(rep a 36)" ;;
    ghs) printf 'ghs_%s' "$(rep a 36)" ;;
    ghpat) printf 'github_pat_%s' "$(rep a 30)" ;;
    slack) printf 'xoxb-%s-%s' "$(rep 1 12)" "$(rep a 12)" ;;
    xapp) printf 'xapp-%s-%s' "$(rep 1 12)" "$(rep a 12)" ;;
    google) printf 'AIza%s' "$(rep a 35)" ;;
    hf) printf 'hf_%s' "$(rep a 32)" ;;
    lsv2) printf 'lsv2_pt_%s' "$(rep a 24)" ;;
    bearer) printf 'Authorization%s Bearer %s' "$colon" "$(rep a 24)" ;;
    url) printf 'https%s%su%sp%sh.example' "$colon" "$sl" "$colon" "@" ;;
    gen-json) printf '{"%s": "ab.cd/ef+gh=ij-klmnop"}' "pass""word" ;;
    gen-plain) printf 'api_key = "%s"' "$(rep a 20)" ;;
  esac
}

KINDS="groq sk stripe stripe-rk aws asia pem pgp ghp gho ghs ghpat slack xapp google hf lsv2"
TEXT_KINDS="bearer url gen-json gen-plain"

if [ "${1:-}" = "--self-test" ]; then
  d="$TMPD/self"
  mkdir "$d"
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
  # the single .env definition (strings only; no env file is ever created)
  for p in .env .env.local a/.env.production .env.; do
    if is_env_path "$p"; then r=yes; else r=no; fi
    expect "is_env_path $p -> yes" yes "$r"
  done
  for p in .envrc .env-notes.md .environment .env.example a/.env.example .env/x x.env; do
    if is_env_path "$p"; then r=yes; else r=no; fi
    expect "is_env_path $p -> no" no "$r"
  done
  # path redaction
  expect "rp redacts key-shaped name" "<redacted-path>" "$(rp "$(fake groq).txt")"
  expect "rp keeps plain name" "a/b.txt" "$(rp a/b.txt)"
  # strict .env.example parsing
  printf 'A_API_KEY=\nB_TOKEN="" # c\nOPENAI_API_KEY: x\nC_SECRET=v\nPLAIN=1\njunk line\n' >"$d/ex.txt"
  got="$(example_check ex "$d/ex.txt" | sort | tr '\n' ' ')"
  expect "example_check parse" "M|ex:3: malformed line M|ex:6: malformed line V|ex: C_SECRET X|ex " "$got"
  if [ "$ok" -eq 1 ]; then
    echo "gate_secrets self-test: PASS (every pattern flagged, clean/generic-binary/symlink not flagged)"
    exit 0
  fi
  echo "gate_secrets self-test: FAIL (positive/negative control mismatch)" >&2
  exit 1
fi

if [ "${1:-}" = "--history" ]; then
  compute_prot
  recs="$({ prot_records; link_records; history_scan; } | sort -u)"
  report "gate_secrets --history" "$recs"
  print_skips
  if [ "$fail" -ne 0 ]; then exit 1; fi
  echo "gate_secrets --history: OK"
  exit 0
fi

if [ -n "${1:-}" ]; then
  echo "usage: gate_secrets.sh [--self-test | --history]" >&2
  exit 2
fi

compute_prot
# Enumerate files; any error (even a warning on stderr) fails closed.
enum_fail=0
if ! git ls-files -z >"$TMPD/tracked" 2>"$TMPD/e1" || [ -s "$TMPD/e1" ]; then enum_fail=1; fi
if ! git ls-files -z --others --exclude-standard >"$TMPD/others" 2>"$TMPD/e2" || [ -s "$TMPD/e2" ]; then enum_fail=1; fi
cat "$TMPD/tracked" "$TMPD/others" | sort -zu >"$TMPD/paths"

records="$(
  {
    if [ "$enum_fail" = 1 ]; then echo "D|working tree listing"; fi
    while IFS= read -r -d '' f; do
      if is_env_path "$f"; then echo "W|$(rp "$f")"; fi
    done <"$TMPD/tracked"
    prot_records
    link_records
    name_hits "$TMPD/paths"
    scan_files <"$TMPD/paths"
    index_scan
  } | sort -u
)"
if [ "$IN_CI" = 1 ]; then # a late-appearing protected path (race) is also a CI failure
  records="$(printf '%s\n' "$records" | sed 's/^K|/P|/' | sort -u)"
fi
report "gate_secrets" "$records"

# Stdout notes: .env.example skips and successful checks.
print_skips
while IFS= read -r p; do # fallback: became unreadable after the up-front check (race)
  [ -z "$p" ] || echo "gate_secrets: skipped content scan of $p (protected path; checked in CI)"
done <<<"$(printf '%s\n' "$records" | sed -n 's/^K|//p')"
if [ "$fail" -eq 0 ]; then
  while IFS= read -r p; do
    [ -z "$p" ] || echo "gate_secrets: $p checked (strict format; secret-named variables are empty)"
  done <<<"$(printf '%s\n' "$records" | sed -n 's/^X|//p')"
fi

if [ "$fail" -ne 0 ]; then exit 1; fi
echo "gate_secrets: OK"
