/* Pure helpers (no DOM). Loaded before app.js; also require()-able from node for offline tests. */
(function (root) {
  "use strict";

  var DOC_RE = /^[A-Z]{2,3}-\d{3}$/;
  var SEC_RE = /^\d+(\.\d+)*$/;

  function sanitizeDocId(v) {
    return typeof v === "string" && DOC_RE.test(v) ? v : null;
  }

  function sanitizeSection(v) {
    return typeof v === "string" && SEC_RE.test(v) ? v : null;
  }

  /* Only entries with valid === true (strict) and well-formed ids become chips. Deduplicated.
     The answer text is never consulted. */
  function validCitations(citations) {
    var out = [];
    var seen = {};
    if (!Array.isArray(citations)) return out;
    for (var i = 0; i < citations.length; i++) {
      var c = citations[i];
      if (!c || c.valid !== true) continue;
      var d = sanitizeDocId(c.doc_id);
      var s = sanitizeSection(c.section);
      if (!d || !s) continue;
      var key = d + "#" + s;
      if (seen[key]) continue;
      seen[key] = true;
      out.push({ doc_id: d, section: s, label: d + " §" + s });
    }
    return out;
  }

  function docAnchorId(docId) {
    var d = sanitizeDocId(docId);
    return d ? "doc-" + d : null;
  }

  function sectionAnchorId(docId, section) {
    var d = sanitizeDocId(docId);
    var s = sanitizeSection(section);
    return d && s ? "sec-" + d + "-" + s : null;
  }

  /* "a\nb" -> ["a", "b"]; used so text is rendered as separate text nodes, never as markup. */
  function splitLines(text) {
    return String(text === null || text === undefined ? "" : text).split(/\r?\n/);
  }

  function clampPct(n) {
    var x = Number(n);
    if (!isFinite(x)) return 0;
    return Math.max(0, Math.min(100, x));
  }

  function fmtNum(n, digits) {
    var x = Number(n);
    return isFinite(x) ? x.toFixed(digits) : "–";
  }

  function chunkDocSection(id) {
    /* chunk id is "<doc_id>#<section>#<n>" */
    var parts = typeof id === "string" ? id.split("#") : [];
    var d = sanitizeDocId(parts[0]);
    var s = sanitizeSection(parts[1]);
    return d && s ? d + " §" + s : d || "–";
  }

  function rankSort(chunks, mode) {
    var arr = Array.isArray(chunks) ? chunks.slice() : [];
    function rk(v) {
      return typeof v === "number" ? v : Infinity;
    }
    if (mode === "bm25") {
      arr.sort(function (a, b) { return rk(a.bm25_rank) - rk(b.bm25_rank); });
    } else if (mode === "dense") {
      arr.sort(function (a, b) { return rk(a.dense_rank) - rk(b.dense_rank); });
    } else {
      arr.sort(function (a, b) { return Number(b.rrf) - Number(a.rrf); });
    }
    return arr;
  }

  /* Eval thresholds: returns "pass" | "fail" | "none". */
  function judge(kind, value) {
    var v = Number(value);
    if (value === null || value === undefined || !isFinite(v)) return "none";
    if (kind === "faithfulness") return v >= 0.9 ? "pass" : "fail";
    if (kind === "hit4") return v >= 0.85 ? "pass" : "fail";
    if (kind === "mrr") return v >= 0.7 ? "pass" : "fail";
    if (kind === "p95s") return v < 8 ? "pass" : "fail";
    return "none";
  }

  /* Look a key up at top level and inside common summary containers. */
  function pick(obj, keys) {
    if (!obj || typeof obj !== "object") return undefined;
    var containers = [obj, obj.metrics, obj.summary, obj.aggregate, obj.scores];
    for (var c = 0; c < containers.length; c++) {
      var o = containers[c];
      if (!o || typeof o !== "object") continue;
      for (var k = 0; k < keys.length; k++) {
        if (Object.prototype.hasOwnProperty.call(o, keys[k]) && o[keys[k]] !== null) return o[keys[k]];
      }
    }
    return undefined;
  }

  var api = {
    sanitizeDocId: sanitizeDocId,
    sanitizeSection: sanitizeSection,
    validCitations: validCitations,
    docAnchorId: docAnchorId,
    sectionAnchorId: sectionAnchorId,
    splitLines: splitLines,
    clampPct: clampPct,
    fmtNum: fmtNum,
    chunkDocSection: chunkDocSection,
    rankSort: rankSort,
    judge: judge,
    pick: pick
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OpsHelpers = api;
})(typeof window !== "undefined" ? window : this);
