/* Meridian Ops Portal. Vanilla JS, no build step.
   Security rules (enforced by tests/test_web_static.py):
   - API-derived strings reach the page only through textContent / createTextNode.
   - No dynamic-HTML sinks, no eval, no inline handlers; same-origin fetch only.
   - Citation chips come only from response.citations entries with valid === true. */
(function () {
  "use strict";

  var H = window.OpsHelpers;
  var GENERIC_ERR = "Something went wrong. Please try again.";
  var SAMPLES = [
    "RCV-5102 from Bluewater Packaging arrived with damaged cartons — what do we have to do and by when?",
    "PO-2291 for $40,000 — who needs to approve it?",
    "Why is Bluewater Packaging on probation, and what happens to a supplier on probation?",
    "How many vacation days do warehouse staff get?"
  ];
  var TITLES = {
    dash: ["Operations", "Dashboard"],
    recv: ["Operations", "Receiving"],
    po: ["Operations", "Purchase orders"],
    sup: ["Operations", "Suppliers"],
    kb: ["Knowledge", "SOPs & policies"],
    console: ["Ops Copilot", "Console"]
  };
  var MODE_LABELS = [["bm25", "Keyword (BM25)"], ["dense", "Meaning (dense)"], ["hybrid", "Hybrid (RRF)"]];
  var TRACE_STEPS = [
    ["embed", "Embed question"], ["bm25", "BM25 search"], ["dense", "Dense search"],
    ["rrf", "RRF merge"], ["gate", "Confidence gate"], ["llm", "LLM answer"]
  ];

  var state = { lastQuestion: null, lastResult: null, mode: "hybrid", busy: false, docs: [] };

  function $(sel) { return document.querySelector(sel); }
  function $$(sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); }
  function byId(id) { return document.getElementById(id); }

  /* ---------- DOM builders (textContent only) ---------- */
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function add(parent) {
    for (var i = 1; i < arguments.length; i++) if (arguments[i]) parent.appendChild(arguments[i]);
    return parent;
  }
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }
  function btn(cls, text, onClick) {
    var b = el("button", cls, text);
    b.type = "button";
    if (onClick) b.addEventListener("click", onClick);
    return b;
  }
  function td(content, cls) {
    var c = el("td", cls);
    if (content instanceof Node) c.appendChild(content);
    else c.textContent = content === null || content === undefined || content === "" ? "–" : String(content);
    return c;
  }
  function th(labels) {
    var tr = el("tr");
    labels.forEach(function (l) { tr.appendChild(el("th", null, l)); });
    return add(el("thead"), tr);
  }
  /* Multi-line text as separate text nodes separated by <br> elements. */
  function multiline(text) {
    var box = el("div", "pre");
    H.splitLines(text).forEach(function (line, i) {
      if (i > 0) box.appendChild(document.createElement("br"));
      box.appendChild(document.createTextNode(line));
    });
    return box;
  }
  function usd(n) {
    var x = Number(n);
    return isFinite(x) ? "$" + x.toLocaleString("en-US") : "–";
  }
  function setPct(node, pct) {
    node.style.setProperty("--w", H.clampPct(pct) + "%"); // numeric only, never an API string
  }

  /* ---------- network (same-origin, JSON only) ---------- */
  function request(url, opts) {
    var o = { credentials: "same-origin", headers: { Accept: "application/json" } };
    if (opts && opts.body !== undefined) {
      o.method = "POST";
      o.headers["Content-Type"] = "application/json";
      o.body = JSON.stringify(opts.body);
    }
    return fetch(url, o).then(function (r) {
      var type = r.headers.get("content-type") || "";
      if (!r.ok || type.indexOf("application/json") === -1) return { ok: false, status: r.status, data: null };
      return r.json().then(
        function (d) { return { ok: true, status: r.status, data: d }; },
        function () { return { ok: false, status: r.status, data: null }; }
      );
    }).catch(function () { return { ok: false, status: 0, data: null }; });
  }

  /* ---------- theme ---------- */
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  function initTheme() {
    var saved = null;
    try { saved = window.localStorage.getItem("meridian-theme"); } catch (e) { saved = null; }
    applyTheme(saved);
    byId("themebtn").addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      if (!cur) cur = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      var next = cur === "dark" ? "light" : "dark";
      applyTheme(next);
      try { window.localStorage.setItem("meridian-theme", next); } catch (e) { /* storage unavailable */ }
    });
  }

  /* ---------- navigation ---------- */
  function go(p) {
    if (!TITLES[p]) return;
    $$(".nav").forEach(function (n) { n.classList.toggle("on", n.getAttribute("data-p") === p); });
    $$(".page").forEach(function (x) { x.classList.toggle("on", x.id === p); });
    byId("crumb").textContent = TITLES[p][0];
    byId("title").textContent = TITLES[p][1];
  }
  function showTab(t) {
    $$(".tabs button").forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-t") === t); });
    $$(".ctab").forEach(function (x) { x.classList.toggle("on", x.id === t); });
  }
  function openCP(focus) {
    byId("app").classList.add("copilot");
    if (focus) byId("in").focus();
  }
  function prefill(text) {
    openCP(true);
    var input = byId("in");
    input.value = String(text).slice(0, 2000);
    input.focus();
  }
  function askButton(text, label) {
    return btn("ask", label || "✦ Ask", function () { prefill(text); });
  }

  /* ---------- dashboard / lists ---------- */
  function statusClass(s) {
    s = String(s || "").toLowerCase();
    if (s.indexOf("quarantin") !== -1 || s.indexOf("held") !== -1 || s.indexOf("probation") !== -1) return "bad";
    if (s.indexOf("awaiting") !== -1 || s.indexOf("provisional") !== -1 || s.indexOf("short") !== -1 || s.indexOf("pending") !== -1) return "warn";
    if (s === "approved" || s === "closed" || s === "received" || s === "open") return "ok";
    return "info";
  }
  function pill(text, cls) { return el("span", "st " + cls, text); }

  function kpi(color, label, value, detail) {
    var d = el("div", "kpi k-" + color);
    return add(d, el("div", "l", label), el("div", "v", value), el("div", "d", detail));
  }

  function exceptionLabel(r) {
    return r.exception && r.exception.type ? String(r.exception.type).replace(/_/g, " ") : "";
  }

  function renderKpis(k) {
    var box = byId("kpis");
    clear(box);
    var fill = k.key_account_fill_rate || {};
    box.appendChild(kpi("teal", "Open purchase orders", k.open_pos, "as of " + (k.as_of || "–")));
    box.appendChild(kpi("violet", "Receipts today", k.receipts_today, "simulated"));
    box.appendChild(kpi("red", "Open discrepancies", k.open_discrepancies, "short · over · damaged · price"));
    box.appendChild(kpi("amber", "Supplier quality", H.fmtNum(k.supplier_quality_pct, 1) + "%", "units accepted, 90 days"));
    box.appendChild(kpi("green", "Key-account fill rate", H.fmtNum(fill.overall_pct, 1) + "%", "target " + H.fmtNum(fill.target_pct, 0) + "%"));
  }

  function receiptRow(r, full) {
    var tr = el("tr");
    tr.appendChild(td(r.receipt_id, "mono"));
    if (full) tr.appendChild(td(r.date, "nowrap"));
    tr.appendChild(td(r.po, "mono nowrap"));
    tr.appendChild(td(r.supplier));
    if (full) {
      tr.appendChild(td(r.item));
      tr.appendChild(td(r.ordered_qty));
      tr.appendChild(td(r.received_qty));
    }
    var ex = r.exception;
    tr.appendChild(td(ex ? (exceptionLabel(r) + (ex.detail ? " · " + ex.detail : "")) : "–"));
    tr.appendChild(td(pill(r.status || "–", ex ? "bad" : statusClass(r.status))));
    tr.appendChild(td(r.ask ? askButton(r.ask) : ""));
    return tr;
  }

  function poRow(p, mode) {
    var tr = el("tr");
    tr.appendChild(td(p.po, "mono nowrap"));
    tr.appendChild(td(p.supplier));
    if (mode === "full") {
      tr.appendChild(td(p.description));
      tr.appendChild(td(usd(p.total_usd), "nowrap"));
      tr.appendChild(td(p.created, "nowrap"));
    } else {
      tr.appendChild(td(usd(p.total_usd), "nowrap"));
      tr.appendChild(td(p.approver));
    }
    tr.appendChild(td(pill(p.status || "–", statusClass(p.status))));
    tr.appendChild(td(p.ask ? askButton(p.ask) : ""));
    return tr;
  }

  function qualityClass(q) {
    var n = Number(q);
    if (n >= 99) return "q-ok";
    if (n >= 98) return "q-mid";
    if (n >= 96) return "q-warn";
    return "q-bad";
  }

  function renderDashboard(d) {
    renderKpis(d.kpis || {});
    var receipts = Array.isArray(d.receipts) ? d.receipts : [];
    var pos = Array.isArray(d.purchase_orders) ? d.purchase_orders : [];
    var sups = Array.isArray(d.suppliers) ? d.suppliers : [];
    byId("rcv-asof").textContent = d.kpis && d.kpis.as_of ? "as of " + d.kpis.as_of : "";

    var rows = byId("rcvrows"); clear(rows);
    receipts.filter(function (r) { return r.exception; }).forEach(function (r) { rows.appendChild(receiptRow(r, false)); });

    var bars = byId("bars"); clear(bars);
    sups.slice().sort(function (a, b) { return Number(a.quality_pct) - Number(b.quality_pct); }).forEach(function (s) {
      var row = el("div", "bar");
      var track = el("div", "t");
      var fill = el("i", qualityClass(s.quality_pct));
      setPct(fill, s.quality_pct);
      track.appendChild(fill);
      add(row, el("span", null, s.name), track, el("b", null, H.fmtNum(s.quality_pct, 1)));
      bars.appendChild(row);
    });

    var porows = byId("porows"); clear(porows);
    pos.filter(function (p) { return String(p.status || "").indexOf("awaiting") === 0; })
      .forEach(function (p) { porows.appendChild(poRow(p, "short")); });

    var rl = byId("recvlist"); clear(rl);
    receipts.forEach(function (r) { rl.appendChild(receiptRow(r, true)); });
    var pl = byId("polist"); clear(pl);
    pos.forEach(function (p) { pl.appendChild(poRow(p, "full")); });
    var sl = byId("suplist"); clear(sl);
    sups.forEach(function (s) {
      var tr = el("tr");
      tr.appendChild(td(s.name));
      tr.appendChild(td(pill(s.status || "–", statusClass(s.status))));
      tr.appendChild(td(s.units_received_90d));
      tr.appendChild(td(s.units_rejected_90d));
      tr.appendChild(td(H.fmtNum(s.quality_pct, 1) + "%"));
      tr.appendChild(td(Array.isArray(s.ncr_dates) ? s.ncr_dates.length : 0));
      tr.appendChild(td(s.ask ? askButton(s.ask) : ""));
      sl.appendChild(tr);
    });
  }

  function loadDashboard() {
    request("/api/dashboard").then(function (r) {
      var err = byId("dash-err");
      if (!r.ok || !r.data) {
        err.textContent = "Dashboard data is unavailable.";
        err.hidden = false;
        return;
      }
      err.hidden = true;
      renderDashboard(r.data);
    });
  }

  /* ---------- SOP cards ---------- */
  function renderDocs(docs) {
    var grid = byId("docs"); clear(grid);
    docs.forEach(function (d) {
      var id = H.sanitizeDocId(d.doc_id);
      if (!id) return;
      var card = el("div", "doc");
      card.id = H.docAnchorId(id);
      add(card, el("span", "code", id), el("b", null, d.title));
      var meta = [d.owner_role, d.version ? "v" + d.version : "", d.effective_date ? "effective " + d.effective_date : ""]
        .filter(function (x) { return x; }).join(" · ");
      card.appendChild(el("div", "m", meta));
      var secs = Array.isArray(d.sections) ? d.sections : [];
      var ul = el("ul");
      secs.forEach(function (s) {
        var sec = H.sanitizeSection(s.section);
        if (!sec) return;
        var li = el("li");
        li.id = H.sectionAnchorId(id, sec);
        add(li, el("span", "sn", "§" + sec), document.createTextNode(String(s.title)));
        ul.appendChild(li);
      });
      if (ul.firstChild) card.appendChild(ul);
      grid.appendChild(card);
    });
  }
  function loadDocs() {
    request("/api/docs").then(function (r) {
      var err = byId("kb-err");
      if (!r.ok || !Array.isArray(r.data)) {
        err.textContent = "Documents are unavailable.";
        err.hidden = false;
        return;
      }
      err.hidden = true;
      state.docs = r.data;
      renderDocs(r.data);
    });
  }
  function flash(node) {
    node.classList.remove("hl");
    void node.offsetWidth;
    node.classList.add("hl");
  }
  function gotoCitation(docId, section) {
    go("kb");
    var card = byId(H.docAnchorId(docId) || "");
    if (!card) return;
    var sec = byId(H.sectionAnchorId(docId, section) || "");
    var target = sec || card;
    target.scrollIntoView({ block: "center" });
    flash(card);
    if (sec) flash(sec);
  }

  /* ---------- Copilot panel ---------- */
  function scrollMsgs() {
    var m = byId("msgs");
    m.scrollTop = m.scrollHeight;
  }
  function addUserMsg(text) {
    var m = byId("msgs");
    m.appendChild(multilineBubble("q", text));
    scrollMsgs();
  }
  function multilineBubble(cls, text) {
    var d = el("div", cls);
    d.appendChild(multiline(text));
    return d;
  }
  function addAnswer(res) {
    var refused = res.refused === true;
    var box = el("div", refused ? "a refuse" : "a");
    var ans = multiline(res.answer);
    box.appendChild(ans);
    var meta = el("div", "meta");
    if (!refused) {
      /* chips: ONLY response.citations with valid === true; the answer text is never parsed */
      H.validCitations(res.citations).forEach(function (c) {
        meta.appendChild(btn("cite", c.label, function () { gotoCitation(c.doc_id, c.section); }));
      });
      var nUsed = Array.isArray(res.chunks) ? res.chunks.filter(function (c) { return c.used === true; }).length : 0;
      meta.appendChild(el("span", "pill ok", "grounded · " + nUsed + " source" + (nUsed === 1 ? "" : "s")));
    } else {
      meta.appendChild(el("span", "pill no", "refused"));
    }
    meta.appendChild(el("span", "pill", "confidence " + H.fmtNum(res.confidence, 2)));
    meta.appendChild(btn("link", "View in Console", function () { go("console"); showTab("insp"); }));
    box.appendChild(meta);
    byId("msgs").appendChild(box);
    scrollMsgs();
  }
  function addNote(text) {
    byId("msgs").appendChild(el("div", "a refuse", text));
    scrollMsgs();
  }

  function postAsk(question, mode) {
    return request("/api/ask", { body: { question: question, mode: mode } });
  }
  function validResult(d) {
    return d && typeof d === "object" && typeof d.answer === "string" && typeof d.refused === "boolean" &&
      Array.isArray(d.chunks) && d.timings_ms && typeof d.timings_ms === "object";
  }

  function submitQuestion(q) {
    q = String(q).trim();
    if (!q || state.busy) return;
    state.busy = true;
    byId("send").disabled = true;
    addUserMsg(q);
    var typing = el("div", "typing", "Searching SOPs · ranking · checking confidence…");
    byId("msgs").appendChild(typing);
    scrollMsgs();
    state.lastQuestion = q;
    state.mode = "hybrid";
    postAsk(q, "hybrid").then(function (r) {
      typing.remove();
      state.busy = false;
      byId("send").disabled = false;
      if (!r.ok || !validResult(r.data)) {
        addNote(GENERIC_ERR);
        return;
      }
      state.lastResult = r.data;
      addAnswer(r.data);
      renderConsole();
    });
  }

  function initCopilot() {
    var chips = byId("chips");
    SAMPLES.forEach(function (q, i) {
      var label = ["Damaged delivery", "Who approves $40k?", "Supplier on probation", "Vacation days? (refuses)"][i];
      chips.appendChild(btn(null, label, function () { openCP(false); submitQuestion(q); }));
    });
    var msgs = byId("msgs");
    msgs.appendChild(el("div", "a", "Ask me how to handle anything you see on the dashboard. I answer only from Meridian's SOPs and policies, and I'll tell you when something isn't covered."));
    byId("f").addEventListener("submit", function (e) {
      e.preventDefault();
      var input = byId("in");
      var v = input.value;
      input.value = "";
      submitQuestion(v);
    });
    byId("cpbtn").addEventListener("click", function () { openCP(true); });
    byId("opencp").addEventListener("click", function () { openCP(true); });
    byId("cpclose").addEventListener("click", function () { byId("app").classList.remove("copilot"); });
  }

  /* ---------- Console ---------- */
  function renderInspector() {
    var box = byId("inspbody");
    clear(box);
    var res = state.lastResult;
    if (!res) {
      box.appendChild(el("div", "empty", "Ask Ops Copilot a question. This view then shows which chunks were found, how keyword and meaning search ranked them, and whether they passed the confidence gate."));
      return;
    }
    var h2 = el("h2", null, "Last question");
    h2.appendChild(el("span", "sub", "mode: " + state.mode));
    box.appendChild(h2);
    box.appendChild(multilineBubble("empty", state.lastQuestion));

    var toggle = el("div", "toggle");
    MODE_LABELS.forEach(function (m) {
      var b = btn(state.mode === m[0] ? "on" : null, m[1], function () { reask(m[0]); });
      if (state.busy) b.disabled = true;
      toggle.appendChild(b);
    });
    box.appendChild(toggle);

    var refused = res.refused === true;
    var gate = el("div", "gate " + (refused ? "fail" : "pass"));
    var conf = H.fmtNum(res.confidence, 2);
    gate.textContent = refused
      ? "✕ Confidence " + conf + " → refused" + (res.refusal_reason ? " (" + String(res.refusal_reason) + ")" : "") + (Number(res.timings_ms && res.timings_ms.llm) === 0 ? ", LLM not called" : "")
      : "✓ Confidence " + conf + " → passed the gate, answered from the chunks marked used";
    box.appendChild(gate);

    var chunks = H.rankSort(res.chunks, state.mode);
    if (!chunks.length) {
      box.appendChild(el("div", "empty", "No chunks were retrieved."));
      return;
    }
    var table = el("table");
    table.appendChild(th(["Chunk id", "Doc §", "Text", "BM25 rank", "Dense rank", "RRF", "Used"]));
    var tb = el("tbody");
    chunks.forEach(function (c) {
      var tr = el("tr");
      tr.appendChild(td(String(c.id), "mono"));
      tr.appendChild(td(H.chunkDocSection(c.id), "nowrap"));
      var t = el("td", "chunk-text");
      t.appendChild(multiline(c.text));
      tr.appendChild(t);
      tr.appendChild(td(typeof c.bm25_rank === "number" ? "#" + c.bm25_rank : "–"));
      tr.appendChild(td(typeof c.dense_rank === "number" ? "#" + c.dense_rank : "–"));
      tr.appendChild(td(H.fmtNum(c.rrf, 4), "mono"));
      tr.appendChild(td(pill(c.used === true ? "used" : "no", c.used === true ? "ok" : "info")));
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    box.appendChild(table);
  }

  function renderTrace() {
    var box = byId("tracebody");
    clear(box);
    var res = state.lastResult;
    if (!res) {
      box.appendChild(el("div", "empty", "No run yet."));
      return;
    }
    var t = res.timings_ms || {};
    var total = 0;
    TRACE_STEPS.forEach(function (s) { var v = Number(t[s[0]]); if (isFinite(v)) total += v; });
    var h2 = el("h2", null, "Run trace");
    h2.appendChild(el("span", "sub", "total " + H.fmtNum(total / 1000, 2) + " s" + (res.refused === true ? " · LLM skipped (refused)" : "")));
    box.appendChild(h2);
    var wrap = el("div", "trace");
    TRACE_STEPS.forEach(function (s) {
      var v = Number(t[s[0]]);
      if (!isFinite(v)) v = 0;
      var row = el("div", "tr");
      var track = el("div", "t");
      var fill = el("i");
      setPct(fill, total > 0 ? Math.max(v > 0 ? 1 : 0, (v / total) * 100) : 0);
      track.appendChild(fill);
      add(row, el("span", null, s[1]), track, el("span", "mono", Math.round(v) + " ms"));
      wrap.appendChild(row);
    });
    box.appendChild(wrap);
  }

  function renderConsole() {
    renderInspector();
    renderTrace();
  }

  function reask(mode) {
    if (!state.lastQuestion || state.busy) return;
    state.busy = true;
    state.mode = mode;
    renderInspector();
    postAsk(state.lastQuestion, mode).then(function (r) {
      state.busy = false;
      if (r.ok && validResult(r.data)) state.lastResult = r.data;
      renderConsole();
      if (!(r.ok && validResult(r.data))) {
        var note = el("div", "notice", GENERIC_ERR);
        byId("inspbody").insertBefore(note, byId("inspbody").firstChild);
      }
    });
  }

  /* ---------- Evals ---------- */
  function evalKpi(color, label, text, target, verdict) {
    var d = el("div", "kpi k-" + color + (verdict === "pass" || verdict === "fail" ? " " + verdict : ""));
    return add(d, el("div", "l", label), el("div", "v", text), el("div", "d", target + (verdict === "pass" ? " · pass" : verdict === "fail" ? " · fail" : "")));
  }
  function p95Seconds(d) {
    var s = H.pick(d, ["p95_latency_s", "p95_s", "p95_seconds", "latency_p95_s"]);
    if (s !== undefined) return Number(s);
    var ms = H.pick(d, ["p95_latency_ms", "p95_ms", "latency_p95_ms"]);
    return ms !== undefined ? Number(ms) / 1000 : undefined;
  }
  function cellText(v) {
    if (v === null || v === undefined) return "";
    if (Array.isArray(v)) return v.map(String).join(", ");
    if (typeof v === "object") return "";
    return String(v);
  }
  function renderEval(d) {
    var box = byId("evalbody");
    clear(box);
    var faith = H.pick(d, ["faithfulness"]);
    var hit = H.pick(d, ["hit_at_4", "hit@4", "hit4", "retrieval_hit_at_4"]);
    var mrr = H.pick(d, ["mrr", "MRR"]);
    var refOk = H.pick(d, ["refusal_correct", "refusals_correct", "refusal_ok"]);
    var refTotal = H.pick(d, ["refusal_total", "refusals_total", "refusal_expected"]);
    var p95 = p95Seconds(d);

    var k = el("div", "ekpis");
    k.appendChild(evalKpi("teal", "Faithfulness", H.fmtNum(faith, 2), "target ≥ 0.90", H.judge("faithfulness", faith)));
    k.appendChild(evalKpi("violet", "Retrieval hit@4", H.fmtNum(hit, 2), "target ≥ 0.85", H.judge("hit4", hit)));
    k.appendChild(evalKpi("amber", "MRR", H.fmtNum(mrr, 2), "target ≥ 0.70", H.judge("mrr", mrr)));
    var refVerdict = "none";
    var refText = "–";
    if (refOk !== undefined) {
      var total = refTotal !== undefined ? Number(refTotal) : 2;
      refText = Number(refOk) + " / " + total;
      refVerdict = Number(refOk) >= 2 && Number(refOk) >= total ? "pass" : "fail";
    }
    k.appendChild(evalKpi("red", "Refusal correct", refText, "target 2 / 2", refVerdict));
    k.appendChild(evalKpi("green", "p95 latency", p95 === undefined ? "–" : H.fmtNum(p95, 1) + " s", "target < 8 s", H.judge("p95s", p95)));
    box.appendChild(k);

    var rows = null;
    ["questions", "results", "items", "rows", "per_question"].forEach(function (key) {
      if (!rows && Array.isArray(d[key])) rows = d[key];
    });
    var card = el("div", "card");
    card.appendChild(el("h2", null, "Per-question results"));
    if (!rows || !rows.length) {
      card.appendChild(el("div", "empty", "No per-question rows in this eval run."));
      box.appendChild(card);
      return;
    }
    var table = el("table");
    table.appendChild(th(["#", "Type", "Question", "Expected source", "Result"]));
    var tb = el("tbody");
    rows.forEach(function (r, i) {
      r = r && typeof r === "object" ? r : {};
      var tr = el("tr");
      tr.appendChild(td(r.id !== undefined ? cellText(r.id) : i + 1));
      tr.appendChild(td(pill(cellText(r.type || r.category) || "–", "info")));
      tr.appendChild(td(cellText(r.question)));
      tr.appendChild(td(cellText(r.expected_docs || r.expected_source || r.expected || r.source), "mono"));
      var passed = r.passed !== undefined ? r.passed : r.pass !== undefined ? r.pass : r.ok;
      var label = passed === true ? "pass" : passed === false ? "fail" : cellText(r.result) || "–";
      tr.appendChild(td(pill(label, passed === true ? "ok" : passed === false ? "bad" : "neutral")));
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    card.appendChild(table);
    box.appendChild(card);
  }
  function loadEval() {
    request("/api/eval/latest").then(function (r) {
      var box = byId("evalbody");
      clear(box);
      if (r.status === 404) {
        box.appendChild(el("div", "empty", "No eval run yet"));
        return;
      }
      if (!r.ok || !r.data || typeof r.data !== "object") {
        box.appendChild(el("div", "notice", "Eval results are unavailable."));
        return;
      }
      renderEval(r.data);
    });
  }

  /* ---------- init ---------- */
  function init() {
    initTheme();
    $$(".nav").forEach(function (n) {
      n.addEventListener("click", function () { go(n.getAttribute("data-p")); });
    });
    $$(".tabs button").forEach(function (b) {
      b.addEventListener("click", function () {
        var t = b.getAttribute("data-t");
        showTab(t);
        if (t === "eval") loadEval();
      });
    });
    initCopilot();
    renderConsole();
    loadDashboard();
    loadDocs();
    loadEval();
  }
  init();
})();
