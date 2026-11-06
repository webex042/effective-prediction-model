/* EPM Chat client. No dependencies, no inline script (strict CSP), and every
   server string is rendered with textContent / createElement — never as HTML. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var log = $("log"), thread = $("thread"), input = $("input"), sendBtn = $("send-btn"), composer = $("composer");
  var fileInput = $("file"), attachBtn = $("attach-btn");
  var params = new URLSearchParams(location.search);
  var EMBED = params.get("embed") === "1" && window.parent !== window;
  var cfg = { max_message_chars: 1000, max_upload_bytes: 5 * 1024 * 1024, embed_origins: [] };
  var token = null, busy = false, lastUser = null, parentOrigin = null, chipsEl = null;

  // ------------------------------------------------------------------ storage (per tab; may be unavailable)
  var store = {
    get: function (k, area) { try { return (area || sessionStorage).getItem(k); } catch (e) { return null; } },
    set: function (k, v, area) { try { (area || sessionStorage).setItem(k, v); } catch (e) { /* ignore */ } },
    del: function (k, area) { try { (area || sessionStorage).removeItem(k); } catch (e) { /* ignore */ } }
  };

  // ------------------------------------------------------------------ theme
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    $("theme-btn").setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  var savedTheme = params.get("theme") || store.get("epm-chat-theme", localStorageSafe());
  applyTheme(savedTheme === "dark" || savedTheme === "light" ? savedTheme :
    (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("theme-btn").addEventListener("click", function () {
    var t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(t);
    store.set("epm-chat-theme", t, localStorageSafe());
  });
  function localStorageSafe() { try { return window.localStorage; } catch (e) { return null; } }

  // ------------------------------------------------------------------ icons
  var ICON = {
    copy: "M9 9h10v10H9zM5 15V5h10",
    retry: "M4 12a8 8 0 1 0 2.3-5.6M4 4v4.5h4.5",
    check: "M5 12.5l4.5 4.5L19 7.5",
    info: "M12 8h.01M11 12h1v5h1M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18Z",
    warn: "M12 9v4M12 17h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z",
    error: "M15 9l-6 6M9 9l6 6M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18Z",
    download: "M12 4v11M7 10l5 5 5-5M5 20h14"
  };
  function icon(name) {
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    var p = document.createElementNS(ns, "path");
    p.setAttribute("d", ICON[name]);
    svg.appendChild(p);
    return svg;
  }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  // ------------------------------------------------------------------ safe mini-markdown
  function inline(parent, s) {
    var re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g, last = 0, m;
    while ((m = re.exec(s))) {
      if (m.index > last) parent.appendChild(document.createTextNode(s.slice(last, m.index)));
      var t = m[0];
      if (t.slice(0, 2) === "**") parent.appendChild(el("strong", null, t.slice(2, -2)));
      else if (t[0] === "`") parent.appendChild(el("code", null, t.slice(1, -1)));
      else parent.appendChild(el("em", null, t.slice(1, -1)));
      last = m.index + t.length;
    }
    if (last < s.length) parent.appendChild(document.createTextNode(s.slice(last)));
  }
  function renderText(t) {
    var wrap = el("div", "txt"), ul = null;
    String(t || "").split("\n").forEach(function (line) {
      if (/^\s*[-•]\s+/.test(line)) {
        if (!ul) { ul = el("ul"); wrap.appendChild(ul); }
        var li = el("li"); inline(li, line.replace(/^\s*[-•]\s+/, "")); ul.appendChild(li);
      } else if (line.trim()) {
        ul = null;
        var p = el("p"); inline(p, line); wrap.appendChild(p);
      }
    });
    return wrap;
  }

  // ------------------------------------------------------------------ number formatting
  function fmt(v, d) {
    if (v === null || v === undefined || v === "") return "—";
    if (typeof v !== "number") return String(v);
    if (!isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e7)) return v.toExponential(2);
    var digits = d === undefined ? (a >= 1000 ? 1 : a >= 1 ? 3 : 4) : d;
    return v.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  function pct(p) { return (p * 100).toFixed(p >= 0.995 || p < 0.005 ? 1 : 1) + "%"; }

  // ------------------------------------------------------------------ block renderers
  function renderTable(b) {
    var frag = el("div");
    if (b.title) frag.appendChild(el("p", "tbl-title", b.title));
    var box = el("div", "tbl"); box.tabIndex = 0;
    box.setAttribute("role", "region");
    box.setAttribute("aria-label", b.title || "Table");
    var t = el("table"), thead = el("thead"), tr = el("tr");
    b.columns.forEach(function (c) { var th = el("th", null, c); th.scope = "col"; tr.appendChild(th); });
    thead.appendChild(tr); t.appendChild(thead);
    var tb = el("tbody");
    b.rows.forEach(function (r) {
      var row = el("tr");
      r.forEach(function (v) { var td = el("td", typeof v === "number" ? "num" : null, typeof v === "number" ? fmt(v) : (v === null ? "—" : v)); row.appendChild(td); });
      tb.appendChild(row);
    });
    t.appendChild(tb); box.appendChild(t); frag.appendChild(box);
    return frag;
  }

  function renderBars(b) {
    var wrap = el("figure", "bars");
    wrap.style.margin = "0";
    wrap.appendChild(el("h3", null, b.title));
    var vals = b.items.map(function (i) { return Math.abs(i.value || 0) + Math.abs(i.err || 0); });
    var max = Math.max.apply(null, vals.concat([1e-12]));
    b.items.forEach(function (it) {
      var row = el("div", "bar-row");
      var lab = el("span", "bar-label", it.label); lab.title = it.label;
      var track = el("span", "bar-track");
      var v = it.value || 0, fill = el("span", "bar-fill" + (v < 0 ? " neg" : ""));
      if (b.signed) {
        var axis = el("span", "bar-axis"); axis.style.left = "50%"; track.appendChild(axis);
        var w = 50 * Math.abs(v) / max;
        fill.style.left = (v < 0 ? 50 - w : 50) + "%"; fill.style.width = w + "%";
      } else {
        fill.style.width = Math.max(0, 100 * v / max) + "%";
        if (it.err) {
          var e = el("span", "bar-err");
          e.style.left = Math.max(0, 100 * (v - it.err) / max) + "%";
          e.style.width = (100 * 2 * it.err / max) + "%";
          track.appendChild(e);
        }
      }
      track.insertBefore(fill, track.firstChild);
      var val = el("span", "bar-val", (b.signed && v > 0 ? "+" : "") + fmt(v));
      row.appendChild(lab); row.appendChild(track); row.appendChild(val);
      row.setAttribute("aria-label", it.label + ": " + fmt(v));
      wrap.appendChild(row);
    });
    if (b.note) wrap.appendChild(el("p", "note", b.note));
    return wrap;
  }

  function scale(lo, hi, value, band) {
    var s = el("div", "scale"); s.setAttribute("aria-hidden", "true");
    s.appendChild(el("div", "rail"));
    var span = hi - lo || 1, pos = function (v) { return Math.min(100, Math.max(0, 100 * (v - lo) / span)); };
    for (var k = 0; k <= 4; k++) {
      var v = lo + span * k / 4, t = el("div", "tick"); t.style.left = pos(v) + "%"; s.appendChild(t);
      var tl = el("div", "tick-label", fmt(v, Math.abs(span) < 10 ? 2 : 0)); tl.style.left = pos(v) + "%"; s.appendChild(tl);
    }
    if (band) {
      var bd = el("div", "band"); bd.style.left = pos(band[0]) + "%"; bd.style.width = (pos(band[1]) - pos(band[0])) + "%"; s.appendChild(bd);
    }
    var n = el("div", "needle"); n.style.left = pos(value) + "%"; s.appendChild(n);
    return s;
  }

  function renderPrediction(b) {
    var card = el("section", "readout");
    card.setAttribute("aria-label", "Prediction result");
    if (b.task === "classification") {
      card.appendChild(el("div", "what", "Predicted " + b.target));
      card.appendChild(el("div", "value", b.label));
      if (b.probabilities) {
        card.appendChild(el("div", "sub", "Model probability by class"));
        var list = el("div", "probs");
        b.probabilities.slice(0, 8).forEach(function (p, i) {
          var row = el("div", "prob" + (i === 0 ? " top" : ""));
          var tr = el("span", "bar-track"), f = el("span", "bar-fill");
          f.style.width = (100 * p.p) + "%"; tr.appendChild(f);
          row.appendChild(el("span", "bar-label", p.label)); row.appendChild(tr); row.appendChild(el("span", "bar-val", pct(p.p)));
          list.appendChild(row);
        });
        card.appendChild(list);
      } else {
        card.appendChild(el("div", "sub", "This model does not output probabilities."));
      }
    } else {
      card.appendChild(el("div", "what", "Predicted " + b.target));
      card.appendChild(el("div", "value", fmt(b.value)));
      if (b.interval) {
        card.appendChild(el("div", "sub", Math.round(b.interval_level * 100) + "% interval: " + fmt(b.interval[0]) + " to " + fmt(b.interval[1])));
        var pad = (b.interval[1] - b.interval[0]) * 0.35 || 1;
        card.appendChild(scale(b.interval[0] - pad, b.interval[1] + pad, b.value, b.interval));
      }
    }
    return card;
  }

  function renderForecast(b) {
    var wrap = el("figure", "forecast"); wrap.style.margin = "0";
    wrap.appendChild(el("h3", null, "Forecast of " + b.target));
    var ns = "http://www.w3.org/2000/svg", W = 600, H = 220, P = { l: 46, r: 12, t: 12, b: 26 };
    var hist = (b.history || []).filter(function (v) { return typeof v === "number"; });
    var all = hist.concat(b.values);
    (b.intervals || []).forEach(function (iv) { all.push(iv[0], iv[1]); });
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), pad = (hi - lo) * 0.08 || 1;
    lo -= pad; hi += pad;
    var n = hist.length + b.values.length;
    var x = function (i) { return P.l + (W - P.l - P.r) * i / Math.max(1, n - 1); };
    var y = function (v) { return P.t + (H - P.t - P.b) * (1 - (v - lo) / (hi - lo)); };
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Line chart: last " + hist.length + " observations and " + b.values.length + " forecast steps");
    function mk(tag, attrs, cls) { var e = document.createElementNS(ns, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (cls) e.setAttribute("class", cls); svg.appendChild(e); return e; }
    for (var g = 0; g <= 3; g++) {
      var gv = lo + (hi - lo) * g / 3;
      mk("line", { x1: P.l, x2: W - P.r, y1: y(gv), y2: y(gv) }, "grid");
      var t = mk("text", { x: P.l - 6, y: y(gv) + 3, "text-anchor": "end" }, "ax"); t.textContent = fmt(gv, 1);
    }
    if (b.intervals && b.intervals.length) {
      var up = b.intervals.map(function (iv, i) { return x(hist.length + i) + "," + y(iv[1]); });
      var dn = b.intervals.map(function (iv, i) { return x(hist.length + i) + "," + y(iv[0]); }).reverse();
      mk("polygon", { points: up.concat(dn).join(" ") }, "band");
    }
    if (hist.length) mk("polyline", { points: hist.map(function (v, i) { return x(i) + "," + y(v); }).join(" ") }, "hist");
    var pts = (hist.length ? [x(hist.length - 1) + "," + y(hist[hist.length - 1])] : []).concat(
      b.values.map(function (v, i) { return x(hist.length + i) + "," + y(v); }));
    mk("polyline", { points: pts.join(" ") }, "pred");
    b.values.forEach(function (v, i) { mk("circle", { cx: x(hist.length + i), cy: y(v), r: 3 }, "pred-pt"); });
    var t1 = mk("text", { x: P.l, y: H - 8 }, "ax"); t1.textContent = "past " + hist.length + " steps";
    var t2 = mk("text", { x: W - P.r, y: H - 8, "text-anchor": "end" }, "ax"); t2.textContent = "next " + b.values.length;
    wrap.appendChild(svg);
    var lg = el("div", "legend");
    var a = el("span"); var ai = el("i"); ai.style.background = "var(--muted)"; a.appendChild(ai); a.appendChild(document.createTextNode("observed"));
    var c = el("span"); var ci = el("i"); ci.style.background = "var(--accent)"; c.appendChild(ci); c.appendChild(document.createTextNode("forecast"));
    lg.appendChild(a); lg.appendChild(c);
    if (b.level) { var d = el("span"); var di = el("i"); di.style.background = "var(--accent-soft)"; di.style.height = "8px"; d.appendChild(di); d.appendChild(document.createTextNode(Math.round(b.level * 100) + "% interval")); lg.appendChild(d); }
    wrap.appendChild(lg);
    return wrap;
  }

  function renderModel(b) {
    var d = el("details", "disc");
    var s = el("summary"); s.appendChild(el("span", null, "Model details")); s.appendChild(el("span", "sum-sub", b.title));
    d.appendChild(s);
    var body = el("div", "disc-body");
    var mg = el("div", "metrics");
    b.metrics.forEach(function (m) {
      var c = el("div", "metric"); c.appendChild(el("div", "m-name", m[0])); c.appendChild(el("div", "m-val", m[1])); mg.appendChild(c);
    });
    body.appendChild(mg);
    if (b.metrics_note) body.appendChild(el("p", "small", b.metrics_note));
    var dl = el("dl", "kv");
    b.rows.forEach(function (r) { dl.appendChild(el("dt", null, r[0])); dl.appendChild(el("dd", null, r[1])); });
    body.appendChild(dl);
    if (b.preprocessing && b.preprocessing.length) {
      body.appendChild(el("p", "small", "Preprocessing (fitted on training rows only)"));
      body.appendChild(renderText(b.preprocessing.map(function (p) { return "- " + p; }).join("\n")));
    }
    d.appendChild(body);
    return d;
  }

  function renderDetails(b) {
    var d = el("details", "disc");
    var s = el("summary"); s.appendChild(el("span", null, b.title)); d.appendChild(s);
    var body = el("div", "disc-body");
    b.blocks.forEach(function (x) { var r = renderBlock(x); if (r) body.appendChild(r); });
    d.appendChild(body);
    return d;
  }

  function renderNotice(b) {
    var n = el("div", "notice " + (b.level || "info"));
    n.setAttribute("role", b.level === "error" ? "alert" : "note");
    n.appendChild(icon(b.level === "error" ? "error" : b.level === "warning" ? "warn" : "info"));
    n.appendChild(el("span", null, b.text));
    return n;
  }

  function renderDownload(b) {
    var btn = el("button", "download"); btn.type = "button";
    btn.appendChild(icon("download")); btn.appendChild(document.createTextNode(b.label));
    btn.addEventListener("click", function () {
      api("GET", "/api/download/" + encodeURIComponent(b.file_id)).then(function (r) {
        if (!r.ok) throw new Error("expired");
        var name = (r.headers.get("Content-Disposition") || "").match(/filename="([^"]+)"/);
        return r.blob().then(function (blob) {
          var a = document.createElement("a");
          a.href = URL.createObjectURL(blob); a.download = name ? name[1] : "predictions.csv";
          document.body.appendChild(a); a.click();
          setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
        });
      }).catch(function () { btn.textContent = "Download expired — run the prediction again"; btn.disabled = true; });
    });
    return btn;
  }

  function renderBlock(b) {
    if (!b) return null;
    switch (b.type) {
      case "text": return renderText(b.text);
      case "notice": return renderNotice(b);
      case "table": return renderTable(b);
      case "bars": return renderBars(b);
      case "prediction": return renderPrediction(b);
      case "forecast": return renderForecast(b);
      case "model": return renderModel(b);
      case "details": return renderDetails(b);
      case "download": return renderDownload(b);
      case "source": return el("p", "source", b.text);
      default: return null;
    }
  }

  // ------------------------------------------------------------------ plain text (copy)
  function plain(m) {
    var out = [];
    (m.blocks || []).forEach(function walk(b) {
      if (!b) return;
      if (b.type === "text" || b.type === "notice" || b.type === "source") out.push(String(b.text).replace(/\*\*|`/g, ""));
      else if (b.type === "table") { out.push(b.columns.join("\t")); b.rows.forEach(function (r) { out.push(r.map(function (v) { return v === null ? "" : v; }).join("\t")); }); }
      else if (b.type === "bars") { out.push(b.title); b.items.forEach(function (i) { out.push(i.label + "\t" + i.value); }); }
      else if (b.type === "prediction") out.push("Predicted " + b.target + ": " + (b.label !== undefined ? b.label : b.value) + (b.interval ? " (interval " + b.interval[0] + " to " + b.interval[1] + ")" : ""));
      else if (b.type === "forecast") out.push("Forecast " + b.target + ": " + b.values.join(", "));
      else if (b.type === "model") { out.push(b.title); b.metrics.forEach(function (x) { out.push(x[0] + ": " + x[1]); }); }
      else if (b.type === "details") b.blocks.forEach(walk);
    });
    return out.join("\n");
  }
  function copyText(text, btn) {
    var done = function () {
      var label = btn.lastChild; var old = label.textContent;
      btn.replaceChild(icon("check"), btn.firstChild); label.textContent = "Copied";
      setTimeout(function () { btn.replaceChild(icon("copy"), btn.firstChild); label.textContent = old; }, 1500);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { legacyCopy(text) && done(); });
    } else if (legacyCopy(text)) done();
  }
  function legacyCopy(text) {
    var ta = el("textarea"); ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    var ok = false; try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    ta.remove(); return ok;
  }

  // ------------------------------------------------------------------ message rendering
  function scrollDown() { thread.scrollTop = thread.scrollHeight; }

  function addUser(text) {
    var m = el("div", "msg user");
    m.appendChild(el("div", "bubble", text));
    log.appendChild(m); scrollDown();
    return m;
  }

  function addAssistant(m, opts) {
    opts = opts || {};
    var wrap = el("article", "msg assistant" + (m.state === "error" ? " error" : ""));
    wrap.setAttribute("aria-label", "EPM Chat reply");
    var body = el("div", "body");
    (m.blocks || []).forEach(function (b) { var r = renderBlock(b); if (r) body.appendChild(r); });
    wrap.appendChild(body);
    if (m.job && m.job.state && (m.job.state === "queued" || m.job.state === "running")) {
      wrap._job = renderJob(m.job); body.appendChild(wrap._job.el);
    }
    var meta = el("div", "meta");
    var copy = el("button"); copy.type = "button"; copy.appendChild(icon("copy")); copy.appendChild(el("span", null, "Copy"));
    copy.setAttribute("aria-label", "Copy this reply");
    copy.addEventListener("click", function () { copyText(plain(m), copy); });
    meta.appendChild(copy);
    if (opts.retry !== false && lastUser) {
      var retry = el("button"); retry.type = "button"; retry.appendChild(icon("retry")); retry.appendChild(el("span", null, "Retry"));
      retry.setAttribute("aria-label", "Send the previous message again");
      retry.className = "retry";
      retry.addEventListener("click", function () { if (!busy && lastUser) send(lastUser.text, lastUser.kind === "file" ? null : undefined); });
      meta.appendChild(retry);
    }
    wrap.appendChild(meta);
    log.appendChild(wrap);
    setChips(m.suggestions || []);
    scrollDown();
    return wrap;
  }

  function renderJob(job) {
    var box = el("div", "job"); box.setAttribute("role", "status");
    var head = el("div", "job-head"); var title = el("span", null, job.label); var p = el("span", "job-pct", "0%");
    head.appendChild(title); head.appendChild(p);
    var track = el("div", "job-track"); var fill = el("div", "job-fill"); track.appendChild(fill);
    track.setAttribute("role", "progressbar"); track.setAttribute("aria-valuemin", "0"); track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-label", job.label);
    var msg = el("div", "job-msg"); var mt = el("span", null, job.message); var cancel = el("button", null, "Cancel");
    cancel.type = "button";
    cancel.addEventListener("click", function () {
      cancel.disabled = true; cancel.textContent = "Cancelling…";
      api("POST", "/api/jobs/" + job.id + "/cancel").catch(function () { /* the stream reports the outcome */ });
    });
    msg.appendChild(mt); msg.appendChild(cancel);
    box.appendChild(head); box.appendChild(track); box.appendChild(msg);
    function update(j) {
      var v = Math.round((j.progress || 0) * 100);
      fill.style.width = v + "%"; p.textContent = v + "%"; track.setAttribute("aria-valuenow", String(v));
      mt.textContent = j.message + (j.elapsed_s ? " (" + Math.round(j.elapsed_s) + " s)" : "");
      if (j.state !== "queued" && j.state !== "running") { cancel.remove(); }
    }
    update(job);
    return { el: box, update: update };
  }

  function setChips(list) {
    if (chipsEl) chipsEl.remove();
    chipsEl = null;
    if (!list.length) return;
    chipsEl = el("div", "chips");
    chipsEl.setAttribute("role", "group"); chipsEl.setAttribute("aria-label", "Suggested messages");
    list.forEach(function (s) {
      var c = el("button", "chip", s); c.type = "button";
      c.addEventListener("click", function () { if (!busy) send(s); });
      chipsEl.appendChild(c);
    });
    log.appendChild(chipsEl);
  }

  function typing() {
    var t = el("div", "typing"); t.setAttribute("aria-label", "EPM Chat is working");
    t.appendChild(el("i")); t.appendChild(el("i")); t.appendChild(el("i"));
    log.appendChild(t); scrollDown();
    return t;
  }

  function systemError(text, retryable) {
    var m = { blocks: [{ type: "notice", level: "error", text: text }], suggestions: [], state: "error" };
    addAssistant(m, { retry: retryable });
  }

  function clearRetries() {
    Array.prototype.forEach.call(log.querySelectorAll(".meta .retry"), function (b) { b.remove(); });
  }

  // ------------------------------------------------------------------ networking
  function api(method, path, body, extra) {
    var h = { "Accept": "application/json" };
    if (token) h["Authorization"] = "Bearer " + token;
    var init = { method: method, headers: h, credentials: "omit", cache: "no-store" };
    if (body !== undefined && body !== null) {
      if (body instanceof Blob || body instanceof ArrayBuffer) { init.body = body; }
      else { h["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
    }
    if (extra) for (var k in extra) h[k] = extra[k];
    return fetch(path, init);
  }

  function errorText(r) {
    return r.json().then(function (j) { return j.error || ("Request failed (" + r.status + ")."); },
      function () { return "Request failed (" + r.status + ")."; });
  }

  function openSession(code) {
    return api("POST", "/api/session", code ? { access_code: code } : {}).then(function (r) {
      if (r.status === 401 && cfg.access_code_required) { token = null; showGate(); throw new Error("gate"); }
      if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function (j) {
      token = j.token; store.set("epm-chat-token", token);
      log.textContent = ""; chipsEl = null;
      j.messages.forEach(function (m) {
        if (m.role === "user") addUser(plain(m)); else addAssistant(m, { retry: false });
      });
      return j;
    });
  }

  function ensureSession() {
    if (token) return Promise.resolve();
    return openSession();
  }

  // Parses a text/event-stream response body incrementally.
  function readStream(resp, on) {
    var reader = resp.body.getReader(), dec = new TextDecoder(), buf = "";
    function pump() {
      return reader.read().then(function (r) {
        if (r.done) return;
        buf += dec.decode(r.value, { stream: true });
        var idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          var chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
          var ev = "message", data = "";
          chunk.split("\n").forEach(function (l) {
            if (l.indexOf("event:") === 0) ev = l.slice(6).trim();
            else if (l.indexOf("data:") === 0) data += l.slice(5).trim();
          });
          try { on(ev, data ? JSON.parse(data) : {}); } catch (e) { /* ignore malformed event */ }
        }
        return pump();
      });
    }
    return pump();
  }

  function send(text, fileMode) {
    text = String(text || "").trim();
    if (!text || busy) return;
    if (text.length > cfg.max_message_chars) { systemError("Messages are limited to " + cfg.max_message_chars + " characters.", false); return; }
    busy = true; setBusy(true); clearRetries(); setChips([]);
    lastUser = { text: text, kind: "text" };
    addUser(text);
    input.value = ""; autosize(); updateSend();
    var dots = typing(), live = null, liveText = "", replyEl = null, retried = false;

    function attempt() {
      return ensureSession().then(function () {
        return api("POST", "/api/chat?stream=1", { message: text }, { "Accept": "text/event-stream" });
      }).then(function (r) {
        if (r.status === 401 && !retried) { retried = true; token = null; store.del("epm-chat-token"); return attempt(); }
        if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
        var ctype = r.headers.get("Content-Type") || "";
        if (ctype.indexOf("text/event-stream") < 0 || !r.body) {
          return r.json().then(function (j) { dots.remove(); addAssistant(j.message); });
        }
        return readStream(r, function (ev, d) {
          if (ev === "delta") {
            if (!live) { dots.remove(); live = el("article", "msg assistant"); var b = el("div", "body"); live.appendChild(b); log.appendChild(live); }
            liveText += d.text; live.firstChild.textContent = liveText; scrollDown();
          } else if (ev === "message") {
            dots.remove(); if (live) { live.remove(); live = null; }
            replyEl = addAssistant(d.message);
          } else if (ev === "job") {
            if (replyEl && replyEl._job) replyEl._job.update(d.job);
          } else if (ev === "reply") {
            if (replyEl && replyEl._job) { replyEl._job.update(d.message.job || {}); replyEl._job.el.remove(); }
            addAssistant(d.message);
          }
        });
      });
    }
    attempt().catch(function (e) {
      dots.remove();
      if (e && e.message === "gate") return;
      var msg = (e && e.message && !/Failed to fetch|NetworkError|Load failed/i.test(e.message)) ? e.message :
        "Can't reach the EPM Chat server. Check your connection, then use Retry.";
      systemError(msg, true);
    }).then(function () {
      busy = false; setBusy(false);
      if (!EMBED || document.hasFocus()) input.focus();
    });
  }

  function upload(file) {
    if (busy || !file) return;
    if (!/\.csv$/i.test(file.name)) { systemError("Only .csv files can be uploaded.", false); return; }
    if (file.size > cfg.max_upload_bytes) {
      systemError("That file is " + (file.size / 1048576).toFixed(1) + " MB; the limit is " + Math.round(cfg.max_upload_bytes / 1048576) + " MB.", false);
      return;
    }
    busy = true; setBusy(true); clearRetries(); setChips([]);
    addUser("📎 " + file.name);
    var dots = typing();
    ensureSession().then(function () {
      return api("POST", "/api/upload", file, { "Content-Type": "text/csv", "X-Filename": file.name.replace(/[^\x20-\x7e]/g, "_") });
    }).then(function (r) {
      if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function (j) { dots.remove(); addAssistant(j.message, { retry: false }); })
      .catch(function (e) { dots.remove(); systemError(e.message || "Upload failed.", false); })
      .then(function () { busy = false; setBusy(false); fileInput.value = ""; });
  }

  // ------------------------------------------------------------------ composer
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; }
  function updateSend() {
    var n = input.value.length, c = $("counter");
    sendBtn.disabled = busy || !input.value.trim() || n > cfg.max_message_chars;
    c.textContent = n > cfg.max_message_chars * 0.8 ? n + " / " + cfg.max_message_chars : "";
    c.className = n > cfg.max_message_chars ? "over" : "";
  }
  function setBusy(b) {
    composer.setAttribute("aria-busy", b ? "true" : "false");
    attachBtn.disabled = b;
    Array.prototype.forEach.call(document.querySelectorAll(".chip"), function (c) { c.disabled = b; });
    updateSend();
  }
  input.addEventListener("input", function () { autosize(); updateSend(); });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!sendBtn.disabled) send(input.value); }
    if (e.key === "Escape" && EMBED) postParent({ type: "epm-chat:close" });
  });
  composer.addEventListener("submit", function (e) { e.preventDefault(); send(input.value); });
  attachBtn.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { upload(fileInput.files && fileInput.files[0]); });
  // drag & drop onto the conversation
  thread.addEventListener("dragover", function (e) { e.preventDefault(); });
  thread.addEventListener("drop", function (e) { e.preventDefault(); var f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) upload(f); });

  $("clear-btn").addEventListener("click", function () {
    if (busy) return;
    if (!window.confirm("Start a new chat? This clears the conversation, your uploaded files and models trained in this chat.")) return;
    busy = true; setBusy(true);
    ensureSession().then(function () { return api("DELETE", "/api/session"); }).then(function (r) {
      if (!r.ok) return errorText(r).then(function (t) { throw new Error(t); });
      return r.json();
    }).then(function (j) {
      log.textContent = ""; chipsEl = null; lastUser = null; addAssistant(j.message, { retry: false });
    }).catch(function (e) { systemError(e.message || "Couldn't clear the chat.", false); })
      .then(function () { busy = false; setBusy(false); input.focus(); });
  });

  // ------------------------------------------------------------------ access gate
  function showGate() {
    $("gate").hidden = false; $("gate-code").focus();
  }
  $("gate-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var code = $("gate-code").value;
    $("gate-error").textContent = "";
    openSession(code).then(function () { $("gate").hidden = true; input.focus(); })
      .catch(function (err) { if (err.message !== "gate") $("gate-error").textContent = err.message; else $("gate-error").textContent = "That code isn't valid."; });
  });

  // ------------------------------------------------------------------ embedding (iframe ⇄ host page)
  function postParent(msg) { if (EMBED && parentOrigin) window.parent.postMessage(msg, parentOrigin); }
  function setupEmbed() {
    if (!EMBED) return;
    document.body.classList.add("embed");
    var claimed = (params.get("parent") || "").replace(/\/$/, "").toLowerCase();
    var ancestor = (location.ancestorOrigins && location.ancestorOrigins.length) ? location.ancestorOrigins[0].toLowerCase() : null;
    var allowed = (cfg.embed_origins || []).map(function (o) { return o.toLowerCase(); });
    // Only talk to a parent that is on the server's allowlist (and, where the browser tells us, really is our parent).
    if (claimed && allowed.indexOf(claimed) >= 0 && (!ancestor || ancestor === claimed)) parentOrigin = claimed;
    $("close-btn").addEventListener("click", function () { postParent({ type: "epm-chat:close" }); });
    if (!parentOrigin) $("close-btn").hidden = true;
    window.addEventListener("message", function (e) {
      if (!parentOrigin || e.origin !== parentOrigin || e.source !== window.parent) return;
      var d = e.data || {};
      if (d.type === "epm-chat:focus") input.focus();
      else if (d.type === "epm-chat:theme" && (d.theme === "light" || d.theme === "dark")) applyTheme(d.theme);
      else if (d.type === "epm-chat:prefill" && typeof d.text === "string") { input.value = d.text.slice(0, cfg.max_message_chars); autosize(); updateSend(); input.focus(); }
    });
    postParent({ type: "epm-chat:ready" });
  }

  // ------------------------------------------------------------------ boot
  fetch("/api/config", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (c) {
    cfg = c; input.maxLength = c.max_message_chars;
    setupEmbed();
    token = store.get("epm-chat-token");
    return openSession().catch(function (e) {
      if (e.message === "gate") return;
      if (token) { token = null; store.del("epm-chat-token"); return openSession(); }
      throw e;
    });
  }).catch(function (e) {
    if (e && e.message === "gate") return;
    systemError((e && e.message && e.message !== "Failed to fetch") ? e.message : "Can't reach the EPM Chat server. Reload the page to try again.", false);
  }).then(function () { updateSend(); if (!EMBED) input.focus(); });
})();
