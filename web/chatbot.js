/**
 * EPM — Chat panel ("EPM CHAT // SESSION …")
 * Floating, collapsible terminal-style assistant that talks to POST /api/chat on the
 * local server (`python run.py --serve`). Long runs come back with a job_id; the panel
 * polls GET /api/chat/job until the formatted results are ready.
 * Vanilla JS, no dependencies. All server text is escaped before rendering.
 */
(function () {
  'use strict';

  const REDUCED_MOTION = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const C = { sessionId: null, live: false, busy: false, pollTimer: null, history: [] };

  // ------------------------------------------------------------- utils
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v === null || v === undefined ? '' : v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  const fmt = (v) => {
    if (v === null || v === undefined || v === '') return '—';
    if (typeof v === 'number') {
      if (!isFinite(v)) return '—';
      if (Number.isInteger(v)) return v.toLocaleString();
      const a = Math.abs(v);
      return a !== 0 && (a < 1e-3 || a >= 1e6) ? v.toExponential(2) : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
    }
    return esc(v);
  };
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch (_) { /* storage unavailable */ } },
  };

  /** Minimal, safe markdown: escape first, then **bold**, `code`, "- " bullets, line breaks. */
  function md(text) {
    const lines = esc(text).split('\n');
    let html = '', inList = false;
    for (const raw of lines) {
      let l = raw.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+?)`/g, '<code>$1</code>');
      const bullet = /^\s*- /.test(raw);
      if (bullet && !inList) { html += '<ul>'; inList = true; }
      if (!bullet && inList) { html += '</ul>'; inList = false; }
      html += bullet ? `<li>${l.replace(/^\s*- /, '')}</li>` : (l.trim() ? `<p>${l}</p>` : '');
    }
    return html + (inList ? '</ul>' : '');
  }

  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts || {}));
    let body = null;
    try { body = await res.json(); } catch (_) { body = null; }
    if (!res.ok) {
      const err = new Error((body && (body.reply || body.error)) || `HTTP ${res.status}`);
      err.status = res.status; err.body = body; throw err;
    }
    return body;
  }

  // ------------------------------------------------------------- rendering
  function scrollDown() { const log = $('chat-log'); log.scrollTop = log.scrollHeight; }

  function addUser(text) {
    const el = document.createElement('div');
    el.className = 'chat-msg chat-user';
    el.innerHTML = `<span class="chat-prompt">you ›</span> ${esc(text)}`;
    $('chat-log').appendChild(el);
    scrollDown();
  }

  function tableHtml(t) {
    const head = t.columns.map((c) => `<th>${esc(c)}</th>`).join('');
    const rows = t.rows.map((r) => `<tr>${r.map((v) => `<td>${fmt(v)}</td>`).join('')}</tr>`).join('');
    return `<div class="chat-table-title">${esc(t.title || '')}</div>
      <div class="chat-table-wrap"><table class="chat-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      ${t.note ? `<div class="chat-note">${esc(t.note)}</div>` : ''}`;
  }

  function typeInto(el, html) {
    if (REDUCED_MOTION) { el.innerHTML = html; return Promise.resolve(); }
    // reveal whole blocks quickly (keeps markup valid) — a light "typing" effect
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const blocks = Array.from(tmp.childNodes);
    el.innerHTML = '';
    return new Promise((resolve) => {
      let i = 0;
      const step = () => {
        if (i >= blocks.length) { resolve(); return; }
        el.appendChild(blocks[i++]);
        scrollDown();
        setTimeout(step, 35);
      };
      step();
    });
  }

  async function addBot(resp, existing) {
    const el = existing || document.createElement('div');
    el.className = 'chat-msg chat-bot' + (resp.kind === 'error' ? ' chat-error' : '');
    if (!existing) $('chat-log').appendChild(el);
    const body = document.createElement('div');
    body.className = 'chat-body';
    el.innerHTML = '<span class="chat-prompt">epm ›</span>';
    el.appendChild(body);
    let html = md(resp.reply || resp.text || '');
    (resp.tables || []).forEach((t) => { html += tableHtml(t); });
    await typeInto(body, html);
    if (resp.data && resp.data.samples_csv && resp.data.run_id) {
      const b = document.createElement('button');
      b.className = 'chat-pill';
      b.textContent = 'DOWNLOAD SAMPLES CSV';
      b.addEventListener('click', () => downloadSamples(resp.data.run_id));
      body.appendChild(b);
    }
    renderSuggestions(resp.suggestions || []);
    $('chat-state').textContent = (resp.state || 'idle').toUpperCase();
    scrollDown();
    return el;
  }

  function renderSuggestions(list) {
    const box = $('chat-suggestions');
    box.innerHTML = '';
    list.slice(0, 4).forEach((s) => {
      const b = document.createElement('button');
      b.className = 'chat-pill';
      b.type = 'button';
      b.textContent = s;
      b.addEventListener('click', () => send(s));
      box.appendChild(b);
    });
  }

  function progressEl(text) {
    const el = document.createElement('div');
    el.className = 'chat-msg chat-bot chat-progress';
    el.innerHTML = `<span class="chat-prompt">epm ›</span><div class="chat-body">${md(text)}
      <div class="chat-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="chat-bar-fill"></div></div>
      <div class="chat-bar-msg">Queued…</div></div>`;
    $('chat-log').appendChild(el);
    scrollDown();
    return el;
  }

  // ------------------------------------------------------------- networking
  async function send(text) {
    const msg = (text || '').trim();
    if (!msg || C.busy && !/^(cancel|stop|status)\b/i.test(msg)) return;
    if (!C.live) { addUser(msg); addBot({ reply: offlineText(), kind: 'error', suggestions: [] }); return; }
    addUser(msg);
    $('chat-input').value = '';
    renderSuggestions([]);
    try {
      const r = await api('/api/chat', { method: 'POST', body: JSON.stringify({ session_id: C.sessionId, message: msg }) });
      C.sessionId = r.session_id; store.set('epm-chat-session', r.session_id);
      $('chat-session').textContent = r.session_id;
      if (r.job_id && r.state === 'executing' && r.kind === 'progress') {
        poll(r.job_id, progressEl(r.reply));
      } else {
        await addBot(r);
      }
    } catch (e) {
      addBot({ reply: e.status === 501 ? offlineText() : `Request failed: ${e.message}`, kind: 'error', suggestions: ['help'] });
    }
  }

  function poll(jobId, el) {
    C.busy = true;
    setBusy(true);
    clearTimeout(C.pollTimer);
    const tick = async () => {
      try {
        const r = await api(`/api/chat/job?session_id=${encodeURIComponent(C.sessionId)}&job_id=${encodeURIComponent(jobId)}`);
        if (r.state === 'executing') {
          const pct = Math.round(((r.data && r.data.progress) || 0) * 100);
          const fill = el.querySelector('.chat-bar-fill');
          if (fill) fill.style.width = pct + '%';
          const bar = el.querySelector('.chat-bar');
          if (bar) bar.setAttribute('aria-valuenow', String(pct));
          const m = el.querySelector('.chat-bar-msg');
          if (m) m.textContent = `${pct}% · ${r.reply || ''}`;
          C.pollTimer = setTimeout(tick, 700);
          return;
        }
        const fill = el.querySelector('.chat-bar-fill');
        if (fill) fill.style.width = '100%';
        const doneMsg = el.querySelector('.chat-bar-msg');
        if (doneMsg) doneMsg.textContent = r.intent === 'cancel' ? 'Cancelled' : (r.kind === 'error' ? 'Failed' : '100% · Done');
        el.classList.remove('chat-progress');
        const note = document.createElement('div');
        note.className = 'chat-msg chat-bot';
        $('chat-log').appendChild(note);
        await addBot(r, note);
      } catch (e) {
        addBot({ reply: `Lost contact with the local server: ${e.message}`, kind: 'error', suggestions: ['status'] });
      }
      C.busy = false;
      setBusy(false);
    };
    C.pollTimer = setTimeout(tick, 500);
  }

  function setBusy(on) {
    $('chat-send').textContent = on ? 'WAIT' : 'SEND';
    $('chat-panel').classList.toggle('is-busy', on);
  }

  async function downloadSamples(runId) {
    try {
      const r = await api(`/api/chat/samples?run_id=${encodeURIComponent(runId)}`);
      const blob = new Blob([r.csv], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = r.filename || 'samples.csv';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    } catch (e) { addBot({ reply: `Could not download samples: ${e.message}`, kind: 'error' }); }
  }

  function offlineText() {
    return 'The chatbot trains real models, so it needs the local server. Run `python run.py --serve` and open this page from http://localhost:3000 (or use `python run.py --chat` in a terminal).';
  }

  // ------------------------------------------------------------- panel
  function toggle(open) {
    const p = $('chat-panel');
    const willOpen = open === undefined ? p.hidden : open;
    p.hidden = !willOpen;
    $('chat-launcher').setAttribute('aria-expanded', String(willOpen));
    store.set('epm-chat-open', willOpen ? '1' : '0');
    if (willOpen) setTimeout(() => $('chat-input').focus(), 30);
  }

  async function init() {
    if (!$('chat-panel')) return;
    $('chat-launcher').addEventListener('click', () => toggle());
    $('chat-close').addEventListener('click', () => toggle(false));
    const nav = $('nav-chat');
    if (nav) nav.addEventListener('click', (e) => { e.preventDefault(); toggle(true); });
    $('chat-form').addEventListener('submit', (e) => { e.preventDefault(); send($('chat-input').value); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('chat-panel').hidden) toggle(false); });

    try {
      const cap = await api('/api/capabilities');
      C.live = !!(cap && cap.training);
    } catch (_) { C.live = false; }
    $('chat-mode').textContent = C.live ? 'LIVE' : 'OFFLINE';
    $('chat-mode').className = 'chat-mode ' + (C.live ? 'is-live' : 'is-offline');
    C.sessionId = store.get('epm-chat-session') || null;
    $('chat-session').textContent = C.sessionId || 'NEW';

    let restored = false;
    if (C.live && C.sessionId) {
      try {
        const h = await api(`/api/chat/sessions?session_id=${encodeURIComponent(C.sessionId)}`);
        (h.history || []).slice(-12).forEach((m) => {
          if (m.role === 'user') addUser(m.text);
          else {
            const el = document.createElement('div');
            el.className = 'chat-msg chat-bot';
            el.innerHTML = `<span class="chat-prompt">epm ›</span><div class="chat-body">${md(m.text)}</div>`;
            $('chat-log').appendChild(el);
          }
        });
        restored = (h.history || []).length > 0;
      } catch (_) { /* session expired on the server: start fresh */ }
    }
    if (!restored) {
      addBot(C.live
        ? { reply: 'Hi — describe an experiment in plain English, e.g. **classify iris with random forest**. Say **help** for more.',
            suggestions: ['classify iris with random forest', 'list datasets', 'explain flow matching', 'help'] }
        : { reply: offlineText(), kind: 'error', suggestions: [] });
    }
    if (store.get('epm-chat-open') === '1') toggle(true);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
