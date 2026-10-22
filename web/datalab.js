/**
 * EPM — Experiment Lab (dataset catalogue × algorithm benchmarking)
 * Vanilla JS, no dependencies. Talks to the local experiment API served by
 * `python run.py --serve`; falls back to the static catalogue in catalog/*.json
 * (e.g. on Vercel), where training is disabled and that is stated explicitly.
 */
(function () {
  'use strict';

  const TASKS = [
    ['classification', 'CLASSIFICATION'],
    ['regression', 'REGRESSION'],
    ['clustering', 'CLUSTERING'],
    ['forecasting', 'FORECASTING'],
    ['anomaly_detection', 'ANOMALY DETECTION'],
  ];
  const PRIMARY = { regression: ['rmse', 'min'], classification: ['f1', 'max'], clustering: ['silhouette', 'max'],
    forecasting: ['rmse', 'min'], anomaly_detection: ['pr_auc', 'max'] };
  const COLS = {
    regression: ['rmse', 'mae', 'r2'],
    classification: ['f1', 'accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc'],
    clustering: ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'n_clusters', 'ari_external'],
    forecasting: ['rmse', 'mae', 'smape', 'mase', 'skill_vs_naive'],
    anomaly_detection: ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1'],
  };
  const LOWER_BETTER = new Set(['rmse', 'mae', 'mse', 'smape', 'mase', 'davies_bouldin']);

  const S = {
    mode: 'loading', task: 'classification', datasets: [], algorithms: [], datasetId: null,
    selected: new Set(), compat: null, job: null, pollTimer: null, result: null, sortBudget: 'quick',
  };

  // ---------------------------------------------------------------- utilities
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v === null || v === undefined ? '' : v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  const fmt = (v, d = 4) => {
    if (v === null || v === undefined || v === '') return '—';
    if (typeof v === 'number') {
      if (!isFinite(v)) return '—';
      if (Number.isInteger(v)) return v.toLocaleString();
      const a = Math.abs(v);
      return a !== 0 && (a < 1e-3 || a >= 1e6) ? v.toExponential(2) : v.toFixed(d);
    }
    return esc(v);
  };
  const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : null);
  const unit = (v, d, u) => (typeof v === 'number' && isFinite(v) ? fmt(v, d) + u : '—');
  const fmtRows = (n) => (n ? Number(n).toLocaleString() : '—');

  async function api(path, opts = {}) {
    const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts));
    let body = null;
    try { body = await res.json(); } catch (_) { body = null; }
    if (!res.ok) {
      const msg = (body && body.error) || `HTTP ${res.status}`;
      const err = new Error(msg); err.status = res.status; err.body = body; throw err;
    }
    return body;
  }

  function setNotice(html, kind) {
    const el = $('dl-notice');
    if (!el) return;
    el.className = 'dl-notice' + (kind ? ' dl-notice-' + kind : '');
    el.innerHTML = html;
    el.hidden = !html;
  }

  // ---------------------------------------------------------------- bootstrap
  async function init() {
    if (!$('datalab')) return;
    renderTaskTabs();
    bindStaticControls();
    try {
      const cap = await api('/api/capabilities');
      S.mode = cap && cap.training ? 'local' : 'static';
    } catch (_) { S.mode = 'static'; }
    $('dl-mode').textContent = S.mode === 'local' ? 'LOCAL API: TRAINING ENABLED' : 'STATIC CATALOGUE: READ-ONLY';
    $('dl-mode').className = 'dl-mode-pill ' + (S.mode === 'local' ? 'is-live' : 'is-static');
    if (S.mode === 'static') {
      setNotice('Static mode: the catalogue and past runs below are read from <code>catalog/*.json</code>. ' +
        'Previewing, downloading, uploading and training need the local server: <code>python run.py --serve</code>.', 'info');
    }
    await Promise.all([loadAlgorithms(), loadDatasets()]);
    loadHistory();
  }

  async function loadAlgorithms() {
    try {
      const d = S.mode === 'local' ? await api('/api/algorithms') : await api('catalog/algorithms.json');
      S.algorithms = d.algorithms || [];
    } catch (e) { S.algorithms = []; setNotice('Could not load the algorithm catalogue: ' + esc(e.message), 'error'); }
    renderAlgoFinder();
  }

  async function loadDatasets() {
    $('dl-dataset-tbody').innerHTML = '<tr><td colspan="6">Loading catalogue…</td></tr>';
    try {
      const d = S.mode === 'local' ? await api('/api/datasets') : await api('catalog/datasets.json');
      S.datasets = d.datasets || [];
    } catch (e) {
      S.datasets = [];
      setNotice('Could not load the dataset catalogue: ' + esc(e.message), 'error');
    }
    renderDatasetTable();
  }

  // ---------------------------------------------------------------- task tabs
  function renderTaskTabs() {
    $('dl-task-tabs').innerHTML = TASKS.map(([id, label]) =>
      `<button class="lb-tab${id === S.task ? ' active' : ''}" data-task="${id}" role="tab" aria-selected="${id === S.task}">${label}</button>`).join('');
    $('dl-task-tabs').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      S.task = b.dataset.task; S.datasetId = null; S.compat = null; S.selected.clear();
      renderTaskTabs(); renderDatasetTable(); renderDetail(); renderAlgorithms(); renderConfig(); renderAlgoFinder();
    }));
  }

  // ---------------------------------------------------------------- datasets
  function datasetsForTask() {
    const list = S.datasets.filter((d) => (d.tasks || []).includes(S.task));
    const rank = { small: 0, medium: 1, large: 2 };
    list.sort((a, b) => {
      const r = (rank[a.size_class] - rank[b.size_class]) || ((a.n_samples || 0) - (b.n_samples || 0));
      return S.sortBudget === 'realistic' ? -r : r;
    });
    return list;
  }

  function statusBadge(d) {
    const st = (d.status && d.status.state) || 'static';
    const label = { available: 'READY', cached: 'CACHED', not_downloaded: 'DOWNLOAD', error: 'ERROR', static: d.requires_network ? 'REMOTE' : 'BUNDLED' }[st] || st.toUpperCase();
    return `<span class="dl-badge dl-badge-${esc(st)}" title="${esc((d.status && d.status.last_error) || '')}">${label}</span>`;
  }

  function renderDatasetTable() {
    const rows = datasetsForTask();
    if (!rows.length) {
      $('dl-dataset-tbody').innerHTML = '<tr><td colspan="6">No registered dataset supports this task.</td></tr>';
      return;
    }
    $('dl-dataset-tbody').innerHTML = rows.map((d) => `
      <tr class="dl-row${d.id === S.datasetId ? ' row-active' : ''}" data-id="${esc(d.id)}" tabindex="0">
        <td><strong>${esc(d.name)}</strong></td>
        <td>${esc(d.source)}</td>
        <td>${fmtRows(d.n_samples)}</td>
        <td>${fmt(d.n_features, 0)}</td>
        <td><span class="dl-size dl-size-${esc(d.size_class)}">${esc((d.size_class || '').toUpperCase())}</span></td>
        <td>${statusBadge(d)}</td>
      </tr>`).join('');
    $('dl-dataset-tbody').querySelectorAll('tr.dl-row').forEach((tr) => {
      const pick = () => selectDataset(tr.dataset.id);
      tr.addEventListener('click', pick);
      tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    });
  }

  async function selectDataset(id) {
    S.datasetId = id; S.result = null;
    renderDatasetTable(); renderDetail(); renderConfig();
    await refreshCompatibility();
  }

  function currentDataset() { return S.datasets.find((d) => d.id === S.datasetId) || null; }

  function renderDetail() {
    const d = currentDataset();
    const box = $('dl-detail');
    if (!d) { box.innerHTML = '<p class="dl-empty">Select a dataset to see its source, licence, schema and compatible algorithms.</p>'; return; }
    const src = safeUrl(d.source_url);
    const dl = safeUrl(d.download_url);
    const prep = (d.preprocessing || []).map((p) => `<li>${esc(p)}</li>`).join('');
    const caveats = (d.caveats || []).map((c) => `<li>${esc(c)}</li>`).join('');
    const metrics = ((d.metrics || {})[S.task] || []).map((m) => `<code>${esc(m)}</code>`).join(' ');
    const isLocal = S.mode === 'local';
    const st = (d.status && d.status.state) || 'static';
    const canPreview = isLocal && (st === 'available' || st === 'cached');
    const needsDl = isLocal && (st === 'not_downloaded' || st === 'error');
    box.innerHTML = `
      <div class="panel-tag">${esc(d.id)}</div>
      <h3 class="dl-detail-title">${esc(d.name)}</h3>
      <p class="panel-desc">${esc(d.description)}</p>
      <table class="dl-kv">
        <tr><th>SOURCE</th><td>${src ? `<a href="${esc(src)}" target="_blank" rel="noopener noreferrer">${esc(d.source)}</a>` : esc(d.source)}</td></tr>
        <tr><th>DOWNLOAD</th><td>${esc(d.download_method)}${dl ? ` · <a href="${esc(dl)}" target="_blank" rel="noopener noreferrer">link</a>` : ''}</td></tr>
        <tr><th>LICENCE</th><td>${esc(d.license)}</td></tr>
        <tr><th>TARGET</th><td>${d.target ? `<code>${esc(d.target)}</code> (${esc(d.target_type)}${d.label_role !== 'target' ? `, ${esc(d.label_role.replace(/_/g, ' '))}` : ''})` : '—'}</td></tr>
        <tr><th>SIZE</th><td>${fmtRows(d.n_samples)} rows × ${fmt(d.n_features, 0)} features · ~${fmt(d.approx_memory_mb, 1)} MB in memory${d.approx_download_mb ? ` · ${fmt(d.approx_download_mb, 1)} MB download` : ''}</td></tr>
        <tr><th>MISSING</th><td>${esc(d.missing_values)}</td></tr>
        <tr><th>SPLIT</th><td>${esc(d.split_strategy)}${d.seasonal_period ? ` · seasonal period ${esc(d.seasonal_period)}` : ''}</td></tr>
        <tr><th>METRICS</th><td>${metrics || '—'}</td></tr>
      </table>
      ${prep ? `<div class="dl-sub">PREPROCESSING (AUTOMATIC, FIT ON TRAIN ONLY)</div><ul class="dl-list">${prep}</ul>` : ''}
      ${caveats ? `<div class="dl-sub">CAVEATS</div><ul class="dl-list">${caveats}</ul>` : ''}
      ${d.status && d.status.last_error ? `<div class="dl-notice dl-notice-error">Last download error: ${esc(d.status.last_error)}</div>` : ''}
      <div class="control-actions dl-detail-actions">
        <button class="ctrl-btn" id="dl-btn-preview" ${canPreview ? '' : 'disabled'}>PREVIEW &amp; STATS</button>
        ${needsDl ? `<button class="ctrl-btn action-primary" id="dl-btn-download">DOWNLOAD (${fmt(d.approx_download_mb, 1)} MB)</button>` : ''}
      </div>
      <div id="dl-preview"></div>`;
    const pb = $('dl-btn-preview');
    if (pb) pb.addEventListener('click', loadPreview);
    const db = $('dl-btn-download');
    if (db) db.addEventListener('click', () => startDownload(d));
  }

  async function loadPreview() {
    const d = currentDataset();
    const box = $('dl-preview');
    box.innerHTML = '<p class="dl-empty">Loading preview…</p>';
    try {
      const p = await api(`/api/datasets/${encodeURIComponent(d.id)}/preview?rows=12`);
      const head = p.columns.map((c) => `<th>${esc(c)}</th>`).join('');
      const body = p.rows.map((r) => `<tr>${r.map((v) => `<td>${fmt(v, 3)}</td>`).join('')}</tr>`).join('');
      const stats = p.column_stats.map((c) => `<tr><td><code>${esc(c.name)}</code></td><td>${esc(c.kind)}</td><td>${fmt(c.missing_pct, 2)}%</td><td>${fmt(c.unique, 0)}</td>
        <td>${c.kind === 'numeric' ? `${fmt(c.mean, 3)} ± ${fmt(c.std, 3)}` : esc(Object.keys(c.top || {}).slice(0, 3).join(', '))}</td>
        <td>${c.kind === 'numeric' ? `${fmt(c.min, 3)} … ${fmt(c.max, 3)}` : ''}</td></tr>`).join('');
      let target = '';
      if (p.target) {
        target = p.target.kind === 'categorical'
          ? Object.entries(p.target.counts).map(([k, v]) => `<span class="dl-chip">${esc(k)}: ${fmt(v, 0)}</span>`).join(' ')
          : `mean ${fmt(p.target.describe.mean, 3)} · std ${fmt(p.target.describe.std, 3)} · min ${fmt(p.target.describe.min, 3)} · max ${fmt(p.target.describe.max, 3)}`;
        target = `<div class="dl-sub">TARGET (${esc(p.target.role.replace(/_/g, ' '))})</div><div>${target}</div>`;
      }
      box.innerHTML = `
        <div class="metrics-summary-strip dl-strip">
          <div class="metric-pill"><span class="m-label">ROWS</span><span class="m-val">${fmt(p.n_rows, 0)}</span></div>
          <div class="metric-pill"><span class="m-label">COLUMNS</span><span class="m-val">${fmt(p.n_columns, 0)}</span></div>
          <div class="metric-pill"><span class="m-label">MISSING CELLS</span><span class="m-val">${fmt(p.missing_cells, 0)}</span></div>
          <div class="metric-pill"><span class="m-label">DUPLICATE ROWS</span><span class="m-val">${fmt(p.duplicate_rows, 0)}</span></div>
          <div class="metric-pill"><span class="m-label">MEMORY</span><span class="m-val">${fmt(p.memory_mb, 2)} MB</span></div>
        </div>
        ${target}
        <div class="dl-sub">FIRST ROWS</div>
        <div class="data-table-wrap dl-scroll"><table class="research-data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>
        <div class="dl-sub">COLUMN SUMMARY</div>
        <div class="data-table-wrap dl-scroll"><table class="research-data-table"><thead><tr><th>COLUMN</th><th>KIND</th><th>MISSING</th><th>UNIQUE</th><th>MEAN ± STD / TOP</th><th>RANGE</th></tr></thead><tbody>${stats}</tbody></table></div>`;
    } catch (e) {
      box.innerHTML = `<div class="dl-notice dl-notice-error">${esc(e.message)}</div>`;
    }
  }

  async function startDownload(d) {
    const allowLarge = d.size_class === 'large' ? confirm(`${d.name} is a large dataset (~${d.approx_download_mb} MB download, ~${d.approx_memory_mb} MB in memory). Download it?`) : false;
    if (d.size_class === 'large' && !allowLarge) return;
    try {
      const job = await api(`/api/datasets/${encodeURIComponent(d.id)}/download`, { method: 'POST', body: JSON.stringify({ allow_large: allowLarge }) });
      trackJob(job, async (j) => {
        if (j.status === 'done') { setNotice(`Downloaded and verified <strong>${esc(d.name)}</strong>.`, 'ok'); await loadDatasets(); await selectDataset(d.id); }
      });
    } catch (e) { setNotice('Download could not start: ' + esc(e.message), 'error'); }
  }

  // ---------------------------------------------------------------- algorithms
  async function refreshCompatibility() {
    const d = currentDataset();
    S.compat = null;
    if (!d) { renderAlgorithms(); return; }
    if (S.mode === 'local') {
      const q = new URLSearchParams({ dataset: d.id, task: S.task, allow_large: $('dl-allow-large') && $('dl-allow-large').checked ? '1' : '0' });
      const mtr = $('dl-max-rows') && $('dl-max-rows').value; if (mtr) q.set('max_train_rows', mtr);
      const w = $('dl-window') && $('dl-window').value; if (w) q.set('window', w);
      try { S.compat = await api('/api/compatibility?' + q.toString()); } catch (e) { setNotice(esc(e.message), 'error'); }
    } else {
      const ok = ((d.algorithms_by_task || {})[S.task]) || [];
      S.compat = { compatible: ok, recommended: S.algorithms.filter((a) => ok.includes(a.id) && a.recommended).map((a) => a.id), excluded: {} };
    }
    S.selected = new Set((S.compat && S.compat.recommended) || []);
    renderAlgorithms();
  }

  function renderAlgorithms() {
    const box = $('dl-algos');
    const d = currentDataset();
    if (!d) { box.innerHTML = '<p class="dl-empty">Choose a dataset first; only algorithms that can process its format are enabled.</p>'; updateRunButton(); return; }
    const algos = S.algorithms.filter((a) => a.task === S.task);
    const ok = new Set((S.compat && S.compat.compatible) || []);
    const excl = (S.compat && S.compat.excluded) || {};
    box.innerHTML = algos.map((a) => {
      const enabled = ok.has(a.id);
      const why = excl[a.id] ? excl[a.id].join('; ') : (enabled ? '' : 'not compatible with this dataset');
      return `<label class="dl-algo${enabled ? '' : ' is-disabled'}" title="${esc(why || a.notes || '')}">
        <input type="checkbox" value="${esc(a.id)}" ${enabled ? '' : 'disabled'} ${S.selected.has(a.id) ? 'checked' : ''}>
        <span class="dl-algo-name">${esc(a.name)}</span>
        <span class="dl-algo-meta">${esc(a.framework)} · cost ${esc(a.cost)} · interp. ${esc(a.interpretability)}</span>
        ${enabled ? '' : `<span class="dl-algo-why">${esc(why)}</span>`}
      </label>`;
    }).join('');
    box.querySelectorAll('input[type=checkbox]').forEach((cb) => cb.addEventListener('change', () => {
      if (cb.checked) S.selected.add(cb.value); else S.selected.delete(cb.value);
      updateRunButton();
    }));
    updateRunButton();
  }

  function renderAlgoFinder() {
    const sel = $('dl-algo-finder');
    if (!sel) return;
    const algos = S.algorithms.filter((a) => a.task === S.task);
    sel.innerHTML = '<option value="">— pick an algorithm —</option>' + algos.map((a) => `<option value="${esc(a.id)}">${esc(a.name)}</option>`).join('');
    $('dl-finder-out').textContent = '';
  }

  async function findDatasetsForAlgo() {
    const id = $('dl-algo-finder').value;
    const out = $('dl-finder-out');
    if (!id) { out.textContent = ''; return; }
    if (S.mode !== 'local') {
      const names = datasetsForTask().filter((d) => ((d.algorithms_by_task || {})[S.task] || []).includes(id)).map((d) => d.name);
      out.innerHTML = names.length ? 'Compatible: ' + names.map(esc).join(', ') : 'No compatible dataset in the catalogue.';
      return;
    }
    try {
      const r = await api(`/api/recommend?algorithm=${encodeURIComponent(id)}&budget=${S.sortBudget}&allow_large=${$('dl-allow-large').checked ? 1 : 0}`);
      const name = (x) => (S.datasets.find((d) => d.id === x) || { name: x }).name;
      out.innerHTML = r.datasets.length
        ? `${S.sortBudget === 'quick' ? 'Quickest first' : 'Most realistic first'}: ` + r.datasets.map((x, i) => `<a href="#" data-id="${esc(x)}" class="dl-link">${i + 1}. ${esc(name(x))}</a>`).join(' · ')
        : 'No compatible dataset (enable large datasets or lower limits).';
      out.querySelectorAll('a.dl-link').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); selectDataset(a.dataset.id); }));
    } catch (e) { out.textContent = e.message; }
  }

  // ---------------------------------------------------------------- config
  function renderConfig() {
    const d = currentDataset();
    const t = S.task;
    const show = (id, on) => { const el = $(id); if (el) el.closest('.control-group').hidden = !on; };
    show('dl-window', t === 'forecasting' || (t === 'anomaly_detection' && d && d.split_strategy === 'chronological'));
    show('dl-horizon', t === 'forecasting');
    show('dl-nclusters', t === 'clustering');
    show('dl-contamination', t === 'anomaly_detection');
    show('dl-classweight', t === 'classification');
    show('dl-tune', t === 'classification' || t === 'regression');
    show('dl-cv', t !== 'clustering' && t !== 'anomaly_detection');
    show('dl-val', t !== 'clustering' && t !== 'anomaly_detection');
    show('dl-test', t !== 'clustering');
    if (d) {
      if (d.default_window) $('dl-window').placeholder = String(d.default_window);
      if (d.default_horizon) $('dl-horizon').placeholder = String(d.default_horizon);
      if (d.n_clusters_hint) $('dl-nclusters').placeholder = String(d.n_clusters_hint);
      if (d.contamination_prior) $('dl-contamination').placeholder = String(d.contamination_prior);
    }
  }

  function readConfig() {
    const num = (id, int = true) => { const v = $(id).value.trim(); if (v === '') return null; const n = int ? parseInt(v, 10) : parseFloat(v); return isFinite(n) ? n : null; };
    const cfg = {
      dataset_id: S.datasetId, task: S.task, algorithms: Array.from(S.selected),
      seed: num('dl-seed') ?? 42, test_size: num('dl-test', false) ?? 0.2, val_size: num('dl-val', false) ?? 0.1,
      cv_folds: num('dl-cv') ?? 0, tune: $('dl-tune').checked, max_train_rows: num('dl-max-rows'),
      allow_large: $('dl-allow-large').checked, window: num('dl-window'), horizon: num('dl-horizon'),
      n_clusters: num('dl-nclusters'), contamination: num('dl-contamination', false), epochs: num('dl-epochs'),
      preprocessing: {
        scale: $('dl-scale').value, clip_outliers: $('dl-clip').checked, drop_duplicates: $('dl-dedup').checked,
        class_weight: $('dl-classweight').value,
      },
    };
    if (S.task === 'clustering' || S.task === 'anomaly_detection') cfg.val_size = 0.0;
    return cfg;
  }

  function updateRunButton() {
    const b = $('dl-run');
    if (!b) return;
    const busy = S.job && (S.job.status === 'queued' || S.job.status === 'running');
    b.disabled = S.mode !== 'local' || !S.datasetId || S.selected.size === 0 || busy;
    b.title = S.mode !== 'local' ? 'Training needs the local server: python run.py --serve' : '';
    $('dl-run-hint').textContent = S.mode !== 'local' ? 'Read-only deployment: run `python run.py --serve` locally to train.'
      : !S.datasetId ? 'Select a dataset.' : S.selected.size === 0 ? 'Select at least one compatible algorithm.'
      : `${S.selected.size} algorithm(s) on ${currentDataset().name}, same split & seed for all.`;
  }

  async function runBenchmark() {
    const cfg = readConfig();
    setNotice('', null);
    try {
      const job = await api('/api/benchmarks/run', { method: 'POST', body: JSON.stringify({ config: cfg }) });
      trackJob(job, (j) => {
        if (j.status === 'done' && j.result) { S.result = j.result; renderResults(); loadHistory(); }
      });
    } catch (e) { setNotice('Could not start the benchmark: ' + esc(e.message), 'error'); }
  }

  // ---------------------------------------------------------------- jobs
  function trackJob(job, onFinish) {
    S.job = job;
    clearTimeout(S.pollTimer);
    renderProgress();
    updateRunButton();
    const tick = async () => {
      try {
        const j = await api('/api/jobs/' + encodeURIComponent(job.id));
        S.job = j; renderProgress();
        if (j.status === 'done' || j.status === 'error') {
          updateRunButton();
          if (j.status === 'error') setNotice(`${esc(j.kind)} failed: ${esc(j.error)}`, 'error');
          onFinish && onFinish(j);
          return;
        }
      } catch (e) { setNotice('Lost contact with the local server: ' + esc(e.message), 'error'); updateRunButton(); return; }
      S.pollTimer = setTimeout(tick, 800);
    };
    S.pollTimer = setTimeout(tick, 400);
  }

  function renderProgress() {
    const j = S.job;
    const box = $('dl-progress');
    if (!j) { box.hidden = true; return; }
    box.hidden = false;
    const pct = Math.round((j.progress || 0) * 100);
    box.innerHTML = `<div class="dl-progress-label"><span>${esc((j.kind || '').toUpperCase())} · ${esc(j.status.toUpperCase())}</span><span>${pct}%</span></div>
      <div class="dl-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><div class="dl-progress-fill" style="width:${pct}%"></div></div>
      <div class="dl-progress-msg">${esc(j.message || '')}</div>`;
  }

  // ---------------------------------------------------------------- results
  function bars(rows, key, lowerBetter, label, unit) {
    const vals = rows.map((r) => (key === 'latency' ? r.latency_us_per_sample : (r.metrics || {})[key])).filter((v) => typeof v === 'number' && isFinite(v));
    if (!vals.length) return '';
    const max = Math.max(...vals.map(Math.abs)) || 1;
    return `<div class="dl-chart"><div class="dl-sub">${esc(label)} ${lowerBetter ? '(LOWER IS BETTER)' : '(HIGHER IS BETTER)'}</div>` +
      rows.map((r) => {
        const v = key === 'latency' ? r.latency_us_per_sample : (r.metrics || {})[key];
        const ok = typeof v === 'number' && isFinite(v);
        const w = ok ? Math.max(1.5, (Math.abs(v) / max) * 100) : 0;
        return `<div class="dl-bar-row"><span class="dl-bar-name" title="${esc(r.name)}">${esc(r.name)}</span>
          <span class="dl-bar-track"><span class="dl-bar-fill${v < 0 ? ' is-neg' : ''}" style="width:${w}%"></span></span>
          <span class="dl-bar-val">${ok ? fmt(v, key === 'latency' ? 2 : 4) + (unit || '') : 'n/a'}</span></div>`;
      }).join('') + '</div>';
  }

  function renderResults() {
    const r = S.result;
    const box = $('dl-results');
    if (!r) { box.innerHTML = ''; return; }
    const task = r.task;
    const [pk, goal] = PRIMARY[task] || ['', 'max'];
    const cols = COLS[task] || [];
    const okRows = (r.models || []).filter((m) => m.status === 'ok').sort((a, b) => (a.rank || 99) - (b.rank || 99));
    const failed = (r.models || []).filter((m) => m.status !== 'ok');
    const s = r.summary || {};
    const nm = (id) => ((r.models || []).find((m) => m.algorithm === id) || { name: id || '—' }).name;
    const cvHead = okRows.some((m) => m.cv) ? '<th>CV MEAN ± STD</th>' : '';
    const table = okRows.map((m) => `<tr class="dl-res-row" data-alg="${esc(m.algorithm)}">
        <td><span class="rank-marker">${esc(m.rank || '-')}</span></td>
        <td><strong>${esc(m.name)}</strong>${m.pareto_optimal ? ' <span class="dl-chip" title="No other model is both more accurate and faster">PARETO</span>' : ''}${m.beats_baseline === false ? ' <span class="verdict-bad">≤ BASELINE</span>' : ''}</td>
        <td><code>${esc(m.framework)}</code></td>
        ${cols.map((c) => `<td${c === pk ? ' class="dl-primary"' : ''}>${fmt((m.metrics || {})[c])}</td>`).join('')}
        ${cvHead ? `<td>${m.cv ? `${fmt(m.cv.mean)} ± ${fmt(m.cv.std)}` : '—'}</td>` : ''}
        <td>${unit(m.train_time_s, 3, ' s')}</td><td>${unit(m.latency_us_per_sample, 2, ' µs')}</td>
        <td>${unit(m.model_size_kb, 1, ' KB')}</td><td>${fmt(m.n_parameters, 0)}</td></tr>`).join('');
    const notes = [];
    (r.warnings || []).forEach((w) => notes.push(`<li>${esc(w)}</li>`));
    Object.entries(r.skipped_algorithms || {}).forEach(([k, v]) => notes.push(`<li>Skipped <code>${esc(k)}</code>: ${esc(v.join('; '))}</li>`));
    failed.forEach((m) => notes.push(`<li>Failed <code>${esc(m.algorithm)}</code>: ${esc(m.error)}</li>`));
    okRows.forEach((m) => (m.notes || []).forEach((n) => notes.push(`<li><code>${esc(m.algorithm)}</code>: ${esc(n)}</li>`)));
    const split = r.split || {};
    const splitTxt = Object.entries(split).filter(([k, v]) => v !== null && v !== undefined && typeof v !== 'object')
      .map(([k, v]) => `${esc(k.replace(/_/g, ' '))}: <strong>${fmt(v, 0)}</strong>`).join(' · ');
    const extra = r.forecast_setup ? `Naive baseline test RMSE: <strong>${fmt(r.forecast_setup.naive_test_rmse)}</strong> · MASE scale: ${fmt(r.forecast_setup.mase_scale)}`
      : r.anomaly_setup ? `Threshold rule: ${esc(r.anomaly_setup.threshold_rule)} (contamination prior ${fmt(r.anomaly_setup.contamination_prior)}). Labels used for training: no.`
      : r.clustering_setup ? `k = ${fmt(r.clustering_setup.n_clusters, 0)} · DBSCAN eps = ${fmt(r.clustering_setup.dbscan_eps)} · reference labels used only for ARI/NMI.` : '';
    box.innerHTML = `
      <div class="h1-verdict-banner">
        <strong>${esc(r.dataset_name)}</strong> · ${esc(task.replace(/_/g, ' '))} · run <code>${esc(r.run_id)}</code> · ${esc(r.status)} in ${fmt(r.duration_s, 1)} s<br>
        <span class="dl-small">${splitTxt}</span>${extra ? `<br><span class="dl-small">${extra}</span>` : ''}
      </div>
      <div class="metrics-summary-strip">
        <div class="metric-pill"><span class="m-label">BEST ${esc(pk.toUpperCase())} (${goal === 'min' ? 'LOWEST' : 'HIGHEST'})</span><span class="m-val">${esc(nm(s.best_by_metric))}</span></div>
        <div class="metric-pill"><span class="m-label">VALUE</span><span class="m-val">${fmt(s.best_value)}</span></div>
        <div class="metric-pill"><span class="m-label">FASTEST INFERENCE</span><span class="m-val">${esc(nm(s.fastest_inference))}</span></div>
        <div class="metric-pill"><span class="m-label">SMALLEST MODEL</span><span class="m-val">${esc(nm(s.smallest_model))}</span></div>
        <div class="metric-pill"><span class="m-label">PARETO FRONT</span><span class="m-val">${(s.pareto_front || []).length}</span></div>
      </div>
      <p class="dl-small">${esc(s.note || '')} ${esc(s.latency_note || '')}</p>
      <div class="data-table-wrap dl-scroll"><table class="research-data-table">
        <thead><tr><th>#</th><th>MODEL</th><th>FRAMEWORK</th>${cols.map((c) => `<th>${esc(c.toUpperCase())}</th>`).join('')}${cvHead}<th>TRAIN</th><th>LATENCY</th><th>SIZE</th><th>PARAMS</th></tr></thead>
        <tbody>${table || '<tr><td colspan="12">No model completed.</td></tr>'}</tbody></table></div>
      <div class="dl-charts">${bars(okRows, pk, LOWER_BETTER.has(pk), pk.toUpperCase())}${bars(okRows, 'latency', true, 'INFERENCE LATENCY / SAMPLE', ' µs')}</div>
      <div id="dl-model-detail"></div>
      ${notes.length ? `<div class="dl-sub">NOTES, LIMITATIONS &amp; SKIPPED MODELS</div><ul class="dl-list">${notes.join('')}</ul>` : ''}`;
    box.querySelectorAll('tr.dl-res-row').forEach((tr) => tr.addEventListener('click', () => {
      box.querySelectorAll('tr.dl-res-row').forEach((x) => x.classList.remove('row-active'));
      tr.classList.add('row-active');
      renderModelDetail(okRows.find((m) => m.algorithm === tr.dataset.alg));
    }));
    if (okRows[0]) renderModelDetail(okRows[0]);
  }

  function renderModelDetail(m) {
    const box = $('dl-model-detail');
    if (!box || !m) return;
    let html = `<div class="dl-sub">MODEL DETAIL — ${esc(m.name)}</div>`;
    const cm = m.metrics && m.metrics.confusion_matrix;
    if (cm && cm.matrix && cm.matrix.length <= 12) {
      const max = Math.max(1, ...cm.matrix.flat());
      html += `<div class="dl-cm-wrap"><table class="dl-cm"><thead><tr><th>TRUE \\ PRED</th>${cm.labels.map((l) => `<th>${esc(l)}</th>`).join('')}</tr></thead><tbody>` +
        cm.matrix.map((row, i) => `<tr><th>${esc(cm.labels[i])}</th>${row.map((v) => {
          const a = v / max; return `<td style="background:rgba(0,0,0,${(a * 0.85).toFixed(2)});color:${a > 0.5 ? '#fff' : '#000'}">${fmt(v, 0)}</td>`;
        }).join('')}</tr>`).join('') + '</tbody></table></div>';
    }
    if (m.metrics && m.metrics.rmse_per_horizon && m.metrics.rmse_per_horizon.length > 1) {
      html += `<p class="dl-small">RMSE per horizon step: ${m.metrics.rmse_per_horizon.map((v) => fmt(v, 3)).join(' · ')}</p>`;
    }
    if (m.metrics && m.metrics.detection) {
      const d = m.metrics.detection;
      html += `<p class="dl-small">Anomaly windows detected: <strong>${fmt(d.detected, 0)}/${fmt(d.windows, 0)}</strong> · mean detection delay: ${d.mean_delay_steps === null ? 'n/a' : fmt(d.mean_delay_steps, 1) + ' steps'}</p>`;
    }
    if (m.fairness) {
      Object.entries(m.fairness).forEach(([col, f]) => {
        html += `<p class="dl-small"><strong>Fairness · ${esc(col)}</strong>: demographic-parity gap ${fmt(f.demographic_parity_diff, 3)}, equal-opportunity (TPR) gap ${fmt(f.equal_opportunity_diff, 3)} — ` +
          Object.entries(f.groups).map(([g, v]) => `${esc(g)} (n=${fmt(v.n, 0)}): sel ${fmt(v.selection_rate, 3)}, TPR ${fmt(v.tpr, 3)}`).join('; ') + '</p>';
      });
    }
    if (m.tuning) html += `<p class="dl-small">Tuned (${esc(m.tuning.cv)}): best params <code>${esc(JSON.stringify(m.tuning.best_params))}</code></p>`;
    if (m.preprocessing) html += `<p class="dl-small">Preprocessing: ${m.preprocessing.map(esc).join(' → ')}</p>`;
    box.innerHTML = html;
  }

  // ---------------------------------------------------------------- history
  async function loadHistory() {
    const box = $('dl-history');
    try {
      const d = S.mode === 'local' ? await api('/api/benchmarks/history') : await api('catalog/history.json');
      const runs = (d.runs || []).slice(0, 15);
      if (!runs.length) { box.innerHTML = '<p class="dl-empty">No recorded runs yet.</p>'; return; }
      box.innerHTML = `<div class="data-table-wrap dl-scroll"><table class="research-data-table"><thead><tr><th>WHEN (UTC)</th><th>DATASET</th><th>TASK</th><th>MODELS</th><th>BEST</th><th>PRIMARY</th><th>STATUS</th></tr></thead><tbody>` +
        runs.map((h) => `<tr class="dl-hist-row" data-id="${esc(h.run_id)}" tabindex="0"><td>${esc((h.created_at || '').replace('T', ' ').slice(0, 16))}</td><td>${esc(h.dataset_name)}</td><td>${esc((h.task || '').replace(/_/g, ' '))}</td><td>${fmt(h.n_models, 0)}</td><td>${esc(h.best || '—')}</td><td>${h.primary_metric ? esc(h.primary_metric.name) + ' = ' + fmt(h.best_value) : '—'}</td><td>${esc(h.status)}</td></tr>`).join('') +
        '</tbody></table></div>';
      box.querySelectorAll('tr.dl-hist-row').forEach((tr) => tr.addEventListener('click', async () => {
        try {
          S.result = S.mode === 'local' ? await api('/api/benchmarks/' + encodeURIComponent(tr.dataset.id))
            : await api('catalog/runs/' + encodeURIComponent(tr.dataset.id) + '.json');
          renderResults();
          $('dl-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (e) { setNotice('Could not load run: ' + esc(e.message), 'error'); }
      }));
    } catch (_) { box.innerHTML = '<p class="dl-empty">No run history available.</p>'; }
  }

  // ---------------------------------------------------------------- upload
  async function uploadCsv() {
    const f = $('dl-up-file').files[0];
    const out = $('dl-up-out');
    if (!f) { out.textContent = 'Choose a CSV file first.'; return; }
    if (f.size > 20 * 1024 * 1024) { out.textContent = 'File exceeds the 20 MB limit.'; return; }
    const buf = new Uint8Array(await f.arrayBuffer());
    let bin = '';
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    const body = { filename: f.name, content_b64: btoa(bin), task: $('dl-up-task').value,
      target: $('dl-up-target').value.trim() || null, time_column: $('dl-up-time').value.trim() || null };
    out.textContent = 'Validating…';
    try {
      const meta = await api('/api/datasets/upload', { method: 'POST', body: JSON.stringify(body) });
      out.innerHTML = `Accepted as <code>${esc(meta.id)}</code>: ${fmt(meta.report.rows, 0)} rows, ${fmt(meta.report.columns, 0)} columns` +
        (Object.keys(meta.report.renamed_columns || {}).length ? ' (some column names were sanitised)' : '') + '.';
      S.task = meta.task; renderTaskTabs(); await loadDatasets(); await selectDataset(meta.id);
    } catch (e) { out.innerHTML = `<span class="verdict-bad">REJECTED</span> ${esc(e.message)}`; }
  }

  // ---------------------------------------------------------------- wiring
  function bindStaticControls() {
    $('dl-run').addEventListener('click', runBenchmark);
    $('dl-sel-rec').addEventListener('click', () => { S.selected = new Set((S.compat && S.compat.recommended) || []); renderAlgorithms(); });
    $('dl-sel-all').addEventListener('click', () => { S.selected = new Set((S.compat && S.compat.compatible) || []); renderAlgorithms(); });
    $('dl-sel-none').addEventListener('click', () => { S.selected.clear(); renderAlgorithms(); });
    $('dl-budget').addEventListener('change', (e) => { S.sortBudget = e.target.value; renderDatasetTable(); findDatasetsForAlgo(); });
    $('dl-algo-finder').addEventListener('change', findDatasetsForAlgo);
    ['dl-allow-large', 'dl-max-rows', 'dl-window'].forEach((id) => $(id).addEventListener('change', () => { if (S.datasetId) refreshCompatibility(); }));
    $('dl-up-btn').addEventListener('click', uploadCsv);
    renderConfig();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
