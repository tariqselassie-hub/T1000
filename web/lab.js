// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

'use strict';
(function () {
    const $ = (id) => document.getElementById(id);
    const NT = window.NT, G = window.Geom;
    const GREEN = '#00e676', GOLD = '#ffb300', RED = '#ef4444', MUTED = '#94a3b8', FAINT = '#334155', TEXT = '#e2e8f0';
    const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    const num = (x, d = 5) => (typeof x === 'number' ? (Math.abs(x) >= 1e5 || (x !== 0 && Math.abs(x) < 1e-4) ? x.toExponential(3) : +x.toFixed(d)) : x);

    async function post(url, body) {
        const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const j = await r.json();
        if (j.error) throw new Error(j.error);
        return j;
    }
    const show = (id, html) => { $(id).innerHTML = html; };
    const fail = (id, e) => { $(id).innerHTML = '<span class="err">' + esc(e.message || e) + '</span>'; };
    const kv = (k, v) => '<span class="k">' + esc(k) + '</span> ' + v + '\n';

    // ---- tabs and status ----------------------------------------------------------------------------------
    let geoActive = false;
    $('tabs').addEventListener('click', (ev) => {
        const b = ev.target.closest('button[data-tab]');
        if (!b) return;
        document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('active', x === b));
        document.querySelectorAll('section.tab').forEach((s) => s.classList.toggle('active', s.id === 'tab-' + b.dataset.tab));
        geoActive = b.dataset.tab === 'geo';
        if (geoActive) { drawBloch(); drawHopf(); drawPoly(); }
    });
    fetch('/api/status').then((r) => r.json()).then((s) => {
        const f = s.fpga_status || {};
        const pill = $('pill-board');
        pill.textContent = f.online ? 'board: online (' + (f.port || 'COM6') + ')' : 'board: not connected';
        pill.className = 'pill ' + (f.online ? 'on' : 'off');
    }).catch(() => { $('pill-board').textContent = 'board: unknown'; });

    // ---- chart helpers ------------------------------------------------------------------------------------------
    // Canvases keep their HTML width/height as the logical drawing size and are backed at the device pixel
    // ratio, so lines and text stay sharp on high-density screens.
    const MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace";
    const AXIS = '#475569';
    const dims = (canvas) => [+canvas.dataset.w || canvas.width, +canvas.dataset.h || canvas.height];
    function prep(canvas) {
        if (!canvas.dataset.w) { canvas.dataset.w = canvas.width; canvas.dataset.h = canvas.height; }
        const [w, h] = dims(canvas), dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(w * dpr)) {
            canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
            canvas.style.width = w + 'px';
        }
        const c = canvas.getContext('2d');
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        c.clearRect(0, 0, w, h);
        c.fillStyle = '#080c14'; c.fillRect(0, 0, w, h);
        c.font = '10.5px ' + MONO; c.lineWidth = 1; c.setLineDash([]); c.textAlign = 'left';
        return c;
    }

    // Dashed graticule, solid axes with outward ticks, and y labels right-aligned to the axis.
    function frame(c, W, H, L, R, T, B, yTicks) {
        c.save();
        c.strokeStyle = FAINT; c.setLineDash([2, 3]);
        yTicks.forEach(([y]) => { c.beginPath(); c.moveTo(L, y + 0.5); c.lineTo(W - R, y + 0.5); c.stroke(); });
        c.setLineDash([]); c.strokeStyle = AXIS;
        c.beginPath(); c.moveTo(L + 0.5, T); c.lineTo(L + 0.5, H - B + 0.5); c.lineTo(W - R, H - B + 0.5); c.stroke();
        c.fillStyle = MUTED; c.textAlign = 'right';
        yTicks.forEach(([y, label]) => {
            c.beginPath(); c.moveTo(L - 4, y + 0.5); c.lineTo(L, y + 0.5); c.stroke();
            c.fillText(label, L - 6, y + 3.5);
        });
        c.restore();
    }

    // Hover probe: a chart registers how to redraw itself and what to read out at a point.
    function readoutBox(c, W, H, mx, my, lines) {
        const w = 12 + 6.4 * Math.max(...lines.map((s) => s.length)), h = 8 + 14 * lines.length;
        const x = mx + 14 + w > W ? mx - 14 - w : mx + 14, y = Math.min(Math.max(4, my - h / 2), H - h - 4);
        c.fillStyle = 'rgba(5, 8, 13, 0.92)'; c.fillRect(x, y, w, h);
        c.strokeStyle = GREEN; c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        c.fillStyle = TEXT; c.textAlign = 'left';
        lines.forEach((s, i) => c.fillText(s, x + 6, y + 15 + 14 * i));
    }
    function setProbe(canvas, redraw, pick) {
        canvas._probe = { redraw, pick };
        if (canvas._probeBound) return;
        canvas._probeBound = true;
        canvas.classList.add('probe');
        canvas.addEventListener('mousemove', (e) => {
            const p = canvas._probe, r = canvas.getBoundingClientRect(), [W, H] = dims(canvas);
            const mx = (e.clientX - r.left) * W / r.width, my = (e.clientY - r.top) * H / r.height;
            const hit = p.pick(mx, my);
            p.redraw();
            if (!hit) return;
            const c = canvas.getContext('2d');
            c.save(); c.font = '10.5px ' + MONO; c.strokeStyle = 'rgba(0, 230, 118, 0.55)'; c.setLineDash([3, 3]);
            c.beginPath(); c.moveTo(hit.x + 0.5, hit.top); c.lineTo(hit.x + 0.5, hit.bottom); c.stroke();
            if (hit.y !== undefined) { c.beginPath(); c.moveTo(hit.left, hit.y + 0.5); c.lineTo(hit.right, hit.y + 0.5); c.stroke(); }
            c.setLineDash([]);
            readoutBox(c, W, H, hit.x, hit.y !== undefined ? hit.y : my, hit.lines);
            c.restore();
        });
        canvas.addEventListener('mouseleave', () => canvas._probe.redraw());
    }

    function barChart(canvas, values, o = {}) {
        const c = prep(canvas), [W, H] = dims(canvas);
        const L = 50, R = 10, T = 16, B = o.labels ? 34 : 22;
        const n = values.length;
        const top = Math.max(1e-12, o.ymax || Math.max(...values, ...(o.expected || [0])) * 1.08);
        frame(c, W, H, L, R, T, B, [0, 1, 2, 3, 4].map((i) => [T + (H - T - B) * (1 - i / 4), String(num(top * i / 4, 3))]));
        const bw = (W - L - R) / n;
        values.forEach((v, i) => {
            const h = (H - T - B) * (v / top);
            c.fillStyle = (o.highlight && o.highlight.has(i)) ? GOLD : GREEN;
            c.fillRect(L + 1 + i * bw + Math.min(1, bw * 0.1), H - B - h, Math.max(1, bw * 0.8), h);
            if (o.labels && bw > 14) {
                c.fillStyle = MUTED; c.textAlign = 'center';
                c.fillText(o.labels[i], L + (i + 0.5) * bw, H - B + 13); c.textAlign = 'left';
            }
        });
        if (o.expected) {
            c.strokeStyle = RED; c.lineWidth = 2;
            o.expected.forEach((e, i) => {
                const y = H - B - (H - T - B) * (e / top);
                c.beginPath(); c.moveTo(L + i * bw, y); c.lineTo(L + (i + 1) * bw, y); c.stroke();
            });
            c.lineWidth = 1;
        }
        if (o.title) { c.fillStyle = TEXT; c.fillText(o.title.toUpperCase(), L + 6, T - 4); }
        if (o.xlabel) { c.fillStyle = MUTED; c.textAlign = 'right'; c.fillText(o.xlabel, W - R, H - 4); c.textAlign = 'left'; }
        setProbe(canvas, () => barChart(canvas, values, o), (mx) => {
            const i = Math.floor((mx - L) / bw);
            if (i < 0 || i >= n) return null;
            const lines = ['[' + (o.labels ? o.labels[i] : i) + ']', 'value  ' + num(values[i], 6)];
            if (o.expected) lines.push('expect ' + num(o.expected[i], 6));
            return { x: L + (i + 0.5) * bw, top: T, bottom: H - B, lines };
        });
    }

    function lineChart(canvas, xs, series, o = {}) {
        const c = prep(canvas), [W, H] = dims(canvas);
        const L = 50, R = 10, T = 16, B = 26;
        const ys = series.flatMap((s) => s.y);
        const ymax = o.ymax || Math.max(1e-12, ...ys) * 1.05, ymin = o.ymin !== undefined ? o.ymin : 0;
        const xmin = xs[0], xmax = xs[xs.length - 1];
        const X = (x) => L + (W - L - R) * (x - xmin) / ((xmax - xmin) || 1);
        const Y = (y) => H - B - (H - T - B) * (y - ymin) / ((ymax - ymin) || 1);
        frame(c, W, H, L, R, T, B, [0, 1, 2, 3, 4].map((i) => { const v = ymin + (ymax - ymin) * i / 4; return [Y(v), String(num(v, 3))]; }));
        c.fillStyle = MUTED; c.strokeStyle = AXIS; c.textAlign = 'center';
        for (let i = 0; i <= 4; i++) {
            const v = xmin + (xmax - xmin) * i / 4, x = X(v);
            c.beginPath(); c.moveTo(x + 0.5, H - B); c.lineTo(x + 0.5, H - B + 4); c.stroke();
            c.fillText(String(num(v, 3)), x, H - B + 15);
        }
        c.textAlign = 'left';
        series.forEach((s, k) => {
            c.strokeStyle = s.color; c.fillStyle = s.color; c.lineWidth = 1.75;
            if (s.dots) { s.y.forEach((y, i) => { c.beginPath(); c.arc(X(xs[i]), Y(y), 2.5, 0, 6.3); c.fill(); }); }
            else { c.beginPath(); s.y.forEach((y, i) => (i ? c.lineTo(X(xs[i]), Y(y)) : c.moveTo(X(xs[i]), Y(y)))); c.stroke(); }
            c.lineWidth = 1;
            c.fillRect(L + 8 + k * 160, T - 9, 10, 2); c.fillText(s.label || '', L + 22 + k * 160, T - 5);
        });
        if (o.xlabel) { c.fillStyle = MUTED; c.textAlign = 'right'; c.fillText(o.xlabel, W - R, H - 2); c.textAlign = 'left'; }
        setProbe(canvas, () => lineChart(canvas, xs, series, o), (mx) => {
            if (mx < L || mx > W - R) return null;
            let i = 0;
            xs.forEach((x, k) => { if (Math.abs(X(x) - mx) < Math.abs(X(xs[i]) - mx)) i = k; });
            const lines = ['x = ' + num(xs[i], 6)].concat(series.map((s) => (s.label || 'y').slice(0, 22) + '  ' + num(s.y[i], 6)));
            return { x: X(xs[i]), y: Y(series[0].y[i]), top: T, bottom: H - B, left: L, right: W - R, lines };
        });
    }

    function heatmap(canvas, m, title) {
        const c = prep(canvas), [W, H] = dims(canvas);
        const rows = m.length, cols = m[0].length, T = 16;
        let mx = 1e-12; m.forEach((r) => r.forEach((v) => { if (v > mx) mx = v; }));
        const cw = (W - 4) / cols, rh = (H - T - 4) / rows;
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
            const v = Math.min(1, m[i][j] / mx);
            c.fillStyle = `rgb(${Math.round(255 * v ** 1.5)},${Math.round(40 + 190 * v)},${Math.round(30 + 80 * v ** 3)})`;
            c.fillRect(2 + j * cw, T + i * rh, Math.ceil(cw), Math.ceil(rh));
        }
        c.fillStyle = TEXT; c.fillText(title + '  (rows: time down, columns: node, max ' + num(mx, 3) + ')', 4, 11);
    }

    // ====================================================================== ALGORITHMS
    let lastGrover = null;
    const algoForms = {
        grover: '<label>Qubits <input type="number" id="ap-n" value="5" min="2" max="16"></label>' +
            '<label>Marked items (list) <input type="text" id="ap-marked" value="19" style="width:150px"></label>' +
            '<label>Iterations (-1 = optimal) <input type="number" id="ap-iter" value="-1"></label>' +
            '<label>Show up to <input type="number" id="ap-max" value="14"></label>',
        bernstein_vazirani: '<label>Bits <input type="number" id="ap-n" value="6" min="1" max="16"></label><label>Secret <input type="number" id="ap-s" value="37"></label>',
        deutsch_jozsa: '<label>Bits <input type="number" id="ap-n" value="5" min="1" max="16"></label><label>Function <select id="ap-kind"><option>constant0</option><option>constant1</option><option selected>balanced_parity</option><option>balanced_random</option></select></label><label>Seed <input type="number" id="ap-seed" value="1"></label>',
        qpe: '<label>Phase (0 to 1) <input type="text" id="ap-phase" value="0.3333333" style="width:110px"></label><label>Counting qubits <input type="number" id="ap-t" value="7" min="1" max="16"></label>',
        shor_order: '<label>a <input type="number" id="ap-a" value="2"></label><label>N <input type="number" id="ap-N" value="15" min="2" max="256"></label><label>Counting qubits (0 = 2 log N) <input type="number" id="ap-t" value="0"></label><label>Shots <input type="number" id="ap-shots" value="300"></label><label>Seed <input type="number" id="ap-seed" value="1"></label>',
    };
    const renderAlgoForm = () => { $('algo-params').innerHTML = algoForms[$('algo-name').value]; };
    $('algo-name').addEventListener('change', renderAlgoForm);
    renderAlgoForm();
    const val = (id) => $(id).value;

    $('algo-run').addEventListener('click', async () => {
        const name = val('algo-name');
        let body = { name };
        try {
            if (name === 'grover') body = { name, n: +val('ap-n'), marked: val('ap-marked').split(/[ ,;]+/).filter(Boolean).map(Number), iterations: +val('ap-iter'), max_iterations: +val('ap-max') };
            if (name === 'bernstein_vazirani') body = { name, n: +val('ap-n'), secret: +val('ap-s') };
            if (name === 'deutsch_jozsa') body = { name, n: +val('ap-n'), kind: val('ap-kind'), seed: +val('ap-seed') };
            if (name === 'qpe') body = { name, phase: +val('ap-phase'), t: +val('ap-t') };
            if (name === 'shor_order') body = { name, a: +val('ap-a'), N: +val('ap-N'), t: +val('ap-t'), shots: +val('ap-shots'), seed: +val('ap-seed') };
            $('algo-out').textContent = 'running...';
            const r = await post('/api/lab/algorithm', body);
            const c1 = $('algo-c1'), c2 = $('algo-c2');
            c2.style.display = 'none';
            let out = '';
            if (name === 'grover') {
                const xs = r.success_by_iteration.map((_, i) => i);
                lineChart(c1, xs, [{ y: r.success_by_iteration, color: GREEN, label: 'simulated' }, { y: r.closed_form, color: GOLD, dots: true, label: 'sin^2((2k+1) theta)' }], { ymax: 1.05, xlabel: 'iteration' });
                if (r.probabilities) { c2.style.display = 'block'; barChart(c2, r.probabilities, { highlight: new Set(r.marked), title: 'distribution after ' + r.iterations_used + ' iterations (marked in gold)', labels: r.probabilities.length <= 32 ? r.probabilities.map((_, i) => i) : null }); lastGrover = r.probabilities; }
                out += kv('success probability', num(r.success_probability));
                out += kv('optimal iterations', r.optimal_iterations + ' (pi/4 * sqrt(N/M), floored)');
                out += kv('largest gap to closed form', r.max_abs_error_vs_closed_form.toExponential(2));
            } else if (name === 'qpe' || name === 'shor_order') {
                if (r.probabilities) barChart(c1, r.probabilities, { title: 'outcome distribution of the counting register', xlabel: 'outcome k' });
                else barChart(c1, r.top.map((t) => t.p), { labels: r.top.map((t) => t.index), title: 'most likely outcomes' });
                if (name === 'qpe') {
                    out += kv('best outcome', r.best_outcome + '  (estimate ' + num(r.estimate, 6) + ')');
                    out += kv('absolute error', r.abs_error.toExponential(2) + '   (resolution 1/2^t = ' + (1 / 2 ** r.counting_qubits).toExponential(2) + ')');
                    out += kv('probability of best outcome', num(r.probability_of_best));
                } else {
                    const cands = Object.entries(r.order_candidates);
                    if (cands.length) { c2.style.display = 'block'; barChart(c2, cands.map((e) => e[1]), { labels: cands.map((e) => 'r=' + e[0]), title: 'orders recovered by continued fractions (counts)' }); }
                    out += kv('true order of a mod N', r.true_order + '   (classical check)');
                    out += kv('estimated order', r.estimated_order + '   (' + r.shots_giving_true_order + ' of ' + r.shots + ' shots gave the true order)');
                    out += kv('peak spacing 2^t / r', r.peak_spacing);
                    out += kv('factors of N found', r.factors_found.length ? r.factors_found.join(', ') : 'none (needs even order and a^(r/2) != -1)');
                    out += kv('qubits', r.counting_qubits + ' counting + ' + r.work_qubits + ' work');
                }
            } else {
                barChart(c1, r.top.map((t) => t.p), { labels: r.top.map((t) => t.index), title: 'most likely measurement outcomes' });
                if (name === 'bernstein_vazirani') { out += kv('secret', r.secret + ' (' + r.secret.toString(2) + ')'); out += kv('recovered', r.recovered + ' with probability ' + num(r.success_probability)); out += kv('queries', '1 quantum, ' + r.classical_queries_needed + ' classical'); }
                else { out += kv('P(all zeros)', num(r.probability_all_zeros)); out += kv('verdict', r.verdict); }
            }
            out += '<span class="k">model</span> ' + esc(r.model);
            show('algo-out', out);
        } catch (e) { fail('algo-out', e); }
    });

    // ---- circuit composer ---------------------------------------------------------------------------------------
    const PRESETS = {
        bell: ['2', 'H 0\nCX 0 1'],
        ghz: ['4', 'H 0\nCX 0 1\nCX 1 2\nCX 2 3'],
        rot: ['3', 'RY 0 0.8\nRY 1 1.4\nCZ 0 1\nRX 2 0.6\nCX 1 2\nRZ 0 1.1\nH 2'],
    };
    document.querySelectorAll('[data-circ]').forEach((b) => b.addEventListener('click', () => {
        const [n, t] = PRESETS[b.dataset.circ];
        $('circ-n').value = n; $('circ-text').value = t;
    }));

    function parseCircuit(text, n) {
        const layers = [];
        text.split('\n').forEach((raw, ln) => {
            const line = raw.replace(/#.*/, '').trim();
            if (!line) return;
            const t = line.split(/\s+/), op = t[0].toUpperCase();
            const q = (k) => { const v = Number(t[k]); if (!Number.isInteger(v) || v < 0 || v >= n) throw new Error('line ' + (ln + 1) + ': qubit "' + t[k] + '" is outside 0..' + (n - 1)); return v; };
            if (['H', 'X', 'Z', 'S', 'T'].includes(op)) layers.push({ gates: [{ type: op, target: q(1) }] });
            else if (['RX', 'RY', 'RZ'].includes(op)) layers.push({ gates: [{ type: op, target: q(1), theta: Number(t[2]) }] });
            else if (['CX', 'CZ', 'SWAP'].includes(op)) { const a = q(1), b = q(2); if (a === b) throw new Error('line ' + (ln + 1) + ': needs two different qubits'); layers.push({ gates: [{ type: op, control: a, target: b }] }); }
            else throw new Error('line ' + (ln + 1) + ': unknown gate "' + t[0] + '"');
        });
        return layers;
    }

    $('circ-run').addEventListener('click', async () => {
        try {
            const n = +val('circ-n');
            const layers = parseCircuit(val('circ-text'), n);
            $('circ-out').textContent = 'running...';
            const r = await post('/api/run_circuit', { qubits: n, layers, target: val('circ-target') });
            const top = r.top_states || [];
            barChart($('circ-c'), top.map((s) => s.prob), { labels: top.map((s) => s.basis.replace(/[|⟩]/g, '')), title: 'probabilities of the most likely basis states' });
            let out = kv('executed on', esc(r.target_executed || ''));
            out += kv('mode', esc(r.hardware_mode || ''));
            if (r.hardware_error) out += '<span class="warn">' + esc(r.hardware_error) + '</span>\n';
            out += kv('gates / depth', r.total_gates + ' / ' + r.depth);
            out += kv('<Z> per qubit', '[' + r.pauli_z.join(', ') + ']');
            out += kv('purity / entropy of half', num(r.purity) + ' / ' + num(r.bipartite_entropy_bits, 3) + ' bits');
            if (r.board_gates_sent !== undefined) {
                out += kv('board', r.board_gates_sent + ' gates sent, norm ' + r.board_norm_squared + ', largest <Z> gap to host ' + r.max_abs_pauli_z_error_vs_host);
                out += kv('board time', r.wall_time_ms + ' ms in total, ' + num(r.gate_phase_us / 1000, 2) + ' ms for the gates (UART-limited)');
            } else out += kv('time', r.wall_time_ms + ' ms (host simulation)');
            show('circ-out', out);
        } catch (e) { fail('circ-out', e); }
    });

    // ====================================================================== PROBABILITY
    $('shot-run').addEventListener('click', async () => {
        try {
            let probs;
            const src = val('shot-src'), txt = val('shot-custom');
            if (src === 'die') probs = Array(6).fill(1 / 6);
            else if (src === 'coin') { const p = Number(txt.split(/[ ,]+/)[0]); if (!(p >= 0 && p <= 1)) throw new Error('bias must be between 0 and 1'); probs = [1 - p, p]; }
            else if (src === 'bell') probs = [0.5, 0, 0, 0.5];
            else if (src === 'grover') { if (!lastGrover) throw new Error('run Grover on the first tab first'); probs = lastGrover; }
            else probs = txt.split(/[ ,;]+/).filter(Boolean).map(Number);
            const r = await post('/api/lab/shots', { probabilities: probs, shots: +val('shot-n'), seed: +val('shot-seed'), repeats: +val('shot-rep') });
            const total = probs.reduce((a, b) => a + b, 0);
            const exp = probs.map((p) => (p / total) * r.shots);
            if (r.counts) barChart($('shot-c'), r.counts, { expected: exp, title: 'observed counts (green) and expected (red)', labels: r.counts.map((_, i) => i) });
            let out = kv('shots', r.shots) + kv('chi-square', num(r.chi_square, 3) + ' on ' + r.degrees_of_freedom + ' degrees of freedom');
            out += kv('p-value', r.p_value === null ? 'n/a' : num(r.p_value, 4));
            out += kv('total variation distance to the model', num(r.total_variation_distance, 5));
            const pc = $('shot-p');
            if (r.repeat_p_values && r.repeat_p_values.length) {
                const bins = new Array(10).fill(0);
                r.repeat_p_values.forEach((p) => { bins[Math.min(9, Math.floor(p * 10))]++; });
                const e = r.repeat_p_values.length / 10;
                barChart(pc, bins, { expected: bins.map(() => e), title: 'p-values of ' + r.repeat_p_values.length + ' repeated experiments (uniform expected)', labels: bins.map((_, i) => (i / 10).toFixed(1)) });
                const low = r.repeat_p_values.filter((p) => p < 0.05).length / r.repeat_p_values.length;
                out += kv('fraction of repeats with p < 0.05', num(low, 3) + ' (about 0.05 if the model is right)');
                pc.style.display = 'block';
            } else pc.style.display = 'none';
            show('shot-out', out);
        } catch (e) { fail('shot-out', e); }
    });

    $('rc-run').addEventListener('click', async () => {
        try {
            $('rc-out').textContent = 'running...';
            const r = await post('/api/lab/random_circuit', { n: +val('rc-n'), depth: +val('rc-d'), seed: +val('rc-seed') });
            barChart($('rc-c'), r.histogram, { expected: r.exponential_expected, labels: r.bin_edges.slice(0, -1).map((x) => x.toFixed(1)), title: 'histogram of N * p over the 2^n outcomes (red: Porter-Thomas)', xlabel: 'N*p' });
            let out = kv('linear XEB of the output with itself', num(r.linear_xeb_self, 4) + '   (1 for a chaotic circuit)');
            out += kv('KS distance to exp(-x)', num(r.ks_statistic, 5) + '   (5% critical value ' + num(r.ks_critical_5pct, 5) + ')');
            out += kv('entropy', num(r.entropy_nats, 4) + ' nats   (Porter-Thomas value ' + num(r.ideal_entropy_nats, 4) + ')');
            show('rc-out', out + '<span class="k">model</span> ' + esc(r.model));
        } catch (e) { fail('rc-out', e); }
    });

    const wkHelp = {
        cycle: 'n nodes in a ring. The quantum walk spreads ballistically; the classical one diffusively.',
        path: 'n nodes in a line.', complete: 'every pair connected.', star: 'node 0 joined to all others.',
        hypercube: 'enter the dimension d in the extra box (n is ignored); 2^d nodes.',
        grid: 'enter rows,cols in the extra box (n is ignored).',
        circulant: 'Cayley graph of Z_n: enter the generators (for example 1 or 1,3); node i joins i + g and i - g.',
        unit_multiplication: 'enter k,c in the extra box: node x is joined to k*x + c mod n (n = 9, k = 2 gives the 1-2-4-8-7-5 hexagon and the pair 3-6).',
        glued_trees: 'enter the depth (1 to 5) in the extra box; start at 0 and aim at the other root (node 2^(d+1) - 1).',
    };
    const updWk = () => { $('wk-help').textContent = wkHelp[val('wk-kind')]; };
    $('wk-kind').addEventListener('change', updWk); updWk();

    function wkGraph() {
        const kind = val('wk-kind'), extra = val('wk-extra'), n = +val('wk-n');
        const parts = extra.split(/[ ,;]+/).filter(Boolean).map(Number);
        const g = { kind, n };
        if (kind === 'hypercube') g.d = parts[0] || 3;
        if (kind === 'grid') { g.rows = parts[0] || 4; g.cols = parts[1] || 4; }
        if (kind === 'circulant') g.generators = parts.length ? parts : [1];
        if (kind === 'unit_multiplication') { g.k = parts[0] === undefined ? 2 : parts[0]; g.offset = parts[1] || 0; }
        if (kind === 'glued_trees') g.depth = parts[0] || 3;
        return g;
    }

    async function runWalk(graph, start, target, tmax, outId, lineC, hq, hc) {
        const r = await post('/api/lab/walk', { graph, start, target, t_max: tmax, steps: 120 });
        if (lineC) {
            lineChart($(lineC), r.times, [{ y: r.quantum.target_probability, color: GREEN, label: 'quantum' }, { y: r.classical.target_probability, color: GOLD, label: 'classical' }], { xlabel: 't' });
            if (r.heat_quantum) { heatmap($(hq), r.heat_quantum, 'quantum'); heatmap($(hc), r.heat_classical, 'classical'); }
        }
        let out = kv('graph', r.nodes + ' nodes, ' + r.edges + ' edges');
        out += kv('peak at target', 'quantum ' + num(r.quantum.peak_target, 4) + ' at t = ' + num(r.quantum.time_of_peak, 3) + ';  classical ' + num(r.classical.peak_target, 4) + ' at t = ' + num(r.classical.time_of_peak, 3));
        out += kv('distance to uniform at the end (TV)', 'quantum ' + num(r.quantum.tv_to_uniform.slice(-1)[0], 4) + ';  classical ' + num(r.classical.tv_to_uniform.slice(-1)[0], 4));
        out += kv('probability conservation', 'quantum ' + r.norm_check_quantum.toExponential(1) + ',  classical ' + r.norm_check_classical.toExponential(1));
        if (r.adjacency_eigenvalues) out += kv('adjacency eigenvalues', r.adjacency_eigenvalues.slice().reverse().slice(0, 8).map((x) => num(x, 3)).join(', ') + (r.nodes > 8 ? ', ...' : ''));
        show(outId, out + '<span class="k">model</span> ' + esc(r.model));
        return r;
    }
    $('wk-run').addEventListener('click', async () => {
        try { $('wk-out').textContent = 'running...'; await runWalk(wkGraph(), +val('wk-start'), +val('wk-target'), +val('wk-t'), 'wk-out', 'wk-line', 'wk-hq', 'wk-hc'); }
        catch (e) { fail('wk-out', e); }
    });

    // ====================================================================== NUMBER THEORY
    const big = (id) => { const s = val(id).trim(); if (!/^-?\d+$/.test(s)) throw new Error('"' + s + '" is not an integer'); return BigInt(s); };
    const fmtFactors = (f) => f.map(([p, e]) => (e > 1 ? p + '^' + e : String(p))).join(' x ');

    document.querySelectorAll('[data-nt]').forEach((b) => b.addEventListener('click', () => {
        try {
            const n = big('nt-n'), a = big('nt-a');
            let out = '';
            const t0 = performance.now();
            switch (b.dataset.nt) {
                case 'factor': { const f = NT.factor(n); out = kv('n', n) + kv('factorisation', (n === 1n ? '1' : fmtFactors(f.factors)) + (f.complete ? '' : '   (a cofactor could not be split)')); break; }
                case 'phi': out = kv('phi(n)', NT.eulerPhi(n)); break;
                case 'prime': out = kv('n is prime', NT.isPrime(n) ? '<span class="good">yes</span>' : '<span class="bad">no</span>') + kv('method', n < 3317044064679887385961981n ? 'Miller-Rabin, deterministic at this size' : 'strong probable prime (13 bases)'); break;
                case 'digital': out = kv('digital root (base 10)', NT.digitalRoot(n)) + kv('digit sum', NT.digitSum(n)) + kv('n mod 9', NT.mod(n, 9n)) + '<span class="k">The digital root is 1 + (n-1) mod 9, so it is fixed by n mod 9.</span>\n'; break;
                case 'order': { const o = NT.order(a, n); out = o === null ? '<span class="bad">a and n are not coprime: no order</span>' : kv('order of a mod n', o) + kv('check', 'a^order mod n = ' + NT.modpow(a, o, n)); break; }
                case 'root': { if (n > 10n ** 12n) throw new Error('n is too large for the search'); const g = NT.primitiveRoot(n); out = g === null ? '<span class="warn">(Z/n)* is not cyclic: no primitive root</span>' : kv('smallest primitive root', g); break; }
                case 'jacobi': out = kv('Jacobi symbol (a/n)', NT.jacobi(a, n)); break;
                case 'inv': { const v = NT.modinv(a, n); out = v === null ? '<span class="bad">no inverse: gcd(a,n) != 1</span>' : kv('a^-1 mod n', v) + kv('check', (a * v) % n); break; }
                case 'gcd': { const [g, x, y] = NT.egcd(a, n); out = kv('gcd', g) + kv('Bezout', a + ' * ' + x + ' + ' + n + ' * ' + y + ' = ' + g); break; }
                case 'qr': { if (n > 5000n) throw new Error('n must be at most 5000'); const q = NT.quadraticResidues(Number(n)); out = kv('squares mod n', q.join(', ')) + kv('count', q.length); break; }
                case 'pisano': { if (n > 1000000n) throw new Error('n must be at most 1,000,000'); out = kv('Pisano period', NT.pisano(n)); break; }
            }
            show('nt-out', out + '<span class="k">' + (performance.now() - t0).toFixed(1) + ' ms</span>');
        } catch (e) { fail('nt-out', e); }
    }));

    const ROOT_COLORS = ['#00e676', '#00ff88', '#ffb300', '#10b981', '#e2e8f0', '#ffb300', '#94a3b8', '#64748b', '#ffb300'];
    $('dr-run').addEventListener('click', () => {
        try {
            const base = +val('dr-base'), count = +val('dr-n'), name = val('dr-seq'), c = +val('dr-c');
            const vals = NT.sequence(name, count, c);
            const roots = vals.map((v) => NT.digitalRoot(v, base));
            $('dr-cells').innerHTML = roots.slice(0, 72).map((r) => '<span style="background:' + (base === 10 ? ROOT_COLORS[r - 1] : '#4ade80') + '">' + r.toString(base) + '</span>').join('');
            const h = NT.rootHistogram(vals, base);
            const k = base - 1, total = h.counts.reduce((a, b) => a + b, 0);
            barChart($('dr-c1'), h.counts, { expected: h.counts.map(() => total / k), labels: h.counts.map((_, i) => i + 1), title: 'digital roots (red: uniform)' });
            const test = NT.chiSquare(h.counts, h.counts.map(() => 1 / k));
            let period = null;
            for (let p = 1; p <= Math.min(300, roots.length >> 1) && period === null; p++) {
                let ok = true; for (let i = 8; i + p < roots.length; i++) if (roots[i] !== roots[i + p]) { ok = false; break; }
                if (ok) period = p;
            }
            let out = kv('terms', count) + kv('counts for roots 1..' + k, h.counts.join(', '));
            out += kv('chi-square against uniform', num(test.stat, 3) + ' on ' + test.df + ' df, p = ' + (test.p < 1e-4 ? test.p.toExponential(2) : num(test.p, 4)));
            out += kv('period of the root sequence', period === null ? 'none up to 300' : '<span class="good">' + period + '</span>');
            const b1 = BigInt(base - 1);
            if (name === 'powers' && NT.gcd(c, b1) === 1n) out += kv('order of ' + c + ' mod ' + (base - 1), NT.order(c, b1) + '   (equals the period: ' + c + '^n mod ' + (base - 1) + ' repeats with the multiplicative order)');
            if (name === 'fibonacci') out += kv('Pisano period mod ' + (base - 1), NT.pisano(base - 1) + '   (the root sequence repeats with it)');
            if (name === 'primes') out += kv('primes divisible by ' + (base - 1) + ' or its factors', 'the root classes that share a factor with ' + (base - 1) + ' are empty for primes above that factor');
            out += '<span class="k">A non-random sequence is expected to fail the uniformity test; the interest is in which roots occur and with what period.</span>';
            show('dr-out', out);
        } catch (e) { fail('dr-out', e); }
    });

    function drawOrbits() {
        const m = +val('ob-m'), k = +val('ob-k'), off = +val('ob-c');
        const fn = (x) => Number(NT.mod(BigInt(k) * BigInt(x) + BigInt(off), BigInt(m)));
        const orb = NT.orbits(m, fn);
        const cv = $('ob-c1'), c = prep(cv), W = dims(cv)[0], R = W / 2 - 34, cx = W / 2, cy = W / 2;
        const pos = (i) => [cx + R * Math.sin((2 * Math.PI * i) / m), cy - R * Math.cos((2 * Math.PI * i) / m)];
        const palette = ['#00e676', '#ffb300', '#e2e8f0', '#ef4444', '#10b981', '#ffd700', '#94a3b8', '#00ff88'];
        const small = m <= 60;
        for (let x = 0; x < m; x++) {
            const y = fn(x);
            if (y === x) continue;
            const [x0, y0] = pos(x), [x1, y1] = pos(y);
            const cyc = orb.cycleOf[x];
            c.strokeStyle = cyc >= 0 ? palette[cyc % palette.length] : '#475569';
            c.globalAlpha = small ? 0.95 : (cyc >= 0 ? 0.7 : 0.25);
            c.lineWidth = small ? 2 : 1;
            c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
            if (small) { const ang = Math.atan2(y1 - y0, x1 - x0), mx = (x0 + x1) / 2, my = (y0 + y1) / 2; c.beginPath(); c.moveTo(mx, my); c.lineTo(mx - 8 * Math.cos(ang - 0.4), my - 8 * Math.sin(ang - 0.4)); c.lineTo(mx - 8 * Math.cos(ang + 0.4), my - 8 * Math.sin(ang + 0.4)); c.closePath(); c.fillStyle = c.strokeStyle; c.fill(); }
        }
        c.globalAlpha = 1; c.lineWidth = 1;
        for (let x = 0; x < m; x++) {
            const [px, py] = pos(x);
            const three = $('ob-369').checked && x % 3 === 0;
            c.fillStyle = three ? GOLD : '#e2e8f0';
            c.beginPath(); c.arc(px, py, small ? 5 : 2, 0, 6.3); c.fill();
            if (small) { c.fillStyle = MUTED; const lx = cx + (R + 18) * Math.sin((2 * Math.PI * x) / m), ly = cy - (R + 18) * Math.cos((2 * Math.PI * x) / m); c.fillText(String(x), lx - 4, ly + 4); }
        }
        const lens = orb.cycles.map((cy2) => cy2.length).sort((a, b) => b - a);
        const tails = orb.depth.filter((d) => d > 0).length;
        let out = kv('map', 'x -> ' + k + 'x + ' + off + '  (mod ' + m + ')');
        out += kv('cycles', orb.cycles.length + '  (lengths ' + (lens.length > 16 ? lens.slice(0, 16).join(', ') + ', ...' : lens.join(', ')) + ')');
        if (m <= 60) orb.cycles.forEach((cc) => { out += '   ' + cc.join(' -> ') + ' -> ' + cc[0] + '\n'; });
        out += kv('nodes on tails (not on a cycle)', tails);
        if (NT.gcd(k, m) === 1n && off === 0) {
            const o = NT.order(k, m);
            out += kv('multiplicative order of k mod m', o + '   (every cycle length of an invertible map divides it)');
            out += kv('all cycle lengths divide the order', lens.every((l) => BigInt(l) !== 0n && o % BigInt(l) === 0n) ? '<span class="good">yes</span>' : '<span class="bad">no</span>');
        }
        show('ob-out', out);
        return { m, k, off };
    }
    $('ob-run').addEventListener('click', () => { try { drawOrbits(); } catch (e) { fail('ob-out', e); } });
    $('ob-shor').addEventListener('click', async () => {
        try {
            const { m, k } = drawOrbits();
            const a = ((k % m) + m) % m;
            const r = await post('/api/lab/algorithm', { name: 'shor_order', a, N: m, t: 0, shots: 300, seed: 1 });
            const prev = $('ob-out').innerHTML;
            show('ob-out', prev + '\n<span class="warn">Quantum order finding (exact simulation, ' + r.counting_qubits + ' counting qubits, 300 shots):</span>\n' +
                kv('true order', r.true_order) + kv('estimated from the shots', r.estimated_order + ' (' + r.shots_giving_true_order + ' of 300 shots)') +
                kv('factors of N found', r.factors_found.length ? r.factors_found.join(', ') : 'none (N may be a prime power, or the order is odd)'));
        } catch (e) { fail('ob-out', e); }
    });
    $('ob-walk').addEventListener('click', async () => {
        try {
            const { m, k, off } = drawOrbits();
            const prev = $('ob-out').innerHTML;
            const start = m > 1 ? 1 % m : 0;
            const fn = (x) => Number(NT.mod(BigInt(k) * BigInt(x) + BigInt(off), BigInt(m)));
            let target = fn(start); if (target === start) target = (start + 1) % m;
            const tmp = document.createElement('div'); tmp.id = 'ob-tmp'; document.body.appendChild(tmp);
            await runWalk({ kind: 'unit_multiplication', n: m, k, offset: off }, start, target, 12, 'ob-tmp');
            const html = tmp.innerHTML; tmp.remove();
            show('ob-out', prev + '\n<span class="warn">Walks on this graph, from node ' + start + ' towards node ' + target + ' (t up to 12):</span>\n' + html);
        } catch (e) { fail('ob-out', e); }
    });

    // ---- claim checker ----------------------------------------------------------------------------------------------
    const histLine = (h) => h.counts.map((c, i) => (i + 1) + ':' + c).join('  ');
    const CLAIMS = [
        { text: 'Primes above 3 never have digital root 3, 6 or 9', run() {
            const h = NT.rootHistogram(NT.sequence('primes', 10000).slice(2));
            const bad = h.counts[2] + h.counts[5] + h.counts[8];
            return { ok: bad === 0, detail: 'first 10,000 primes: ' + histLine(h) + '\nReason: a digital root of 3, 6 or 9 means divisible by 3.' };
        } },
        { text: 'Powers of 2 have digital roots 1, 2, 4, 8, 7, 5 repeating', run() {
            const r = NT.sequence('powers', 600, 2).map((v) => NT.digitalRoot(v));
            const ok = r.every((x, i) => x === [1, 2, 4, 8, 7, 5][i % 6]);
            return { ok, detail: '600 powers checked. The order of 2 mod 9 is ' + NT.order(2, 9) + ': 2^n mod 9 cycles through the six units of Z/9.' };
        } },
        { text: 'Powers of 3 have digital roots 3, 9, 9, 9, ...', run() {
            const r = NT.sequence('powers', 40, 3).map((v) => NT.digitalRoot(v));
            return { ok: r[0] === 1 && r[1] === 3 && r.slice(2).every((x) => x === 9), detail: 'sequence ' + r.slice(0, 12).join(' ') + ' ...  (3^n is divisible by 9 for n >= 2)' };
        } },
        { text: 'Fibonacci digital roots repeat with period 24', run() {
            const r = NT.sequence('fibonacci', 2000).map((v) => NT.digitalRoot(v));
            let ok = true; for (let i = 1; i + 24 < r.length; i++) if (r[i] !== r[i + 24]) { ok = false; break; }
            return { ok, detail: 'Pisano period mod 9 = ' + NT.pisano(9) + '; the first 24 roots: ' + r.slice(1, 25).join(' ') };
        } },
        { text: 'Squares only have digital roots 1, 4, 7 or 9', run() {
            const h = NT.rootHistogram(NT.sequence('squares', 5000));
            const ok = [1, 2, 4, 5, 7].every((i) => h.counts[i] === 0);
            return { ok, detail: histLine(h) + '\nSquares mod 9: ' + NT.quadraticResidues(9).join(', ') };
        } },
        { text: 'Cubes only have digital roots 1, 8 or 9', run() {
            const h = NT.rootHistogram(NT.sequence('cubes', 5000));
            const ok = [1, 2, 3, 4, 5, 6].every((i) => h.counts[i] === 0);
            return { ok, detail: histLine(h) };
        } },
        { text: 'Doubling mod 9 splits the residues into cycles of length 6, 2 and 1', run() {
            const o = NT.orbits(9, (x) => (2 * x) % 9);
            const lens = o.cycles.map((c) => c.length).sort((a, b) => b - a);
            return { ok: lens.join() === '6,2,1', detail: o.cycles.map((c) => c.join(' -> ')).join('\n') + '\n(0 is the fixed point; 3 <-> 6; the six units form the long cycle)' };
        } },
        { text: 'The digital root of a product is the digital root of the product of the roots', run() {
            let bad = 0; const N = 20000;
            for (let i = 1; i <= N; i++) { const a = BigInt(i * 7919 + 13), b = BigInt(i * 104729 + 7); if (NT.digitalRoot(a * b) !== NT.digitalRoot(BigInt(NT.digitalRoot(a)) * BigInt(NT.digitalRoot(b)))) bad++; }
            return { ok: bad === 0, detail: N + ' pairs, ' + bad + ' counterexamples' };
        } },
        { text: 'Primes are equally spread over the six units mod 9 (Dirichlet)', run() {
            const ps = NT.sequence('primes', 20000).slice(2);
            const cls = [1, 2, 4, 5, 7, 8], cnt = cls.map((c) => ps.filter((p) => p % 9n === BigInt(c)).length);
            const t = NT.chiSquare(cnt, cls.map(() => 1 / 6));
            return { ok: t.p > 0.01, detail: cls.map((c, i) => c + ':' + cnt[i]).join('  ') + '\nchi-square ' + num(t.stat, 3) + ' on ' + t.df + ' df, p = ' + num(t.p, 4) + '.  Asymptotically equal; small differences (Chebyshev bias) are expected.' };
        } },
        { text: 'Integers 1..N have exactly equal numbers of each digital root (up to a remainder)', run() {
            const N = 100000, h = NT.rootHistogram(NT.sequence('naturals', N));
            return { ok: Math.max(...h.counts) - Math.min(...h.counts) <= 1, detail: histLine(h) + '\nUniform by construction (period 9), not a statistical fact.' };
        } },
    ];
    $('cl-sel').innerHTML = CLAIMS.map((c, i) => '<option value="' + i + '">' + esc(c.text) + '</option>').join('');
    $('cl-run').addEventListener('click', () => {
        try {
            const c = CLAIMS[+val('cl-sel')], r = c.run();
            show('cl-out', '<span class="' + (r.ok ? 'good' : 'bad') + '">' + (r.ok ? 'HOLDS on the range tested' : 'FAILS') + '</span>\n' + esc(r.detail) + '\n<span class="k">A finite check supports a claim; it does not prove it. The reasons given are proofs.</span>');
        } catch (e) { fail('cl-out', e); }
    });

    // ====================================================================== GEOMETRY
    let spin = G.spinor(+val('hp-th'), +val('hp-ph'), 0);
    let view = { yaw: 0.6, pitch: -0.4 };
    const bloch = () => G.hopf(spin[0], spin[1]);
    function syncSliders() {
        const b = bloch();
        $('hp-th').value = Math.acos(Math.max(-1, Math.min(1, b[2])));
        $('hp-ph').value = Math.atan2(b[1], b[0]);
    }
    const rot3 = (p, yaw, pitch) => {
        const [x, y, z] = p;
        const x1 = x * Math.cos(yaw) + z * Math.sin(yaw), z1 = -x * Math.sin(yaw) + z * Math.cos(yaw);
        const y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch), z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);
        return [x1, y2, z2];
    };

    const cstr = (z) => num(z[0], 3) + (z[1] < 0 ? ' - ' : ' + ') + num(Math.abs(z[1]), 3) + 'i';

    function drawBloch() {
        if (!geoActive && $('tab-geo') && !$('tab-geo').classList.contains('active')) return;
        const cv = $('hp-bloch'), c = prep(cv), W = dims(cv)[0], R = W / 2 - 30, cx = W / 2, cy = W / 2;
        const P = (p) => { const q = rot3([p[0], p[2], p[1]], 0.5, -0.35); return [cx + R * q[0], cy - R * q[1], q[2]]; };
        c.strokeStyle = FAINT; c.beginPath(); c.arc(cx, cy, R, 0, 6.3); c.stroke();
        [[[1, 0, 0], 'x'], [[0, 1, 0], 'y'], [[0, 0, 1], 'z']].forEach(([a, l]) => {
            const p0 = P([-a[0], -a[1], -a[2]]), p1 = P(a);
            c.strokeStyle = '#475569'; c.beginPath(); c.moveTo(p0[0], p0[1]); c.lineTo(p1[0], p1[1]); c.stroke();
            c.fillStyle = MUTED; c.fillText(l, p1[0] + 4, p1[1] - 4);
        });
        c.strokeStyle = '#1f2f46'; c.beginPath();
        for (let i = 0; i <= 64; i++) { const a = (2 * Math.PI * i) / 64, p = P([Math.cos(a), Math.sin(a), 0]); i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1]); }
        c.stroke();
        const b = bloch(), pt = P(b), o = P([0, 0, 0]);
        c.strokeStyle = GOLD; c.lineWidth = 3; c.beginPath(); c.moveTo(o[0], o[1]); c.lineTo(pt[0], pt[1]); c.stroke(); c.lineWidth = 1;
        c.fillStyle = GOLD; c.beginPath(); c.arc(pt[0], pt[1], 5, 0, 6.3); c.fill();
        c.fillStyle = TEXT; c.fillText('|0>', P([0, 0, 1])[0] + 6, P([0, 0, 1])[1] + 12); c.fillText('|1>', P([0, 0, -1])[0] + 6, P([0, 0, -1])[1] - 4);
        const th = Math.acos(Math.max(-1, Math.min(1, b[2]))), ph = Math.atan2(b[1], b[0]);
        show('hp-out', kv('Bloch vector', '(' + b.map((x) => num(x, 4)).join(', ') + ')') + kv('theta, phi', num(th, 4) + ', ' + num(ph, 4)) +
            kv('P(0), P(1)', num(Math.cos(th / 2) ** 2, 4) + ', ' + num(Math.sin(th / 2) ** 2, 4)) +
            kv('spinor (z1, z2)', cstr(spin[0]) + ',  ' + cstr(spin[1])));
    }

    function drawHopf() {
        if (!$('tab-geo').classList.contains('active')) return;
        const cv = $('hp-fib'), c = prep(cv), [W, H] = dims(cv), S = 36;
        const lat = +val('hp-lat'), per = +val('hp-per');
        const proj = (p) => { const q = rot3(p, view.yaw, view.pitch), k = 4.5 / (4.5 - q[2] * 0.6); return [W / 2 + S * q[0] * k, H / 2 - S * q[1] * k, q[2]]; };
        const drawFiber = (th, ph, color, w) => {
            const pts = G.hopfFiber(th, ph, 120).filter((p) => Math.hypot(...p) < 9);
            if (pts.length < 3) return;
            c.strokeStyle = color; c.lineWidth = w; c.beginPath();
            pts.forEach((p, i) => { const q = proj(p); i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]); });
            c.closePath(); c.stroke();
        };
        for (let i = 0; i < lat; i++) {
            const th = 0.3 * Math.PI + (0.4 * Math.PI * (lat === 1 ? 0.5 : i / (lat - 1)));
            // Latitudes are told apart by emerald lightness (palette: no cyan or violet hues).
            const light = Math.round(28 + 40 * (i / Math.max(1, lat - 1)));
            for (let j = 0; j < per; j++) drawFiber(th, (2 * Math.PI * j) / per, `hsla(152,90%,${light}%,0.6)`, 1.3);
        }
        const b = bloch();
        drawFiber(Math.min(Math.acos(Math.max(-1, Math.min(1, b[2]))), 2.9), Math.atan2(b[1], b[0]), GOLD, 3);
        c.fillStyle = MUTED; c.fillText('Hopf fibres (gold: over your state)', 6, 12);
    }

    ['hp-th', 'hp-ph'].forEach((id) => $(id).addEventListener('input', () => { spin = G.spinor(+val('hp-th'), +val('hp-ph'), 0); drawBloch(); drawHopf(); }));
    ['hp-lat', 'hp-per'].forEach((id) => $(id).addEventListener('input', drawHopf));
    document.querySelectorAll('[data-gate]').forEach((b) => b.addEventListener('click', () => {
        const g = b.dataset.gate;
        if (g === 'RESET') spin = [[1, 0], [0, 0]];
        else if (g.length === 2 && g[0] === 'R') spin = G.applyGate(G.rotation(g[1].toLowerCase(), 0.4), spin);
        else spin = G.applyGate(G.GATES[g], spin);
        syncSliders(); drawBloch(); drawHopf();
    }));
    let drag = null;
    const hf = $('hp-fib');
    hf.addEventListener('mousedown', (e) => { drag = [e.clientX, e.clientY]; });
    window.addEventListener('mouseup', () => { drag = null; });
    window.addEventListener('mousemove', (e) => { if (!drag) return; view.yaw += (e.clientX - drag[0]) * 0.01; view.pitch += (e.clientY - drag[1]) * 0.01; drag = [e.clientX, e.clientY]; drawHopf(); });

    // ---- 4-polytopes ---------------------------------------------------------------------------------------------
    const PLANES = ['xy', 'xz', 'xw', 'yz', 'yw', 'zw'];
    const angles = [0, 0, 0.4, 0, 0.7, 0];
    $('pt-sliders').innerHTML = PLANES.map((p, i) => '<label>' + p + ' <input type="range" min="-3.1416" max="3.1416" step="0.01" value="' + angles[i] + '" data-pl="' + i + '"></label>').join('');
    $('pt-sliders').addEventListener('input', (e) => { if (e.target.dataset.pl !== undefined) { angles[+e.target.dataset.pl] = +e.target.value; drawPoly(); } });
    let poly = G.polytope('tesseract');
    const describePoly = () => {
        const deg = new Array(poly.vertices.length).fill(0);
        poly.edges.forEach(([i, j]) => { deg[i]++; deg[j]++; });
        show('pt-out', kv('vertices, edges', poly.vertices.length + ', ' + poly.edges.length) + kv('edge length', num(poly.edgeLength, 4)) + kv('vertex degree', [...new Set(deg)].join(', ') + '   (regular polytopes have one value)'));
    };
    $('pt-sel').addEventListener('change', () => { poly = G.polytope(val('pt-sel')); describePoly(); drawPoly(); });
    function drawPoly() {
        if (!$('tab-geo').classList.contains('active')) return;
        const cv = $('pt-c'), c = prep(cv), [W, H] = dims(cv);
        const r = G.rotation4(angles), S = 105;
        const pts = poly.vertices.map((v) => { const p3 = G.project4(G.matVec(r, v), 4); const q = rot3(p3, 0.4, -0.3), k = 6 / (6 - q[2]); return [W / 2 + S * q[0] * k, H / 2 - S * q[1] * k, q[2]]; });
        poly.edges.forEach(([i, j]) => { const d = (pts[i][2] + pts[j][2]) / 2; c.strokeStyle = `rgba(0,230,118,${0.25 + 0.5 * (d + 2) / 4})`; c.beginPath(); c.moveTo(pts[i][0], pts[i][1]); c.lineTo(pts[j][0], pts[j][1]); c.stroke(); });
        pts.forEach((p) => { c.fillStyle = GOLD; c.beginPath(); c.arc(p[0], p[1], 3, 0, 6.3); c.fill(); });
    }
    describePoly();
    let last = 0;
    function tick(ts) {
        if (geoActive && $('pt-auto').checked && ts - last > 30) {
            last = ts; angles[2] += 0.012; angles[4] += 0.008; angles[5] += 0.005;
            document.querySelectorAll('[data-pl]').forEach((s) => { const a = ((+angles[+s.dataset.pl] + Math.PI) % (2 * Math.PI)) - Math.PI; s.value = a; });
            drawPoly();
        }
        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    // initial draws for the geometry tab happen when it is opened
    window.LabChart = { prep, dims };
})();
