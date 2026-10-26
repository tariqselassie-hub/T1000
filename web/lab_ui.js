// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

// Instrument chrome for the lab bench: panel designators, per-panel run state and timing, the UTC clock,
// the API round-trip readout and an idle frame on canvases that have not been drawn yet. Nothing here
// computes a result; it only watches the panels that lab.js fills in.
'use strict';
(function () {
    const $ = (id) => document.getElementById(id);
    const PREFIX = { algo: 'A', prob: 'P', nt: 'N', geo: 'G' };

    // ---- panel headers and run state ------------------------------------------------------------------------
    function stateOf(out) {
        const t = out.textContent.trim();
        if (!t) return 'idle';
        if (/^running/i.test(t)) return 'busy';
        return out.querySelector('.err') ? 'fault' : 'ok';
    }

    function wirePanel(card, pid) {
        const h2 = card.querySelector('h2');
        if (!h2) return;
        const head = document.createElement('div');
        head.className = 'card-head';
        head.innerHTML = '<span class="pid"></span><span class="state"><span class="led"></span><span class="txt">IDLE</span></span>';
        head.querySelector('.pid').textContent = pid;
        card.insertBefore(head, h2);
        head.insertBefore(h2, head.querySelector('.state'));
        card.dataset.state = 'idle';

        const txt = head.querySelector('.txt');
        let t0 = 0;
        const update = (out) => {
            const s = stateOf(out);
            card.dataset.state = s;
            if (s === 'busy') { t0 = performance.now(); txt.textContent = 'RUNNING'; return; }
            if (s === 'idle') { txt.textContent = 'IDLE'; return; }
            const dt = t0 ? performance.now() - t0 : null;
            t0 = 0;
            txt.textContent = (s === 'fault' ? 'FAULT' : 'OK') + (dt !== null ? '  ' + fmtMs(dt) : '');
        };
        card.querySelectorAll('.out').forEach((out) => {
            new MutationObserver(() => update(out)).observe(out, { childList: true, characterData: true, subtree: true });
        });
    }

    const fmtMs = (ms) => (ms < 1000 ? ms.toFixed(0) + ' ms' : (ms / 1000).toFixed(2) + ' s');

    document.querySelectorAll('section.tab').forEach((sec) => {
        const p = PREFIX[sec.id.replace('tab-', '')] || 'X';
        sec.querySelectorAll('.card').forEach((card, i) => wirePanel(card, p + '-' + String(i + 1).padStart(2, '0')));
    });

    // On narrow screens the tab strip scrolls; keep the active tab visible.
    $('tabs').addEventListener('click', (ev) => {
        const b = ev.target.closest('button[data-tab]');
        if (b) b.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    });

    // ---- clock and API round trip -----------------------------------------------------------------------------
    const clock = $('ro-clock');
    const tickClock = () => { clock.textContent = new Date().toISOString().slice(11, 19); };
    tickClock();
    setInterval(tickClock, 1000);

    // The server handles one request at a time and /api/status probes the serial port, so the readout times
    // the bench's own API calls (last round trip) instead of polling.
    const api = $('ro-api');
    const fetch0 = window.fetch.bind(window);
    window.fetch = async (url, opts) => {
        const isApi = typeof url === 'string' && url.startsWith('/api/');
        const t0 = performance.now();
        try {
            const r = await fetch0(url, opts);
            if (isApi) { api.textContent = fmtMs(performance.now() - t0); api.style.color = r.ok ? '' : '#ef4444'; }
            return r;
        } catch (e) {
            if (isApi) { api.textContent = 'offline'; api.style.color = '#ef4444'; }
            throw e;
        }
    };

    // ---- idle frame on canvases that nothing has drawn yet ----------------------------------------------------
    function idleFrame(cv) {
        const chart = window.LabChart;
        if (!chart || cv.dataset.w) return;
        const c = chart.prep(cv), [W, H] = chart.dims(cv);
        c.strokeStyle = '#1e293b';
        for (let x = 0.5; x < W; x += 40) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); }
        for (let y = 0.5; y < H; y += 40) { c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
        c.strokeStyle = '#334155';
        c.beginPath(); c.moveTo(W / 2 - 8, H / 2 + 0.5); c.lineTo(W / 2 + 8, H / 2 + 0.5); c.moveTo(W / 2 + 0.5, H / 2 - 8); c.lineTo(W / 2 + 0.5, H / 2 + 8); c.stroke();
        c.fillStyle = '#475569'; c.textAlign = 'center';
        c.fillText('NO SIGNAL  //  RUN TO ACQUIRE', W / 2, H / 2 + 24);
        c.textAlign = 'left';
    }
    // Geometry canvases are drawn when their tab opens; everything else waits for a run. lab.js loads after
    // this file (so its first API call is timed), hence the wait for the load event.
    window.addEventListener('load', () => {
        document.querySelectorAll('section.tab:not(#tab-geo) canvas').forEach(idleFrame);
    });
})();
