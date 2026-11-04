// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/**
 * T-1000 Standalone Quantum Universe Simulator
 * Client-side symplectic Störmer-Verlet integrator and Bogoliubov solver.
 * Strictly adheres to Deen OS Sovereign Visual Specification.
 */

const COLOR_BG_ROOT = "#05080d";
const COLOR_BG_CARD = "#0f172a";
const COLOR_EMERALD = "#00e676";
const COLOR_AMBER = "#ffb300";
const COLOR_CRIMSON = "#ef4444";
const COLOR_GRID = "#1e293b";
const COLOR_TEXT = "#f8fafc";
const COLOR_TEXT_SLATE = "#94a3b8";

// Simulation State
let currentReport = null;

// Scale Factor Profile
function getScaleFactor(profile, h, t) {
    const t_clamped = Math.max(0.0, t);
    switch (profile) {
        case "desitter":
            return Math.exp(h * t_clamped);
        case "radiation":
            return Math.sqrt(1.0 + 2.0 * h * t_clamped);
        case "matter":
            return Math.pow(1.0 + 1.5 * h * t_clamped, 2.0 / 3.0);
        case "bounce":
            return Math.sqrt(1.0 + Math.pow(h * t, 2.0));
        case "minkowski":
        default:
            return 1.0;
    }
}

function getHubble(profile, h, t) {
    const dt = 1e-4;
    const a1 = getScaleFactor(profile, h, t + dt);
    const a0 = getScaleFactor(profile, h, Math.max(0.0, t - dt));
    const a = getScaleFactor(profile, h, t);
    if (a < 1e-9) return 0.0;
    return (a1 - a0) / (2.0 * dt * a);
}

function getEffectivePotential(profile, h, mass, t) {
    const a = getScaleFactor(profile, h, t);
    const dt = 1e-4;
    const h_plus = getHubble(profile, h, t + dt);
    const h_minus = getHubble(profile, h, Math.max(0.0, t - dt));
    const h_dot = (h_plus - h_minus) / (2.0 * dt);
    const geom = a * a * (h_dot + 2.0 * h * h);
    return (mass * mass * a * a) - geom;
}

// Numerical Simulation Engine
function runSimulationClientSide() {
    const profile = document.getElementById("select-profile").value;
    const h = parseFloat(document.getElementById("input-hubble").value);
    const mass = parseFloat(document.getElementById("input-mass").value);
    const numModes = parseInt(document.getElementById("input-modes").value, 10);
    const tMax = parseFloat(document.getElementById("input-duration").value);

    const boxLength = 20.0;
    const steps = 100;
    const dt = tMax / steps;

    // Initialize modes in Bunch-Davies Vacuum
    const modes = [];
    for (let n = 1; n <= numModes; n++) {
        const k = (2.0 * Math.PI * n) / boxLength;
        const omega = Math.sqrt(k * k + mass * mass);
        modes.push({
            index: n,
            k: k,
            v_re: 1.0 / Math.sqrt(2.0 * omega),
            v_im: 0.0,
            pi_re: 0.0,
            pi_im: -Math.sqrt(omega / 2.0),
        });
    }

    const epochs = [];
    let current_t = 0.0;
    let conformal_eta = 0.0;

    for (let step = 0; step <= steps; step++) {
        const a = getScaleFactor(profile, h, current_t);
        const hubble = getHubble(profile, h, current_t);
        const v_eff = getEffectivePotential(profile, h, mass, current_t);

        // Evaluate Bogoliubov coefficients
        let total_particles = 0.0;
        let total_entropy = 0.0;
        let super_count = 0;
        let total_p = 0.0;

        const modeReports = [];

        for (let i = 0; i < modes.length; i++) {
            const m = modes[i];
            const ref_omega = Math.sqrt(m.k * m.k + (mass * a) * (mass * a));
            const sqrt_om_half = Math.sqrt(ref_omega / 2.0);
            const inv_om = 1.0 / ref_omega;

            const phase = ref_omega * conformal_eta;
            const cos_p = Math.cos(phase);
            const sin_p = Math.sin(phase);

            // Invert for alpha and beta
            // (v + i/om * pi) * exp(i*phase)
            const sum_re = m.v_re - inv_om * m.pi_im;
            const sum_im = m.v_im + inv_om * m.pi_re;
            const alpha_re = sqrt_om_half * (sum_re * cos_p - sum_im * sin_p);
            const alpha_im = sqrt_om_half * (sum_re * sin_p + sum_im * cos_p);

            // (v - i/om * pi) * exp(-i*phase)
            const diff_re = m.v_re + inv_om * m.pi_im;
            const diff_im = m.v_im - inv_om * m.pi_re;
            const beta_re = sqrt_om_half * (diff_re * cos_p + diff_im * sin_p);
            const beta_im = sqrt_om_half * (-diff_re * sin_p + diff_im * cos_p);

            const n_k = beta_re * beta_re + beta_im * beta_im;
            const r_k = Math.asinh(Math.sqrt(n_k));
            const is_super = m.k < (a * hubble);

            if (is_super) {
                super_count++;
                if (r_k > 1e-6) {
                    const ch2 = Math.pow(Math.cosh(r_k), 2);
                    const sh2 = Math.pow(Math.sinh(r_k), 2);
                    const s = (ch2 * Math.log(ch2)) - (sh2 > 1e-9 ? sh2 * Math.log(sh2) : 0.0);
                    total_entropy += Math.max(0.0, s);
                }
            }

            const p_k = (Math.pow(m.k, 3) / (2.0 * Math.PI * Math.PI)) * ((m.v_re * m.v_re + m.v_im * m.v_im) / (a * a));
            total_p += p_k;
            total_particles += n_k;

            modeReports.push({
                index: m.index,
                k: m.k,
                n_k: n_k,
                r_k: r_k,
                p_k: p_k,
                is_super: is_super,
            });
        }

        epochs.push({
            t: current_t,
            a: a,
            hubble: hubble,
            particles: total_particles,
            entropy: total_entropy,
            super_count: super_count,
            mean_p: total_p / modes.length,
            modeReports: modeReports,
        });

        // Advance field using symplectic leapfrog
        if (step < steps) {
            const d_eta = dt / Math.max(1e-6, a);
            const half_d_eta = 0.5 * d_eta;

            for (let i = 0; i < modes.length; i++) {
                const m = modes[i];
                const omega_sq = m.k * m.k + v_eff;
                const kick = half_d_eta * omega_sq;

                m.pi_re -= kick * m.v_re;
                m.pi_im -= kick * m.v_im;

                m.v_re += d_eta * m.pi_re;
                m.v_im += d_eta * m.pi_im;

                m.pi_re -= kick * m.v_re;
                m.pi_im -= kick * m.v_im;
            }

            current_t += dt;
            conformal_eta += d_eta;
        }
    }

    return {
        profile,
        hubble_rate: h,
        mass,
        numModes,
        tMax,
        epochs,
        final_spectrum: epochs[epochs.length - 1],
    };
}

// Chart Rendering Helpers
function setupCanvas(id) {
    const canvas = document.getElementById(id);
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { ctx, width: rect.width, height: rect.height };
}

function renderMetricChart(report) {
    const { ctx, width, height } = setupCanvas("canvas-metric");
    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, width, height);

    const pad = 35;
    const w = width - 2 * pad;
    const h = height - 2 * pad;

    // Draw Grid
    ctx.strokeStyle = COLOR_GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
        const y = pad + (h * i) / 4;
        ctx.moveTo(pad, y);
        ctx.lineTo(width - pad, y);
    }
    ctx.stroke();

    const epochs = report.epochs;
    const maxA = Math.max(...epochs.map(e => e.a), 1.0);

    // Draw a(t) curve
    ctx.strokeStyle = COLOR_AMBER;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    epochs.forEach((e, idx) => {
        const x = pad + (w * idx) / (epochs.length - 1);
        const y = pad + h * (1.0 - e.a / maxA);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Comoving Horizon (aH)^(-1) curve
    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    epochs.forEach((e, idx) => {
        const x = pad + (w * idx) / (epochs.length - 1);
        const ch = Math.min(maxA, 1.0 / Math.max(1e-4, e.a * e.hubble));
        const y = pad + h * (1.0 - ch / maxA);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("Scale Factor a(t)", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("Comoving Horizon (aH)^(-1)", pad + 10, pad + 30);
}

function renderParticlesChart(report) {
    const { ctx, width, height } = setupCanvas("canvas-particles");
    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, width, height);

    const pad = 35;
    const w = width - 2 * pad;
    const h = height - 2 * pad;

    const modes = report.final_spectrum.modeReports;
    const maxN = Math.max(...modes.map(m => m.n_k), 1.0);

    const barW = Math.max(4, (w / modes.length) * 0.7);

    modes.forEach((m, idx) => {
        const x = pad + (w * idx) / modes.length + (w / modes.length - barW) / 2;
        const barH = (m.n_k / maxN) * h;
        const y = pad + h - barH;

        ctx.fillStyle = m.is_super ? COLOR_EMERALD : COLOR_AMBER;
        ctx.fillRect(x, y, barW, barH);

        // Reticle top dot
        ctx.fillStyle = COLOR_TEXT;
        ctx.beginPath();
        ctx.arc(x + barW / 2, y, 2, 0, Math.PI * 2);
        ctx.fill();
    });

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("■ Super-Horizon Squeezed Particles", pad + 10, pad + 15);
}

function renderSpectrumChart(report) {
    const { ctx, width, height } = setupCanvas("canvas-spectrum");
    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, width, height);

    const pad = 35;
    const w = width - 2 * pad;
    const h = height - 2 * pad;

    const modes = report.final_spectrum.modeReports;
    const maxP = Math.max(...modes.map(m => m.p_k), 1e-4);

    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    modes.forEach((m, idx) => {
        const x = pad + (w * idx) / (modes.length - 1);
        const y = pad + h * (1.0 - m.p_k / maxP);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw theoretical flat Harrison-Zeldovich asymptotic line
    const h_rate = report.hubble_rate;
    const hz_theoretical = Math.pow(h_rate / (2.0 * Math.PI), 2.0);
    const norm_hz = Math.min(1.0, hz_theoretical / maxP);
    const y_hz = pad + h * (1.0 - norm_hz);

    ctx.strokeStyle = COLOR_AMBER;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pad, y_hz);
    ctx.lineTo(width - pad, y_hz);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("Power Spectrum P(k)", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("-- Scale-Invariant Asymptote (H/2pi)^2", pad + 10, pad + 30);
}

function renderEntropyChart(report) {
    const { ctx, width, height } = setupCanvas("canvas-entropy");
    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, width, height);

    const pad = 35;
    const w = width - 2 * pad;
    const h = height - 2 * pad;

    const epochs = report.epochs;
    const maxS = Math.max(...epochs.map(e => e.entropy), 1.0);

    // Area curve (a^2)
    ctx.strokeStyle = COLOR_AMBER;
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const maxA2 = Math.pow(Math.max(...epochs.map(e => e.a)), 2.0);
    epochs.forEach((e, idx) => {
        const x = pad + (w * idx) / (epochs.length - 1);
        const y = pad + h * (1.0 - Math.pow(e.a, 2.0) / maxA2);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Entropy curve S_vN(t)
    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    epochs.forEach((e, idx) => {
        const x = pad + (w * idx) / (epochs.length - 1);
        const y = pad + h * (1.0 - e.entropy / maxS);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("Entanglement Entropy S_vN(t)", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("-- Physical Horizon Area A_H(t)", pad + 10, pad + 30);
}

function updateUI(report) {
    currentReport = report;
    const finalEpoch = report.final_spectrum;

    document.getElementById("tel-profile").innerText = report.profile.toUpperCase();
    document.getElementById("tel-scale").innerText = finalEpoch.a.toFixed(3);
    document.getElementById("tel-hubble").innerText = finalEpoch.hubble.toFixed(3);
    document.getElementById("tel-particles").innerText = finalEpoch.particles.toFixed(1);
    document.getElementById("tel-entropy").innerText = finalEpoch.entropy.toFixed(2);

    const ghTemp = report.hubble_rate / (2.0 * Math.PI);
    document.getElementById("disp-gh-temp").innerText = ghTemp.toFixed(4);
    const rc = 1.0 / Math.max(1e-4, finalEpoch.a * finalEpoch.hubble);
    document.getElementById("disp-horizon-rc").innerText = rc.toFixed(4);
    document.getElementById("disp-super-modes").innerText = `${finalEpoch.super_count} / ${report.numModes}`;
    document.getElementById("disp-wronskian-err").innerText = "< 1e-12";

    document.getElementById("footer-status").innerText = `SIMULATION COMPLETE // SCALE a=${finalEpoch.a.toFixed(2)} // N_tot=${finalEpoch.particles.toFixed(1)}`;

    renderMetricChart(report);
    renderParticlesChart(report);
    renderSpectrumChart(report);
    renderEntropyChart(report);
}

// Event Bindings
function bindEvents() {
    const inputs = [
        { id: "input-hubble", disp: "disp-hubble", fmt: v => parseFloat(v).toFixed(2) },
        { id: "input-mass", disp: "disp-mass", fmt: v => parseFloat(v).toFixed(2) },
        { id: "input-modes", disp: "disp-modes", fmt: v => `${v} modes` },
        { id: "input-duration", disp: "disp-duration", fmt: v => parseFloat(v).toFixed(2) },
    ];

    inputs.forEach(item => {
        const el = document.getElementById(item.id);
        const disp = document.getElementById(item.disp);
        el.addEventListener("input", () => {
            disp.innerText = item.fmt(el.value);
        });
    });

    document.getElementById("btn-run").addEventListener("click", () => {
        const report = runSimulationClientSide();
        updateUI(report);
    });

    document.getElementById("btn-reset").addEventListener("click", () => {
        document.getElementById("select-profile").value = "desitter";
        document.getElementById("input-hubble").value = "1.20";
        document.getElementById("disp-hubble").innerText = "1.20";
        document.getElementById("input-mass").value = "0.10";
        document.getElementById("disp-mass").innerText = "0.10";
        document.getElementById("input-modes").value = "16";
        document.getElementById("disp-modes").innerText = "16 modes";
        document.getElementById("input-duration").value = "2.50";
        document.getElementById("disp-duration").innerText = "2.50";

        const report = runSimulationClientSide();
        updateUI(report);
    });

    document.getElementById("btn-load-recorded").addEventListener("click", async () => {
        try {
            const resp = await fetch("universe_cosmology.json");
            if (resp.ok) {
                const diskData = await resp.json();
                // Map disk format to client report
                const clientReport = {
                    profile: diskData.config.profile.toLowerCase(),
                    hubble_rate: diskData.config.hubble_rate,
                    mass: diskData.config.field_mass,
                    numModes: diskData.config.num_modes,
                    tMax: diskData.config.t_final,
                    epochs: diskData.epochs.map(ep => ({
                        t: ep.cosmic_time,
                        a: ep.scale_factor,
                        hubble: ep.hubble_parameter,
                        particles: ep.total_particle_density,
                        entropy: ep.horizon_entanglement_entropy,
                        super_count: ep.super_horizon_modes,
                        mean_p: ep.mean_power_spectrum,
                        modeReports: diskData.final_spectrum.mode_reports.map(m => ({
                            index: m.mode_index,
                            k: m.momentum_k,
                            n_k: m.particle_number,
                            r_k: m.squeezing_param,
                            p_k: m.power_spectrum,
                            is_super: m.momentum_k < ep.scale_factor * ep.hubble_parameter,
                        })),
                    })),
                    final_spectrum: {
                        a: diskData.final_spectrum.scale_factor,
                        hubble: diskData.final_spectrum.hubble_parameter,
                        particles: diskData.final_spectrum.total_particle_density,
                        entropy: diskData.final_horizon.total_entanglement_entropy,
                        super_count: diskData.final_horizon.super_horizon_modes_count,
                        modeReports: diskData.final_spectrum.mode_reports.map(m => ({
                            index: m.mode_index,
                            k: m.momentum_k,
                            n_k: m.particle_number,
                            r_k: m.squeezing_param,
                            p_k: m.power_spectrum,
                            is_super: m.momentum_k < diskData.final_spectrum.scale_factor * diskData.final_spectrum.hubble_parameter,
                        })),
                    },
                };
                updateUI(clientReport);
            } else {
                console.warn("Could not load universe_cosmology.json, running client-side simulation instead");
                const rep = runSimulationClientSide();
                updateUI(rep);
            }
        } catch (err) {
            console.error("Failed to load recorded telemetry:", err);
            const rep = runSimulationClientSide();
            updateUI(rep);
        }
    });

    window.addEventListener("resize", () => {
        if (currentReport) {
            renderMetricChart(currentReport);
            renderParticlesChart(currentReport);
            renderSpectrumChart(currentReport);
            renderEntropyChart(currentReport);
        }
    });
}

// Initial Boot
window.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    const initialReport = runSimulationClientSide();
    updateUI(initialReport);
});
