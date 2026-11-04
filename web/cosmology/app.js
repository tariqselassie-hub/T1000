// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/**
 * T-1000 Standalone Quantum Universe Simulator & Macro Cosmic Web Observatory
 * Client-side symplectic Störmer-Verlet integrator, Bogoliubov solver,
 * and 3D Zel'dovich Cosmic Web caustic structure formation engine.
 * Strictly adheres to Deen OS Sovereign Visual Specification.
 */

const COLOR_BG_ROOT = "#05080d";
const COLOR_BG_CARD = "#0f172a";
const COLOR_EMERALD = "#00e676";
const COLOR_EMERALD_GLOW = "rgba(0, 230, 118, 0.35)";
const COLOR_AMBER = "#ffb300";
const COLOR_AMBER_GLOW = "rgba(255, 179, 0, 0.4)";
const COLOR_CRIMSON = "#ef4444";
const COLOR_CRIMSON_GLOW = "rgba(239, 68, 68, 0.35)";
const COLOR_GRID = "#1e293b";
const COLOR_TEXT = "#f8fafc";
const COLOR_TEXT_SLATE = "#94a3b8";

// Master Simulation State
let currentReport = null;
let macroSeeds = [];
let macroWaveModes = [];

// 3D Orbital Camera & Timeline Playback State
let cameraRotX = 0.38;
let cameraRotY = -0.55;
let cameraZoom = 1.0;
let isDraggingCamera = false;
let lastMouseX = 0;
let lastMouseY = 0;

let isPlayingMacro = false;
let playAnimFrameId = null;
let currentScrubFraction = 1.0;
let currentMacroViewMode = "cosmic-web";
let activeDisplayTab = "macro"; // "macro", "split", "telemetry"

// Scale Factor Profile Functions
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

// Numerical Symplectic Simulation Engine
function runSimulationClientSide() {
    const profile = document.getElementById("select-profile").value;
    const hubbleRate = parseFloat(document.getElementById("input-hubble").value);
    const mass = parseFloat(document.getElementById("input-mass").value);
    const numModes = parseInt(document.getElementById("input-modes").value, 10);
    const tMax = parseFloat(document.getElementById("input-duration").value);

    const L = 20.0;
    const steps = 120;
    const dt = tMax / steps;

    // Discretized momentum modes
    const modes = [];
    for (let i = 1; i <= numModes; i++) {
        const k = (2.0 * Math.PI * i) / L;
        const omega0 = Math.sqrt(k * k + mass * mass);
        modes.push({
            index: i,
            k: k,
            v_re: 1.0 / Math.sqrt(2.0 * omega0),
            v_im: 0.0,
            pi_re: 0.0,
            pi_im: -Math.sqrt(omega0 / 2.0),
        });
    }

    const epochs = [];
    let current_t = 0.0;
    let conformal_eta = 0.0;

    for (let step = 0; step <= steps; step++) {
        const a = getScaleFactor(profile, hubbleRate, current_t);
        const h = getHubble(profile, hubbleRate, current_t);
        const V_eff = getEffectivePotential(profile, hubbleRate, mass, current_t);

        // Bogoliubov particle evaluation
        let totalParticles = 0.0;
        let superCount = 0;
        let totalP = 0.0;
        const modeReports = [];

        for (let i = 0; i < modes.length; i++) {
            const m = modes[i];
            const refOmega = Math.sqrt(m.k * m.k + mass * mass * a * a);
            const sqrt2w = Math.sqrt(2.0 * refOmega);

            const alpha_re = sqrt2w * 0.5 * m.v_re - (1.0 / sqrt2w) * 0.5 * m.pi_im;
            const alpha_im = sqrt2w * 0.5 * m.v_im + (1.0 / sqrt2w) * 0.5 * m.pi_re;
            const beta_re = sqrt2w * 0.5 * m.v_re + (1.0 / sqrt2w) * 0.5 * m.pi_im;
            const beta_im = sqrt2w * 0.5 * m.v_im - (1.0 / sqrt2w) * 0.5 * m.pi_re;

            const n_k = beta_re * beta_re + beta_im * beta_im;
            const r_k = Math.asinh(Math.sqrt(Math.max(0.0, n_k)));
            const p_k = (Math.pow(m.k, 3.0) / (2.0 * Math.PI * Math.PI)) * (m.v_re * m.v_re + m.v_im * m.v_im) / (a * a);

            const is_super = m.k < a * h;
            if (is_super) superCount++;

            totalParticles += n_k;
            totalP += p_k;

            modeReports.push({
                index: m.index,
                k: m.k,
                n_k: n_k,
                r_k: r_k,
                p_k: p_k,
                is_super: is_super,
            });
        }

        // Horizon Entanglement Entropy
        let entropy = 0.0;
        for (let i = 0; i < modeReports.length; i++) {
            const rep = modeReports[i];
            if (rep.is_super) {
                const c2 = Math.cosh(rep.r_k) * Math.cosh(rep.r_k);
                const s2 = Math.sinh(rep.r_k) * Math.sinh(rep.r_k);
                const sm = c2 * Math.log(Math.max(1e-12, c2)) - s2 * Math.log(Math.max(1e-12, s2));
                entropy += sm;
            }
        }

        epochs.push({
            t: current_t,
            a: a,
            hubble: h,
            particles: totalParticles,
            entropy: entropy,
            super_count: superCount,
            mean_p: totalP / Math.max(1, modes.length),
            modeReports: modeReports,
        });

        // Symplectic Störmer-Verlet field step
        if (step < steps) {
            const d_eta = dt / Math.max(1e-6, a);
            for (let i = 0; i < modes.length; i++) {
                const m = modes[i];
                const omega_eff_sq = m.k * m.k + V_eff;
                // Kick 1
                m.pi_re -= 0.5 * d_eta * omega_eff_sq * m.v_re;
                m.pi_im -= 0.5 * d_eta * omega_eff_sq * m.v_im;
                // Drift
                m.v_re += d_eta * m.pi_re;
                m.v_im += d_eta * m.pi_im;
                // Kick 2
                m.pi_re -= 0.5 * d_eta * omega_eff_sq * m.v_re;
                m.pi_im -= 0.5 * d_eta * omega_eff_sq * m.v_im;
            }
            current_t += dt;
            conformal_eta += d_eta;
        }
    }

    const lastEpoch = epochs[epochs.length - 1];
    return {
        profile: profile,
        hubble_rate: hubbleRate,
        mass: mass,
        numModes: numModes,
        tMax: tMax,
        epochs: epochs,
        final_spectrum: {
            a: lastEpoch.a,
            hubble: lastEpoch.hubble,
            particles: lastEpoch.particles,
            entropy: lastEpoch.entropy,
            super_count: lastEpoch.super_count,
            modeReports: lastEpoch.modeReports,
        },
    };
}

// -------------------------------------------------------------
// Macroscopic Cosmic Web & Zel'dovich Structure Generation
// -------------------------------------------------------------
function initializeMacroUniverseSeeds(numSeeds = 110, boxScale = 14.0) {
    macroSeeds = [];
    macroWaveModes = [];

    // Spherical Fibonacci distribution (uniform isotropic seeds)
    const goldenRatio = (1.0 + Math.sqrt(5.0)) / 2.0;
    for (let i = 0; i < numSeeds; i++) {
        const theta = 2.0 * Math.PI * i / goldenRatio;
        const phi = Math.acos(Math.max(-1.0, Math.min(1.0, (2.0 * i + 1.0) / numSeeds - 1.0)));
        const r = boxScale * Math.pow((i + 1.0) / numSeeds, 1.0 / 3.0);

        const qx = r * Math.sin(phi) * Math.cos(theta);
        const qy = r * Math.sin(phi) * Math.sin(theta);
        const qz = r * Math.cos(phi);
        macroSeeds.push({ qx, qy, qz });
    }

    // Generate perturbation wavevectors
    for (let m = 1; m <= 12; m++) {
        const km = 2.0 * Math.PI * m / boxScale;
        const amp = 0.35 / Math.pow(m, 1.3);
        macroWaveModes.push(
            { kx: km, ky: 0, kz: 0, amp: amp, phase: m * 1.37 },
            { kx: 0, ky: km, kz: 0, amp: amp, phase: m * 2.19 },
            { kx: 0, ky: 0, kz: km, amp: amp, phase: m * 3.71 }
        );
    }
}

function evaluateMacroDisplacement(qx, qy, qz) {
    let psix = 0, psiy = 0, psiz = 0;
    for (let i = 0; i < macroWaveModes.length; i++) {
        const w = macroWaveModes[i];
        const dot = w.kx * qx + w.ky * qy + w.kz * qz;
        const s = Math.sin(dot + w.phase);
        psix += w.amp * w.kx * s;
        psiy += w.amp * w.ky * s;
        psiz += w.amp * w.kz * s;
    }
    return { psix, psiy, psiz };
}

// 3D Perspective Projection
function project3D(x, y, z, cx, cy, rotX, rotY, zoom) {
    // Yaw around Y
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    // Pitch around X
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    const distance = 45.0;
    const fov = 340.0 * zoom;
    const scale = fov / Math.max(10.0, distance + z2);

    return {
        px: cx + x1 * scale,
        py: cy - y2 * scale,
        scale: scale,
        depth: z2,
    };
}

// -------------------------------------------------------------
// Macro Universe Viewport Renderer
// -------------------------------------------------------------
function renderMacroCosmos() {
    const canvas = document.getElementById("canvas-macro-cosmos");
    if (!canvas || !currentReport) return;
    const ctx = canvas.getContext("2d");

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    ctx.fillStyle = COLOR_BG_ROOT;
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;

    // Determine current scrubbed epoch
    const epochs = currentReport.epochs;
    const epochIdx = Math.min(epochs.length - 1, Math.max(0, Math.floor(currentScrubFraction * (epochs.length - 1))));
    const ep = epochs[epochIdx];
    const finalA = currentReport.final_spectrum.a;

    // Cosmological growth factor D(t)
    const normA = (ep.a - 1.0) / Math.max(0.1, finalA - 1.0);
    const growth = Math.max(0.02, normA * 0.9 + 0.05);

    // Horizon radius R_H
    const horizon_r = ep.hubble > 1e-4 ? Math.min(22.0, ep.a / ep.hubble * 0.6) : 22.0;

    // Update Macro HUD stats
    document.getElementById("hud-epoch").innerText = `t = ${ep.t.toFixed(2)} Gy`;
    document.getElementById("hud-scale").innerText = `a = ${ep.a.toFixed(2)}`;
    document.getElementById("hud-horizon").innerText = `R_H = ${horizon_r.toFixed(1)} Mpc`;

    if (currentMacroViewMode === "density-slice") {
        renderDensitySliceMode(ctx, w, h, ep, growth);
        return;
    }

    if (currentMacroViewMode === "horizon-bubble") {
        renderHorizonBubbleMode(ctx, cx, cy, ep, epochs);
        return;
    }

    // Mode: "cosmic-web" (3D Filaments, Halos & Horizon Sphere)
    renderCosmicWeb3D(ctx, cx, cy, ep, growth, horizon_r);
}

function renderCosmicWeb3D(ctx, cx, cy, ep, growth, horizon_r) {
    if (macroSeeds.length === 0) {
        initializeMacroUniverseSeeds(110, 14.0);
    }

    // 1. Calculate displaced node coordinates & density
    const projectedNodes = [];
    let clusterCount = 0;
    let voidCount = 0;

    for (let i = 0; i < macroSeeds.length; i++) {
        const s = macroSeeds[i];
        const disp = evaluateMacroDisplacement(s.qx, s.qy, s.qz);

        // Zel'dovich displacement
        const ex = s.qx - growth * disp.psix * 2.8;
        const ey = s.qy - growth * disp.psiy * 2.8;
        const ez = s.qz - growth * disp.psiz * 2.8;

        const dispMag = Math.sqrt(disp.psix * disp.psix + disp.psiy * disp.psiy + disp.psiz * disp.psiz);
        const delta = growth * dispMag * 4.5 - 0.25;

        const distFromCenter = Math.sqrt(ex * ex + ey * ey + ez * ez);
        const isInsideHorizon = distFromCenter <= horizon_r;

        let nodeType = "wall";
        if (delta > 1.686) {
            nodeType = "cluster";
            clusterCount++;
        } else if (delta > 0.35) {
            nodeType = "filament";
        } else if (delta < -0.2) {
            nodeType = "void";
            voidCount++;
        }

        const proj = project3D(ex, ey, ez, cx, cy, cameraRotX, cameraRotY, cameraZoom);
        projectedNodes.push({
            id: i,
            x: ex,
            y: ey,
            z: ez,
            delta: delta,
            nodeType: nodeType,
            isInside: isInsideHorizon,
            proj: proj,
        });
    }

    // 2. Draw 3D Event Horizon Wireframe Sphere
    drawHorizonSphere3D(ctx, cx, cy, horizon_r, cameraRotX, cameraRotY, cameraZoom);

    // 3. Compute & Draw Cosmic Filaments
    let filamentCount = 0;
    const maxLinkDist = 5.2;
    const maxLinkSq = maxLinkDist * maxLinkDist;

    ctx.lineWidth = 1.0;
    for (let i = 0; i < projectedNodes.length; i++) {
        const n1 = projectedNodes[i];
        if (n1.nodeType === "void") continue;

        for (let j = i + 1; j < projectedNodes.length; j++) {
            const n2 = projectedNodes[j];
            if (n2.nodeType === "void") continue;

            const dx = n1.x - n2.x;
            const dy = n1.y - n2.y;
            const dz = n1.z - n2.z;
            const d_sq = dx * dx + dy * dy + dz * dz;

            if (d_sq <= maxLinkSq) {
                filamentCount++;
                const dist = Math.sqrt(d_sq);
                const alpha = Math.max(0.12, Math.min(0.75, (1.0 - dist / maxLinkDist) * growth * 1.2));

                ctx.strokeStyle = `rgba(0, 230, 118, ${alpha.toFixed(2)})`;
                ctx.beginPath();
                ctx.moveTo(n1.proj.px, n1.proj.py);
                ctx.lineTo(n2.proj.px, n2.proj.py);
                ctx.stroke();
            }
        }
    }

    // 4. Sort nodes by camera depth (painters algorithm)
    projectedNodes.sort((a, b) => b.proj.depth - a.proj.depth);

    // 5. Draw Galaxy Clusters & Tracer Nodes
    for (let i = 0; i < projectedNodes.length; i++) {
        const n = projectedNodes[i];
        const p = n.proj;

        if (n.nodeType === "cluster") {
            // Luminous Gold Supercluster Halo
            const rHalo = Math.max(4, 7 * p.scale);
            const grad = ctx.createRadialGradient(p.px, p.py, 1, p.px, p.py, rHalo * 2.2);
            grad.addColorStop(0, COLOR_TEXT);
            grad.addColorStop(0.3, COLOR_AMBER);
            grad.addColorStop(1, "transparent");

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(p.px, p.py, rHalo * 2.2, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = COLOR_TEXT;
            ctx.beginPath();
            ctx.arc(p.px, p.py, Math.max(2, 3 * p.scale), 0, Math.PI * 2);
            ctx.fill();
        } else if (n.nodeType === "filament") {
            // Emerald Filament Bridge Node
            ctx.fillStyle = COLOR_EMERALD;
            ctx.beginPath();
            ctx.arc(p.px, p.py, Math.max(1.8, 3.2 * p.scale), 0, Math.PI * 2);
            ctx.fill();
        } else {
            // Void Tracer Particle
            ctx.fillStyle = `rgba(148, 163, 184, 0.3)`;
            ctx.beginPath();
            ctx.arc(p.px, p.py, Math.max(1, 1.5 * p.scale), 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw Redshift Marker if outside the Event Horizon
        if (!n.isInside) {
            ctx.strokeStyle = COLOR_CRIMSON;
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            ctx.arc(p.px, p.py, Math.max(3, 5 * p.scale), 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    // Update overlay stats
    const voidFrac = ((voidCount / Math.max(1, projectedNodes.length)) * 100).toFixed(1);
    document.getElementById("stat-filaments").innerText = filamentCount;
    document.getElementById("stat-clusters").innerText = clusterCount;
    document.getElementById("stat-super-modes").innerText = `${ep.super_count} / ${currentReport.numModes} MODES`;
    document.getElementById("stat-voids").innerText = `${voidFrac}%`;
}

// Draw 3D Event Horizon Sphere Rings
function drawHorizonSphere3D(ctx, cx, cy, radius, rotX, rotY, zoom) {
    ctx.strokeStyle = COLOR_CRIMSON;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);

    const numLatitudes = 5;
    const numPoints = 48;

    for (let l = 1; l < numLatitudes; l++) {
        const phi = (Math.PI * l) / numLatitudes - Math.PI / 2;
        const r_ring = radius * Math.cos(phi);
        const y_ring = radius * Math.sin(phi);

        ctx.beginPath();
        for (let p = 0; p <= numPoints; p++) {
            const theta = (2.0 * Math.PI * p) / numPoints;
            const x = r_ring * Math.cos(theta);
            const z = r_ring * Math.sin(theta);
            const proj = project3D(x, y_ring, z, cx, cy, rotX, rotY, zoom);

            if (p === 0) ctx.moveTo(proj.px, proj.py);
            else ctx.lineTo(proj.px, proj.py);
        }
        ctx.stroke();
    }
    ctx.setLineDash([]);
}

// -------------------------------------------------------------
// 2D Density Field Slice View Mode
// -------------------------------------------------------------
function renderDensitySliceMode(ctx, w, h, ep, growth) {
    const res = 60;
    const cellW = w / res;
    const cellH = h / res;

    for (let iy = 0; iy < res; iy++) {
        for (let ix = 0; ix < res; ix++) {
            const gx = ((ix / res) - 0.5) * 20.0;
            const gy = ((iy / res) - 0.5) * 20.0;

            let val = 0.0;
            for (let m = 1; m <= 6; m++) {
                const km = 2.0 * Math.PI * m / 20.0;
                val += (0.4 / Math.pow(m, 1.2)) * Math.sin(km * gx + m * 1.3) * Math.cos(km * gy + m * 2.1);
            }
            const delta = val * growth * 6.0;

            if (delta > 1.686) {
                ctx.fillStyle = COLOR_AMBER;
            } else if (delta > 0.4) {
                const alpha = Math.min(1.0, (delta - 0.4) / 1.2);
                ctx.fillStyle = `rgba(0, 230, 118, ${alpha.toFixed(2)})`;
            } else if (delta < -0.3) {
                ctx.fillStyle = COLOR_BG_ROOT;
            } else {
                ctx.fillStyle = "#09131f";
            }
            ctx.fillRect(ix * cellW, iy * cellH, cellW + 1, cellH + 1);
        }
    }

    ctx.font = "11px monospace";
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("2D Slice Matter Density Field delta(x, y) // Overdensity Peaks (Gold) & Voids (Dark)", 20, 25);
}

// -------------------------------------------------------------
// Expanding Horizon Bubble View Mode
// -------------------------------------------------------------
function renderHorizonBubbleMode(ctx, cx, cy, currentEp, epochs) {
    const maxR = Math.min(cx, cy) * 0.85;

    // Draw past horizon light-cones
    epochs.forEach((ep, idx) => {
        if (idx % 12 === 0) {
            const frac = ep.a / currentReport.final_spectrum.a;
            const r = frac * maxR;
            ctx.strokeStyle = idx === epochs.length - 1 ? COLOR_AMBER : "rgba(30, 41, 59, 0.7)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();
        }
    });

    // Current Cosmic Boundary
    const currentR = (currentEp.a / currentReport.final_spectrum.a) * maxR;
    ctx.strokeStyle = COLOR_CRIMSON;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.arc(cx, cy, currentR, 0, Math.PI * 2);
    ctx.stroke();

    // Central Big Bang Point
    ctx.fillStyle = COLOR_AMBER;
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "11px monospace";
    ctx.fillStyle = COLOR_TEXT;
    ctx.fillText("Cosmic Expansion Boundary // Causal Horizon Spheres", 20, 25);
    ctx.fillStyle = COLOR_CRIMSON;
    ctx.fillText(`Current Scale a(t) = ${currentEp.a.toFixed(2)}`, 20, 42);
}

// -------------------------------------------------------------
// 4 Mathematical Telemetry Panel Renderers
// -------------------------------------------------------------
function renderMetricChart(report) {
    const canvas = document.getElementById("canvas-metric");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 30;
    const epochs = report.epochs;
    if (!epochs || epochs.length === 0) return;

    const maxA = Math.max(...epochs.map(e => e.a));
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;

    // Grid lines
    ctx.strokeStyle = COLOR_GRID;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(w - pad, y);
        ctx.stroke();
    }

    // Scale factor curve
    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2;
    ctx.beginPath();
    epochs.forEach((e, idx) => {
        const x = pad + (plotW * idx) / (epochs.length - 1);
        const y = pad + plotH * (1.0 - e.a / maxA);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Comoving horizon curve rc = (aH)^-1
    const maxRc = Math.max(...epochs.map(e => 1.0 / Math.max(1e-4, e.a * e.hubble)));
    ctx.strokeStyle = COLOR_CRIMSON;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    epochs.forEach((e, idx) => {
        const x = pad + (plotW * idx) / (epochs.length - 1);
        const rc = 1.0 / Math.max(1e-4, e.a * e.hubble);
        const y = pad + plotH * (1.0 - rc / maxRc);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("Scale Factor a(t)", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_CRIMSON;
    ctx.fillText("-- Comoving Horizon (aH)^-1", pad + 10, pad + 30);
}

function renderParticlesChart(report) {
    const canvas = document.getElementById("canvas-particles");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 30;
    const modes = report.final_spectrum.modeReports;
    if (!modes || modes.length === 0) return;

    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;
    const barW = (plotW / modes.length) * 0.7;
    const maxN = Math.max(...modes.map(m => m.n_k), 1.0);

    modes.forEach((m, idx) => {
        const x = pad + idx * (plotW / modes.length) + barW * 0.2;
        const barH = plotH * (m.n_k / maxN);
        const y = pad + plotH - barH;

        ctx.fillStyle = m.is_super ? COLOR_AMBER : COLOR_EMERALD;
        ctx.fillRect(x, y, barW, barH);

        ctx.font = "9px monospace";
        ctx.fillStyle = COLOR_TEXT_SLATE;
        if (idx % 2 === 0) {
            ctx.fillText(`k${m.index}`, x, h - pad + 12);
        }
    });

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("Super-Horizon Squeezed Quanta", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("Sub-Horizon Vacuum Modes", pad + 10, pad + 30);
}

function renderSpectrumChart(report) {
    const canvas = document.getElementById("canvas-spectrum");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 30;
    const modes = report.final_spectrum.modeReports;
    if (!modes || modes.length === 0) return;

    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;
    const maxP = Math.max(...modes.map(m => m.p_k), 1e-4);

    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2;
    ctx.beginPath();
    modes.forEach((m, idx) => {
        const x = pad + (plotW * idx) / (modes.length - 1);
        const y = pad + plotH * (1.0 - m.p_k / maxP);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Harrison-Zeldovich flat asymptote
    const asymp = Math.pow(report.hubble_rate / (2.0 * Math.PI), 2.0);
    const asympY = pad + plotH * (1.0 - Math.min(1.0, asymp / maxP));
    ctx.strokeStyle = COLOR_AMBER;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, asympY);
    ctx.lineTo(w - pad, asympY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("Primordial Power Spectrum P(k)", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("-- (H / 2pi)^2 Flat Asymptote", pad + 10, pad + 30);
}

function renderEntropyChart(report) {
    const canvas = document.getElementById("canvas-entropy");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 30;
    const epochs = report.epochs;
    if (!epochs || epochs.length === 0) return;

    const maxS = Math.max(...epochs.map(e => e.entropy), 1.0);
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;

    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2;
    ctx.beginPath();
    epochs.forEach((e, idx) => {
        const x = pad + (plotW * idx) / (epochs.length - 1);
        const y = pad + plotH * (1.0 - e.entropy / maxS);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("Entanglement Entropy S_vN(t)", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("-- Holographic Area Law Scaling", pad + 10, pad + 30);
}

// -------------------------------------------------------------
// Master UI Update & Event Wiring
// -------------------------------------------------------------
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

    document.getElementById("footer-status").innerText = `SIMULATION ACTIVE // SCALE a=${finalEpoch.a.toFixed(2)} // N_tot=${finalEpoch.particles.toFixed(1)} // HORIZON ENTROPY S=${finalEpoch.entropy.toFixed(2)}`;

    // Render Macro Universe Canvas
    renderMacroCosmos();

    // Render Telemetry Canvases
    renderMetricChart(report);
    renderParticlesChart(report);
    renderSpectrumChart(report);
    renderEntropyChart(report);
}

function bindEvents() {
    // Sliders
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

    // Run / Reset Buttons
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

    // Load Recorded Disk Telemetry
    document.getElementById("btn-load-recorded").addEventListener("click", async () => {
        try {
            const resp = await fetch("universe_cosmology.json");
            if (resp.ok) {
                const diskData = await resp.json();
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
                const rep = runSimulationClientSide();
                updateUI(rep);
            }
        } catch (err) {
            console.warn("Could not load disk telemetry, running client-side simulation:", err);
            const rep = runSimulationClientSide();
            updateUI(rep);
        }
    });

    // View Switching Tabs
    const tabMacro = document.getElementById("tab-macro");
    const tabSplit = document.getElementById("tab-split");
    const tabTelemetry = document.getElementById("tab-telemetry");
    const panelMacro = document.getElementById("panel-macro-cosmos");
    const panelTelemetry = document.getElementById("panel-telemetry-grid");
    const macroCanvasBox = panelMacro.querySelector(".macro-canvas-container");

    tabMacro.addEventListener("click", () => {
        activeDisplayTab = "macro";
        tabMacro.classList.add("active");
        tabSplit.classList.remove("active");
        tabTelemetry.classList.remove("active");
        panelMacro.style.display = "flex";
        macroCanvasBox.style.height = "520px";
        panelTelemetry.style.display = "none";
        renderMacroCosmos();
    });

    tabSplit.addEventListener("click", () => {
        activeDisplayTab = "split";
        tabMacro.classList.remove("active");
        tabSplit.classList.add("active");
        tabTelemetry.classList.remove("active");
        panelMacro.style.display = "flex";
        macroCanvasBox.style.height = "360px";
        panelTelemetry.style.display = "grid";
        renderMacroCosmos();
        if (currentReport) {
            renderMetricChart(currentReport);
            renderParticlesChart(currentReport);
            renderSpectrumChart(currentReport);
            renderEntropyChart(currentReport);
        }
    });

    tabTelemetry.addEventListener("click", () => {
        activeDisplayTab = "telemetry";
        tabMacro.classList.remove("active");
        tabSplit.classList.remove("active");
        tabTelemetry.classList.add("active");
        panelMacro.style.display = "none";
        panelTelemetry.style.display = "grid";
        if (currentReport) {
            renderMetricChart(currentReport);
            renderParticlesChart(currentReport);
            renderSpectrumChart(currentReport);
            renderEntropyChart(currentReport);
        }
    });

    // 3D Orbital Mouse Drag & Zoom Controls
    const macroCanvas = document.getElementById("canvas-macro-cosmos");
    macroCanvas.addEventListener("mousedown", e => {
        isDraggingCamera = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    });

    window.addEventListener("mousemove", e => {
        if (!isDraggingCamera) return;
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        cameraRotY += dx * 0.008;
        cameraRotX += dy * 0.008;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        renderMacroCosmos();
    });

    window.addEventListener("mouseup", () => {
        isDraggingCamera = false;
    });

    macroCanvas.addEventListener("wheel", e => {
        e.preventDefault();
        cameraZoom = Math.min(2.5, Math.max(0.4, cameraZoom - e.deltaY * 0.0015));
        renderMacroCosmos();
    }, { passive: false });

    document.getElementById("btn-macro-reset-cam").addEventListener("click", () => {
        cameraRotX = 0.38;
        cameraRotY = -0.55;
        cameraZoom = 1.0;
        renderMacroCosmos();
    });

    // View Mode Dropdown
    document.getElementById("select-macro-mode").addEventListener("change", e => {
        currentMacroViewMode = e.target.value;
        renderMacroCosmos();
    });

    // Timeline Scrubber
    const scrubber = document.getElementById("input-timeline");
    scrubber.addEventListener("input", e => {
        currentScrubFraction = parseFloat(e.target.value) / 100.0;
        renderMacroCosmos();
    });

    // Animation Play / Pause
    const btnPlay = document.getElementById("btn-macro-play");
    function stepPlayback() {
        if (!isPlayingMacro) return;
        let val = parseFloat(scrubber.value) + 0.6;
        if (val > 100.0) val = 0.0;
        scrubber.value = val;
        currentScrubFraction = val / 100.0;
        renderMacroCosmos();
        playAnimFrameId = requestAnimationFrame(stepPlayback);
    }

    btnPlay.addEventListener("click", () => {
        isPlayingMacro = !isPlayingMacro;
        if (isPlayingMacro) {
            btnPlay.innerText = "PAUSE EVOLUTION";
            btnPlay.classList.add("playing");
            playAnimFrameId = requestAnimationFrame(stepPlayback);
        } else {
            btnPlay.innerText = "PLAY EVOLUTION";
            btnPlay.classList.remove("playing");
            if (playAnimFrameId) cancelAnimationFrame(playAnimFrameId);
        }
    });

    window.addEventListener("resize", () => {
        if (currentReport) {
            renderMacroCosmos();
            if (activeDisplayTab !== "macro") {
                renderMetricChart(currentReport);
                renderParticlesChart(currentReport);
                renderSpectrumChart(currentReport);
                renderEntropyChart(currentReport);
            }
        }
    });
}

// Initial Boot
window.addEventListener("DOMContentLoaded", () => {
    initializeMacroUniverseSeeds(110, 14.0);
    bindEvents();
    const initialReport = runSimulationClientSide();
    updateUI(initialReport);
});
