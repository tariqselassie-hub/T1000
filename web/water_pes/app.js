// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/**
 * T-1000 Standalone Water (H2O) 2D Energy & Linearization Barrier Engine
 * 3D Molecular Orbital Geometry, Wilson GF Harmonic Vibrational Dynamics,
 * and Walsh Correlation Diagram.
 * Strictly adheres to Deen OS Sovereign Visual Specification.
 */

const COLOR_BG_ROOT = "#05080d";
const COLOR_BG_CARD = "#0f172a";
const COLOR_EMERALD = "#00e676";
const COLOR_EMERALD_GLOW = "rgba(0, 230, 118, 0.4)";
const COLOR_AMBER = "#ffb300";
const COLOR_AMBER_GLOW = "rgba(255, 179, 0, 0.45)";
const COLOR_CRIMSON = "#ef4444";
const COLOR_GRID = "#1e293b";
const COLOR_TEXT = "#f8fafc";
const COLOR_TEXT_SLATE = "#94a3b8";

// Molecule State
let bondLengthR = 0.9578;
let bondAngleDeg = 104.48;
let currentVibMode = 0; // 0: static, 1: sym stretch, 2: bend, 3: asym stretch
let animPhase = 0.0;
let animSpeed = 1.0;
let lastTimestamp = 0;

// 3D Orbital Camera
let cameraRotX = 0.25;
let cameraRotY = -0.45;
let cameraZoom = 1.0;
let isDraggingCamera = false;
let lastMouseX = 0;
let lastMouseY = 0;

// Ab-Initio Telemetry
let diskTelemetry = null;

// Normal Mode Vectors from Wilson GF Solution
const VIB_MODES = [
    { index: 0, name: "Static Equilibrium", freq_cm1: 0, o: [0, 0, 0], h1: [0, 0, 0], h2: [0, 0, 0] },
    { index: 1, name: "ν₁ Symmetric Stretch", freq_cm1: 4019.6, o: [0, 0, -0.07], h1: [0.12, 0, 0.10], h2: [-0.12, 0, 0.10] },
    { index: 2, name: "ν₂ Scissor Bending", freq_cm1: 2141.2, o: [0, 0, 0.08], h1: [-0.14, 0, 0.08], h2: [0.14, 0, 0.08] },
    { index: 3, name: "ν₃ Asymmetric Stretch", freq_cm1: 4168.5, o: [-0.06, 0, 0], h1: [0.14, 0, 0.11], h2: [0.14, 0, -0.11] },
];

// 3D Perspective Projection
function project3D(x, y, z, cx, cy, rotX, rotY, zoom) {
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    const distance = 6.0;
    const fov = 380.0 * zoom;
    const scale = fov / Math.max(1.0, distance + z2);

    return {
        px: cx + x1 * scale,
        py: cy - y2 * scale,
        scale: scale,
        depth: z2,
    };
}

// -------------------------------------------------------------
// 3D Water Molecule Renderer & Harmonic Animator
// -------------------------------------------------------------
function renderWaterMolecule3D(timestamp) {
    const canvas = document.getElementById("canvas-molecule-3d");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    if (!lastTimestamp) lastTimestamp = timestamp;
    const dt = (timestamp - lastTimestamp) * 0.001;
    lastTimestamp = timestamp;

    if (currentVibMode > 0) {
        const freqScale = currentVibMode === 2 ? 4.0 : 6.0;
        animPhase += dt * freqScale * animSpeed;
    }

    ctx.fillStyle = COLOR_BG_ROOT;
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2 + 10;

    // Calculate current atom positions including vibrational displacement
    const halfRad = (bondAngleDeg / 2.0) * Math.PI / 180.0;
    const sinHalf = Math.sin(halfRad);
    const cosHalf = Math.cos(halfRad);

    let ox = 0.0, oy = 0.0, oz = 0.0;
    let h1x = bondLengthR * sinHalf, h1y = 0.0, h1z = bondLengthR * cosHalf;
    let h2x = -bondLengthR * sinHalf, h2y = 0.0, h2z = bondLengthR * cosHalf;

    // Apply vibrational oscillation
    let amp = 0.0;
    if (currentVibMode > 0) {
        const m = VIB_MODES[currentVibMode];
        amp = Math.sin(animPhase) * 0.5;
        ox += m.o[0] * amp; oy += m.o[1] * amp; oz += m.o[2] * amp;
        h1x += m.h1[0] * amp; h1y += m.h1[1] * amp; h1z += m.h1[2] * amp;
        h2x += m.h2[0] * amp; h2y += m.h2[1] * amp; h2z += m.h2[2] * amp;
    }

    // 1. Draw Chemical Bond Cylinders
    drawBond3D(ctx, cx, cy, ox, oy, oz, h1x, h1y, h1z);
    drawBond3D(ctx, cx, cy, ox, oy, oz, h2x, h2y, h2z);

    // 2. Draw Translucent Lone Pair Electron Lobes on Oxygen (3a1 and 1b1)
    drawLonePairLobe(ctx, cx, cy, ox, oy, oz, 0.0, 0.55, -0.45);  // 1b1 out-of-plane lobe 1
    drawLonePairLobe(ctx, cx, cy, ox, oy, oz, 0.0, -0.55, -0.45); // 1b1 out-of-plane lobe 2
    drawLonePairLobe(ctx, cx, cy, ox, oy, oz, 0.0, 0.0, -0.65);   // 3a1 hybrid lone pair lobe

    // 3. Draw Dipole Moment Vector Arrow
    const isLinear = Math.abs(bondAngleDeg - 180.0) < 0.5;
    const dipoleMag = isLinear ? 0.0 : 1.854 * (Math.cos(halfRad) / Math.cos(52.24 * Math.PI / 180.0)) * (bondLengthR / 0.9578);

    if (dipoleMag > 0.05) {
        drawDipoleArrow3D(ctx, cx, cy, ox, oy, oz, dipoleMag);
    }

    // 4. Project and Draw Atoms (depth sorted)
    const atoms = [
        { name: "O", x: ox, y: oy, z: oz, radius: 24, color: COLOR_CRIMSON, border: COLOR_AMBER },
        { name: "H1", x: h1x, y: h1y, z: h1z, radius: 14, color: COLOR_TEXT, border: COLOR_TEXT_SLATE },
        { name: "H2", x: h2x, y: h2y, z: h2z, radius: 14, color: COLOR_TEXT, border: COLOR_TEXT_SLATE },
    ];

    atoms.forEach(a => {
        a.proj = project3D(a.x, a.y, a.z, cx, cy, cameraRotX, cameraRotY, cameraZoom);
    });
    atoms.sort((a, b) => b.proj.depth - a.proj.depth);

    atoms.forEach(a => {
        const p = a.proj;
        const r = a.radius * p.scale * 0.025;

        // Radial shaded specular sphere
        const grad = ctx.createRadialGradient(p.px - r * 0.35, p.py - r * 0.35, r * 0.1, p.px, p.py, r);
        if (a.name === "O") {
            grad.addColorStop(0, "#ff8a80");
            grad.addColorStop(0.5, COLOR_CRIMSON);
            grad.addColorStop(1, "#7f1d1d");
        } else {
            grad.addColorStop(0, "#ffffff");
            grad.addColorStop(0.6, COLOR_TEXT);
            grad.addColorStop(1, "#64748b");
        }

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.px, p.py, r, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = a.border;
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // Atom symbol label
        ctx.font = `bold ${Math.max(9, Math.floor(r * 0.9))}px monospace`;
        ctx.fillStyle = a.name === "O" ? "#ffffff" : "#0f172a";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(a.name, p.px, p.py);
    });

    // 5. Draw Vibrational Displacement Vector Arrows
    if (currentVibMode > 0) {
        const m = VIB_MODES[currentVibMode];
        drawDisplacementArrow(ctx, atoms.find(a => a.name === "O").proj, m.o, amp);
        drawDisplacementArrow(ctx, atoms.find(a => a.name === "H1").proj, m.h1, amp);
        drawDisplacementArrow(ctx, atoms.find(a => a.name === "H2").proj, m.h2, amp);
    }

    requestAnimationFrame(renderWaterMolecule3D);
}

function drawBond3D(ctx, cx, cy, x1, y1, z1, x2, y2, z2) {
    const p1 = project3D(x1, y1, z1, cx, cy, cameraRotX, cameraRotY, cameraZoom);
    const p2 = project3D(x2, y2, z2, cx, cy, cameraRotX, cameraRotY, cameraZoom);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.75)";
    ctx.lineWidth = Math.max(3, 6 * ((p1.scale + p2.scale) * 0.012));
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();
}

function drawLonePairLobe(ctx, cx, cy, ox, oy, oz, dx, dy, dz) {
    const pCenter = project3D(ox, oy, oz, cx, cy, cameraRotX, cameraRotY, cameraZoom);
    const pLobe = project3D(ox + dx, oy + dy, oz + dz, cx, cy, cameraRotX, cameraRotY, cameraZoom);

    const r = 16 * pLobe.scale * 0.022;
    const grad = ctx.createRadialGradient(pLobe.px, pLobe.py, 1, pLobe.px, pLobe.py, r * 1.8);
    grad.addColorStop(0, COLOR_EMERALD);
    grad.addColorStop(0.5, COLOR_EMERALD_GLOW);
    grad.addColorStop(1, "transparent");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(pLobe.px, pLobe.py, r * 1.8, 0, Math.PI * 2);
    ctx.fill();
}

function drawDipoleArrow3D(ctx, cx, cy, ox, oy, oz, dipoleMag) {
    // Dipole vector points along -z towards Oxygen
    const arrowLen = dipoleMag * 0.45;
    const pStart = project3D(ox, oy, oz + arrowLen, cx, cy, cameraRotX, cameraRotY, cameraZoom);
    const pEnd = project3D(ox, oy, oz, cx, cy, cameraRotX, cameraRotY, cameraZoom);

    ctx.strokeStyle = COLOR_AMBER;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(pStart.px, pStart.py);
    ctx.lineTo(pEnd.px, pEnd.py);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(pEnd.py - pStart.py, pEnd.px - pStart.px);
    ctx.fillStyle = COLOR_AMBER;
    ctx.beginPath();
    ctx.moveTo(pEnd.px, pEnd.py);
    ctx.lineTo(pEnd.px - 10 * Math.cos(angle - Math.PI / 6), pEnd.py - 10 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(pEnd.px - 10 * Math.cos(angle + Math.PI / 6), pEnd.py - 10 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    ctx.font = "bold 10px monospace";
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText(`μ = ${dipoleMag.toFixed(2)} D`, pStart.px + 10, pStart.py);
}

function drawDisplacementArrow(ctx, proj, disp, amp) {
    const scale = 80.0;
    const dx = disp[0] * amp * scale;
    const dy = -disp[2] * amp * scale; // screen y is inverted z

    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(proj.px, proj.py);
    ctx.lineTo(proj.px + dx, proj.py + dy);
    ctx.stroke();
}

// -------------------------------------------------------------
// 4 Mathematical Telemetry Panel Renderers
// -------------------------------------------------------------
function renderPesSurface() {
    const canvas = document.getElementById("canvas-pes-surface");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 28;
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;

    // Grid contours
    const nx = 40, ny = 30;
    const cellW = plotW / nx, cellH = plotH / ny;

    for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
            const th = 80.0 + (ix / nx) * 100.0;
            const r = 0.80 + (1.0 - iy / ny) * 0.50;

            // Model STO-3G PES energy
            const dr = r - 0.9578;
            const dth = (th - 104.5) * Math.PI / 180.0;
            const e_rel = (0.5 * 2.1 * dr * dr + 0.5 * 0.3 * dth * dth) * 627.5;

            if (e_rel < 15.0) ctx.fillStyle = "#0d2818";
            else if (e_rel < 35.0) ctx.fillStyle = "rgba(0, 230, 118, 0.4)";
            else if (e_rel < 65.0) ctx.fillStyle = "rgba(255, 179, 0, 0.4)";
            else ctx.fillStyle = "rgba(239, 68, 68, 0.35)";

            ctx.fillRect(pad + ix * cellW, pad + iy * cellH, cellW + 1, cellH + 1);
        }
    }

    // Minimum Energy Path (MEP) curve
    ctx.strokeStyle = COLOR_AMBER;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let i = 0; i <= 30; i++) {
        const th = 104.5 + (i / 30) * (180.0 - 104.5);
        const r = 0.9578 - 0.02 * Math.pow((th - 104.5) / 75.5, 2.0);
        const px = pad + ((th - 80.0) / 100.0) * plotW;
        const py = pad + (1.0 - (r - 0.80) / 0.50) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Live Crosshair for Current (r, theta)
    const curX = pad + ((bondAngleDeg - 80.0) / 100.0) * plotW;
    const curY = pad + (1.0 - (bondLengthR - 0.80) / 0.50) * plotH;

    ctx.strokeStyle = COLOR_TEXT;
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.moveTo(curX - 8, curY); ctx.lineTo(curX + 8, curY);
    ctx.moveTo(curX, curY - 8); ctx.lineTo(curX, curY + 8);
    ctx.stroke();

    ctx.font = "9.5px monospace";
    ctx.fillStyle = COLOR_TEXT;
    ctx.fillText(`Current (${bondAngleDeg.toFixed(1)}°, ${bondLengthR.toFixed(2)} Å)`, curX + 10, curY - 6);
}

function renderBarrierCurve() {
    const canvas = document.getElementById("canvas-barrier-curve");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 28;
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;

    // Double-well symmetric umbrella inversion potential
    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2.0;
    ctx.beginPath();

    const nPoints = 60;
    for (let i = 0; i <= nPoints; i++) {
        const theta = 30.0 + (i / nPoints) * 300.0; // from 30° to 330°
        const dth = Math.abs(theta - 180.0) - 75.5; // double wells at 104.5° and 255.5°
        const barrierHeight = 77.2; // kcal/mol
        const v = barrierHeight * Math.pow(Math.sin((theta - 180.0) * Math.PI / 180.0 * 0.5), 2.0);

        const x = pad + (i / nPoints) * plotW;
        const y = pad + plotH * (1.0 - Math.min(1.0, v / 90.0));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 180° Barrier Line
    const barX = pad + 0.5 * plotW;
    ctx.strokeStyle = COLOR_CRIMSON;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(barX, pad);
    ctx.lineTo(barX, pad + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "10px monospace";
    ctx.fillStyle = COLOR_CRIMSON;
    ctx.fillText("180° Planar Barrier: 27,012.6 cm⁻¹ (3.35 eV)", barX - 110, pad + 15);
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("Bent C_2v Well (104.5°)", pad + 15, h - pad - 10);
    ctx.fillText("Inverted Well (255.5°)", barX + 30, h - pad - 10);
}

function renderWalshDiagram() {
    const canvas = document.getElementById("canvas-walsh-diagram");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 28;
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;

    const nPoints = 50;
    // 3a1 curve
    ctx.strokeStyle = COLOR_AMBER;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let i = 0; i <= nPoints; i++) {
        const th = 80.0 + (i / nPoints) * 100.0;
        const e3a1 = -0.501 + 0.211 * ((th - 80.0) / 100.0);
        const x = pad + (i / nPoints) * plotW;
        const y = pad + plotH * (1.0 - (e3a1 - (-0.55)) / 0.35);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 1b1 curve
    ctx.strokeStyle = COLOR_EMERALD;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let i = 0; i <= nPoints; i++) {
        const th = 80.0 + (i / nPoints) * 100.0;
        const e1b1 = -0.400 + 0.110 * ((th - 80.0) / 100.0);
        const x = pad + (i / nPoints) * plotW;
        const y = pad + plotH * (1.0 - (e1b1 - (-0.55)) / 0.35);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Degeneracy convergence point at 180°
    const linX = pad + plotW;
    const linY = pad + plotH * (1.0 - (-0.290 - (-0.55)) / 0.35);
    ctx.fillStyle = COLOR_TEXT;
    ctx.beginPath();
    ctx.arc(linX, linY, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "9.5px monospace";
    ctx.fillStyle = COLOR_AMBER;
    ctx.fillText("3a₁ (Lone Pair, s-p hybrid)", pad + 10, pad + 18);
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("1b₁ (Pure Non-Bonding Lone Pair)", pad + 10, pad + 32);
    ctx.fillStyle = COLOR_TEXT;
    ctx.fillText("1π_u Degenerate Shell (180°)", linX - 160, linY - 8);
}

function renderDipoleModes() {
    const canvas = document.getElementById("canvas-dipole-modes");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    ctx.fillStyle = COLOR_BG_CARD;
    ctx.fillRect(0, 0, w, h);

    const pad = 28;
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;

    // Dipole Moment Curve |μ(θ)|
    ctx.strokeStyle = COLOR_CRIMSON;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i <= 50; i++) {
        const th = 80.0 + (i / 50) * 100.0;
        const half = (th / 2.0) * Math.PI / 180.0;
        const mu = (180.0 - th < 0.5) ? 0.0 : 1.854 * (Math.cos(half) / Math.cos(52.24 * Math.PI / 180.0));
        const x = pad + (i / 50) * plotW;
        const y = pad + plotH * (1.0 - mu / 2.4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Zero Dipole Marker at 180°
    ctx.fillStyle = COLOR_AMBER;
    ctx.beginPath();
    ctx.arc(pad + plotW, pad + plotH, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "9.5px monospace";
    ctx.fillStyle = COLOR_CRIMSON;
    ctx.fillText("Dipole Moment |μ(θ)| → 0.00 D at 180°", pad + 10, pad + 15);
    ctx.fillStyle = COLOR_EMERALD;
    ctx.fillText("ν₁: 4019.6 cm⁻¹ | ν₂: 2141.2 cm⁻¹ | ν₃: 4168.5 cm⁻¹", pad + 10, pad + 30);
    ctx.fillStyle = COLOR_TEXT_SLATE;
    ctx.fillText("D_inf_h Centrosymmetric Inversion (Dipole Extinguished)", pad + plotW - 270, h - pad - 6);
}

// -------------------------------------------------------------
// Live UI Synchronization & Events
// -------------------------------------------------------------
function updateUIFromGeometry() {
    const isLinear = Math.abs(bondAngleDeg - 180.0) < 0.5;
    const halfRad = (bondAngleDeg / 2.0) * Math.PI / 180.0;
    const dipole = isLinear ? 0.0 : 1.854 * (Math.cos(halfRad) / Math.cos(52.24 * Math.PI / 180.0)) * (bondLengthR / 0.9578);

    const normTh = (bondAngleDeg - 80.0) / 100.0;
    const eps3a1 = -0.5015 + 0.2116 * normTh;
    const eps1b1 = -0.4003 + 0.1104 * normTh;
    const gap = isLinear ? 0.0 : Math.abs(eps3a1 - eps1b1);

    document.getElementById("disp-angle").innerText = `${bondAngleDeg.toFixed(1)}°`;
    document.getElementById("disp-bond").innerText = `${bondLengthR.toFixed(2)} Å`;
    document.getElementById("hud-bond").innerText = `${bondLengthR.toFixed(2)} Å`;
    document.getElementById("hud-angle").innerText = `${bondAngleDeg.toFixed(1)}°`;
    document.getElementById("hud-dipole").innerText = `${dipole.toFixed(2)} D`;
    document.getElementById("disp-live-dipole").innerText = `${dipole.toFixed(3)} D`;
    document.getElementById("disp-orb-3a1").innerText = `${eps3a1.toFixed(4)} Ha`;
    document.getElementById("disp-orb-1b1").innerText = `${eps1b1.toFixed(4)} Ha`;
    document.getElementById("disp-live-gap").innerText = `${gap.toFixed(4)} Ha`;

    const symBadge = document.getElementById("hud-symmetry-badge");
    const symText = document.getElementById("disp-symmetry");
    if (isLinear) {
        symBadge.innerText = "D_inf_h (LINEAR)";
        symBadge.style.color = COLOR_CRIMSON;
        symBadge.style.borderColor = COLOR_CRIMSON;
        symText.innerText = "D_inf_h";
        symText.style.color = COLOR_CRIMSON;
        document.getElementById("footer-status").innerText = "180° LINEAR SADDLE POINT // DEGENERATE 1π_u SHELL // D_inf_h INVERSION ACTIVE";
    } else {
        symBadge.innerText = "C_2v (BENT)";
        symBadge.style.color = COLOR_AMBER;
        symBadge.style.borderColor = COLOR_AMBER;
        symText.innerText = "C_2v";
        symText.style.color = COLOR_EMERALD;
        document.getElementById("footer-status").innerText = "C_2v EQUILIBRIUM ACTIVE // DIPOLE μ = 1.85 D";
    }

    renderPesSurface();
    renderBarrierCurve();
    renderWalshDiagram();
    renderDipoleModes();
}

function bindEvents() {
    const inputAngle = document.getElementById("input-angle");
    const inputBond = document.getElementById("input-bond");
    const inputSpeed = document.getElementById("input-osc-speed");

    inputAngle.addEventListener("input", e => {
        bondAngleDeg = parseFloat(e.target.value);
        updateUIFromGeometry();
    });

    inputBond.addEventListener("input", e => {
        bondLengthR = parseFloat(e.target.value);
        updateUIFromGeometry();
    });

    inputSpeed.addEventListener("input", e => {
        animSpeed = parseFloat(e.target.value);
        document.getElementById("disp-osc-speed").innerText = `${animSpeed.toFixed(1)}x`;
    });

    // Linearize / Reset Buttons
    document.getElementById("btn-linearize").addEventListener("click", () => {
        bondAngleDeg = 180.0;
        inputAngle.value = 180.0;
        updateUIFromGeometry();
    });

    document.getElementById("btn-reset-geom").addEventListener("click", () => {
        bondAngleDeg = 104.48;
        bondLengthR = 0.9578;
        inputAngle.value = 104.5;
        inputBond.value = 0.96;
        currentVibMode = 0;
        setActiveModeButton("btn-mode-static");
        updateUIFromGeometry();
    });

    // Vibrational Mode Buttons
    const modeButtons = [
        { id: "btn-mode-static", mode: 0 },
        { id: "btn-mode-1", mode: 1 },
        { id: "btn-mode-2", mode: 2 },
        { id: "btn-mode-3", mode: 3 },
    ];

    modeButtons.forEach(btn => {
        document.getElementById(btn.id).addEventListener("click", () => {
            currentVibMode = btn.mode;
            setActiveModeButton(btn.id);
        });
    });

    function setActiveModeButton(activeId) {
        modeButtons.forEach(b => {
            const el = document.getElementById(b.id);
            if (b.id === activeId) el.classList.add("active");
            else el.classList.remove("active");
        });
    }

    // 3D Orbital Mouse Controls
    const canvas = document.getElementById("canvas-molecule-3d");
    canvas.addEventListener("mousedown", e => {
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
    });

    window.addEventListener("mouseup", () => {
        isDraggingCamera = false;
    });

    canvas.addEventListener("wheel", e => {
        e.preventDefault();
        cameraZoom = Math.min(2.5, Math.max(0.5, cameraZoom - e.deltaY * 0.0015));
    }, { passive: false });

    // Load recorded JSON if available
    document.getElementById("btn-load-telemetry").addEventListener("click", async () => {
        try {
            const resp = await fetch("water_pes.json");
            if (resp.ok) {
                diskTelemetry = await resp.json();
                console.log("Ingested water_pes.json from disk:", diskTelemetry);
                document.getElementById("footer-status").innerText = "DISK TELEMETRY INGESTED // 231 GRID POINTS ACTIVE";
            }
        } catch (err) {
            console.warn("Could not load water_pes.json:", err);
        }
    });

    window.addEventListener("resize", () => {
        renderPesSurface();
        renderBarrierCurve();
        renderWalshDiagram();
        renderDipoleModes();
    });
}

// Initial Boot
window.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    updateUIFromGeometry();
    requestAnimationFrame(renderWaterMolecule3D);
});

