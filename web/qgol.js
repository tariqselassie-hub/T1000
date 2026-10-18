// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/**
 * T-1000 Sovereign Quantum Conway Game of Life Visualizer Engine.
 * Implements pure-client 60 FPS unitary QCA evolution, interaction holonomy cycles,
 * 3D isometric pillar rendering with phase needles, Friedel wavefields, and 
 * confined Hilbert space chamber (CP^2 simplex) projections.
 */

// =========================================================================
// 1. Quantum Conway Physics Engine (Client-Side Dual of crates/t1000_conway)
// =========================================================================

class QuantumConwayEngine {
    constructor(width = 16, height = 16) {
        this.width = width;
        this.height = height;
        this.totalCells = width * height;
        
        // Structure of Arrays (SoA) Amplitudes
        this.re0 = new Float64Array(this.totalCells); // Vacuum |0>
        this.im0 = new Float64Array(this.totalCells);
        this.re1 = new Float64Array(this.totalCells); // Living |1>
        this.im1 = new Float64Array(this.totalCells);
        this.re2 = new Float64Array(this.totalCells); // FFS l=2 |2>
        this.im2 = new Float64Array(this.totalCells);
        
        // Scratch buffers for smooth interpolation
        this.prevRe1 = new Float64Array(this.totalCells);
        this.prevIm1 = new Float64Array(this.totalCells);
        this.prevRe2 = new Float64Array(this.totalCells);
        
        this.nbrBuffer = new Float64Array(this.totalCells);
        this.currentStep = 0;
        this.lQuantum = 2; // alpha = l = 2
        this.cirWidth = 0.5;
        this.birthSharpness = 2.0;
        
        this.initVacuum();
    }

    initVacuum() {
        this.re0.fill(1.0);
        this.im0.fill(0.0);
        this.re1.fill(0.0);
        this.im1.fill(0.0);
        this.re2.fill(0.0);
        this.im2.fill(0.0);
        this.currentStep = 0;
    }

    coordToIdx(x, y) {
        return y * this.width + x;
    }

    idxToCoord(idx) {
        return [idx % this.width, Math.floor(idx / this.width)];
    }

    probLiving(idx) {
        return this.re1[idx] * this.re1[idx] + this.im1[idx] * this.im1[idx];
    }

    probFFS(idx) {
        return this.re2[idx] * this.re2[idx] + this.im2[idx] * this.im2[idx];
    }

    cellDensity(idx) {
        return this.probLiving(idx) + 2.0 * this.probFFS(idx);
    }

    cellPhase(idx) {
        return Math.atan2(this.im1[idx], this.re1[idx]);
    }

    getMooreNeighbors(idx) {
        const [x, y] = this.idxToCoord(idx);
        const nbrs = [];
        const offsets = [
            [-1, -1], [0, -1], [1, -1],
            [-1,  0],          [1,  0],
            [-1,  1], [0,  1], [1,  1]
        ];
        for (let i = 0; i < 8; i++) {
            const [dx, dy] = offsets[i];
            const nx = (x + dx + this.width) % this.width;
            const ny = (y + dy + this.height) % this.height;
            nbrs.push(ny * this.width + nx);
        }
        return nbrs;
    }

    computeNeighborhoodOccupations() {
        const densities = new Float64Array(this.totalCells);
        for (let i = 0; i < this.totalCells; i++) {
            densities[i] = this.cellDensity(i);
        }
        for (let i = 0; i < this.totalCells; i++) {
            const nbrs = this.getMooreNeighbors(i);
            let sum = 0.0;
            for (let k = 0; k < 8; k++) {
                sum += densities[nbrs[k]];
            }
            this.nbrBuffer[i] = sum;
        }
    }

    buildHolonomyUnitary(nbrOcc) {
        const gammaGeom = Math.PI * (1.0 - 1.0 / this.lQuantum); // pi/2 for l=2
        const deltaBirth = nbrOcc - 3.0;
        const birthWeight = Math.exp(-this.birthSharpness * deltaBirth * deltaBirth);
        const thetaB = (Math.PI * 0.5) * birthWeight;

        const survivalActive = nbrOcc >= 1.8 && nbrOcc <= 3.2;
        const cirDelta = Math.max(0.0, nbrOcc - 4.0) / this.cirWidth;
        const thetaCir = (Math.PI * 0.5) * (1.0 - Math.exp(-cirDelta));
        const underpopDelta = Math.max(0.0, 1.8 - nbrOcc) / this.cirWidth;
        const thetaDecay = (Math.PI * 0.5) * (1.0 - Math.exp(-underpopDelta));

        // Return 3x3 complex unitary matrix components [row][col] {re, im}
        const u = Array.from({ length: 3 }, () => 
            Array.from({ length: 3 }, () => ({ re: 0.0, im: 0.0 }))
        );

        if (survivalActive) {
            const cosB = Math.cos(thetaB);
            const sinB = Math.sin(thetaB);
            const phaseAng = 0.1 * deltaBirth;
            const cosP = Math.cos(phaseAng);
            const sinP = Math.sin(phaseAng);

            u[0][0] = { re: cosB, im: 0.0 };
            u[0][1] = { re: 0.0, im: -sinB };
            // Row 1 multiplied by phase exp(i*phaseAng)
            u[1][0] = { re: sinP * sinB, im: -cosP * sinB };
            u[1][1] = { re: cosP * cosB, im: sinP * cosB };
            u[2][2] = { re: 1.0, im: 0.0 };
        } else if (nbrOcc >= 3.8) {
            const cosC = Math.cos(thetaCir);
            const sinC = Math.sin(thetaCir);
            const cosG = Math.cos(gammaGeom);
            const sinG = Math.sin(gammaGeom);

            u[0][0] = { re: 1.0, im: 0.0 };
            u[1][1] = { re: cosC, im: 0.0 };
            u[1][2] = { re: 0.0, im: -sinC };
            // Row 2 multiplied by holonomy phase exp(i*gammaGeom)
            u[2][1] = { re: sinG * sinC, im: -cosG * sinC };
            u[2][2] = { re: cosG * cosC, im: sinG * cosC };
        } else {
            const cosD = Math.cos(thetaDecay);
            const sinD = Math.sin(thetaDecay);
            u[0][0] = { re: cosD, im: 0.0 };
            u[0][1] = { re: sinD, im: 0.0 };
            u[1][0] = { re: -sinD, im: 0.0 };
            u[1][1] = { re: cosD, im: 0.0 };
            u[2][2] = { re: 1.0, im: 0.0 };
        }
        return u;
    }

    applySublattice(indices) {
        for (let k = 0; k < indices.length; k++) {
            const idx = indices[k];
            const u = this.buildHolonomyUnitary(this.nbrBuffer[idx]);
            
            const r0 = this.re0[idx], i0 = this.im0[idx];
            const r1 = this.re1[idx], i1 = this.im1[idx];
            const r2 = this.re2[idx], i2 = this.im2[idx];

            // Complex matrix-vector multiply
            const mul = (r, c_re, c_im) => ({
                re: r.re * c_re - r.im * c_im,
                im: r.re * c_im + r.im * c_re
            });

            const next0 = {
                re: mul(u[0][0], r0, i0).re + mul(u[0][1], r1, i1).re + mul(u[0][2], r2, i2).re,
                im: mul(u[0][0], r0, i0).im + mul(u[0][1], r1, i1).im + mul(u[0][2], r2, i2).im
            };
            const next1 = {
                re: mul(u[1][0], r0, i0).re + mul(u[1][1], r1, i1).re + mul(u[1][2], r2, i2).re,
                im: mul(u[1][0], r0, i0).im + mul(u[1][1], r1, i1).im + mul(u[1][2], r2, i2).im
            };
            const next2 = {
                re: mul(u[2][0], r0, i0).re + mul(u[2][1], r1, i1).re + mul(u[2][2], r2, i2).re,
                im: mul(u[2][0], r0, i0).im + mul(u[2][1], r1, i1).im + mul(u[2][2], r2, i2).im
            };

            this.re0[idx] = next0.re;
            this.im0[idx] = next0.im;
            this.re1[idx] = next1.re;
            this.im1[idx] = next1.im;
            this.re2[idx] = next2.re;
            this.im2[idx] = next2.im;
        }
    }

    step() {
        // Cache previous state for visual interpolation
        this.prevRe1.set(this.re1);
        this.prevIm1.set(this.im1);
        this.prevRe2.set(this.re2);

        // Partition Sublattice A & B
        const subA = [];
        const subB = [];
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const idx = this.coordToIdx(x, y);
                if ((x + y) % 2 === 0) subA.push(idx);
                else subB.push(idx);
            }
        }

        // Half-Step 1: Sublattice A
        this.computeNeighborhoodOccupations();
        this.applySublattice(subA);

        // Half-Step 2: Sublattice B
        this.computeNeighborhoodOccupations();
        this.applySublattice(subB);

        this.currentStep++;
        return this.gatherTelemetry();
    }

    bornRuleMeasurement() {
        for (let i = 0; i < this.totalCells; i++) {
            const p1 = this.probLiving(i);
            const p2 = this.probFFS(i);
            const pAlive = p1 + p2;
            const rand = Math.random();

            if (rand < pAlive) {
                // Collapsed alive
                if (p2 > p1) {
                    this.re0[i] = 0; this.im0[i] = 0;
                    this.re1[i] = 0; this.im1[i] = 0;
                    this.re2[i] = 1; this.im2[i] = 0;
                } else {
                    this.re0[i] = 0; this.im0[i] = 0;
                    this.re1[i] = 1; this.im1[i] = 0;
                    this.re2[i] = 0; this.im2[i] = 0;
                }
            } else {
                // Collapsed dead
                this.re0[i] = 1; this.im0[i] = 0;
                this.re1[i] = 0; this.im1[i] = 0;
                this.re2[i] = 0; this.im2[i] = 0;
            }
        }
    }

    gatherTelemetry() {
        let totalPop = 0.0;
        let livingPop = 0.0;
        let ffsPop = 0.0;
        let maxNormErr = 0.0;

        for (let i = 0; i < this.totalCells; i++) {
            const p0 = this.re0[i] ** 2 + this.im0[i] ** 2;
            const p1 = this.probLiving(i);
            const p2 = this.probFFS(i);
            
            livingPop += p1;
            ffsPop += p2;
            totalPop += (p1 + 2.0 * p2);

            const err = Math.abs((p0 + p1 + p2) - 1.0);
            if (err > maxNormErr) maxNormErr = err;
        }

        // Bipartite Entanglement Entropy across W/2 cut
        const midX = Math.floor(this.width / 2);
        let probL = 0.0, probR = 0.0;
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const idx = this.coordToIdx(x, y);
                const p = this.probLiving(idx) + this.probFFS(idx);
                if (x < midX) probL += p;
                else probR += p;
            }
        }
        const tot = probL + probR;
        let entropy = 0.0;
        if (tot > 1e-12) {
            const pL = probL / tot;
            const pR = probR / tot;
            if (pL > 1e-12) entropy -= pL * Math.log(pL);
            if (pR > 1e-12) entropy -= pR * Math.log(pR);
        }

        // Central Momentum Density (FFS plateau indicator)
        const meanMomentum = this.computeCentralMomentumDensity();

        return {
            step: this.currentStep,
            totalPop,
            livingPop,
            ffsPop,
            maxNormErr,
            entropy,
            meanMomentum
        };
    }

    computeCentralMomentumDensity() {
        let sum = 0.0, count = 0;
        const w = this.width, h = this.height;
        for (let kx = Math.floor(w / 4); kx < Math.floor(3 * w / 4); kx++) {
            for (let ky = Math.floor(h / 4); ky < Math.floor(3 * h / 4); ky++) {
                let reF = 0.0, imF = 0.0;
                for (let y = 0; y < h; y++) {
                    for (let x = 0; x < w; x++) {
                        const idx = this.coordToIdx(x, y);
                        const phase = 2.0 * Math.PI * ((kx * x) / w + (ky * y) / h);
                        const cosP = Math.cos(phase), sinP = Math.sin(phase);
                        reF += this.re1[idx] * cosP + this.im1[idx] * sinP;
                        imF += -this.re1[idx] * sinP + this.im1[idx] * cosP;
                    }
                }
                sum += (reF * reF + imF * imF) / (w * h);
                count++;
            }
        }
        return count > 0 ? (sum / count) : 0.0;
    }

    // Pattern Injection
    injectPattern(type) {
        this.initVacuum();
        const midX = Math.floor(this.width / 2);
        const midY = Math.floor(this.height / 2);

        if (type === 'glider') {
            const coords = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]];
            coords.forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX - 2 + dx, midY - 2 + dy);
                this.re0[idx] = 0; this.re1[idx] = 1;
            });
        } else if (type === 'ffs_bubble') {
            const coords = [[0, 0], [1, 0], [0, 1], [1, 1]];
            coords.forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX - 1 + dx, midY - 1 + dy);
                this.re0[idx] = 0; this.re2[idx] = 1; // FFS branch
            });
        } else if (type === 'cat_blinker') {
            const centerIdx = this.coordToIdx(midX, midY);
            this.re0[centerIdx] = 0; this.re1[centerIdx] = 1;
            const invSqrt2 = 1.0 / Math.sqrt(2);
            [[0, -1], [0, 1], [-1, 0], [1, 0]].forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX + dx, midY + dy);
                this.re0[idx] = invSqrt2;
                this.re1[idx] = invSqrt2;
            });
        } else if (type === 'block') {
            [[0, 0], [1, 0], [0, 1], [1, 1]].forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX - 1 + dx, midY - 1 + dy);
                this.re0[idx] = 0; this.re1[idx] = 1;
            });
        } else if (type === 'soup') {
            for (let i = 0; i < this.totalCells; i++) {
                if (Math.random() < 0.25) {
                    const ang = Math.random() * 2 * Math.PI;
                    this.re0[i] = 0;
                    this.re1[i] = Math.cos(ang);
                    this.im1[i] = Math.sin(ang);
                }
            }
        }
    }
}

// =========================================================================
// 2. Rendering Pipelines (Isometric 3D Columns, Wavefield, Hilbert Simplex)
// =========================================================================

class QGolVisualizerDeck {
    constructor() {
        this.engine = new QuantumConwayEngine(16, 16);
        this.engine.injectPattern('glider');

        // Canvases
        this.cvsIso = document.getElementById('canvas-iso');
        this.ctxIso = this.cvsIso.getContext('2d');
        
        this.cvsWave = document.getElementById('canvas-wave');
        this.ctxWave = this.cvsWave.getContext('2d');

        this.cvsMomentum = document.getElementById('canvas-momentum');
        this.ctxMomentum = this.cvsMomentum.getContext('2d');

        this.cvsHilbert = document.getElementById('canvas-hilbert');
        this.ctxHilbert = this.cvsHilbert.getContext('2d');

        // Isometric Camera State
        this.isoYaw = Math.PI / 4;
        this.isoPitch = Math.PI / 6;
        this.isoZoom = 1.0;
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };

        // Simulation Loop State
        this.isPlaying = true;
        this.ticksPerSec = 4;
        this.lastTickTime = performance.now();
        // Active View State (ONLY ONE PANEL AT A TIME)
        this.activeView = 'panel-iso';

        this.setupEventListeners();
        this.resizeCanvases();
        window.addEventListener('resize', () => this.resizeCanvases());

        // Kickoff Render Loop
        requestAnimationFrame((t) => this.renderLoop(t));
    }

    switchView(viewId) {
        this.activeView = viewId;
        document.querySelectorAll('.view-tab').forEach(t => {
            t.classList.toggle('active', t.getAttribute('data-view') === viewId);
        });
        document.querySelectorAll('.panel-card').forEach(p => {
            p.classList.toggle('active-view', p.id === viewId);
        });
        setTimeout(() => this.resizeCanvases(), 10);
    }

    resizeCanvases() {
        const dpr = window.devicePixelRatio || 1;
        [this.cvsIso, this.cvsWave, this.cvsMomentum, this.cvsHilbert].forEach(c => {
            if (!c || !c.parentElement) return;
            const rect = c.parentElement.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                c.width = rect.width * dpr;
                c.height = rect.height * dpr;
            }
        });
    }

    setupEventListeners() {
        // Dashboard Switcher Tabs (Only 1 Dashboard at once)
        const viewTabs = document.querySelectorAll('.view-tab');
        viewTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetView = tab.getAttribute('data-view');
                this.switchView(targetView);
            });
        });

        // Keyboard shortcuts: 1, 2, 3
        window.addEventListener('keydown', (e) => {
            if (e.key === '1') this.switchView('panel-iso');
            else if (e.key === '2') this.switchView('panel-wave');
            else if (e.key === '3') this.switchView('panel-hilbert');
        });

        // Isometric Drag rotation
        this.cvsIso.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
        });
        window.addEventListener('mouseup', () => this.isDragging = false);
        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.isoYaw += dx * 0.008;
            this.isoPitch = Math.max(0.1, Math.min(Math.PI / 2.2, this.isoPitch + dy * 0.008));
            this.lastMouse = { x: e.clientX, y: e.clientY };
        });
        this.cvsIso.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.isoZoom = Math.max(0.5, Math.min(2.5, this.isoZoom - e.deltaY * 0.001));
        });

        // Controls
        const btnPlay = document.getElementById('btn-play');
        btnPlay.addEventListener('click', () => {
            this.isPlaying = !this.isPlaying;
            btnPlay.textContent = this.isPlaying ? 'PAUSE' : 'PLAY';
            btnPlay.classList.toggle('primary', this.isPlaying);
        });

        document.getElementById('btn-step').addEventListener('click', () => {
            this.engine.step();
            this.updateTelemetryHUD(this.engine.gatherTelemetry());
        });

        document.getElementById('btn-reset').addEventListener('click', () => {
            const pat = document.getElementById('select-pattern').value;
            this.engine.injectPattern(pat);
            this.updateTelemetryHUD(this.engine.gatherTelemetry());
        });

        document.getElementById('btn-born-probe').addEventListener('click', () => {
            this.engine.bornRuleMeasurement();
            this.updateTelemetryHUD(this.engine.gatherTelemetry());
        });

        document.getElementById('select-pattern').addEventListener('change', (e) => {
            this.engine.injectPattern(e.target.value);
            this.updateTelemetryHUD(this.engine.gatherTelemetry());
        });

        const sliderSpeed = document.getElementById('slider-speed');
        const speedVal = document.getElementById('speed-val');
        sliderSpeed.addEventListener('input', (e) => {
            this.ticksPerSec = parseInt(e.target.value);
            speedVal.textContent = `${this.ticksPerSec} TPS`;
        });
    }

    renderLoop(now) {
        // Step execution logic
        const interval = 1000.0 / this.ticksPerSec;
        const elapsed = now - this.lastTickTime;
        if (this.isPlaying && elapsed >= interval) {
            const telem = this.engine.step();
            this.updateTelemetryHUD(telem);
            this.lastTickTime = now;
            this.interpFraction = 0.0;
        } else {
            this.interpFraction = Math.min(1.0, elapsed / interval);
        }

        // Render ONLY the single active panel to maximize performance and visual focus
        if (this.activeView === 'panel-iso') {
            this.renderIsometricPanel();
        } else if (this.activeView === 'panel-wave') {
            this.renderDualDomainPanel();
        } else if (this.activeView === 'panel-hilbert') {
            this.renderHilbertChamberPanel();
        }

        requestAnimationFrame((t) => this.renderLoop(t));
    }

    // ---------------------------------------------------------------------
    // Viewport 1: 3D Isometric Columns & Phase Needles
    // ---------------------------------------------------------------------
    renderIsometricPanel() {
        const ctx = this.ctxIso;
        const w = this.cvsIso.width;
        const h = this.cvsIso.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h / 2 + 30;
        const size = Math.min(w, h) * 0.038 * this.isoZoom;

        // Depth sort grid cells back-to-front based on camera yaw
        const cells = [];
        for (let y = 0; y < this.engine.height; y++) {
            for (let x = 0; x < this.engine.width; x++) {
                const idx = this.engine.coordToIdx(x, y);
                // Center-origin coords
                const nx = x - this.engine.width / 2;
                const ny = y - this.engine.height / 2;
                // Rotate by yaw
                const rotX = nx * Math.cos(this.isoYaw) - ny * Math.sin(this.isoYaw);
                const rotY = nx * Math.sin(this.isoYaw) + ny * Math.cos(this.isoYaw);
                // Depth value
                const depth = rotY * Math.cos(this.isoPitch);
                cells.push({ x, y, idx, rotX, rotY, depth });
            }
        }
        cells.sort((a, b) => a.depth - b.depth);

        // Render each cell column
        cells.forEach(cell => {
            const screenX = cx + cell.rotX * size * 2.2;
            const screenY = cy + cell.rotY * size * 1.2 * Math.sin(this.isoPitch);

            const p1 = this.engine.probLiving(cell.idx);
            const p2 = this.engine.probFFS(cell.idx);
            const phase = this.engine.cellPhase(cell.idx);

            const colHeight = (p1 * 70 + p2 * 110) * this.isoZoom;
            const ffsRatio = (p1 + p2) > 0 ? (p2 / (p1 + p2)) : 0;

            if (colHeight > 2) {
                // Column Base (Laser Emerald)
                ctx.fillStyle = ffsRatio > 0.4 ? 'rgba(255, 179, 0, 0.75)' : 'rgba(0, 230, 118, 0.75)';
                ctx.strokeStyle = ffsRatio > 0.4 ? '#ffb300' : '#00e676';
                ctx.lineWidth = 1;

                // Simple 3D Prism
                const r = size * 0.7;
                ctx.beginPath();
                ctx.moveTo(screenX - r, screenY);
                ctx.lineTo(screenX + r, screenY);
                ctx.lineTo(screenX + r, screenY - colHeight);
                ctx.lineTo(screenX - r, screenY - colHeight);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Top Face (Phase Reticle Compass)
                ctx.fillStyle = '#0f172a';
                ctx.beginPath();
                ctx.arc(screenX, screenY - colHeight, r, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // Needle
                const needleLen = r * 0.85;
                ctx.strokeStyle = '#f8fafc';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(screenX, screenY - colHeight);
                ctx.lineTo(
                    screenX + needleLen * Math.cos(phase),
                    screenY - colHeight + needleLen * Math.sin(phase)
                );
                ctx.stroke();
            } else {
                // Vacuum grid point
                ctx.fillStyle = '#1e293b';
                ctx.beginPath();
                ctx.arc(screenX, screenY, 2, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    }

    // ---------------------------------------------------------------------
    // Viewport 2: Continuous Real-Space Wavefield & Reciprocal Momentum Sea
    // ---------------------------------------------------------------------
    renderDualDomainPanel() {
        // 1. Spatial Wavefield
        const ctxW = this.ctxWave;
        const wW = this.cvsWave.width;
        const hW = this.cvsWave.height;
        ctxW.clearRect(0, 0, wW, hW);

        const imgData = ctxW.createImageData(wW, hW);
        const data = imgData.data;

        const scaleX = this.engine.width / wW;
        const scaleY = this.engine.height / hW;

        for (let py = 0; py < hW; py += 3) {
            for (let px = 0; px < wW; px += 3) {
                const gx = px * scaleX;
                const gy = py * scaleY;

                let field = 0.0;
                // Accumulate Friedel wave ripples from live cells
                for (let i = 0; i < this.engine.totalCells; i++) {
                    const p = this.engine.probLiving(i) + this.engine.probFFS(i);
                    if (p > 0.1) {
                        const [cx, cy] = this.engine.idxToCoord(i);
                        const dist = Math.hypot(gx - cx, gy - cy);
                        if (dist < 4.5) {
                            field += p * Math.cos(dist * 3.5) / (1.0 + dist);
                        }
                    }
                }

                const intensity = Math.min(255, Math.max(0, Math.floor(field * 200)));
                const idx = (py * wW + px) * 4;
                data[idx] = 0;                          // R
                data[idx + 1] = intensity;              // G (#00e676 Emerald)
                data[idx + 2] = Math.floor(intensity * 0.4); // B
                data[idx + 3] = 255;
            }
        }
        ctxW.putImageData(imgData, 0, 0);

        // 2. Momentum Space n(k) Plateau Bar
        const ctxM = this.ctxMomentum;
        const wM = this.cvsMomentum.width;
        const hM = this.cvsMomentum.height;
        ctxM.clearRect(0, 0, wM, hM);

        const meanNk = this.engine.computeCentralMomentumDensity();
        const barW = wM * 0.45;
        const barH = Math.min(hM * 0.85, (meanNk / 0.5) * (hM * 0.65));
        const barX = (wM - barW) / 2;
        const barY = hM * 0.85 - barH;

        // Draw Amber Plateau Bar
        ctxM.fillStyle = 'rgba(255, 179, 0, 0.25)';
        ctxM.fillRect(barX, barY, barW, barH);
        ctxM.strokeStyle = '#ffb300';
        ctxM.lineWidth = 2;
        ctxM.strokeRect(barX, barY, barW, barH);

        // Theoretical Plateau Reticle n(k) = 0.5000
        const platY = hM * 0.85 - (hM * 0.65);
        ctxM.strokeStyle = '#ef4444';
        ctxM.setLineDash([4, 4]);
        ctxM.beginPath();
        ctxM.moveTo(10, platY);
        ctxM.lineTo(wM - 10, platY);
        ctxM.stroke();
        ctxM.setLineDash([]);

        ctxM.fillStyle = '#f8fafc';
        ctxM.font = '10px monospace';
        ctxM.fillText(`n(k) = ${meanNk.toFixed(4)}`, barX, barY - 8);
        ctxM.fillStyle = '#ef4444';
        ctxM.fillText(`1/ℓ = 0.5000`, 12, platY - 4);
    }

    // ---------------------------------------------------------------------
    // Viewport 3: Confined Hilbert Space Chamber (Bloch Simplex / CP^2)
    // ---------------------------------------------------------------------
    renderHilbertChamberPanel() {
        const ctx = this.ctxHilbert;
        const w = this.cvsHilbert.width;
        const h = this.cvsHilbert.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h / 2 + 20;
        const scale = Math.min(w, h) * 0.38;

        // Equilateral Simplex Vertices in 3D projection
        const v0 = { x: cx - scale, y: cy + scale * 0.6 };       // Vacuum |0>
        const v1 = { x: cx + scale, y: cy + scale * 0.6 };       // Living |1>
        const v2 = { x: cx, y: cy - scale * 0.85 };              // FFS |2> (Apex)

        // Draw Wireframe Pyramid Chamber
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(v0.x, v0.y);
        ctx.lineTo(v1.x, v1.y);
        ctx.lineTo(v2.x, v2.y);
        ctx.closePath();
        ctx.stroke();

        // Vertex Labels
        ctx.font = '11px monospace';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText('|0⟩ VACUUM', v0.x - 30, v0.y + 18);
        ctx.fillStyle = '#00e676';
        ctx.fillText('|1⟩ LIVING', v1.x - 20, v1.y + 18);
        ctx.fillStyle = '#ffb300';
        ctx.fillText('FFS |2⟩', v2.x - 22, v2.y - 10);

        // Center CIR Axis (Confinement-Induced Resonance Singularity)
        ctx.strokeStyle = 'rgba(255, 179, 0, 0.35)';
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy + scale * 0.6);
        ctx.lineTo(v2.x, v2.y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Plot All Cells as Floating Quantum Beads with Trails
        for (let i = 0; i < this.engine.totalCells; i++) {
            const p0 = this.engine.re0[i] ** 2 + this.engine.im0[i] ** 2;
            const p1 = this.engine.probLiving(i);
            const p2 = this.engine.probFFS(i);

            if (p1 > 0.05 || p2 > 0.05) {
                // Barycentric position inside triangle
                const px = p0 * v0.x + p1 * v1.x + p2 * v2.x;
                const py = p0 * v0.y + p1 * v1.y + p2 * v2.y;

                // Color based on FFS excitation
                const isFFS = p2 > 0.3;
                ctx.fillStyle = isFFS ? '#ffb300' : '#00e676';
                ctx.beginPath();
                ctx.arc(px, py, isFFS ? 3.5 : 2.5, 0, Math.PI * 2);
                ctx.fill();

                // Faint Holonomy Trail
                ctx.strokeStyle = isFFS ? 'rgba(255, 179, 0, 0.25)' : 'rgba(0, 230, 118, 0.25)';
                ctx.beginPath();
                ctx.arc(cx, py, Math.abs(px - cx), 0, Math.PI * 2);
                ctx.stroke();
            }
        }
    }

    updateTelemetryHUD(telem) {
        document.getElementById('hud-step').textContent = String(telem.step).padStart(4, '0');
        document.getElementById('hud-tot-pop').textContent = telem.totalPop.toFixed(2);
        document.getElementById('hud-live-pop').textContent = telem.livingPop.toFixed(2);
        document.getElementById('hud-ffs-pop').textContent = telem.ffsPop.toFixed(2);
        document.getElementById('hud-norm-err').textContent = telem.maxNormErr.toExponential(2);
        document.getElementById('hud-entropy').textContent = `${telem.entropy.toFixed(4)} nats`;
        document.getElementById('hud-momentum').textContent = telem.meanMomentum.toFixed(4);
    }
}

// Instantiate visualizer deck when DOM loads
window.addEventListener('DOMContentLoaded', () => {
    new QGolVisualizerDeck();
});
