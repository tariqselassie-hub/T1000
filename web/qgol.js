// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/**
 * T-1000 Sovereign Quantum Conway Game of Life Visualizer Engine.
 * Dual-mission client implementation matching crates/t1000_conway:
 *  1. Unitary QCA evolution across 6 sovereign rule sets (CIR B3/S23, HighLife,
 *     Gross-Pitaevskii BEC, Floquet Chaos, Dirac-Weyl, Cyclic Lotka-Volterra).
 *  2. 3D Isometric columns with phase needles, Friedel wavefields, and CP^2 Hilbert simplex.
 *  3. Macroscopic Cosmic Viewport with pan/zoom, probability current streamlines,
 *     vorticity reticles (+1 / -1 circulation), and OTOC scrambling causal wavefront.
 */

// =========================================================================
// 1. Quantum Conway Physics Engine (Client-Side Dual of crates/t1000_conway)
// =========================================================================

class QuantumConwayEngine {
    constructor(width = 32, height = 32) {
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

        // Rule parameters mirroring HolonomyParams
        this.rule = 'b3s23';
        this.lQuantum = 2; // alpha = l = 2
        this.cirWidth = 0.5;
        this.birthSharpness = 2.0;
        this.chaosK = 1.2;
        this.gaugeFlux = 0.0;
        this.nonLinearG = 0.8;
        this.cyclePeriod = 1.0;

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
        return ((y % this.height + this.height) % this.height) * this.width +
               ((x % this.width + this.width) % this.width);
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
        switch (this.rule) {
            case 'highlife':
                return this.buildHighLifeUnitary(nbrOcc);
            case 'bec':
                return this.buildBecUnitary(nbrOcc);
            case 'floquet':
                return this.buildFloquetChaosUnitary(nbrOcc);
            case 'dirac':
                return this.buildDiracUnitary(nbrOcc);
            case 'lotka':
                return this.buildCyclicLotkaUnitary(nbrOcc);
            case 'b3s23':
            default:
                return this.buildB3S23Unitary(nbrOcc);
        }
    }

    buildB3S23Unitary(nbrOcc) {
        const gammaGeom = Math.PI * (1.0 - 1.0 / this.lQuantum);
        const deltaBirth = nbrOcc - 3.0;
        const birthWeight = Math.exp(-this.birthSharpness * deltaBirth * deltaBirth);
        const thetaB = (Math.PI * 0.5) * birthWeight;
        const survivalActive = (nbrOcc >= 1.8 && nbrOcc <= 3.2);
        const cirDelta = Math.max(0.0, nbrOcc - 4.0) / this.cirWidth;
        const thetaCir = (Math.PI * 0.5) * (1.0 - Math.exp(-cirDelta));
        const underpopDelta = Math.max(0.0, 1.8 - nbrOcc) / this.cirWidth;
        const thetaDecay = (Math.PI * 0.5) * (1.0 - Math.exp(-underpopDelta));

        const u = [
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }]
        ];

        if (survivalActive) {
            const cosB = Math.cos(thetaB);
            const sinB = Math.sin(thetaB);
            const phaseAng = 0.1 * deltaBirth + this.gaugeFlux * Math.PI;
            const cosP = Math.cos(phaseAng);
            const sinP = Math.sin(phaseAng);

            u[0][0] = { re: cosB, im: 0.0 };
            u[0][1] = { re: 0.0, im: -sinB };
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

    buildHighLifeUnitary(nbrOcc) {
        const d3 = nbrOcc - 3.0;
        const d6 = nbrOcc - 6.0;
        const w3 = Math.exp(-this.birthSharpness * d3 * d3);
        const w6 = Math.exp(-this.birthSharpness * d6 * d6);
        const thetaB = (Math.PI * 0.5) * Math.min(1.0, w3 + w6);
        const survivalActive = (nbrOcc >= 1.8 && nbrOcc <= 3.2) || (nbrOcc >= 5.5 && nbrOcc <= 6.5);

        const u = [
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }]
        ];

        if (survivalActive) {
            const phaseAng = (w6 > 0.5) ? (Math.PI * 0.5) : (0.05 * d3);
            const cosP = Math.cos(phaseAng);
            const sinP = Math.sin(phaseAng);
            const cosB = Math.cos(thetaB);
            const sinB = Math.sin(thetaB);

            u[0][0] = { re: cosB, im: 0.0 };
            u[0][1] = { re: 0.0, im: -sinB };
            u[1][0] = { re: sinP * sinB, im: -cosP * sinB };
            u[1][1] = { re: cosP * cosB, im: sinP * cosB };
            u[2][2] = { re: 1.0, im: 0.0 };
        } else if (nbrOcc >= 6.8) {
            const delta = Math.max(0.0, nbrOcc - 6.8);
            const theta = (Math.PI * 0.5) * (1.0 - Math.exp(-delta));
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta);
            u[0][0] = { re: 1.0, im: 0.0 };
            u[1][1] = { re: cosT, im: 0.0 };
            u[1][2] = { re: 0.0, im: -sinT };
            u[2][1] = { re: 0.0, im: -sinT };
            u[2][2] = { re: cosT, im: 0.0 };
        } else {
            const decay = Math.max(0.0, 1.8 - nbrOcc) / this.cirWidth;
            const thetaD = (Math.PI * 0.5) * (1.0 - Math.exp(-decay));
            const cosD = Math.cos(thetaD);
            const sinD = Math.sin(thetaD);
            u[0][0] = { re: cosD, im: 0.0 };
            u[0][1] = { re: sinD, im: 0.0 };
            u[1][0] = { re: -sinD, im: 0.0 };
            u[1][1] = { re: cosD, im: 0.0 };
            u[2][2] = { re: 1.0, im: 0.0 };
        }
        return u;
    }

    buildBecUnitary(nbrOcc) {
        const g = this.nonLinearG;
        const mu = g * (nbrOcc - 3.0);
        const phiBec = mu * this.cyclePeriod;
        const thetaTunnel = (Math.PI * 0.25) * Math.exp(-0.5 * (nbrOcc - 3.0) * (nbrOcc - 3.0));
        const cosT = Math.cos(thetaTunnel);
        const sinT = Math.sin(thetaTunnel);

        const cos1 = Math.cos(phiBec), sin1 = Math.sin(phiBec);
        const cos2 = Math.cos(2.0 * phiBec), sin2 = Math.sin(2.0 * phiBec);

        const u = [
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }]
        ];

        u[0][0] = { re: cosT, im: 0.0 };
        u[0][1] = { re: 0.0, im: -sinT };
        u[1][0] = { re: sin1 * sinT, im: -cos1 * sinT };
        u[1][1] = { re: cos1 * cosT, im: sin1 * cosT };
        u[2][2] = { re: cos2, im: sin2 };
        return u;
    }

    buildFloquetChaosUnitary(nbrOcc) {
        const k = this.chaosK;
        const vKick = k * Math.cos(Math.PI * nbrOcc);
        const thetaMix = (Math.PI * 0.35) * (1.0 / (1.0 + Math.abs(nbrOcc - 3.0)));
        const cosM = Math.cos(thetaMix);
        const sinM = Math.sin(thetaMix);

        const cosK = Math.cos(vKick), sinK = Math.sin(vKick);
        const cos2K = Math.cos(-0.5 * vKick), sin2K = Math.sin(-0.5 * vKick);
        const cos3K = Math.cos(1.5 * vKick), sin3K = Math.sin(1.5 * vKick);

        const u = [
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }]
        ];

        u[0][0] = { re: cosK * cosM, im: sinK * cosM };
        u[0][1] = { re: sinK * sinM, im: -cosK * sinM };
        u[1][0] = { re: sin2K * sinM, im: -cos2K * sinM };
        u[1][1] = { re: cos2K * cosM, im: sin2K * cosM };
        u[2][2] = { re: cos3K, im: sin3K };
        return u;
    }

    buildDiracUnitary(nbrOcc) {
        const mass = 0.5 * (nbrOcc - 3.0);
        const pEff = 1.0;
        const energy = Math.hypot(pEff, mass);
        const thetaDirac = Math.asin(pEff / energy);
        const timePhase = energy * this.cyclePeriod;

        const cosD = Math.cos(thetaDirac);
        const sinD = Math.sin(thetaDirac);
        const cosP = Math.cos(-timePhase), sinP = Math.sin(-timePhase);
        const cosM = Math.cos(timePhase), sinM = Math.sin(timePhase);

        const u = [
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }]
        ];

        u[0][0] = { re: 1.0, im: 0.0 };
        u[1][1] = { re: cosP * cosD, im: sinP * cosD };
        u[1][2] = { re: sinP * sinD, im: -cosP * sinD };
        u[2][1] = { re: sinM * sinD, im: -cosM * sinD };
        u[2][2] = { re: cosM * cosD, im: sinM * cosD };
        return u;
    }

    buildCyclicLotkaUnitary(nbrOcc) {
        const rate = (Math.PI / 3.0) / (1.0 + Math.exp(-this.birthSharpness * (nbrOcc - 3.0)));
        const cosR = Math.cos(rate);
        const sinR = Math.sin(rate);
        const phaseGauge = this.gaugeFlux * 2.0 * Math.PI;
        const cosG = Math.cos(phaseGauge), sinG = Math.sin(phaseGauge);

        const u = [
            [{ re: cosR, im: 0 }, { re: 0, im: -sinR }, { re: 0, im: 0 }],
            [{ re: 0, im: 0 }, { re: cosR, im: 0 }, { re: 0, im: -sinR }],
            [{ re: sinG * sinR, im: -cosG * sinR }, { re: 0, im: 0 }, { re: cosR, im: 0 }]
        ];

        // Gram-Schmidt Orthonormalization across columns
        for (let c = 0; c < 3; c++) {
            for (let prev = 0; prev < c; prev++) {
                let dotRe = 0, dotIm = 0;
                for (let r = 0; r < 3; r++) {
                    const pr = u[r][prev];
                    const cur = u[r][c];
                    dotRe += pr.re * cur.re + pr.im * cur.im;
                    dotIm += pr.re * cur.im - pr.im * cur.re;
                }
                for (let r = 0; r < 3; r++) {
                    const pr = u[r][prev];
                    const subRe = dotRe * pr.re - dotIm * pr.im;
                    const subIm = dotRe * pr.im + dotIm * pr.re;
                    u[r][c].re -= subRe;
                    u[r][c].im -= subIm;
                }
            }
            let normSq = 0;
            for (let r = 0; r < 3; r++) {
                normSq += u[r][c].re * u[r][c].re + u[r][c].im * u[r][c].im;
            }
            const norm = Math.max(1e-15, Math.sqrt(normSq));
            for (let r = 0; r < 3; r++) {
                u[r][c].re /= norm;
                u[r][c].im /= norm;
            }
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

            const mul = (m, c_re, c_im) => ({
                re: m.re * c_re - m.im * c_im,
                im: m.re * c_im + m.im * c_re
            });

            const m00 = mul(u[0][0], r0, i0), m01 = mul(u[0][1], r1, i1), m02 = mul(u[0][2], r2, i2);
            const m10 = mul(u[1][0], r0, i0), m11 = mul(u[1][1], r1, i1), m12 = mul(u[1][2], r2, i2);
            const m20 = mul(u[2][0], r0, i0), m21 = mul(u[2][1], r1, i1), m22 = mul(u[2][2], r2, i2);

            this.re0[idx] = m00.re + m01.re + m02.re;
            this.im0[idx] = m00.im + m01.im + m02.im;
            this.re1[idx] = m10.re + m11.re + m12.re;
            this.im1[idx] = m10.im + m11.im + m12.im;
            this.re2[idx] = m20.re + m21.re + m22.re;
            this.im2[idx] = m20.im + m21.im + m22.im;
        }
    }

    step() {
        this.prevRe1.set(this.re1);
        this.prevIm1.set(this.im1);
        this.prevRe2.set(this.re2);

        const subA = [];
        const subB = [];
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const idx = this.coordToIdx(x, y);
                if ((x + y) % 2 === 0) subA.push(idx);
                else subB.push(idx);
            }
        }

        // Sublattice A
        this.computeNeighborhoodOccupations();
        this.applySublattice(subA);

        // Sublattice B
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
                this.re0[i] = 1; this.im0[i] = 0;
                this.re1[i] = 0; this.im1[i] = 0;
                this.re2[i] = 0; this.im2[i] = 0;
            }
        }
    }

    computeProbabilityCurrent() {
        const w = this.width;
        const h = this.height;
        const currents = new Array(this.totalCells);

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = this.coordToIdx(x, y);
                const xNext = this.coordToIdx((x + 1) % w, y);
                const xPrev = this.coordToIdx((x + w - 1) % w, y);
                const yNext = this.coordToIdx(x, (y + 1) % h);
                const yPrev = this.coordToIdx(x, (y + h - 1) % h);

                const c1_re = this.re1[idx], c1_im = this.im1[idx];
                const c2_re = this.re2[idx], c2_im = this.im2[idx];

                const dc1_dx_re = (this.re1[xNext] - this.re1[xPrev]) * 0.5;
                const dc1_dx_im = (this.im1[xNext] - this.im1[xPrev]) * 0.5;
                const dc2_dx_re = (this.re2[xNext] - this.re2[xPrev]) * 0.5;
                const dc2_dx_im = (this.im2[xNext] - this.im2[xPrev]) * 0.5;

                const dc1_dy_re = (this.re1[yNext] - this.re1[yPrev]) * 0.5;
                const dc1_dy_im = (this.im1[yNext] - this.im1[yPrev]) * 0.5;
                const dc2_dy_re = (this.re2[yNext] - this.re2[yPrev]) * 0.5;
                const dc2_dy_im = (this.im2[yNext] - this.im2[yPrev]) * 0.5;

                // Im(c^* * dc) = c_re * dc_im - c_im * dc_re
                const jx1 = c1_re * dc1_dx_im - c1_im * dc1_dx_re;
                const jx2 = 2.0 * (c2_re * dc2_dx_im - c2_im * dc2_dx_re);

                const jy1 = c1_re * dc1_dy_im - c1_im * dc1_dy_re;
                const jy2 = 2.0 * (c2_re * dc2_dy_im - c2_im * dc2_dy_re);

                currents[idx] = [jx1 + jx2, jy1 + jy2];
            }
        }
        return currents;
    }

    computeVorticity(currents) {
        const w = this.width;
        const h = this.height;
        const vort = new Float64Array(this.totalCells);

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = this.coordToIdx(x, y);
                const xNext = this.coordToIdx((x + 1) % w, y);
                const xPrev = this.coordToIdx((x + w - 1) % w, y);
                const yNext = this.coordToIdx(x, (y + 1) % h);
                const yPrev = this.coordToIdx(x, (y + h - 1) % h);

                const djy_dx = (currents[xNext][1] - currents[xPrev][1]) * 0.5;
                const djx_dy = (currents[yNext][0] - currents[yPrev][0]) * 0.5;
                vort[idx] = djy_dx - djx_dy;
            }
        }
        return vort;
    }

    computeScramblingOtoc() {
        const cx = this.width * 0.5;
        const cy = this.height * 0.5;
        let sum = 0.0, total = 0.0;
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const idx = this.coordToIdx(x, y);
                const dens = this.cellDensity(idx);
                const dx = x - cx;
                const dy = y - cy;
                sum += dens * (dx * dx + dy * dy);
                total += dens;
            }
        }
        return total > 1e-12 ? (sum / total) : 0.0;
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

        // Bipartite Entanglement Entropy
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

        const meanMomentum = this.computeCentralMomentumDensity();
        const currents = this.computeProbabilityCurrent();
        const vorticities = this.computeVorticity(currents);

        let maxVort = 0.0;
        let vortexCnt = 0;
        let currentMagSum = 0.0;

        for (let i = 0; i < this.totalCells; i++) {
            const vAbs = Math.abs(vorticities[i]);
            if (vAbs > maxVort) maxVort = vAbs;
            if (vAbs > 0.15) vortexCnt++;
            const c = currents[i];
            currentMagSum += Math.hypot(c[0], c[1]);
        }

        const meanCurrent = this.totalCells > 0 ? (currentMagSum / this.totalCells) : 0.0;
        const scramblingOtoc = this.computeScramblingOtoc();

        return {
            step: this.currentStep,
            totalPop,
            livingPop,
            ffsPop,
            maxNormErr,
            entropy,
            meanMomentum,
            maxVorticity: maxVort,
            vortexCount: vortexCnt,
            meanCurrent,
            scramblingOtoc,
            currents,
            vorticities
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
        } else if (type === 'glider_pair' || type === 'beam_splitter') {
            const phase = (type === 'beam_splitter') ? (Math.PI * 0.5) : (this.relativePhaseRad || (Math.PI * 0.5));
            const invSqrt2 = 1.0 / Math.sqrt(2);
            const coordsA = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]];
            coordsA.forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX - 7 + dx, midY - 7 + dy);
                this.re0[idx] = 0; this.re1[idx] = invSqrt2; this.im1[idx] = 0;
            });
            const coordsB = [[1, 0], [0, 1], [0, 2], [1, 2], [2, 2]];
            coordsB.forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX + 5 + dx, midY - 7 + dy);
                this.re0[idx] = 0;
                this.re1[idx] += invSqrt2 * Math.cos(phase);
                this.im1[idx] += invSqrt2 * Math.sin(phase);
            });
        } else if (type === 'vortex_dipole') {
            const cx1 = midX - 3, cy1 = midY;
            const cx2 = midX + 3, cy2 = midY;
            for (let y = 0; y < this.height; y++) {
                for (let x = 0; x < this.width; x++) {
                    const dx1 = x - cx1, dy1 = y - cy1;
                    const dx2 = x - cx2, dy2 = y - cy2;
                    const r1 = Math.hypot(dx1, dy1);
                    const r2 = Math.hypot(dx2, dy2);

                    if (r1 < 6.5 || r2 < 6.5) {
                        const theta1 = Math.atan2(dy1, dx1);
                        const theta2 = -Math.atan2(dy2, dx2);
                        const phase = theta1 + theta2;
                        const envelope = Math.exp(-0.15 * Math.min(r1, r2)) *
                                         (1.0 - Math.exp(-0.8 * r1)) *
                                         (1.0 - Math.exp(-0.8 * r2));
                        const ampMag = Math.max(0.0, Math.min(0.95, envelope));
                        const rem = Math.sqrt(Math.max(0.0, 1.0 - ampMag * ampMag));
                        const idx = this.coordToIdx(x, y);
                        this.re0[idx] = rem; this.im0[idx] = 0;
                        this.re1[idx] = ampMag * Math.cos(phase);
                        this.im1[idx] = ampMag * Math.sin(phase);
                        this.re2[idx] = 0; this.im2[idx] = 0;
                    }
                }
            }
        } else if (type === 'highlife_replicator') {
            const coords = [
                [1, 0], [2, 0], [3, 0],
                [0, 1], [3, 1],
                [0, 2], [3, 2],
                [0, 3], [1, 3], [2, 3], [3, 3],
                [2, 4]
            ];
            coords.forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX - 2 + dx, midY - 2 + dy);
                this.re0[idx] = 0; this.re1[idx] = 1;
            });
        } else if (type === 'turbulent_soup') {
            for (let i = 0; i < this.totalCells; i++) {
                const pAlive = Math.random();
                if (pAlive > 0.65) {
                    const phase = Math.random() * 2.0 * Math.PI;
                    const isFFS = pAlive > 0.88;
                    this.re0[i] = 0; this.im0[i] = 0;
                    if (isFFS) {
                        this.re1[i] = 0; this.im1[i] = 0;
                        this.re2[i] = Math.cos(phase); this.im2[i] = Math.sin(phase);
                    } else {
                        this.re1[i] = Math.cos(phase); this.im1[i] = Math.sin(phase);
                        this.re2[i] = 0; this.im2[i] = 0;
                    }
                }
            }
        } else if (type === 'dirac_wavepacket') {
            const px = 0.8, py = 0.4, sigma = 3.5;
            for (let y = 0; y < this.height; y++) {
                for (let x = 0; x < this.width; x++) {
                    const dx = x - midX, dy = y - midY;
                    const r2 = dx * dx + dy * dy;
                    const envelope = Math.exp(-r2 / (2.0 * sigma * sigma));
                    if (envelope > 0.04) {
                        const phase = px * dx + py * dy;
                        const amp1 = envelope * 0.8;
                        const amp2 = envelope * 0.4;
                        const remSq = Math.max(0.0, 1.0 - (amp1 * amp1 + amp2 * amp2));
                        const idx = this.coordToIdx(x, y);
                        this.re0[idx] = Math.sqrt(remSq); this.im0[idx] = 0;
                        this.re1[idx] = amp1 * Math.cos(phase);
                        this.im1[idx] = amp1 * Math.sin(phase);
                        this.re2[idx] = amp2 * Math.cos(phase + Math.PI * 0.5);
                        this.im2[idx] = amp2 * Math.sin(phase + Math.PI * 0.5);
                    }
                }
            }
        } else if (type === 'bz_spiral') {
            const armLen = 5;
            for (let d = 0; d < armLen; d++) {
                const idx1 = this.coordToIdx(midX + d, midY);
                this.re0[idx1] = 0; this.re1[idx1] = 1; this.re2[idx1] = 0;
                const idx2 = this.coordToIdx(midX, midY + d);
                this.re0[idx2] = 0; this.re1[idx2] = 0; this.im1[idx2] = 0;
                this.re2[idx2] = 0; this.im2[idx2] = 1;
            }
        } else if (type === 'pentadecathlon') {
            const offsets = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0]];
            offsets.forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX - 5 + dx, midY + dy);
                this.re0[idx] = 0; this.re1[idx] = 1;
            });
        } else if (type === 'ffs_bubble') {
            [[0, 0], [1, 0], [0, 1], [1, 1]].forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX - 1 + dx, midY - 1 + dy);
                this.re0[idx] = 0; this.re2[idx] = 1;
            });
        } else if (type === 'cat_blinker') {
            const centerIdx = this.coordToIdx(midX, midY);
            this.re0[centerIdx] = 0; this.re1[centerIdx] = 1;
            const invSqrt2 = 1.0 / Math.sqrt(2);
            [[0, -1], [0, 1], [-1, 0], [1, 0]].forEach(([dx, dy]) => {
                const idx = this.coordToIdx(midX + dx, midY + dy);
                this.re0[idx] = invSqrt2; this.re1[idx] = invSqrt2;
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
// 2. Rendering Pipelines (Isometric 3D Columns, Wavefield, Hilbert, Cosmic Chaos)
// =========================================================================

class QGolVisualizerDeck {
    constructor() {
        const selectGrid = document.getElementById('select-grid-dim');
        this.gridDim = selectGrid ? parseInt(selectGrid.value, 10) : 32;
        this.relativePhaseRad = Math.PI * 0.5;

        this.engine = new QuantumConwayEngine(this.gridDim, this.gridDim);
        this.engine.relativePhaseRad = this.relativePhaseRad;

        const selectPattern = document.getElementById('select-pattern');
        const initialPattern = selectPattern ? selectPattern.value : 'vortex_dipole';
        this.engine.injectPattern(initialPattern);

        // Canvases
        this.cvsIso = document.getElementById('canvas-iso');
        this.ctxIso = this.cvsIso ? this.cvsIso.getContext('2d') : null;

        this.cvsWave = document.getElementById('canvas-wave');
        this.ctxWave = this.cvsWave ? this.cvsWave.getContext('2d') : null;

        this.cvsMomentum = document.getElementById('canvas-momentum');
        this.ctxMomentum = this.cvsMomentum ? this.cvsMomentum.getContext('2d') : null;

        this.cvsHilbert = document.getElementById('canvas-hilbert');
        this.ctxHilbert = this.cvsHilbert ? this.cvsHilbert.getContext('2d') : null;

        this.cvsChaos = document.getElementById('canvas-chaos');
        this.ctxChaos = this.cvsChaos ? this.cvsChaos.getContext('2d') : null;

        // Isometric Camera State
        this.isoYaw = Math.PI / 4;
        this.isoPitch = Math.PI / 6;
        this.isoZoom = 1.0;
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };

        // Macroscopic Cosmic Viewport Camera State
        this.chaosPanX = 0;
        this.chaosPanY = 0;
        this.chaosZoom = 1.0;
        this.isChaosDragging = false;
        this.lastChaosMouse = { x: 0, y: 0 };
        this.chaosDragDist = 0;
        this.showFlow = true;

        // Streamline Tracer Particle Pool (280 particles)
        this.streamlines = [];
        this.initStreamlineParticles(280);

        // Simulation Loop State
        this.isPlaying = true;
        this.ticksPerSec = 8;
        this.lastTickTime = performance.now();
        this.activeView = 'panel-iso';
        this.cachedTelemetry = this.engine.gatherTelemetry();

        this.setupEventListeners();
        this.resizeCanvases();
        window.addEventListener('resize', () => this.resizeCanvases());

        // Kickoff Render Loop
        requestAnimationFrame((t) => this.renderLoop(t));
    }

    initStreamlineParticles(count) {
        this.streamlines = [];
        for (let i = 0; i < count; i++) {
            this.streamlines.push({
                x: Math.random() * this.engine.width,
                y: Math.random() * this.engine.height,
                life: Math.floor(Math.random() * 80),
                maxLife: 60 + Math.floor(Math.random() * 60),
                trail: []
            });
        }
    }

    respawnParticle(p) {
        // Find a random cell weighted towards living density
        const attempts = 6;
        let chosenX = Math.random() * this.engine.width;
        let chosenY = Math.random() * this.engine.height;
        for (let k = 0; k < attempts; k++) {
            const rx = Math.floor(Math.random() * this.engine.width);
            const ry = Math.floor(Math.random() * this.engine.height);
            const idx = this.engine.coordToIdx(rx, ry);
            if (this.engine.probLiving(idx) + this.engine.probFFS(idx) > 0.05) {
                chosenX = rx + Math.random() * 0.8 - 0.4;
                chosenY = ry + Math.random() * 0.8 - 0.4;
                break;
            }
        }
        p.x = (chosenX + this.engine.width) % this.engine.width;
        p.y = (chosenY + this.engine.height) % this.engine.height;
        p.life = 0;
        p.maxLife = 60 + Math.floor(Math.random() * 60);
        p.trail = [];
    }

    switchView(viewId) {
        this.activeView = viewId;
        document.querySelectorAll('.view-tab').forEach(t => {
            t.classList.toggle('active', t.getAttribute('data-view') === viewId);
        });
        document.querySelectorAll('.panel-card').forEach(p => {
            p.classList.toggle('active-view', p.id === viewId);
        });
        setTimeout(() => this.resizeCanvases(), 15);
    }

    resizeCanvases() {
        const dpr = window.devicePixelRatio || 1;
        [this.cvsIso, this.cvsWave, this.cvsMomentum, this.cvsHilbert, this.cvsChaos].forEach(c => {
            if (!c || !c.parentElement) return;
            const rect = c.parentElement.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                c.width = rect.width * dpr;
                c.height = rect.height * dpr;
            }
        });
    }

    setupEventListeners() {
        // Tab Clicks
        document.querySelectorAll('.view-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const targetView = tab.getAttribute('data-view');
                this.switchView(targetView);
            });
        });

        // Keyboard shortcuts: 1, 2, 3, 4
        window.addEventListener('keydown', (e) => {
            if (e.key === '1') this.switchView('panel-iso');
            else if (e.key === '2') this.switchView('panel-wave');
            else if (e.key === '3') this.switchView('panel-hilbert');
            else if (e.key === '4') this.switchView('panel-chaos');
        });

        // Isometric Drag & Zoom
        if (this.cvsIso) {
            this.dragDistance = 0;
            this.cvsIso.addEventListener('mousedown', (e) => {
                this.isDragging = true;
                this.lastMouse = { x: e.clientX, y: e.clientY };
                this.dragDistance = 0;
            });
            window.addEventListener('mouseup', (e) => {
                if (this.isDragging) {
                    this.isDragging = false;
                    if (this.dragDistance < 6) this.handleCanvasClick(e);
                }
            });
            window.addEventListener('mousemove', (e) => {
                if (!this.isDragging) return;
                const dx = e.clientX - this.lastMouse.x;
                const dy = e.clientY - this.lastMouse.y;
                this.dragDistance += Math.hypot(dx, dy);
                this.isoYaw += dx * 0.008;
                this.isoPitch = Math.max(0.1, Math.min(Math.PI / 2.2, this.isoPitch + dy * 0.008));
                this.lastMouse = { x: e.clientX, y: e.clientY };
            });
            this.cvsIso.addEventListener('wheel', (e) => {
                e.preventDefault();
                this.isoZoom = Math.max(0.5, Math.min(2.5, this.isoZoom - e.deltaY * 0.001));
            });
            this.cvsIso.addEventListener('contextmenu', (e) => e.preventDefault());
        }

        // Macroscopic Chaos Viewport Pan / Zoom / Click
        if (this.cvsChaos) {
            this.cvsChaos.addEventListener('mousedown', (e) => {
                this.isChaosDragging = true;
                this.lastChaosMouse = { x: e.clientX, y: e.clientY };
                this.chaosDragDist = 0;
            });
            window.addEventListener('mouseup', (e) => {
                if (this.isChaosDragging) {
                    this.isChaosDragging = false;
                    if (this.chaosDragDist < 6 && this.activeView === 'panel-chaos') {
                        this.handleChaosClick(e);
                    }
                }
            });
            window.addEventListener('mousemove', (e) => {
                if (!this.isChaosDragging) return;
                const dx = e.clientX - this.lastChaosMouse.x;
                const dy = e.clientY - this.lastChaosMouse.y;
                this.chaosDragDist += Math.hypot(dx, dy);
                this.chaosPanX += dx;
                this.chaosPanY += dy;
                this.lastChaosMouse = { x: e.clientX, y: e.clientY };
            });
            this.cvsChaos.addEventListener('wheel', (e) => {
                e.preventDefault();
                const factor = e.deltaY < 0 ? 1.15 : 0.87;
                this.chaosZoom = Math.max(0.2, Math.min(6.0, this.chaosZoom * factor));
            });
            this.cvsChaos.addEventListener('contextmenu', (e) => e.preventDefault());
        }

        // Toggle Flow Button
        const btnToggleFlow = document.getElementById('btn-toggle-flow');
        if (btnToggleFlow) {
            btnToggleFlow.addEventListener('click', () => {
                this.showFlow = !this.showFlow;
                btnToggleFlow.textContent = this.showFlow ? 'FLOW: ON' : 'FLOW: OFF';
                btnToggleFlow.classList.toggle('active', this.showFlow);
            });
        }

        // Rule Selector Dropdown
        const selectRule = document.getElementById('select-rule');
        if (selectRule) {
            selectRule.addEventListener('change', (e) => {
                this.engine.rule = e.target.value;
                this.cachedTelemetry = this.engine.gatherTelemetry();
                this.updateTelemetryHUD(this.cachedTelemetry);
            });
        }

        // Chaos K Slider
        const sliderChaos = document.getElementById('slider-chaos-k');
        const chaosVal = document.getElementById('chaos-val');
        if (sliderChaos && chaosVal) {
            sliderChaos.addEventListener('input', (e) => {
                const k = parseFloat(e.target.value) / 100.0;
                this.engine.chaosK = k;
                chaosVal.textContent = k.toFixed(2);
            });
        }

        // Gauge Flux Slider
        const sliderGauge = document.getElementById('slider-gauge-flux');
        const gaugeVal = document.getElementById('gauge-val');
        if (sliderGauge && gaugeVal) {
            sliderGauge.addEventListener('input', (e) => {
                const flux = parseFloat(e.target.value) / 100.0;
                this.engine.gaugeFlux = flux;
                gaugeVal.textContent = `${flux.toFixed(2)}Φ₀`;
            });
        }

        // Play / Step / Reset / Probe Buttons
        const btnPlay = document.getElementById('btn-play');
        if (btnPlay) {
            btnPlay.addEventListener('click', () => {
                this.isPlaying = !this.isPlaying;
                btnPlay.textContent = this.isPlaying ? 'PAUSE' : 'PLAY';
                btnPlay.classList.toggle('primary', this.isPlaying);
            });
        }

        const btnStep = document.getElementById('btn-step');
        if (btnStep) {
            btnStep.addEventListener('click', () => {
                this.cachedTelemetry = this.engine.step();
                this.updateTelemetryHUD(this.cachedTelemetry);
            });
        }

        const btnReset = document.getElementById('btn-reset');
        if (btnReset) {
            btnReset.addEventListener('click', () => {
                const pat = document.getElementById('select-pattern').value;
                this.engine.injectPattern(pat);
                this.cachedTelemetry = this.engine.gatherTelemetry();
                this.updateTelemetryHUD(this.cachedTelemetry);
            });
        }

        const btnProbe = document.getElementById('btn-born-probe');
        if (btnProbe) {
            btnProbe.addEventListener('click', () => {
                this.engine.bornRuleMeasurement();
                this.cachedTelemetry = this.engine.gatherTelemetry();
                this.updateTelemetryHUD(this.cachedTelemetry);
            });
        }

        const selectPattern = document.getElementById('select-pattern');
        if (selectPattern) {
            selectPattern.addEventListener('change', (e) => {
                this.engine.injectPattern(e.target.value);
                this.cachedTelemetry = this.engine.gatherTelemetry();
                this.updateTelemetryHUD(this.cachedTelemetry);
            });
        }

        const selectGrid = document.getElementById('select-grid-dim');
        if (selectGrid) {
            selectGrid.addEventListener('change', (e) => {
                this.gridDim = parseInt(e.target.value, 10);
                this.engine = new QuantumConwayEngine(this.gridDim, this.gridDim);
                this.engine.relativePhaseRad = this.relativePhaseRad;
                this.engine.rule = document.getElementById('select-rule')?.value || 'b3s23';
                this.engine.chaosK = parseFloat(document.getElementById('slider-chaos-k')?.value || '120') / 100.0;
                this.engine.gaugeFlux = parseFloat(document.getElementById('slider-gauge-flux')?.value || '0') / 100.0;
                const currentPat = document.getElementById('select-pattern')?.value || 'vortex_dipole';
                this.engine.injectPattern(currentPat);
                this.initStreamlineParticles(280);
                this.cachedTelemetry = this.engine.gatherTelemetry();
                this.updateTelemetryHUD(this.cachedTelemetry);
            });
        }

        const sliderPhase = document.getElementById('slider-phase');
        const phaseVal = document.getElementById('phase-val');
        if (sliderPhase && phaseVal) {
            sliderPhase.addEventListener('input', (e) => {
                const deg = parseFloat(e.target.value);
                this.relativePhaseRad = deg * (Math.PI / 180.0);
                this.engine.relativePhaseRad = this.relativePhaseRad;
                const piFrac = (deg / 180.0).toFixed(2);
                phaseVal.textContent = `${Math.round(deg)}° (${piFrac}π)`;
                const currentPat = document.getElementById('select-pattern').value;
                if (currentPat === 'glider_pair' || currentPat === 'beam_splitter') {
                    this.engine.injectPattern(currentPat);
                    this.cachedTelemetry = this.engine.gatherTelemetry();
                    this.updateTelemetryHUD(this.cachedTelemetry);
                }
            });
        }

        const sliderSpeed = document.getElementById('slider-speed');
        const speedVal = document.getElementById('speed-val');
        if (sliderSpeed && speedVal) {
            sliderSpeed.addEventListener('input', (e) => {
                this.ticksPerSec = parseInt(e.target.value);
                speedVal.textContent = `${this.ticksPerSec} TPS`;
            });
        }
    }

    handleCanvasClick(e) {
        if (!this.cvsIso) return;
        const rect = this.cvsIso.getBoundingClientRect();
        if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
            return;
        }

        const mouseX = (e.clientX - rect.left) * (this.cvsIso.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (this.cvsIso.height / rect.height);
        const w = this.cvsIso.width, h = this.cvsIso.height;
        const cx = w / 2, cy = h / 2 + 30;
        const size = Math.min(w, h) * 0.038 * this.isoZoom;

        const dx = mouseX - cx;
        const dy = mouseY - cy;
        const rotX = dx / (size * 2.2);
        const rotY = dy / (size * 1.2 * Math.sin(this.isoPitch));

        const nx = rotX * Math.cos(this.isoYaw) + rotY * Math.sin(this.isoYaw);
        const ny = -rotX * Math.sin(this.isoYaw) + rotY * Math.cos(this.isoYaw);

        const x = Math.round(nx + this.engine.width / 2);
        const y = Math.round(ny + this.engine.height / 2);

        if (x >= 0 && x < this.engine.width && y >= 0 && y < this.engine.height) {
            this.toggleCellState(x, y, e.shiftKey, e.button === 2 || e.altKey);
        }
    }

    handleChaosClick(e) {
        if (!this.cvsChaos) return;
        const rect = this.cvsChaos.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const screenX = (e.clientX - rect.left) * dpr;
        const screenY = (e.clientY - rect.top) * dpr;

        const w = this.cvsChaos.width;
        const h = this.cvsChaos.height;
        const cx = w * 0.5 + this.chaosPanX * dpr;
        const cy = h * 0.5 + this.chaosPanY * dpr;
        const cellSize = (Math.min(w, h) * 0.8 / Math.max(this.engine.width, this.engine.height)) * this.chaosZoom;

        const worldX = (screenX - cx) / cellSize + this.engine.width * 0.5;
        const worldY = (screenY - cy) / cellSize + this.engine.height * 0.5;

        const gx = Math.floor(worldX);
        const gy = Math.floor(worldY);

        if (gx >= 0 && gx < this.engine.width && gy >= 0 && gy < this.engine.height) {
            this.toggleCellState(gx, gy, e.shiftKey, e.button === 2 || e.altKey);
        }
    }

    toggleCellState(x, y, shiftKey, altKey) {
        const idx = this.engine.coordToIdx(x, y);
        if (shiftKey) {
            this.engine.re0[idx] = 1.0; this.engine.im0[idx] = 0.0;
            this.engine.re1[idx] = 0.0; this.engine.im1[idx] = 0.0;
            this.engine.re2[idx] = 0.0; this.engine.im2[idx] = 0.0;
        } else if (altKey) {
            this.engine.re0[idx] = 0.0; this.engine.im0[idx] = 0.0;
            this.engine.re1[idx] = 0.0; this.engine.im1[idx] = 0.0;
            this.engine.re2[idx] = 1.0; this.engine.im2[idx] = 0.0;
        } else {
            const isLiving = this.engine.probLiving(idx) > 0.5;
            this.engine.re0[idx] = isLiving ? 1.0 : 0.0;
            this.engine.im0[idx] = 0.0;
            this.engine.re1[idx] = isLiving ? 0.0 : 1.0;
            this.engine.im1[idx] = 0.0;
            this.engine.re2[idx] = 0.0;
            this.engine.im2[idx] = 0.0;
        }
        this.cachedTelemetry = this.engine.gatherTelemetry();
        this.updateTelemetryHUD(this.cachedTelemetry);
    }

    renderLoop(now) {
        const interval = 1000.0 / this.ticksPerSec;
        const elapsed = now - this.lastTickTime;
        if (this.isPlaying && elapsed >= interval) {
            this.cachedTelemetry = this.engine.step();
            this.updateTelemetryHUD(this.cachedTelemetry);
            this.lastTickTime = now;
        }

        if (this.activeView === 'panel-iso') {
            this.renderIsometricPanel();
        } else if (this.activeView === 'panel-wave') {
            this.renderDualDomainPanel();
        } else if (this.activeView === 'panel-hilbert') {
            this.renderHilbertChamberPanel();
        } else if (this.activeView === 'panel-chaos') {
            this.renderChaosPanel();
        }

        requestAnimationFrame((t) => this.renderLoop(t));
    }

    // ---------------------------------------------------------------------
    // Viewport 1: 3D Isometric Columns & Phase Needles
    // ---------------------------------------------------------------------
    renderIsometricPanel() {
        if (!this.ctxIso) return;
        const ctx = this.ctxIso;
        const w = this.cvsIso.width, h = this.cvsIso.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2 + 30;
        const size = Math.min(w, h) * 0.038 * this.isoZoom;

        const cells = [];
        for (let y = 0; y < this.engine.height; y++) {
            for (let x = 0; x < this.engine.width; x++) {
                const idx = this.engine.coordToIdx(x, y);
                const nx = x - this.engine.width / 2;
                const ny = y - this.engine.height / 2;
                const rotX = nx * Math.cos(this.isoYaw) - ny * Math.sin(this.isoYaw);
                const rotY = nx * Math.sin(this.isoYaw) + ny * Math.cos(this.isoYaw);
                const depth = rotY * Math.cos(this.isoPitch);
                cells.push({ x, y, idx, rotX, rotY, depth });
            }
        }
        cells.sort((a, b) => a.depth - b.depth);

        cells.forEach(cell => {
            const screenX = cx + cell.rotX * size * 2.2;
            const screenY = cy + cell.rotY * size * 1.2 * Math.sin(this.isoPitch);

            const p1 = this.engine.probLiving(cell.idx);
            const p2 = this.engine.probFFS(cell.idx);
            const phase = this.engine.cellPhase(cell.idx);

            const colHeight = (p1 * 70 + p2 * 110) * this.isoZoom;
            const ffsRatio = (p1 + p2) > 0 ? (p2 / (p1 + p2)) : 0;

            if (colHeight > 2) {
                ctx.fillStyle = ffsRatio > 0.4 ? 'rgba(255, 179, 0, 0.75)' : 'rgba(0, 230, 118, 0.75)';
                ctx.strokeStyle = ffsRatio > 0.4 ? '#ffb300' : '#00e676';
                ctx.lineWidth = 1;

                const r = size * 0.7;
                ctx.beginPath();
                ctx.moveTo(screenX - r, screenY);
                ctx.lineTo(screenX + r, screenY);
                ctx.lineTo(screenX + r, screenY - colHeight);
                ctx.lineTo(screenX - r, screenY - colHeight);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = '#0f172a';
                ctx.beginPath();
                ctx.arc(screenX, screenY - colHeight, r, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

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
        if (!this.ctxWave || !this.ctxMomentum) return;
        const ctxW = this.ctxWave;
        const wW = this.cvsWave.width, hW = this.cvsWave.height;
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
                data[idx] = 0;
                data[idx + 1] = intensity;
                data[idx + 2] = Math.floor(intensity * 0.4);
                data[idx + 3] = 255;
            }
        }
        ctxW.putImageData(imgData, 0, 0);

        const ctxM = this.ctxMomentum;
        const wM = this.cvsMomentum.width, hM = this.cvsMomentum.height;
        ctxM.clearRect(0, 0, wM, hM);

        const meanNk = this.engine.computeCentralMomentumDensity();
        const barW = wM * 0.45;
        const barH = Math.min(hM * 0.85, (meanNk / 0.5) * (hM * 0.65));
        const barX = (wM - barW) / 2;
        const barY = hM * 0.85 - barH;

        ctxM.fillStyle = 'rgba(255, 179, 0, 0.25)';
        ctxM.fillRect(barX, barY, barW, barH);
        ctxM.strokeStyle = '#ffb300';
        ctxM.lineWidth = 2;
        ctxM.strokeRect(barX, barY, barW, barH);

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
        if (!this.ctxHilbert) return;
        const ctx = this.ctxHilbert;
        const w = this.cvsHilbert.width, h = this.cvsHilbert.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2 + 20;
        const scale = Math.min(w, h) * 0.38;

        const v0 = { x: cx - scale, y: cy + scale * 0.6 };
        const v1 = { x: cx + scale, y: cy + scale * 0.6 };
        const v2 = { x: cx, y: cy - scale * 0.85 };

        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(v0.x, v0.y);
        ctx.lineTo(v1.x, v1.y);
        ctx.lineTo(v2.x, v2.y);
        ctx.closePath();
        ctx.stroke();

        ctx.font = '11px monospace';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText('|0⟩ VACUUM', v0.x - 30, v0.y + 18);
        ctx.fillStyle = '#00e676';
        ctx.fillText('|1⟩ LIVING', v1.x - 20, v1.y + 18);
        ctx.fillStyle = '#ffb300';
        ctx.fillText('FFS |2⟩', v2.x - 22, v2.y - 10);

        ctx.strokeStyle = 'rgba(255, 179, 0, 0.35)';
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy + scale * 0.6);
        ctx.lineTo(v2.x, v2.y);
        ctx.stroke();
        ctx.setLineDash([]);

        for (let i = 0; i < this.engine.totalCells; i++) {
            const p0 = this.engine.re0[i] ** 2 + this.engine.im0[i] ** 2;
            const p1 = this.engine.probLiving(i);
            const p2 = this.engine.probFFS(i);

            if (p1 > 0.05 || p2 > 0.05) {
                const px = p0 * v0.x + p1 * v1.x + p2 * v2.x;
                const py = p0 * v0.y + p1 * v1.y + p2 * v2.y;

                const isFFS = p2 > 0.3;
                ctx.fillStyle = isFFS ? '#ffb300' : '#00e676';
                ctx.beginPath();
                ctx.arc(px, py, isFFS ? 3.5 : 2.5, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = isFFS ? 'rgba(255, 179, 0, 0.25)' : 'rgba(0, 230, 118, 0.25)';
                ctx.beginPath();
                ctx.arc(cx, py, Math.abs(px - cx), 0, Math.PI * 2);
                ctx.stroke();
            }
        }
    }

    // ---------------------------------------------------------------------
    // Viewport 4: Macroscopic Cosmic Viewport & Vortex Circulation
    // ---------------------------------------------------------------------
    renderChaosPanel() {
        if (!this.ctxChaos) return;
        const ctx = this.ctxChaos;
        const w = this.cvsChaos.width;
        const h = this.cvsChaos.height;
        const dpr = window.devicePixelRatio || 1;

        // 1. Root Canvas: Deep Obsidian (#05080d)
        ctx.fillStyle = '#05080d';
        ctx.fillRect(0, 0, w, h);

        const cx = w * 0.5 + this.chaosPanX * dpr;
        const cy = h * 0.5 + this.chaosPanY * dpr;
        const maxDim = Math.max(this.engine.width, this.engine.height);
        const cellSize = (Math.min(w, h) * 0.78 / maxDim) * this.chaosZoom;
        const gridW = this.engine.width * cellSize;
        const gridH = this.engine.height * cellSize;
        const startX = cx - gridW * 0.5;
        const startY = cy - gridH * 0.5;

        // 2. Macroscopic Coordinate Reticle Substrate
        ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
        ctx.lineWidth = 1;
        const step = Math.max(1, Math.round(4 / this.chaosZoom));
        for (let gx = 0; gx <= this.engine.width; gx += step) {
            const lx = startX + gx * cellSize;
            ctx.beginPath();
            ctx.moveTo(lx, startY);
            ctx.lineTo(lx, startY + gridH);
            ctx.stroke();
        }
        for (let gy = 0; gy <= this.engine.height; gy += step) {
            const ly = startY + gy * cellSize;
            ctx.beginPath();
            ctx.moveTo(startX, ly);
            ctx.lineTo(startX + gridW, ly);
            ctx.stroke();
        }

        // Outer Lattice Bounding Border with Emerald Corner Reticles
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(startX, startY, gridW, gridH);

        const cornerLen = Math.min(24 * dpr, 0.1 * gridW);
        ctx.strokeStyle = '#00e676';
        ctx.lineWidth = 2;
        // Top-Left
        ctx.beginPath();
        ctx.moveTo(startX, startY + cornerLen);
        ctx.lineTo(startX, startY);
        ctx.lineTo(startX + cornerLen, startY);
        ctx.stroke();
        // Top-Right
        ctx.beginPath();
        ctx.moveTo(startX + gridW - cornerLen, startY);
        ctx.lineTo(startX + gridW, startY);
        ctx.lineTo(startX + gridW, startY + cornerLen);
        ctx.stroke();
        // Bottom-Left
        ctx.beginPath();
        ctx.moveTo(startX, startY + gridH - cornerLen);
        ctx.lineTo(startX, startY + gridH);
        ctx.lineTo(startX + cornerLen, startY + gridH);
        ctx.stroke();
        // Bottom-Right
        ctx.beginPath();
        ctx.moveTo(startX + gridW - cornerLen, startY + gridH);
        ctx.lineTo(startX + gridW, startY + gridH);
        ctx.lineTo(startX + gridW, startY + gridH - cornerLen);
        ctx.stroke();

        // 3. Fluid Wavefunction Plasma Rendering
        for (let y = 0; y < this.engine.height; y++) {
            for (let x = 0; x < this.engine.width; x++) {
                const idx = this.engine.coordToIdx(x, y);
                const p1 = this.engine.probLiving(idx);
                const p2 = this.engine.probFFS(idx);
                const density = p1 + 2.0 * p2;

                if (density > 0.01) {
                    const px = startX + x * cellSize;
                    const py = startY + y * cellSize;
                    const ffsFrac = (p1 + p2) > 0 ? (p2 / (p1 + p2)) : 0;

                    // Blend Laser Emerald (#00e676) and Radiant Amber (#ffb300)
                    const r = Math.round(ffsFrac * 255);
                    const g = Math.round((1.0 - ffsFrac * 0.3) * 230);
                    const b = Math.round((1.0 - ffsFrac) * 118);
                    const alpha = Math.min(0.9, Math.max(0.12, density * 0.8));

                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                    ctx.fillRect(px, py, cellSize, cellSize);

                    // Phase Vector Needle Tick
                    if (cellSize > 8) {
                        const phase = this.engine.cellPhase(idx);
                        const midCellX = px + cellSize * 0.5;
                        const midCellY = py + cellSize * 0.5;
                        const tickLen = cellSize * 0.45;
                        ctx.strokeStyle = 'rgba(248, 250, 252, 0.7)';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(midCellX, midCellY);
                        ctx.lineTo(midCellX + tickLen * Math.cos(phase), midCellY + tickLen * Math.sin(phase));
                        ctx.stroke();
                    }
                }
            }
        }

        // 4. Streamline Particle Advection along Probability Current j(x,y)
        const telem = this.cachedTelemetry;
        if (this.showFlow && telem && telem.currents) {
            const currents = telem.currents;
            ctx.lineWidth = 1.5;

            for (let i = 0; i < this.streamlines.length; i++) {
                const p = this.streamlines[i];
                p.life++;
                if (p.life > p.maxLife || p.x < 0 || p.x >= this.engine.width || p.y < 0 || p.y >= this.engine.height) {
                    this.respawnParticle(p);
                    continue;
                }

                // Sample local current
                const gx = Math.floor(p.x);
                const gy = Math.floor(p.y);
                const idx = this.engine.coordToIdx(gx, gy);
                const [jx, jy] = currents[idx];
                const speed = Math.hypot(jx, jy);

                // Save screen position for trail
                const scrX = startX + p.x * cellSize;
                const scrY = startY + p.y * cellSize;
                p.trail.push({ x: scrX, y: scrY });
                if (p.trail.length > 5) p.trail.shift();

                // Draw glowing streamline trail
                if (p.trail.length > 1) {
                    ctx.strokeStyle = speed > 0.08 ? 'rgba(255, 179, 0, 0.65)' : 'rgba(0, 230, 118, 0.55)';
                    ctx.beginPath();
                    ctx.moveTo(p.trail[0].x, p.trail[0].y);
                    for (let t = 1; t < p.trail.length; t++) {
                        ctx.lineTo(p.trail[t].x, p.trail[t].y);
                    }
                    ctx.stroke();
                }

                // Advance particle position
                const stepScale = 2.4;
                p.x = (p.x + jx * stepScale + this.engine.width) % this.engine.width;
                p.y = (p.y + jy * stepScale + this.engine.height) % this.engine.height;
            }
        }

        // 5. Topological Vortex Reticles (Laser Emerald for +1, Crimson Red for -1)
        if (telem && telem.vorticities) {
            const vorts = telem.vorticities;
            const now = performance.now() * 0.003;

            for (let y = 1; y < this.engine.height - 1; y++) {
                for (let x = 1; x < this.engine.width - 1; x++) {
                    const idx = this.engine.coordToIdx(x, y);
                    const v = vorts[idx];
                    const vAbs = Math.abs(v);

                    // Local extrema threshold
                    if (vAbs > 0.16) {
                        const isMax = v > 0 &&
                            v >= vorts[this.engine.coordToIdx(x+1, y)] &&
                            v >= vorts[this.engine.coordToIdx(x-1, y)] &&
                            v >= vorts[this.engine.coordToIdx(x, y+1)] &&
                            v >= vorts[this.engine.coordToIdx(x, y-1)];
                        const isMin = v < 0 &&
                            v <= vorts[this.engine.coordToIdx(x+1, y)] &&
                            v <= vorts[this.engine.coordToIdx(x-1, y)] &&
                            v <= vorts[this.engine.coordToIdx(x, y+1)] &&
                            v <= vorts[this.engine.coordToIdx(x, y-1)];

                        if (isMax || isMin) {
                            const vx = startX + (x + 0.5) * cellSize;
                            const vy = startY + (y + 0.5) * cellSize;
                            const isPositive = v > 0;
                            const color = isPositive ? '#00e676' : '#ef4444';
                            const rotSign = isPositive ? 1 : -1;
                            const reticleRadius = Math.max(12 * dpr, cellSize * 1.4);

                            // Rotating Reticle Ring
                            ctx.save();
                            ctx.translate(vx, vy);
                            ctx.rotate(now * rotSign);

                            ctx.strokeStyle = color;
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath();
                            ctx.arc(0, 0, reticleRadius, 0, Math.PI * 2);
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Directional Ticks
                            ctx.beginPath();
                            ctx.moveTo(-reticleRadius, 0); ctx.lineTo(-reticleRadius + 4, -4);
                            ctx.moveTo(reticleRadius, 0); ctx.lineTo(reticleRadius - 4, 4);
                            ctx.stroke();

                            ctx.restore();

                            // Core Dot & Winding Number Badge
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.arc(vx, vy, 3.5 * dpr, 0, Math.PI * 2);
                            ctx.fill();

                            ctx.font = `bold ${Math.max(10, Math.round(11 * dpr))}px monospace`;
                            ctx.fillStyle = '#f8fafc';
                            ctx.fillText(isPositive ? '+1' : '-1', vx + 8 * dpr, vy - 8 * dpr);
                        }
                    }
                }
            }
        }

        // 6. OTOC Causal Scrambling Horizon
        if (telem && telem.scramblingOtoc > 0.0) {
            const otocR = Math.sqrt(telem.scramblingOtoc) * cellSize;
            ctx.strokeStyle = 'rgba(255, 179, 0, 0.7)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 6]);
            ctx.beginPath();
            ctx.arc(cx, cy, otocR, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#ffb300';
            ctx.font = `${Math.max(9, Math.round(10 * dpr))}px monospace`;
            ctx.fillText(`OTOC HORIZON R=${Math.sqrt(telem.scramblingOtoc).toFixed(2)}`, cx + otocR + 6, cy);
        }

        // 7. HUD Telemetry Overlay (Screen Space)
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(16, 16, 260 * dpr, 100 * dpr);
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.strokeRect(16, 16, 260 * dpr, 100 * dpr);

        ctx.font = `${Math.max(10, Math.round(11 * dpr))}px monospace`;
        ctx.fillStyle = '#f8fafc';
        ctx.fillText(`VIEWPORT: ${this.engine.width}×${this.engine.height} QUBIT MACRO`, 26, 36 * dpr);
        ctx.fillStyle = '#00e676';
        ctx.fillText(`ZOOM: ${this.chaosZoom.toFixed(2)}x | PAN: [${Math.round(this.chaosPanX)}, ${Math.round(this.chaosPanY)}]`, 26, 54 * dpr);
        ctx.fillStyle = '#ffb300';
        ctx.fillText(`RULE: ${this.engine.rule.toUpperCase()} | K=${this.engine.chaosK.toFixed(2)}`, 26, 72 * dpr);
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`STREAMLINES: ${this.showFlow ? 'ACTIVE' : 'OFF'} (280 TRACERS)`, 26, 90 * dpr);
    }

    updateTelemetryHUD(telem) {
        if (!telem) return;
        const setTxt = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setTxt('hud-step', String(telem.step).padStart(4, '0'));
        setTxt('hud-tot-pop', telem.totalPop.toFixed(2));
        setTxt('hud-live-pop', telem.livingPop.toFixed(2));
        setTxt('hud-ffs-pop', telem.ffsPop.toFixed(2));
        setTxt('hud-norm-err', telem.maxNormErr.toExponential(2));
        setTxt('hud-entropy', `${telem.entropy.toFixed(4)} nats`);
        setTxt('hud-momentum', telem.meanMomentum.toFixed(4));
        setTxt('hud-vorticity', telem.maxVorticity.toFixed(4));
        setTxt('hud-vortex-count', String(telem.vortexCount));
        setTxt('hud-otoc', telem.scramblingOtoc.toFixed(4));

        setTxt('badge-vortices', `${telem.vortexCount} VORTICES`);
        setTxt('badge-scrambling', `OTOC: ${telem.scramblingOtoc.toFixed(4)}`);
    }
}

// Instantiate Visualizer Deck on DOM load
window.addEventListener('DOMContentLoaded', () => {
    new QGolVisualizerDeck();
});
