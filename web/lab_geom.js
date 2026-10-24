// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/*
 * Geometry for the bench: the Hopf map S3 -> S2, Bloch-sphere gates, regular 4-polytopes and 4D rotations.
 * Pure functions (no DOM) so the page and test_lab_geom.js load the same file.
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.Geom = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    // ---- complex numbers as [re, im] ---------------------------------------------------------------
    const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
    const cconj = (a) => [a[0], -a[1]];
    const cabs2 = (a) => a[0] * a[0] + a[1] * a[1];

    /** Hopf map: (z1, z2) on S3 -> Bloch vector on S2. */
    function hopf(z1, z2) {
        const w = cmul(cconj(z1), z2);
        return [2 * w[0], 2 * w[1], cabs2(z1) - cabs2(z2)];
    }

    /** Spinor (z1, z2) for polar angle theta and azimuth phi, with global phase chi (the fibre coordinate). */
    function spinor(theta, phi, chi = 0) {
        const c = Math.cos(theta / 2), s = Math.sin(theta / 2);
        return [[c * Math.cos(chi), c * Math.sin(chi)], [s * Math.cos(phi + chi), s * Math.sin(phi + chi)]];
    }

    /** Point of S3 as 4-vector (Re z1, Im z1, Re z2, Im z2). */
    const s3 = (z1, z2) => [z1[0], z1[1], z2[0], z2[1]];

    /** Stereographic projection of S3 from the pole (0,0,0,1) to R3. */
    function stereo(p) {
        const d = 1 - p[3];
        return [p[0] / d, p[1] / d, p[2] / d];
    }

    /** The fibre over the Bloch point (theta, phi), projected to R3, as a closed polyline of n points. */
    function hopfFiber(theta, phi, n = 96) {
        const pts = [];
        for (let i = 0; i < n; i++) {
            const chi = (2 * Math.PI * i) / n;
            const [z1, z2] = spinor(theta, phi, chi);
            pts.push(stereo(s3(z1, z2)));
        }
        return pts;
    }

    /** Discrete Gauss linking number of two closed polylines in R3. */
    function linkingNumber(a, b) {
        const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
        const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            const a0 = a[i], a1 = a[(i + 1) % a.length];
            const da = sub(a1, a0), ma = [(a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2, (a0[2] + a1[2]) / 2];
            for (let j = 0; j < b.length; j++) {
                const b0 = b[j], b1 = b[(j + 1) % b.length];
                const db = sub(b1, b0), mb = [(b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2, (b0[2] + b1[2]) / 2];
                const r = sub(ma, mb);
                const d = Math.hypot(r[0], r[1], r[2]);
                sum += dot(cross(da, db), r) / (d * d * d);
            }
        }
        return sum / (4 * Math.PI);
    }

    // ---- single-qubit gates on a spinor -----------------------------------------------------------------
    const R2 = Math.SQRT1_2;
    const GATES = {
        H: [[[R2, 0], [R2, 0]], [[R2, 0], [-R2, 0]]],
        X: [[[0, 0], [1, 0]], [[1, 0], [0, 0]]],
        Y: [[[0, 0], [0, -1]], [[0, 1], [0, 0]]],
        Z: [[[1, 0], [0, 0]], [[0, 0], [-1, 0]]],
        S: [[[1, 0], [0, 0]], [[0, 0], [0, 1]]],
        T: [[[1, 0], [0, 0]], [[0, 0], [R2, R2]]],
    };

    function applyGate(u, z) {
        return [cadd(cmul(u[0][0], z[0]), cmul(u[0][1], z[1])), cadd(cmul(u[1][0], z[0]), cmul(u[1][1], z[1]))];
    }

    function rotation(axis, angle) {
        const c = Math.cos(angle / 2), s = Math.sin(angle / 2);
        if (axis === 'x') return [[[c, 0], [0, -s]], [[0, -s], [c, 0]]];
        if (axis === 'y') return [[[c, 0], [-s, 0]], [[s, 0], [c, 0]]];
        return [[[c, -s], [0, 0]], [[0, 0], [c, s]]];
    }

    // ---- 4D polytopes ------------------------------------------------------------------------------------------
    function permutations(arr) {
        if (arr.length <= 1) return [arr];
        const out = new Map();
        arr.forEach((v, i) => permutations([...arr.slice(0, i), ...arr.slice(i + 1)]).forEach((p) => out.set([v, ...p].join(','), [v, ...p])));
        return [...out.values()];
    }

    function polytope(name) {
        let v = [];
        if (name === 'tesseract') {
            for (let m = 0; m < 16; m++) v.push([0, 1, 2, 3].map((b) => ((m >> b) & 1 ? 1 : -1)));
        } else if (name === '16-cell') {
            for (let i = 0; i < 4; i++) for (const s of [-1, 1]) { const p = [0, 0, 0, 0]; p[i] = s; v.push(p); }
        } else if (name === '24-cell') {
            const seen = new Set();
            for (const p of permutations([1, 1, 0, 0])) {
                for (let m = 0; m < 4; m++) {
                    const q = p.slice(); let bit = 0;
                    for (let i = 0; i < 4; i++) if (q[i]) { if ((m >> bit) & 1) q[i] = -1; bit++; }
                    const key = q.join(',');
                    if (!seen.has(key)) { seen.add(key); v.push(q); }
                }
            }
        } else if (name === '5-cell') {
            const s = 1 / Math.sqrt(5);
            v = [[1, 1, 1, -s], [1, -1, -1, -s], [-1, 1, -1, -s], [-1, -1, 1, -s], [0, 0, 0, 4 * s]];
        } else throw new Error('unknown polytope ' + name);
        const edges = [];
        let dmin = Infinity;
        const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]);
        for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) dmin = Math.min(dmin, dist(v[i], v[j]));
        for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) if (Math.abs(dist(v[i], v[j]) - dmin) < 1e-9) edges.push([i, j]);
        return { name, vertices: v, edges, edgeLength: dmin };
    }

    /** 4x4 rotation by `angle` in the coordinate plane (i, j). */
    function planeRotation(i, j, angle) {
        const m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        const c = Math.cos(angle), s = Math.sin(angle);
        m[i][i] = c; m[j][j] = c; m[i][j] = -s; m[j][i] = s;
        return m;
    }

    const matMul = (a, b) => a.map((row, i) => b[0].map((_, j) => row.reduce((acc, _x, k) => acc + a[i][k] * b[k][j], 0)));
    const matVec = (m, v) => m.map((row) => row.reduce((acc, x, k) => acc + x * v[k], 0));

    /** Composite rotation from six plane angles in the order xy, xz, xw, yz, yw, zw. */
    function rotation4(angles) {
        const planes = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
        let m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        planes.forEach(([i, j], k) => { m = matMul(planeRotation(i, j, angles[k] || 0), m); });
        return m;
    }

    /** Perspective projection of a 4D point to 3D from a viewer at distance d on the w axis. */
    function project4(p, d = 3) {
        const k = d / (d - p[3]);
        return [p[0] * k, p[1] * k, p[2] * k];
    }

    return {
        hopf, spinor, s3, stereo, hopfFiber, linkingNumber, GATES, applyGate, rotation,
        polytope, planeRotation, rotation4, project4, matVec, matMul,
    };
});
