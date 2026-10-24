// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

// Run: node web/test_lab_geom.js
const assert = require('assert');
const G = require('./lab_geom.js');

let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok  ' + name); };
const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('Hopf map sends a spinor to the Bloch vector of its angles, on a unit sphere', () => {
    for (const [th, ph] of [[0.3, 1.1], [1.7, -2.0], [Math.PI, 0.4], [2.5, 3.0]]) {
        const [z1, z2] = G.spinor(th, ph, 0.77);
        const b = G.hopf(z1, z2);
        close(b[0], Math.sin(th) * Math.cos(ph));
        close(b[1], Math.sin(th) * Math.sin(ph));
        close(b[2], Math.cos(th));
        close(Math.hypot(...b), 1);
    }
});

test('every point of a fibre maps to the same base point (the fibre is a global-phase circle)', () => {
    const th = 1.2, ph = 0.5;
    const ref = G.hopf(...G.spinor(th, ph, 0));
    for (let k = 0; k < 20; k++) {
        const b = G.hopf(...G.spinor(th, ph, (2 * Math.PI * k) / 20));
        close(b[0], ref[0]); close(b[1], ref[1]); close(b[2], ref[2]);
    }
});

test('S3 points are unit 4-vectors', () => {
    const [z1, z2] = G.spinor(0.9, 2.2, 1.4);
    close(Math.hypot(...G.s3(z1, z2)), 1);
});

test('two Hopf fibres over different base points are linked once', () => {
    const a = G.hopfFiber(1.0, 0.0, 180);
    const b = G.hopfFiber(2.0, 2.0, 180);
    const lk = G.linkingNumber(a, b);
    assert.ok(Math.abs(Math.abs(lk) - 1) < 0.05, 'linking number ' + lk);
    // and a fibre is not linked with a far-away, small circle
    const far = Array.from({ length: 60 }, (_, i) => [40 + Math.cos(i / 60 * 2 * Math.PI), 40 + Math.sin(i / 60 * 2 * Math.PI), 40]);
    assert.ok(Math.abs(G.linkingNumber(a, far)) < 0.05);
});

test('Bloch-sphere gates', () => {
    const plus = G.applyGate(G.GATES.H, [[1, 0], [0, 0]]);
    const b = G.hopf(...plus);
    close(b[0], 1); close(b[2], 0);
    const y = G.hopf(...G.applyGate(G.GATES.S, plus));
    close(y[1], 1);                                       // S takes +x to +y
    const back = G.hopf(...G.applyGate(G.GATES.H, G.applyGate(G.GATES.H, [[1, 0], [0, 0]])));
    close(back[2], 1);                                    // H H = I
    const ry = G.hopf(...G.applyGate(G.rotation('y', Math.PI / 2), [[1, 0], [0, 0]]));
    close(ry[0], 1);                                      // Ry(pi/2) takes +z to +x
});

test('regular 4-polytopes have the right vertex and edge counts and edge length', () => {
    const expected = { '5-cell': [5, 10], 'tesseract': [16, 32], '16-cell': [8, 24], '24-cell': [24, 96] };
    for (const [name, [v, e]] of Object.entries(expected)) {
        const p = G.polytope(name);
        assert.strictEqual(p.vertices.length, v, name + ' vertices');
        assert.strictEqual(p.edges.length, e, name + ' edges');
    }
    close(G.polytope('tesseract').edgeLength, 2);
    close(G.polytope('16-cell').edgeLength, Math.SQRT2);
    close(G.polytope('24-cell').edgeLength, Math.SQRT2);
    // every vertex of a regular polytope has the same degree
    for (const name of ['5-cell', 'tesseract', '16-cell', '24-cell']) {
        const p = G.polytope(name);
        const deg = new Array(p.vertices.length).fill(0);
        p.edges.forEach(([i, j]) => { deg[i]++; deg[j]++; });
        assert.strictEqual(new Set(deg).size, 1, name + ' regular');
    }
});

test('Euler characteristic of the 4-polytope boundary: V - E + F - C = 0', () => {
    // faces and cells from the known counts of the regular polytopes
    const counts = { '5-cell': [5, 10, 10, 5], 'tesseract': [16, 32, 24, 8], '16-cell': [8, 24, 32, 16], '24-cell': [24, 96, 96, 24] };
    for (const [name, [v, e, f, c]] of Object.entries(counts)) {
        assert.strictEqual(v - e + f - c, 0, name);
        assert.strictEqual(G.polytope(name).vertices.length, v);
        assert.strictEqual(G.polytope(name).edges.length, e);
    }
});

test('4D rotations preserve length and are orthogonal', () => {
    const r = G.rotation4([0.3, -0.7, 1.1, 0.2, 2.0, -1.3]);
    const v = [0.4, -1.2, 2.5, 0.9];
    close(Math.hypot(...G.matVec(r, v)), Math.hypot(...v));
    const rt = r[0].map((_, j) => r.map((row) => row[j]));
    const id = G.matMul(rt, r);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) close(id[i][j], i === j ? 1 : 0);
    // the 24-cell maps to itself under a 90-degree rotation in the xy plane
    const p = G.polytope('24-cell');
    const q = G.planeRotation(0, 1, Math.PI / 2);
    const keys = new Set(p.vertices.map((x) => x.join(',')));
    p.vertices.forEach((x) => assert.ok(keys.has(G.matVec(q, x).map((c) => Math.round(c)).join(','))));
});

console.log(`\n${n} test groups passed`);
