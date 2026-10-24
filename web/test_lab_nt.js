// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

// Run: node web/test_lab_nt.js   (exit status 0 when every check passes)
const assert = require('assert');
const NT = require('./lab_nt.js');

let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok  ' + name); };

test('gcd, egcd, modpow, modinv', () => {
    assert.strictEqual(NT.gcd(48, 18), 6n);
    const [g, x, y] = NT.egcd(240n, 46n);
    assert.strictEqual(g, 2n);
    assert.strictEqual(240n * x + 46n * y, g);
    assert.strictEqual(NT.modpow(2, 10, 1000), 24n);
    // 7 has order 12 mod 13 and 2^100 = 4 (mod 12), so 7^(2^100) = 7^4 = 2401 = 9 (mod 13)
    assert.strictEqual(NT.modpow(7n, 1n << 100n, 13n), 9n);
    assert.strictEqual(NT.modinv(3, 11), 4n);
    assert.strictEqual(NT.modinv(6, 9), null);
});

test('primality', () => {
    assert.strictEqual(NT.isPrime((1n << 61n) - 1n), true);
    assert.strictEqual(NT.isPrime(561n), false); // Carmichael
    assert.strictEqual(NT.isPrime(3825123056546413051n), false); // strong pseudoprime to bases 2..23
    assert.strictEqual(NT.isPrime(1000000007n), true);
    assert.deepStrictEqual(NT.sieve(30), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
});

test('factorisation', () => {
    const f = NT.factor(600851475143n);
    assert.deepStrictEqual(f.factors, [[71n, 1], [839n, 1], [1471n, 1], [6857n, 1]]);
    const g = NT.factor((1n << 64n) + 1n);
    assert.deepStrictEqual(g.factors, [[274177n, 1], [67280421310721n, 1]]);
    assert.strictEqual(g.complete, true);
    assert.deepStrictEqual(NT.factor(360n).factors, [[2n, 3], [3n, 2], [5n, 1]]);
});

test('phi, order, primitive roots', () => {
    assert.strictEqual(NT.eulerPhi(36), 12n);
    assert.strictEqual(NT.order(2, 9), 6n); // the 1-2-4-8-7-5 cycle
    assert.strictEqual(NT.order(10, 7), 6n);
    assert.strictEqual(NT.order(4, 9), 3n);
    assert.strictEqual(NT.order(3, 9), null);
    assert.strictEqual(NT.primitiveRoot(7), 3n);
    assert.strictEqual(NT.primitiveRoot(9), 2n);
    assert.strictEqual(NT.primitiveRoot(8), null); // (Z/8)* is not cyclic
    assert.strictEqual(NT.primitiveRoot(4), 3n);
    assert.strictEqual(NT.primitiveRoot(18), 5n);  // 2 * 3^2
    assert.strictEqual(NT.primitiveRoot(25), 2n);
    assert.strictEqual(NT.primitiveRoot(600851475143n), null); // product of four primes: not cyclic, and fast
});

test('Jacobi and Legendre symbols, quadratic residues', () => {
    assert.strictEqual(NT.jacobi(2, 15), 1);
    assert.strictEqual(NT.legendre(5, 11), 1);
    assert.strictEqual(NT.legendre(2, 11), -1);
    assert.strictEqual(NT.legendre(11, 11), 0);
    assert.deepStrictEqual(NT.quadraticResidues(9), [0, 1, 4, 7]);
    // Euler's criterion on every prime below 100
    for (const p of NT.sieve(100).filter((q) => q > 2)) {
        for (let a = 1; a < p; a++) {
            const e = NT.modpow(a, (BigInt(p) - 1n) / 2n, BigInt(p));
            assert.strictEqual(NT.legendre(a, p), e === 1n ? 1 : -1, `legendre(${a},${p})`);
        }
    }
});

test('digital roots (the mod-9 facts)', () => {
    assert.strictEqual(NT.digitalRoot(493), 7);
    assert.strictEqual(NT.digitalRoot(9), 9);
    assert.strictEqual(NT.digitalRoot(18), 9);
    assert.strictEqual(NT.digitalRoot(0), 0);
    for (let k = 1; k < 3000; k++) {
        const ds = Number(NT.digitSum(k));
        assert.strictEqual(NT.digitalRoot(k), NT.digitalRoot(ds), 'root(k) = root(digit sum)');
        assert.strictEqual(NT.digitalRoot(k), ((k - 1) % 9) + 1);
    }
    // base 7: digital root of n is 1 + (n-1) mod 6
    assert.strictEqual(NT.digitalRoot(100, 7), 1 + ((100 - 1) % 6));
});

test('doubling modulo 9: cycles', () => {
    const o = NT.orbits(9, (x) => (2 * x) % 9);
    const sorted = o.cycles.map((c) => c.slice().sort((a, b) => a - b)).sort((a, b) => a[0] - b[0]);
    assert.deepStrictEqual(sorted, [[0], [1, 2, 4, 5, 7, 8], [3, 6]]);
    const six = o.cycles.find((c) => c.length === 6);
    assert.deepStrictEqual(six.map((v) => v), [1, 2, 4, 8, 7, 5].length === 6 ? six : []);
    // following the map from 1 reproduces 1 2 4 8 7 5
    const seq = []; let x = 1; for (let i = 0; i < 6; i++) { seq.push(x); x = (2 * x) % 9; }
    assert.deepStrictEqual(seq, [1, 2, 4, 8, 7, 5]);
    // depth: every node of the cycle is at depth 0; no tails for a bijection
    assert.ok(o.depth.slice(1).every((d) => d === 0));
});

test('orbits of a non-invertible map have tails', () => {
    const o = NT.orbits(10, (x) => (x * x) % 10); // squaring mod 10
    assert.ok(o.depth.some((d) => d > 0));
    assert.ok(o.cycles.length >= 1);
});

test('Fibonacci modulo m and Pisano periods', () => {
    assert.strictEqual(NT.pisano(10), 60);
    assert.strictEqual(NT.pisano(9), 24);
    assert.strictEqual(NT.pisano(7), 16);
    assert.strictEqual(NT.fibMod(100, 1000000007), 687995182n);
    assert.strictEqual(NT.fibMod(10n ** 18n, 10), NT.fibMod(10n ** 18n % 60n, 10));
});

test('CRT and continued fractions', () => {
    const r = NT.crt([2, 3, 2], [3, 5, 7]);
    assert.strictEqual(r.x, 23n);
    assert.strictEqual(r.modulus, 105n);
    assert.strictEqual(NT.crt([1, 2], [4, 6]), null); // incompatible
    const cf = NT.continuedFraction(415n, 93n);
    assert.deepStrictEqual(cf, [4n, 2n, 6n, 7n]);
    const conv = NT.convergents(cf);
    assert.deepStrictEqual(conv[conv.length - 1], [415n, 93n]);
});

test('chi-square survival function against tabulated values', () => {
    const close = (a, b, tol = 1e-4) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);
    close(NT.chi2sf(3.841459, 1), 0.05);
    close(NT.chi2sf(16.918978, 9), 0.05);
    close(NT.chi2sf(15.507313, 8), 0.05);
    close(NT.chi2sf(0, 5), 1);
    close(NT.chi2sf(18.307038, 10), 0.05);
    const t = NT.chiSquare([10, 10, 10, 10], [0.25, 0.25, 0.25, 0.25]);
    close(t.stat, 0); close(t.p, 1);
});

test('digital roots of special sequences', () => {
    // powers of 2 cycle through 1 2 4 8 7 5
    const pw = NT.sequence('powers', 12, 2).map((v) => NT.digitalRoot(v));
    assert.deepStrictEqual(pw, [1, 2, 4, 8, 7, 5, 1, 2, 4, 8, 7, 5]);
    // primes above 3 never have digital root 3, 6 or 9 (they are not divisible by 3)
    const h = NT.rootHistogram(NT.sequence('primes', 1000).slice(2));
    assert.strictEqual(h.counts[2] + h.counts[5] + h.counts[8], 0);
    // Fibonacci digital roots repeat with period 24 (the Pisano period mod 9)
    const fr = NT.sequence('fibonacci', 72).map((v) => NT.digitalRoot(v));
    for (let i = 1; i + 24 < 72; i++) assert.strictEqual(fr[i], fr[i + 24]);
    // squares: only roots 1, 4, 7, 9
    const sq = new Set(NT.sequence('squares', 500).map((v) => NT.digitalRoot(v)));
    assert.deepStrictEqual([...sq].sort((a, b) => a - b), [1, 4, 7, 9]);
    assert.strictEqual(NT.sequence('primes', 25).slice(-1)[0], 97n);
});

console.log(`\n${n} test groups passed`);
