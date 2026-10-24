// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/*
 * Exact number theory for the bench: BigInt arithmetic, no approximation except where a function says so
 * (isPrime is deterministic below 3.3e24 and a strong probable-prime test above it).
 * Pure functions with no DOM access, so the same file is loaded by the page and by test_lab_nt.js.
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.NT = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    const big = (x) => (typeof x === 'bigint' ? x : BigInt(x));
    const abs = (x) => (x < 0n ? -x : x);
    const mod = (a, m) => ((a % m) + m) % m;

    function gcd(a, b) {
        a = abs(big(a)); b = abs(big(b));
        while (b) [a, b] = [b, a % b];
        return a;
    }

    /** [g, x, y] with a*x + b*y = g = gcd(a, b). */
    function egcd(a, b) {
        a = big(a); b = big(b);
        let [r0, r1, s0, s1, t0, t1] = [a, b, 1n, 0n, 0n, 1n];
        while (r1) {
            const q = r0 / r1;
            [r0, r1] = [r1, r0 - q * r1];
            [s0, s1] = [s1, s0 - q * s1];
            [t0, t1] = [t1, t0 - q * t1];
        }
        return [r0, s0, t0];
    }

    function modpow(b, e, m) {
        b = mod(big(b), big(m)); e = big(e); m = big(m);
        if (m === 1n) return 0n;
        let r = 1n;
        while (e > 0n) {
            if (e & 1n) r = (r * b) % m;
            b = (b * b) % m;
            e >>= 1n;
        }
        return r;
    }

    function modinv(a, m) {
        m = big(m);
        const [g, x] = egcd(mod(big(a), m), m);
        return g === 1n ? mod(x, m) : null;
    }

    const MR_BASES = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n, 41n];

    /** Miller-Rabin with the first 13 prime bases: deterministic for n < 3.3e24. */
    function isPrime(n) {
        n = big(n);
        if (n < 2n) return false;
        for (const p of MR_BASES) {
            if (n === p) return true;
            if (n % p === 0n) return false;
        }
        let d = n - 1n, s = 0;
        while ((d & 1n) === 0n) { d >>= 1n; s++; }
        outer: for (const a of MR_BASES) {
            let x = modpow(a, d, n);
            if (x === 1n || x === n - 1n) continue;
            for (let i = 1; i < s; i++) {
                x = (x * x) % n;
                if (x === n - 1n) continue outer;
            }
            return false;
        }
        return true;
    }

    function pollardRho(n) {
        if (n % 2n === 0n) return 2n;
        for (let c = 1n; c < 50n; c++) {
            let x = 2n, y = 2n, d = 1n;
            const f = (v) => (v * v + c) % n;
            let steps = 0;
            while (d === 1n && steps++ < 4_000_000) {
                x = f(x); y = f(f(y));
                d = gcd(x - y, n);
            }
            if (d !== 1n && d !== n) return d;
        }
        return null;
    }

    /** Prime factorisation as [[p, e], ...] sorted; `complete` is false if a cofactor could not be split. */
    function factor(n) {
        n = abs(big(n));
        const out = new Map();
        let complete = true;
        const add = (p, e = 1) => out.set(p, (out.get(p) || 0) + e);
        if (n < 2n) return { factors: [], complete: true };
        for (let p = 2n; p < 1000n && p * p <= n; p += (p === 2n ? 1n : 2n)) {
            while (n % p === 0n) { add(p); n /= p; }
        }
        const stack = n > 1n ? [n] : [];
        while (stack.length) {
            const m = stack.pop();
            if (m === 1n) continue;
            if (isPrime(m)) { add(m); continue; }
            const d = pollardRho(m);
            if (d === null) { add(m); complete = false; continue; }
            stack.push(d, m / d);
        }
        const factors = [...out.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
        return { factors, complete };
    }

    function eulerPhi(n) {
        n = big(n);
        let r = n;
        for (const [p] of factor(n).factors) r = (r / p) * (p - 1n);
        return r;
    }

    /** Multiplicative order of a modulo n (a and n coprime); null if they are not. */
    function order(a, n) {
        a = mod(big(a), big(n)); n = big(n);
        if (n === 1n) return 1n;
        if (gcd(a, n) !== 1n) return null;
        let o = eulerPhi(n);
        for (const [p] of factor(o).factors) {
            while (o % p === 0n && modpow(a, o / p, n) === 1n) o /= p;
        }
        return o;
    }

    /** Smallest primitive root modulo n, or null when (Z/n)* is not cyclic (n is not 1, 2, 4, p^k or 2 p^k). */
    function primitiveRoot(n) {
        n = big(n);
        if (n === 1n) return 0n;
        if (n === 2n) return 1n;
        if (n === 4n) return 3n;
        let fs = factor(n).factors;
        if (fs.length && fs[0][0] === 2n) {
            if (fs[0][1] > 1) return null;
            fs = fs.slice(1);
        }
        if (fs.length !== 1) return null;
        const phi = eulerPhi(n);
        for (let g = 2n; g < n; g++) {
            if (gcd(g, n) === 1n && order(g, n) === phi) return g;
        }
        return null;
    }

    function jacobi(a, n) {
        a = big(a); n = big(n);
        if (n <= 0n || n % 2n === 0n) throw new Error('jacobi needs an odd positive modulus');
        a = mod(a, n);
        let t = 1;
        while (a !== 0n) {
            while (a % 2n === 0n) {
                a /= 2n;
                const r = n % 8n;
                if (r === 3n || r === 5n) t = -t;
            }
            [a, n] = [n, a];
            if (a % 4n === 3n && n % 4n === 3n) t = -t;
            a = a % n;
        }
        return n === 1n ? t : 0;
    }

    const legendre = (a, p) => jacobi(a, p);

    /** Sorted distinct squares modulo n (as Numbers; n must be small). */
    function quadraticResidues(n) {
        const s = new Set();
        for (let x = 0; x < n; x++) s.add((x * x) % n);
        return [...s].sort((a, b) => a - b);
    }

    /** Digital root in `base`: repeated digit sum, i.e. 1 + (n-1) mod (base-1) for n > 0. */
    function digitalRoot(n, base = 10) {
        n = abs(big(n)); const b1 = big(base) - 1n;
        if (n === 0n) return 0;
        return Number(1n + ((n - 1n) % b1));
    }

    function digitSum(n, base = 10) {
        n = abs(big(n)); const b = big(base);
        let s = 0n;
        while (n > 0n) { s += n % b; n /= b; }
        return s;
    }

    /**
     * Decomposes the map f on {0..m-1} into cycles and tails.
     * Returns { cycles: [[...]], depth: Array (steps to reach a cycle), cycleOf: Array }.
     */
    function orbits(m, f) {
        const next = new Array(m);
        for (let x = 0; x < m; x++) next[x] = f(x);
        const state = new Array(m).fill(0); // 0 new, 1 on stack, 2 done
        const cycleOf = new Array(m).fill(-1);
        const cycles = [];
        for (let s = 0; s < m; s++) {
            if (state[s]) continue;
            const path = [];
            let x = s;
            while (state[x] === 0) { state[x] = 1; path.push(x); x = next[x]; }
            if (state[x] === 1) {
                const at = path.indexOf(x);
                const cyc = path.slice(at);
                cyc.forEach((v) => { cycleOf[v] = cycles.length; });
                cycles.push(cyc);
            }
            path.forEach((v) => { state[v] = 2; });
        }
        const depth = new Array(m).fill(-1);
        const resolve = (x) => {
            if (depth[x] >= 0) return depth[x];
            if (cycleOf[x] >= 0) return (depth[x] = 0);
            return (depth[x] = 1 + resolve(next[x]));
        };
        for (let x = 0; x < m; x++) {
            const seen = [];
            let y = x;
            while (depth[y] < 0 && cycleOf[y] < 0) { seen.push(y); y = next[y]; }
            let d = depth[y] >= 0 ? depth[y] : 0;
            depth[y] = d;
            for (let i = seen.length - 1; i >= 0; i--) depth[seen[i]] = ++d;
        }
        return { cycles, depth, cycleOf, next };
    }

    function fibMod(k, m) {
        m = big(m); k = big(k);
        const fib = (n) => {
            if (n === 0n) return [0n, 1n];
            const [a, b] = fib(n >> 1n);
            const c = (a * ((2n * b - a + m) % m)) % m;
            const d = (a * a + b * b) % m;
            return n & 1n ? [d, (c + d) % m] : [c, d];
        };
        return fib(k)[0] % m;
    }

    /** Period of the Fibonacci sequence modulo n (the Pisano period). */
    function pisano(n) {
        n = Number(n);
        if (n === 1) return 1;
        let a = 0, b = 1;
        for (let i = 1; i <= 6 * n; i++) {
            [a, b] = [b, (a + b) % n];
            if (a === 0 && b === 1) return i;
        }
        return null;
    }

    function sieve(limit) {
        const flags = new Uint8Array(limit + 1).fill(1);
        flags[0] = 0; if (limit >= 1) flags[1] = 0;
        for (let i = 2; i * i <= limit; i++) if (flags[i]) for (let j = i * i; j <= limit; j += i) flags[j] = 0;
        const out = [];
        for (let i = 2; i <= limit; i++) if (flags[i]) out.push(i);
        return out;
    }

    function crt(remainders, moduli) {
        let r = 0n, m = 1n;
        for (let i = 0; i < moduli.length; i++) {
            const mi = big(moduli[i]), ri = mod(big(remainders[i]), mi);
            const [g, x] = egcd(m, mi);
            if ((ri - r) % g !== 0n) return null;
            const lcm = (m / g) * mi;
            r = mod(r + m * ((((ri - r) / g) * x) % (mi / g)), lcm);
            m = lcm;
        }
        return { x: r, modulus: m };
    }

    /** Continued-fraction expansion of num/den (BigInt) as an array of partial quotients. */
    function continuedFraction(num, den, maxTerms = 64) {
        num = big(num); den = big(den);
        const out = [];
        while (den !== 0n && out.length < maxTerms) {
            const q = num / den;
            out.push(q);
            [num, den] = [den, num - q * den];
        }
        return out;
    }

    /** Convergents [p, q] of a continued fraction. */
    function convergents(terms) {
        const out = [];
        let [p0, p1, q0, q1] = [0n, 1n, 1n, 0n];
        for (const a of terms) {
            [p0, p1] = [p1, a * p1 + p0];
            [q0, q1] = [q1, a * q1 + q0];
            out.push([p1, q1]);
        }
        return out;
    }

    // ---- statistics ------------------------------------------------------------------------------

    function lgamma(x) {
        const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
            0.1208650973866179e-2, -0.5395239384953e-5];
        let y = x, tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        let ser = 1.000000000190015;
        for (const cj of c) ser += cj / ++y;
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    /** Regularised upper incomplete gamma Q(a, x). */
    function gammaQ(a, x) {
        if (x <= 0) return 1;
        if (x < a + 1) {
            let ap = a, sum = 1 / a, del = sum;
            for (let n = 0; n < 500; n++) {
                ap += 1; del *= x / ap; sum += del;
                if (Math.abs(del) < Math.abs(sum) * 1e-14) break;
            }
            return 1 - sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
        }
        let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
        for (let i = 1; i < 500; i++) {
            const an = -i * (i - a);
            b += 2;
            d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
            c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
            d = 1 / d;
            const del = d * c; h *= del;
            if (Math.abs(del - 1) < 1e-14) break;
        }
        return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
    }

    /** Survival function of the chi-square distribution with k degrees of freedom. */
    const chi2sf = (x, k) => gammaQ(k / 2, x / 2);

    /** Pearson chi-square test of observed counts against expected probabilities. */
    function chiSquare(observed, probs) {
        const n = observed.reduce((a, b) => a + b, 0);
        let stat = 0, df = -1;
        observed.forEach((o, i) => {
            const e = n * probs[i];
            if (e > 0) { stat += (o - e) * (o - e) / e; df++; }
        });
        return { stat, df, p: df > 0 ? chi2sf(stat, df) : NaN, n };
    }

    // ---- sequences -----------------------------------------------------------------------------------

    function sequence(name, count, param = 2) {
        const out = [];
        switch (name) {
            case 'naturals': for (let i = 1; i <= count; i++) out.push(BigInt(i)); break;
            case 'powers': { let v = 1n; for (let i = 0; i < count; i++) { out.push(v); v *= big(param); } break; }
            case 'fibonacci': { let a = 0n, b = 1n; for (let i = 0; i < count; i++) { out.push(a); [a, b] = [b, a + b]; } break; }
            case 'squares': for (let i = 1; i <= count; i++) out.push(BigInt(i) * BigInt(i)); break;
            case 'cubes': for (let i = 1; i <= count; i++) out.push(BigInt(i) ** 3n); break;
            case 'triangular': for (let i = 1; i <= count; i++) out.push(BigInt(i) * BigInt(i + 1) / 2n); break;
            case 'factorials': { let v = 1n; for (let i = 1; i <= count; i++) { v *= BigInt(i); out.push(v); } break; }
            case 'primes': {
                let limit = Math.max(30, Math.ceil(count * (Math.log(count + 2) + Math.log(Math.log(count + 3)) + 2)));
                let ps = sieve(limit);
                while (ps.length < count) { limit *= 2; ps = sieve(limit); }
                ps.slice(0, count).forEach((p) => out.push(BigInt(p)));
                break;
            }
            case 'random': { // deterministic LCG so runs repeat; uniform 64-bit-ish integers
                let s = 0x9E3779B97F4A7C15n;
                for (let i = 0; i < count; i++) {
                    s = (s * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
                    out.push(s >> 11n);
                }
                break;
            }
            default: throw new Error('unknown sequence ' + name);
        }
        return out;
    }

    /** Counts of digital roots 1..base-1 (index root-1) for BigInt values; zero is skipped. */
    function rootHistogram(values, base = 10) {
        const counts = new Array(base - 1).fill(0);
        let zeros = 0;
        for (const v of values) {
            const r = digitalRoot(v, base);
            if (r === 0) zeros++; else counts[r - 1]++;
        }
        return { counts, zeros };
    }

    return {
        gcd, egcd, modpow, modinv, isPrime, factor, eulerPhi, order, primitiveRoot, jacobi, legendre,
        quadraticResidues, digitalRoot, digitSum, orbits, fibMod, pisano, sieve, crt, continuedFraction,
        convergents, chi2sf, chiSquare, sequence, rootHistogram, mod,
    };
});
