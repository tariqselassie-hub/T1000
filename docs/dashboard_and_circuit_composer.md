<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# The lab bench (dashboard)

```bash
python web/serve.py              # then open http://localhost:8000/
python web/serve.py --no-browser
T1000_QPU_PORT=COM6 python web/serve.py   # serial port of the Basys 3 (default COM6)
```

`web/index.html` is a bench for running things, not a gallery of results. Every panel states where its
numbers come from, and each source is tested:

| Tab | What you can do | Source of the numbers | Tests |
| :--- | :--- | :--- | :--- |
| Circuits and algorithms | Grover (with overshoot), Bernstein-Vazirani, Deutsch-Jozsa, phase estimation, quantum order finding (Shor) with shots, continued fractions and factors; a gate-list circuit composer that runs on the host or on the Basys 3 | exact state vectors in numpy (`web/lab_api.py`); the board path is `web/qpu_board.py` (see `docs/fpga_architecture.md`) | `web/test_lab_api.py` |
| Probability | Born-rule sampling with chi-square p-values and repeated experiments; Porter-Thomas statistics of random circuits; quantum and classical walks on cycles, grids, hypercubes, Cayley graphs of Z_n, the map x to kx + c (mod n) and glued trees | seeded multinomial sampling; exact spectral evolution `exp(-iAt)` and `exp(-(D-A)t)` | `web/test_lab_api.py` |
| Number theory and digit roots | Factorisation, phi, order, primitive roots, Jacobi symbols, digital roots in any base, digit-root histograms with period detection, the times-table circle and the cycles of x to kx + c, a claim checker (for example "primes above 3 never have digital root 3, 6 or 9", "powers of 2 cycle 1-2-4-8-7-5", Fibonacci roots have period 24) | exact BigInt arithmetic (`web/lab_nt.js`) | `web/test_lab_nt.js` |
| Geometry | Bloch sphere with gates, the Hopf fibres over your state (fibres over different points are linked once), the six rotation planes of the regular 4-polytopes (5-cell, tesseract, 16-cell, 24-cell) | exact (`web/lab_geom.js`) | `web/test_lab_geom.js` |

## Why the mod-9 facts show up on several tabs

The digital root of n in base 10 is `1 + (n - 1) mod 9`, so everything about digit roots is arithmetic in
`Z/9`. Doubling is a permutation of the six units `{1, 2, 4, 8, 7, 5}` (one 6-cycle, because the order of 2
mod 9 is 6), swaps 3 and 6 (the multiples of 3 that are not 0), and fixes 0 (the root 9). The orbit panel draws
exactly this. The same number, 6, is what quantum order finding returns for `a = 2`, `N = 9`, and it is the
period of the root sequence of the powers of 2 in the digit-root panel. None of these is a coincidence and none
needs more than the group `(Z/9)*`; the claim checker states each fact and checks it over a range.

## What is not here

The earlier panels that only displayed stored results (the QEC and 36-qubit decks, ZINC/QM9 viewers, scaling
tables) are in `web/archive.html`, with a banner saying they are recorded output, and `docs/CLAIMS.md` records
which of those numbers were re-run and corrected. Two endpoints that returned made-up results (`/api/canonical/run`,
which printed hard-coded Bernstein-Vazirani and uniform Grover distributions, and `/api/smiles/evaluate`, which
used invented property formulas) were removed. The lab runs Grover and Bernstein-Vazirani by simulation instead.

## API

All `POST`, JSON in, JSON out; errors come back as `{"error": "..."}` with status 400.

| Endpoint | Body |
| :--- | :--- |
| `/api/lab/algorithm` | `name` = `grover` (`n`, `marked`, `iterations`, `max_iterations`), `bernstein_vazirani` (`n`, `secret`), `deutsch_jozsa` (`n`, `kind`, `seed`), `qpe` (`phase`, `t`), `shor_order` (`a`, `N`, `t`, `shots`, `seed`) |
| `/api/lab/shots` | `probabilities`, `shots`, `seed`, `repeats` |
| `/api/lab/random_circuit` | `n`, `depth`, `seed` |
| `/api/lab/walk` | `graph` (`kind`, `n`, ...), `start`, `target`, `t_max`, `steps` |
| `/api/run_circuit` | `qubits`, `layers`, `target` (`cpu` or `fpga`) |
| `/api/status` | GET: host and board status |

## Tests

```bash
node web/test_lab_nt.js        # 12 groups: gcd, primality, factorisation, order, residues, digit roots, orbits, CRT, chi-square
node web/test_lab_geom.js      # 8 groups: Hopf map and fibre linking, gates, 4-polytope counts, 4D rotations
python web/test_lab_api.py     # 33 checks: closed forms, brute-force QFT, walks, Porter-Thomas, input validation
```
