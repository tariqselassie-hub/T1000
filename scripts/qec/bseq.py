# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
#
# Upstream source: https://github.com/vm6502q/pyqrack-examples/blob/main/qec/bseq.py
# Example of nonlocal measurement collapse with QrackAceBackend
# (Produced by Elara, a custom OpenAI GPT)

import math
import sys
from pyqrack import QrackAceBackend, Pauli


def apply_measurement_basis(sim, q, theta):
    """Rotate qubit q into the measurement basis."""
    sim.r(Pauli.PauliY, theta, q)


def measure_expectation(sim, shots):
    """Measure shots on qubits 0 and 1, calculating <Z_0 Z_1> correlation."""
    results = sim.measure_shots([0, 1], shots)
    total = 0
    for res in results:
        a = (res >> 0) & 1
        b = (res >> 1) & 1
        total += 1 if a == b else -1
    return total / shots


def run_chsh_test(shots=2048):
    """
    Executes CHSH Bell Inequality test on QrackAceBackend.

    NOTE: The upstream script allocates `sim = QrackAceBackend(6, ...)` at the start
    of `run_chsh_test` which is never used because `expectation()` constructs its own
    `s = QrackAceBackend(16, ...)` per angle pair.
    """
    # Upstream Line 30-32 (unused allocation in original bseq.py)
    # sim = QrackAceBackend(6, long_range_columns=1)
    # sim.h(0)
    # sim.mcx([0], 1)

    pi = math.pi

    # CHSH optimal measurement angles (in radians)
    theta_a = 0.0
    theta_ap = pi / 2.0
    theta_b = pi / 4.0
    theta_bp = -pi / 4.0

    def expectation(theta1, theta2):
        # 16-qubit QrackAceBackend grid (4x4 lattice, patch_width = 2)
        s = QrackAceBackend(16, long_range_columns=1)
        s.h(0)
        s.mcx([0], 1)
        apply_measurement_basis(s, 0, theta1)
        apply_measurement_basis(s, 1, theta2)
        return measure_expectation(s, shots)

    E_ab = expectation(theta_a, theta_b)
    E_abp = expectation(theta_a, theta_bp)
    E_apb = expectation(theta_ap, theta_b)
    E_apbp = expectation(theta_ap, theta_bp)

    S = abs(E_ab + E_abp + E_apb - E_apbp)
    print("-------------------------------------------------------------------------")
    print(" CHSH BELL INEQUALITY TEST: QrackAceBackend (N=16)")
    print("-------------------------------------------------------------------------")
    print(f"  E(a, b)     = {E_ab:+.6f}")
    print(f"  E(a, b')    = {E_abp:+.6f}")
    print(f"  E(a', b)    = {E_apb:+.6f}")
    print(f"  E(a', b')   = {E_apbp:+.6f}")
    print("-------------------------------------------------------------------------")
    print(f"  CHSH S-value: {S:.6f} (Tsirelson Bound: 2*sqrt(2) ≈ 2.828427)")
    print(f"  Bell inequality violated? {'Yes (Quantum Non-Local)' if S > 2.0 else 'No (Classical LHV Bound)'}")
    print("-------------------------------------------------------------------------")
    return S


if __name__ == "__main__":
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    run_chsh_test(shots)
