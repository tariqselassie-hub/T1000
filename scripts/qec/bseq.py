# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
#
# Upstream source: https://github.com/vm6502q/pyqrack-examples/blob/main/qec/bseq.py
# Example of nonlocal measurement collapse with QrackAceBackend
# (Produced by Elara, a custom OpenAI GPT)

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pyqrack import QrackAceBackend, Pauli


def apply_measurement_basis(sim, q, theta):
    """Rotate qubit q into the measurement basis."""
    sim.r(Pauli.PauliY, theta, q)


def measure_expectation(sim, qa, qb, shots):
    """Measure shots on qubits qa and qb, calculating <Z_a Z_b> correlation."""
    results = sim.measure_shots([qa, qb], shots)
    total = 0
    for res in results:
        a = (res >> 0) & 1
        b = (res >> 1) & 1
        total += 1 if a == b else -1
    return total / shots


def run_chsh_test(shots=2048, qa=4, qb=13, n_qubits=16):
    """
    Executes CHSH Bell Inequality test on QrackAceBackend boundary bridge pair (qa, qb).
    Offloaded to Intel Arc / Integrated Graphics OpenCL device.
    """
    pi = math.pi

    # CHSH optimal measurement angles (in radians)
    theta_a = 0.0
    theta_ap = pi / 2.0
    theta_b = pi / 4.0
    theta_bp = -pi / 4.0

    def expectation(theta1, theta2):
        s = QrackAceBackend(n_qubits, long_range_columns=1)
        s.h(qa)
        s.mcx([qa], qb)
        apply_measurement_basis(s, qa, theta1)
        apply_measurement_basis(s, qb, theta2)
        return measure_expectation(s, qa, qb, shots)

    E_ab = expectation(theta_a, theta_b)
    E_abp = expectation(theta_a, theta_bp)
    E_apb = expectation(theta_ap, theta_b)
    E_apbp = expectation(theta_ap, theta_bp)

    S = abs(E_ab + E_abp + E_apb - E_apbp)
    print("┌────────────────────────────────────────────────────────────────────────┐")
    print(f"│ CHSH BELL INEQUALITY TEST: QrackAceBackend Boundary Pair (q{qa}, q{qb})    │")
    print(f"│ Target Register: N={n_qubits} Qubits | Shots: {shots:<6} | Device: Intel OpenCL    │")
    print("├────────────────────────────────────────────────────────────────────────┤")
    print(f"│  E(a, b)     = {E_ab:+.6f}                                              │")
    print(f"│  E(a, b')    = {E_abp:+.6f}                                              │")
    print(f"│  E(a', b)    = {E_apb:+.6f}                                              │")
    print(f"│  E(a', b')   = {E_apbp:+.6f}                                              │")
    print("├────────────────────────────────────────────────────────────────────────┤")
    print(f"│  CHSH S-value: {S:.6f} (Tsirelson Bound: 2*sqrt(2) ~= 2.828427)         │")
    violation_str = "VIOLATED (Quantum Entangled)" if S > 2.0 else "RESPECTED (Classical LHV)"
    print(f"│  Bell Inequality: {violation_str:<53}│")
    print("└────────────────────────────────────────────────────────────────────────┘")
    return S


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QrackAceBackend Boundary Bridge CHSH Audit")
    parser.add_argument("shots_pos", nargs="?", type=int, default=None, help="Shot count (positional)")
    parser.add_argument("--shots", type=int, default=2048, help="Monte Carlo shot count")
    parser.add_argument("--qa", type=int, default=4, help="Fragment A bridge qubit anchor")
    parser.add_argument("--qb", type=int, default=13, help="Fragment B bridge qubit anchor")
    parser.add_argument("--qubits", type=int, default=16, help="Total qubits on QrackAceBackend")
    args = parser.parse_args()

    shots = args.shots_pos if args.shots_pos is not None else args.shots
    run_chsh_test(shots=shots, qa=args.qa, qb=args.qb, n_qubits=args.qubits)
