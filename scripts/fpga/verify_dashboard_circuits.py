# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Runs dashboard circuits on the board through `web/serve.py` and compares them with the host.

    .venv\\Scripts\\python.exe scripts\\fpga\\verify_dashboard_circuits.py

Starts the dashboard server on port 8000 (it must be free), posts three circuits with
`target: "fpga"` (a Bell pair, and a 12- and a 14-qubit RY/CZ/T/SWAP circuit) and checks that the
board executed them (`PHYSICAL_SILICON_EXECUTED`), that its norm is 1 within 1e-3, and that the
Pauli-Z expectations from the board's state agree with the host simulation and with the board's
own EXPECT_Z command to 1e-3. Exit status 0 means all three pass.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8000/api/run_circuit"
TOL = 1e-3


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def circuits():
    bell = [{"gates": [{"type": "H", "target": 0}]}, {"gates": [{"type": "CX", "target": 1, "control": 0}]}]
    ring = []
    for layer in range(6):
        ring.append({"gates": [{"type": "RY", "target": q, "theta": 0.3 + 0.17 * q + 0.1 * layer} for q in range(12)]})
        ring.append(
            {
                "gates": [{"type": "CZ", "target": (q + 1) % 12, "control": q} for q in range(0, 12, 2)]
                + [{"type": "T", "target": 3}, {"type": "SWAP", "target": 5, "control": 2}]
            }
        )
    return [("bell pair", 2, bell), ("12 qubits", 12, ring), ("14 qubits", 14, ring)]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "web" / "serve.py"), "--no-browser"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    failed = 0
    try:
        time.sleep(3)
        for name, qubits, layers in circuits():
            r = post({"qubits": qubits, "layers": layers, "target": "fpga"})
            ok = (
                r.get("hardware_mode") == "PHYSICAL_SILICON_EXECUTED"
                and abs(r.get("board_norm_squared", 0.0) - 1.0) < TOL
                and r.get("max_abs_pauli_z_error_vs_host", 1.0) < TOL
                and all(abs(a - b) < TOL for a, b in zip(r["pauli_z"], r["board_expect_z"]))
            )
            failed += not ok
            print(
                f" [{'PASS' if ok else 'FAIL'}] {name}: mode {r.get('hardware_mode')}, "
                f"{r.get('board_gates_sent')} gates, norm {r.get('board_norm_squared')}, "
                f"max |dZ| vs host {r.get('max_abs_pauli_z_error_vs_host')}, "
                f"{r.get('wall_time_ms')} ms total"
                + (f", error: {r['hardware_error']}" if r.get("hardware_error") else "")
            )
    finally:
        server.kill()
    print(" RESULT:", "ALL PASSED" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
