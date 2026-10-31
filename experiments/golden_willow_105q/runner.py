# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Google Willow 105Q Golden Experiment Unified Runner.
Orchestrates:
1. T-1000 pure-Rust simulation execution and telemetry generation.
2. Synchronizes telemetry into the local experiment directory.
3. Automated re-rendering of the 4-panel publication visualization.
"""

import json
from pathlib import Path
import subprocess
import sys
import time

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[1]
TELEMETRY_JSON = CUR_DIR / "willow_105q_telemetry.json"
HARDWARE_SPEC_JSON = CUR_DIR / "willow_hardware_spec.json"


def run_t1000_simulation():
    """Executes the sovereign pure-Rust T-1000 Willow-105Q benchmark binary."""
    print("==========================================================================")
    print("  T-1000 SOVEREIGN ENGINE: EXECUTING WILLOW-105Q BENCHMARK (AVX2+FMA)   ")
    print("==========================================================================")
    t0 = time.perf_counter()
    cmd = ["cargo", "run", "--release", "-p", "t1000", "--bin", "willow_105q_bench", "--", "--record"]
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"[!] Compilation / Execution Error:\n{res.stderr}")
        return False

    print(res.stdout)
    print(f" [+] T-1000 Execution Finished in {elapsed * 1000:.2f} ms")

    # Sync fresh telemetry into this folder
    src_json = REPO_ROOT / "results" / "data" / "willow_105q_telemetry.json"
    if src_json.exists():
        with open(src_json, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)
        with open(TELEMETRY_JSON, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, indent=2)
        print(f" [+] Synced fresh telemetry to {TELEMETRY_JSON.name}")

    # Re-render golden chart
    print(" [+] Re-rendering Golden Experiment visualization figure...")
    from chart_golden_willow import chart_golden_willow
    chart_golden_willow()
    return True


if __name__ == "__main__":
    run_t1000_simulation()

