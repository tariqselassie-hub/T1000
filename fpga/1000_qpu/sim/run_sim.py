# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Run the 1000_QPU testbench in Vivado xsim against the golden vectors.

    python fpga/1000_qpu/sim/run_sim.py

Regenerate the vectors first when the model or protocol changes:

    cargo run --release -p t1000 --bin qpu_vectors -- fpga/1000_qpu/sim/vectors

Exit status 0 means the design's replies matched the Rust model byte for byte.
"""

import os
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
RTL = SIM.parent / "rtl"
VIVADO = Path(os.getenv("VIVADO_DIR", r"C:\AMDDesignTools\2026.1\Vivado")) / "bin"
WORK = SIM / "work"


def tool(name):
    return str(VIVADO / f"{name}.bat")


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=WORK, capture_output=True, text=True, shell=False)


def main():
    WORK.mkdir(exist_ok=True)
    link = WORK / "vectors"
    if not link.exists():
        # xsim reads the vectors relative to its working directory
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(SIM / "vectors")], check=True,
                       capture_output=True)
    sources = [str(p) for p in sorted(RTL.glob("*.v"))] + [str(SIM / "tb_qpu.v")]
    for step in (
        [tool("xvlog")] + sources,
        [tool("xelab"), "tb_qpu", "-s", "tb_qpu_sim", "-timescale", "1ns/1ps", "-debug", "off"],
    ):
        r = run(step)
        if r.returncode != 0:
            print(r.stdout[-4000:], r.stderr[-4000:])
            return 2
    r = run([tool("xsim"), "tb_qpu_sim", "-runall"])
    out = r.stdout
    print(out[-3000:])
    return 0 if "RESULT: PASS" in out else 1


if __name__ == "__main__":
    sys.exit(main())
