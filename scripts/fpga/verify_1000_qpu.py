# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""One-command verification of the loaded 1000_QPU design on the Basys 3.

    .venv\\Scripts\\python.exe scripts\\fpga\\verify_1000_qpu.py
    .venv\\Scripts\\python.exe scripts\\fpga\\verify_1000_qpu.py --port COM6 --quick

It speaks only the opcodes the RTL implements (`fpga/1000_qpu/rtl/qpu_top.v`):

    0xAA            ping                   -> 0x10
    0x10            one 14-qubit layer     -> 0xEE
    0x20 <depth>    autonomous layers      -> 0xEE
    0x30 <depth> <count lo> <count hi>     -> count x 32 bytes of seed, then 0xEE

Stages: 0 port and passive listen, 1 ping, 2 single layer, 3 multi-layer timing (a straight line in
the depth whose slope is the per-layer cost), 4 entropy stream (length, distinct seeds, bit balance),
5 recovery after an unknown opcode. Nothing is written to the FPGA configuration. The report is
printed and saved to results/fpga_runs/verify_1000_qpu.json. Exit status 0 means every stage passed.

Note: the design has no state read-back opcode, so this checks that the hardware executes the
commands and takes the time 1,024-cycle layers should take; it cannot compare amplitudes with the
software simulator.
"""

import argparse
import datetime
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import serial
import serial.tools.list_ports

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = os.getenv("T1000_QPU_PORT", os.getenv("ZENITH_FPGA_PORT", "COM6"))
DEFAULT_BAUD = int(os.getenv("T1000_QPU_BAUD", "3000000"))

PING, PING_OK = 0xAA, 0x10
LAYER, ACK = 0x10, 0xEE
MULTI, CHAOS = 0x20, 0x30
CYCLES_PER_LAYER = 1024
CLOCK_MHZ = 100.0
LAYER_US = CYCLES_PER_LAYER / CLOCK_MHZ  # 10.24 us of pure fabric time per layer


class Stage:
    """Result of one verification stage."""

    def __init__(self, name):
        self.name, self.passed, self.details, self.notes = name, False, {}, []

    def finish(self, passed, **details):
        self.passed = passed
        self.details.update(details)
        return self

    def as_dict(self):
        return {"stage": self.name, "passed": self.passed, "details": self.details, "notes": self.notes}


def exchange(ser, payload, expect, timeout=1.0):
    """Send `payload`, read `expect` bytes; returns (bytes, seconds)."""
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(bytes(payload))
    data = ser.read(expect)
    return data, time.perf_counter() - t0


def stage_port(ser, port):
    s = Stage("0 port and passive listen")
    ports = {p.device: p for p in serial.tools.list_ports.comports()}
    info = ports.get(port)
    s.details["port"] = port
    s.details["description"] = info.description if info else "not enumerated"
    s.details["usb_vid_pid"] = f"{info.vid:04X}:{info.pid:04X}" if info and info.vid else None
    ser.reset_input_buffer()
    time.sleep(0.5)
    stray = ser.read(64)
    s.details["unsolicited_bytes"] = stray.hex()
    if stray:
        s.notes.append("the board is already transmitting before any command was sent")
    return s.finish(info is not None)


def stage_ping(ser, repeats):
    s = Stage("1 ping 0xAA -> 0x10")
    lat, bad = [], 0
    for _ in range(repeats):
        data, dt = exchange(ser, [PING], 1)
        if data == bytes([PING_OK]):
            lat.append(dt * 1e6)
        else:
            bad += 1
    if lat:
        s.details.update(median_us=round(statistics.median(lat), 1), max_us=round(max(lat), 1))
    s.details.update(ok=len(lat), failed=bad)
    if not lat:
        s.notes += [
            "no reply to any ping: check that the 1000_QPU bitstream is the one loaded, the centre",
            "button (reset, pin U18) is not held, and that an on-board LED toggles per byte",
        ]
    return s.finish(bad == 0)


def stage_layer(ser, repeats):
    s = Stage("2 single layer 0x10 -> 0xEE")
    lat, bad = [], 0
    for _ in range(repeats):
        data, dt = exchange(ser, [LAYER], 1)
        if data == bytes([ACK]):
            lat.append(dt * 1e6)
        else:
            bad += 1
    if lat:
        s.details.update(median_us=round(statistics.median(lat), 1), fabric_us=LAYER_US)
    s.details.update(ok=len(lat), failed=bad)
    return s.finish(bad == 0)


def linear_fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys) or 1.0
    return slope, intercept, 1.0 - ss_res / ss_tot


def stage_multilayer(ser, depths, reps):
    s = Stage("3 autonomous layers 0x20 <depth> -> 0xEE, timing")
    points = []
    for depth in depths:
        times = []
        for _ in range(reps):
            data, dt = exchange(ser, [MULTI, depth], 1)
            if data != bytes([ACK]):
                return s.finish(False, failed_depth=depth, got=data.hex())
            times.append(dt * 1e6)
        points.append((depth, statistics.median(times)))
    slope, intercept, r2 = linear_fit([p[0] for p in points], [p[1] for p in points])
    s.details.update(
        points_us={d: round(t, 1) for d, t in points},
        slope_us_per_layer=round(slope, 3),
        fabric_expected_us=LAYER_US,
        overhead_us=round(intercept, 1),
        r_squared=round(r2, 5),
    )
    # Pure fabric time per layer is 10.24 us; allow a few percent of slack for the fixed-latency
    # pipeline fill between layers. A slope far from it means the layers are not 1,024-cycle passes.
    ok = 0.9 * LAYER_US <= slope <= 1.2 * LAYER_US and r2 > 0.99
    if not ok:
        s.notes.append("time per layer is not a straight line near 10.24 us")
    return s.finish(ok)


def shannon_bits_per_byte(data):
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    return -sum(c / len(data) * math.log2(c / len(data)) for c in counts if c)


def stage_entropy(ser, depth, seeds):
    s = Stage(f"4 entropy stream 0x30 depth={depth} seeds={seeds}")
    want = seeds * 32 + 1
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(bytes([CHAOS, depth, seeds & 0xFF, seeds >> 8]))
    data = ser.read(want)
    dt = time.perf_counter() - t0
    s.details.update(bytes_received=len(data), bytes_expected=want, seconds=round(dt, 4))
    if len(data) != want or data[-1] != ACK:
        return s.finish(False)
    payload = data[:-1]
    chunks = [payload[i : i + 32] for i in range(0, len(payload), 32)]
    ones = sum(bin(b).count("1") for b in payload) / (8 * len(payload))
    entropy = shannon_bits_per_byte(payload)
    distinct = len(set(chunks))
    s.details.update(
        distinct_seeds=distinct,
        ones_fraction=round(ones, 4),
        shannon_bits_per_byte=round(entropy, 3),
        seeds_per_second=round(seeds / dt, 1),
    )
    if entropy < 7.0:
        s.notes.append("byte entropy below 7.0 bits; run nist_randomness_audit.py for a full test")
    return s.finish(distinct == seeds and 0.47 <= ones <= 0.53)


def stage_recovery(ser):
    s = Stage("5 recovery after unknown opcodes")
    ser.reset_input_buffer()
    ser.write(bytes([0x7F, 0x7E, 0x7D]))
    time.sleep(0.05)
    stray = ser.read(8)
    data, _ = exchange(ser, [PING], 1)
    s.details.update(reply_to_unknown=stray.hex(), ping_after=data.hex())
    return s.finish(not stray and data == bytes([PING_OK]))


def run(args):
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0, write_timeout=1.0)
    except serial.SerialException as e:
        print(f"cannot open {args.port}: {e}")
        return None
    ser.reset_output_buffer()
    stages = [stage_port(ser, args.port)]
    stages.append(stage_ping(ser, 5 if args.quick else 20))
    if not stages[-1].passed:
        ser.close()
        return stages  # nothing else can work without a reply
    stages.append(stage_layer(ser, 5 if args.quick else 20))
    stages.append(stage_multilayer(ser, [1, 4, 16, 64, 255], 3 if args.quick else 7))
    stages.append(stage_entropy(ser, 12, 16 if args.quick else 128))
    stages.append(stage_recovery(ser))
    ser.close()
    return stages


def report(args, stages):
    print("=" * 79)
    print(f" 1000_QPU HARDWARE VERIFICATION   port {args.port}   {args.baud:,} baud")
    print("=" * 79)
    for s in stages:
        print(f" [{'PASS' if s.passed else 'FAIL'}] {s.name}")
        for k, v in s.details.items():
            print(f"        {k}: {v}")
        for n in s.notes:
            print(f"        note: {n}")
    ok = bool(stages) and all(s.passed for s in stages) and len(stages) == 6
    print("-" * 79)
    print(f" RESULT: {'ALL STAGES PASSED' if ok else 'NOT VERIFIED'}")
    out = ROOT / "results" / "fpga_runs" / "verify_1000_qpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "port": args.port,
                "baud": args.baud,
                "all_passed": ok,
                "stages": [s.as_dict() for s in stages],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f" report: {out}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--quick", action="store_true", help="fewer repetitions")
    args = p.parse_args()
    stages = run(args)
    if stages is None:
        return 2
    return report(args, stages)


if __name__ == "__main__":
    sys.exit(main())
