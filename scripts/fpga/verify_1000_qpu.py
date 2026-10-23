# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Hardware conformance of the loaded 1000_QPU design on the Basys 3.

    .venv\\Scripts\\python.exe scripts\\fpga\\verify_1000_qpu.py
    .venv\\Scripts\\python.exe scripts\\fpga\\verify_1000_qpu.py --port COM6 --vectors path\\to\\vectors

The golden vectors are produced by the bit-exact Rust model (`qpu_vectors`); this script replays
the same command stream the simulation testbench uses and compares every reply byte. A pass means
the board computes exactly what `crates/t1000/src/qpu` computes. The protocol is documented in
`crates/t1000/src/qpu/protocol.rs`.

Stages: 0 port and passive listen, 1 identity (ping, version), 2 golden-vector replay, 3 gate
throughput at 14 qubits (informational timing, every gate must be acknowledged).
Nothing is written to the FPGA configuration. Exit status 0 means every stage passed.
"""

import argparse
import datetime
import json
import os
import struct
import sys
import time
from pathlib import Path

import serial
import serial.tools.list_ports

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTORS = ROOT / "fpga" / "1000_qpu" / "sim" / "vectors"
DEFAULT_PORT = os.getenv("T1000_QPU_PORT", "COM6")
DEFAULT_BAUD = int(os.getenv("T1000_QPU_BAUD", "3000000"))

PING, PING_REPLY = 0xAA, 0x10
VERSION, SET_N, RESET, GATE = 0x01, 0x02, 0x03, 0x06
ACK = 0xEE
IDENT = bytes([0x51, 0x50, 0x55, 0x01])
H_COEF = round(2**23 / 2**0.5)  # 1/sqrt(2) in Q2.23


class Stage:
    def __init__(self, name):
        self.name, self.passed, self.details, self.notes = name, False, {}, []

    def finish(self, passed, **details):
        self.passed = passed
        self.details.update(details)
        return self

    def as_dict(self):
        return {"stage": self.name, "passed": self.passed, "details": self.details, "notes": self.notes}


def read_exact(ser, count, timeout):
    """Read up to `count` bytes, giving up after `timeout` seconds without progress."""
    data = bytearray()
    deadline = time.perf_counter() + timeout
    while len(data) < count and time.perf_counter() < deadline:
        chunk = ser.read(min(4096, count - len(data)))
        if chunk:
            data += chunk
            deadline = time.perf_counter() + timeout
    return bytes(data)


def exchange(ser, payload, reply_len, timeout=1.0):
    ser.reset_input_buffer()
    ser.write(bytes(payload))
    return read_exact(ser, reply_len, timeout)


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
    return s.finish(info is not None and not stray)


def stage_identity(ser):
    s = Stage("1 identity: ping and version")
    ping = exchange(ser, [PING], 1)
    version = exchange(ser, [VERSION], 5)
    s.details.update(ping=ping.hex(), version=version.hex())
    if ping != bytes([PING_REPLY]):
        s.notes.append("no ping reply: is the new 1000_QPU bitstream loaded, and the reset button released?")
    elif version != IDENT + bytes([ACK]):
        s.notes.append("the board answers ping but not VERSION: it is still running the legacy design")
    return s.finish(ping == bytes([PING_REPLY]) and version == IDENT + bytes([ACK]))


def load_vectors(directory):
    stim = (directory / "stim.bin").read_bytes()
    expect = (directory / "expect.bin").read_bytes()
    words = [int(w, 16) for w in (directory / "frames.mem").read_text().split()]
    return stim, expect, list(zip(words[0::2], words[1::2]))


def stage_replay(ser, directory):
    """Frames with a reply longer than one byte are sent stop-and-wait; the others are streamed."""
    s = Stage("2 golden-vector replay against the Rust model")
    stim, expect, frames = load_vectors(directory)
    got = bytearray()
    sent = 0
    due = 0
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    for index, (flen, rlen) in enumerate(frames):
        ser.write(stim[sent : sent + flen])
        sent += flen
        due += rlen
        if rlen > 1 or due - len(got) > 64:
            got += read_exact(ser, due - len(got), 2.0)
            if len(got) < due:
                break
    got += read_exact(ser, len(expect) - len(got), 2.0)
    seconds = time.perf_counter() - t0
    bad = next((i for i in range(min(len(got), len(expect))) if got[i] != expect[i]), None)
    s.details.update(
        frames=len(frames),
        stimulus_bytes=len(stim),
        reply_bytes_expected=len(expect),
        reply_bytes_received=len(got),
        first_mismatch_byte=bad,
        seconds=round(seconds, 3),
    )
    if bad is not None:
        s.details["got_at_mismatch"] = got[bad : bad + 8].hex()
        s.details["expected_at_mismatch"] = expect[bad : bad + 8].hex()
    elif len(got) < len(expect):
        s.notes.append("the board stopped answering before the end of the stream")
    return s.finish(bad is None and len(got) == len(expect))


def h_gate(target):
    words = [H_COEF, 0, H_COEF, 0, H_COEF, 0, -H_COEF, 0]
    return bytes([GATE, target, 0, 0]) + struct.pack("<8i", *words)


def stage_throughput(ser, gates):
    s = Stage("3 gate throughput at 14 qubits")
    setup = exchange(ser, bytes([SET_N, 14]) + bytes([RESET]) + struct.pack("<i", 1 << 28), 2)
    if setup != bytes([ACK, ACK]):
        return s.finish(False, setup=setup.hex())
    batch = b"".join(h_gate(i % 14) for i in range(gates))
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(batch)
    acks = read_exact(ser, gates, 2.0)
    seconds = time.perf_counter() - t0
    # The UART carries 36 bytes per gate: 10 bits each at the line rate.
    uart_us = 36 * 10 / ser.baudrate * 1e6
    s.details.update(
        gates=gates,
        acknowledged=len(acks),
        microseconds_per_gate=round(seconds / gates * 1e6, 1),
        uart_floor_microseconds=round(uart_us, 1),
        fabric_microseconds_at_100mhz=round(8192 / 100, 1),
    )
    s.notes.append("per-gate time is limited by the 36-byte frame on the UART, not by the fabric")
    return s.finish(acks == bytes([ACK]) * gates)


def run(args):
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2, write_timeout=2.0)
    except serial.SerialException as e:
        print(f"cannot open {args.port}: {e}")
        return None
    ser.reset_output_buffer()
    stages = [stage_port(ser, args.port)]
    stages.append(stage_identity(ser))
    if stages[-1].passed:
        stages.append(stage_replay(ser, Path(args.vectors)))
        stages.append(stage_throughput(ser, 50 if args.quick else 400))
    ser.close()
    return stages


def report(args, stages):
    print("=" * 79)
    print(f" 1000_QPU HARDWARE CONFORMANCE   port {args.port}   {args.baud:,} baud")
    print("=" * 79)
    for s in stages:
        print(f" [{'PASS' if s.passed else 'FAIL'}] {s.name}")
        for k, v in s.details.items():
            print(f"        {k}: {v}")
        for n in s.notes:
            print(f"        note: {n}")
    ok = len(stages) == 4 and all(s.passed for s in stages)
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
                "vectors": str(args.vectors),
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
    p.add_argument("--vectors", default=str(DEFAULT_VECTORS), help="directory with stim.bin, expect.bin, frames.mem")
    p.add_argument("--quick", action="store_true", help="fewer throughput gates")
    args = p.parse_args()
    stages = run(args)
    if stages is None:
        return 2
    return report(args, stages)


if __name__ == "__main__":
    sys.exit(main())
