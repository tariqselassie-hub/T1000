# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Runs a dashboard circuit on the 1000_QPU board (Basys 3) through its UART protocol.

The protocol and arithmetic are specified by `crates/t1000/src/qpu` and documented in
`docs/fpga_architecture.md`. A circuit is compiled to controlled 2x2 gates (25-bit coefficients,
Q2.23), the register is reset to |0...0> with 28 fractional bits, every gate is acknowledged, and the
full statevector is read back. Nothing here is estimated: the returned amplitudes are the board's.
"""

import cmath
import math
import struct
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import serial
except ImportError:  # pragma: no cover - pyserial is optional for the dashboard
    serial = None

PING, PING_REPLY = 0xAA, 0x10
VERSION, SET_N, RESET, READ, GATE, EXPECT_Z = 0x01, 0x02, 0x03, 0x05, 0x06, 0x07
ACK = 0xEE
IDENT = bytes([0x51, 0x50, 0x55, 0x01])

COEF_FRAC = 23
COEF_MAX, COEF_MIN = (1 << 24) - 1, -(1 << 24)
STATE_FRAC = 28
READ_CHUNK = 128  # amplitudes per READ: 1 KB reply, well inside what the host buffers

_lock = threading.Lock()

GateSpec = Tuple[int, int, List[complex]]  # (target, control mask, [m00, m01, m10, m11])


class BoardError(RuntimeError):
    """The board did not answer as the protocol specifies."""


def _q(x: float) -> int:
    return max(COEF_MIN, min(COEF_MAX, int(round(x * (1 << COEF_FRAC)))))


def _matrix(g_type: str, theta: float):
    inv = 1.0 / math.sqrt(2.0)
    if g_type == "RY":
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return [c, -s, s, c]
    if g_type == "RX":
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return [c, -1j * s, -1j * s, c]
    if g_type == "RZ":
        return [cmath.exp(-1j * theta / 2.0), 0, 0, cmath.exp(1j * theta / 2.0)]
    return {
        "H": [inv, inv, inv, -inv],
        "X": [0, 1, 1, 0],
        "Z": [1, 0, 0, -1],
        "S": [1, 0, 0, 1j],
        "T": [1, 0, 0, cmath.exp(1j * math.pi / 4.0)],
    }.get(g_type)


def compile_circuit(n_qubits: int, layers: List[Dict[str, Any]]) -> List[GateSpec]:
    """Dashboard layers to board gates; mirrors which gates `simulate_quantum_circuit` applies."""
    gates: List[GateSpec] = []
    for layer in layers:
        for g in layer.get("gates", []):
            g_type = str(g.get("type", "")).upper()
            target = int(g.get("target", 0))
            if target >= n_qubits:
                continue
            m = _matrix(g_type, float(g.get("theta", 0.0)))
            if m is not None:
                gates.append((target, 0, m))
            elif g_type in ("CZ", "CX", "SWAP"):
                ctrl = int(g.get("control", (target + 1) % n_qubits))
                if ctrl == target or ctrl >= n_qubits:
                    continue
                if g_type == "CZ":
                    gates.append((target, 1 << ctrl, _matrix("Z", 0.0)))
                elif g_type == "CX":
                    gates.append((target, 1 << ctrl, _matrix("X", 0.0)))
                else:  # SWAP as three CNOTs
                    x = _matrix("X", 0.0)
                    gates += [(target, 1 << ctrl, x), (ctrl, 1 << target, x), (target, 1 << ctrl, x)]
    return gates


def gate_frame(spec: GateSpec) -> bytes:
    target, ctrl, m = spec
    words = []
    for c in m:
        c = complex(c)
        words += [_q(c.real), _q(c.imag)]
    return bytes([GATE, target]) + struct.pack("<H", ctrl) + struct.pack("<8i", *words)


def _read(ser, count: int, timeout: float = 2.0) -> bytes:
    data = bytearray()
    deadline = time.perf_counter() + timeout
    while len(data) < count and time.perf_counter() < deadline:
        chunk = ser.read(min(4096, count - len(data)))
        if chunk:
            data += chunk
            deadline = time.perf_counter() + timeout
    return bytes(data)


def _exchange(ser, payload: bytes, reply_len: int) -> bytes:
    ser.write(payload)
    return _read(ser, reply_len)


def run_on_board(port: str, baud: int, n_qubits: int, layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Executes the circuit on the board; returns the statevector and measured transport figures."""
    if serial is None:
        raise BoardError("pyserial is not available")
    n_qubits = min(max(n_qubits, 1), 14)
    gates = compile_circuit(n_qubits, layers)
    dim = 1 << n_qubits
    with _lock:
        try:
            ser = serial.Serial(port, baud, timeout=0.2, write_timeout=5.0)
        except serial.SerialException as e:
            raise BoardError(f"cannot open {port}: {e}")
        with ser:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            if _exchange(ser, bytes([VERSION]), 5) != IDENT + bytes([ACK]):
                raise BoardError("the board does not answer VERSION: the 1000_QPU statevector bitstream is not loaded")

            t0 = time.perf_counter()
            setup = _exchange(ser, bytes([SET_N, n_qubits, RESET]) + struct.pack("<i", 1 << STATE_FRAC), 2)
            if setup != bytes([ACK, ACK]):
                raise BoardError(f"SET_N/RESET answered {setup.hex()}")
            t_gates = time.perf_counter()
            batch = b"".join(gate_frame(g) for g in gates)
            ser.write(batch)
            acks = _read(ser, len(gates), timeout=5.0)
            if acks != bytes([ACK]) * len(gates):
                raise BoardError(f"{len(acks)} of {len(gates)} gates acknowledged")
            t_read = time.perf_counter()

            raw = bytearray()
            for addr in range(0, dim, READ_CHUNK):
                count = min(READ_CHUNK, dim - addr)
                reply = _exchange(ser, bytes([READ]) + struct.pack("<HH", addr, count), 8 * count + 1)
                if len(reply) != 8 * count + 1 or reply[-1] != ACK:
                    raise BoardError(f"READ at {addr} answered {len(reply)} bytes")
                raw += reply[:-1]
            words = np.frombuffer(bytes(raw), dtype="<i4").reshape(dim, 2).astype(np.float64)
            psi = (words[:, 0] + 1j * words[:, 1]) / float(1 << STATE_FRAC)

            expect = []
            for q in range(n_qubits):
                r = _exchange(ser, bytes([EXPECT_Z]) + struct.pack("<H", 1 << q), 11)
                if len(r) != 11 or r[-1] != ACK:
                    raise BoardError("EXPECT_Z failed")
                expect.append(r[:10])
            norm_raw = _exchange(ser, bytes([EXPECT_Z]) + struct.pack("<H", 0), 11)
            if len(norm_raw) != 11 or norm_raw[-1] != ACK:
                raise BoardError("EXPECT_Z (norm) failed")
            t_end = time.perf_counter()

    def i80(b: bytes) -> int:
        return int.from_bytes(b, "little", signed=True)

    norm = i80(norm_raw[:10])
    return {
        "psi": psi,
        "gates_sent": len(gates),
        "board_norm_squared": norm / float(1 << (2 * STATE_FRAC)),
        "board_expect_z": [round(i80(e) / norm, 4) for e in expect],
        "wall_time_us": (t_end - t0) * 1e6,
        "gate_phase_us": (t_read - t_gates) * 1e6,
        "uart_bytes_sent": len(batch),
    }
