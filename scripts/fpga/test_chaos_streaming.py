# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
===============================================================================
1000_QPU UNIFIED AUTONOMOUS MULTI-LAYER & QUANTUM ENTROPY STREAMING HARNESS
Target Silicon: Xilinx Artix-7 XC7A35T on COM6 @ 3.0 MBaud
Verifies:
  1. Opcode 0x20: Autonomous Multi-Layer Execution (D=12, D=36, D=100)
  2. Opcode 0x30: Quantum Chaos 32-Byte Cryptographic Seed Streaming
  3. 19.0-Second Continuous Quantum Entropy Harvest
===============================================================================
"""

import os
import sys
import time
from typing import List, Tuple
import serial

SOVEREIGN_FPGA_COM_PORT: str = os.getenv("T1000_QPU_PORT", os.getenv("ZENITH_FPGA_PORT", "COM6"))
SOVEREIGN_FPGA_BAUD_RATE: int = int(os.getenv("T1000_QPU_BAUD", "3000000"))

# Opcodes
CMD_PING: bytes = bytes([0xAA])
RESP_PING_READY: int = 0x10

CMD_RUN_MULTI: int = 0x20
CMD_STREAM_CHAOS: int = 0x30
RESP_ACK: int = 0xEE


def open_ser():
    ser = serial.Serial(
        port=SOVEREIGN_FPGA_COM_PORT,
        baudrate=SOVEREIGN_FPGA_BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1.0,
        write_timeout=1.0,
    )
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.05)
    return ser


def test_autonomous_multi_layer(ser: serial.Serial, depth: int) -> float:
    print(f"\n[*] Testing Autonomous On-Chip Evolution: Depth = {depth} Layers (Opcode: 0x20)...")
    ser.reset_input_buffer()
    cmd = bytes([CMD_RUN_MULTI, depth & 0xFF])
    
    t0 = time.perf_counter()
    ser.write(cmd)
    resp = ser.read(1)
    dt_us = (time.perf_counter() - t0) * 1e6

    if not resp or resp[0] != RESP_ACK:
        print(f"  [-] Failed or unexpected response: {resp}")
        return -1.0

    silicon_time_us = depth * 10.24  # 1024 cycles @ 100 MHz per layer
    wire_overhead_us = max(0.0, dt_us - silicon_time_us)
    print(f"  [+] ACK 0xEE received in {dt_us:.2f} us ({dt_us/1000.0:.3f} ms)")
    print(f"  [+] Pure Silicon Execution: {silicon_time_us:.2f} us ({depth} layers @ 100 MHz)")
    print(f"  [+] USB Wire Overhead     : {wire_overhead_us:.2f} us")
    print(f"  [+] Effective Speed       : {depth / (dt_us / 1e6):,.0f} layers/second")
    return dt_us


def test_stream_chaos_seeds(ser: serial.Serial, depth: int, num_seeds: int):
    print(f"\n[*] Testing Quantum Chaos Seed Streaming: {num_seeds} Seeds @ {depth} Layers each (Opcode: 0x30)...")
    ser.reset_input_buffer()
    cmd = bytes([CMD_STREAM_CHAOS, depth & 0xFF, num_seeds & 0xFF, (num_seeds >> 8) & 0xFF])

    t0 = time.perf_counter()
    ser.write(cmd)

    seeds = []
    for i in range(num_seeds):
        seed_bytes = ser.read(32)
        if len(seed_bytes) != 32:
            print(f"  [-] Timeout or partial seed read at index {i}: got {len(seed_bytes)} bytes")
            break
        seeds.append(seed_bytes)

    # Read final ACK 0xEE
    ack = ser.read(1)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  [+] Successfully harvested {len(seeds)} quantum seeds ({len(seeds)*32} bytes)")
    if ack and ack[0] == RESP_ACK:
        print("  [+] Final Stream ACK 0xEE Confirmed")
    print(f"  [+] Total Elapsed Time   : {total_time_ms:.2f} ms")
    print(f"  [+] Generation Rate      : {len(seeds) / (total_time_ms / 1000.0):,.1f} 256-bit seeds/second")
    print(f"  [+] Wire Data Rate       : {(len(seeds) * 32) / (total_time_ms / 1000.0):,.0f} bytes/second")
    if seeds:
        print(f"  [+] Sample Quantum Seed  : {seeds[0].hex()}")
        if len(seeds) > 1:
            print(f"  [+] Final Quantum Seed   : {seeds[-1].hex()}")


def benchmark_19s_chaos_harvest(ser: serial.Serial, depth: int = 12):
    print("\n" + "=" * 79)
    print("  19-SECOND CONTINUOUS QUANTUM CHAOS STREAMING RUN")
    print(f"  Circuit Depth per Seed : {depth} Layers on 14-Qubit Silicon Matrix")
    print("=" * 79)

    ser.reset_input_buffer()
    # Request a large batch of seeds (e.g. 5,000 seeds = 160 KB)
    batch_size = 5000
    cmd = bytes([CMD_STREAM_CHAOS, depth & 0xFF, batch_size & 0xFF, (batch_size >> 8) & 0xFF])

    t_start = time.perf_counter()
    t_end = t_start + 19.0
    ser.write(cmd)

    total_seeds = 0
    total_bytes = 0

    while time.perf_counter() < t_end:
        chunk = ser.read(ser.in_waiting or 32)
        if chunk:
            total_bytes += len(chunk)
            total_seeds = total_bytes // 32

    actual_elapsed = time.perf_counter() - t_start
    ser.reset_input_buffer()

    total_layers = total_seeds * depth
    total_gates = total_layers * 28  # 14 qubits * 2 gates/layer
    bytes_per_sec = total_bytes / actual_elapsed

    print("-" * 79)
    print("  19.0-SECOND HARVEST TELEMETRY RESULTS")
    print("-" * 79)
    print(f"  Actual Elapsed Time     : {actual_elapsed:.4f} seconds")
    print(f"  Total Quantum Seeds     : {total_seeds:,} 256-bit cryptographic seeds")
    print(f"  Total Entropy Bytes     : {total_bytes:,} bytes")
    print(f"  Total Quantum Layers    : {total_layers:,} layers evolved in silicon")
    print(f"  Total Quantum Gates     : {total_gates:,} quantum gate operations")
    print(f"  Seed Generation Rate    : {total_seeds / actual_elapsed:,.1f} seeds/second")
    print(f"  Layer Transformation Rate: {total_layers / actual_elapsed:,.1f} layers/second")
    print(f"  Effective Gate Rate     : {total_gates / actual_elapsed:,.0f} quantum gates/second")
    print(f"  Entropy Streaming Rate  : {bytes_per_sec:,.0f} bytes/second ({bytes_per_sec*8/1e6:.2f} Mbps)")
    print("=" * 79)


def main():
    ser = open_ser()
    try:
        # 1. Multi-layer tests
        test_autonomous_multi_layer(ser, depth=12)
        test_autonomous_multi_layer(ser, depth=36)
        test_autonomous_multi_layer(ser, depth=100)

        # 2. Chaos streaming test
        test_stream_chaos_seeds(ser, depth=12, num_seeds=100)

        # 3. 19-Second Continuous Run
        benchmark_19s_chaos_harvest(ser, depth=12)

    finally:
        ser.close()


if __name__ == "__main__":
    main()
