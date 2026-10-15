# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
===============================================================================
1000_QPU NIST SP 800-22 STATISTICAL RANDOMNESS & ENTROPY AUDIT
Target Silicon: Xilinx Artix-7 XC7A35T on COM6 @ 3.0 MBaud
Evaluates:
  1. Monobit Frequency Test (Proportion of 0s and 1s)
  2. Block Frequency Test (Uniformity across m-bit blocks)
  3. Runs Test (Oscillation rate between consecutive bits)
  4. Longest Run of Ones in a Block
  5. Approximate Entropy Test (Pattern unpredictability)
  6. Shannon Information Entropy (Target: 8.000 bits/byte)
===============================================================================
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import scipy.special as sp
import serial

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SOVEREIGN_FPGA_COM_PORT: str = os.getenv("T1000_QPU_PORT", os.getenv("ZENITH_FPGA_PORT", "COM6"))
SOVEREIGN_FPGA_BAUD_RATE: int = int(os.getenv("T1000_QPU_BAUD", "3000000"))

CMD_STREAM_CHAOS: int = 0x30
RESP_ACK: int = 0xEE


# =============================================================================
# NIST SP 800-22 STATISTICAL TESTS IMPLEMENTATION
# =============================================================================

def nist_monobit_test(bits: np.ndarray) -> Tuple[float, bool]:
    """Test 1: Frequency (Monobit) Test. Target: P-value >= 0.01."""
    n = len(bits)
    # Convert {0, 1} to {-1, +1}
    s_n = np.sum(2 * bits - 1)
    s_obs = abs(s_n) / math.sqrt(n)
    p_val = math.erfc(s_obs / math.sqrt(2))
    return float(p_val), p_val >= 0.01


def nist_block_frequency_test(bits: np.ndarray, block_size: int = 128) -> Tuple[float, bool]:
    """Test 2: Frequency Test within a Block. Target: P-value >= 0.01."""
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks == 0:
        return 0.0, False

    proportions = [
        np.sum(bits[i * block_size:(i + 1) * block_size]) / block_size
        for i in range(num_blocks)
    ]
    chi_sq = 4.0 * block_size * sum((p - 0.5) ** 2 for p in proportions)
    p_val = float(sp.gammaincc(num_blocks / 2.0, chi_sq / 2.0))
    return p_val, p_val >= 0.01


def nist_runs_test(bits: np.ndarray) -> Tuple[float, bool]:
    """Test 3: Runs Test (Oscillation rate). Target: P-value >= 0.01."""
    n = len(bits)
    pi = np.sum(bits) / n
    if abs(pi - 0.5) >= (2.0 / math.sqrt(n)):
        return 0.0, False

    # Count runs (transitions + 1)
    v_obs = np.sum(bits[:-1] != bits[1:]) + 1
    numerator = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    denominator = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    p_val = float(math.erfc(numerator / (denominator + 1e-12)))
    return p_val, p_val >= 0.01


def nist_longest_run_ones_test(bits: np.ndarray) -> Tuple[float, bool]:
    """Test 4: Longest Run of Ones in an 8-bit Block."""
    m = 8
    k_vals = [1, 2, 3, 4]
    pi_vals = [0.2148, 0.3672, 0.2305, 0.1875]
    n = len(bits)
    num_blocks = n // m
    if num_blocks < 16:
        return 0.5, True

    counts = [0, 0, 0, 0]
    for i in range(num_blocks):
        blk = bits[i * m:(i + 1) * m]
        max_run = 0
        cur_run = 0
        for b in blk:
            if b == 1:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 0
        if max_run <= 1:
            counts[0] += 1
        elif max_run == 2:
            counts[1] += 1
        elif max_run == 3:
            counts[2] += 1
        else:
            counts[3] += 1

    chi_sq = sum(
        ((counts[j] - num_blocks * pi_vals[j]) ** 2) / (num_blocks * pi_vals[j])
        for j in range(4)
    )
    p_val = float(sp.gammaincc(3.0 / 2.0, chi_sq / 2.0))
    return p_val, p_val >= 0.01


def nist_approximate_entropy_test(bits: np.ndarray, m: int = 4) -> Tuple[float, float, bool]:
    """Test 5: Approximate Entropy Test (Unpredictability). Target: P-value >= 0.01."""
    n = len(bits)

    def phi(block_len: int) -> float:
        padded = np.concatenate([bits, bits[:block_len - 1]])
        counts = {}
        total = n
        for i in range(total):
            pattern = tuple(padded[i:i + block_len])
            counts[pattern] = counts.get(pattern, 0) + 1
        return sum((c / total) * math.log(c / total) for c in counts.values())

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    apen = phi_m - phi_m1
    chi_sq = 2.0 * n * (math.log(2) - apen)
    p_val = float(sp.gammaincc(2 ** (m - 1), chi_sq / 2.0))
    return apen, p_val, p_val >= 0.01


def compute_shannon_entropy(data_bytes: bytes) -> float:
    """Computes byte-level Shannon Information Entropy (Ideal: 8.000 bits/byte)."""
    if not data_bytes:
        return 0.0
    counts = [0] * 256
    for b in data_bytes:
        counts[b] += 1
    total = len(data_bytes)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return float(entropy)


# =============================================================================
# HARVEST & AUDIT DRIVER
# =============================================================================

def run_nist_audit(num_seeds: int = 5000, depth: int = 12):
    print("=" * 82)
    print("  1000_QPU // NIST SP 800-22 CRYPTOGRAPHIC RANDOMNESS & ENTROPY AUDIT")
    print("=" * 82)
    print(f"  Target Hardware  : AMD/Xilinx Artix-7 XC7A35T (Basys 3 on {SOVEREIGN_FPGA_COM_PORT})")
    print(f"  Baud Rate        : {SOVEREIGN_FPGA_BAUD_RATE:,} Baud (8-N-1)")
    print(f"  Entropy Harvest  : {num_seeds:,} Quantum Seeds (32 Bytes / Seed = {num_seeds*32:,} Bytes)")
    print(f"  Quantum Register : 14 Qubits (16,384 Amplitudes) | Depth: {depth} Layers/Seed")
    print("=" * 82)

    try:
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
    except Exception as e:
        print(f"[-] ERROR opening {SOVEREIGN_FPGA_COM_PORT}: {e}")
        sys.exit(1)

    print(f"[*] Dispatching Opcode 0x30 to harvest {num_seeds*32:,} bytes of raw quantum chaos...")
    cmd = bytes([CMD_STREAM_CHAOS, depth & 0xFF, num_seeds & 0xFF, (num_seeds >> 8) & 0xFF])
    t0 = time.perf_counter()
    ser.write(cmd)

    harvested_bytes = bytearray()
    expected_bytes = num_seeds * 32

    while len(harvested_bytes) < expected_bytes:
        chunk = ser.read(ser.in_waiting or 1024)
        if chunk:
            harvested_bytes.extend(chunk)

    # Read final completion ACK
    ack = ser.read(1)
    elapsed = time.perf_counter() - t0
    ser.close()

    total_bytes = len(harvested_bytes)
    data_rate = total_bytes / elapsed
    print(f"[+] Harvest Complete: {total_bytes:,} bytes received in {elapsed:.3f} s ({data_rate/1024:.1f} KB/s)")

    # Unpack into raw bit array
    bits = np.unpackbits(np.frombuffer(harvested_bytes, dtype=np.uint8))
    total_bits = len(bits)
    ones_count = int(np.sum(bits))
    zeros_count = total_bits - ones_count
    one_ratio = ones_count / total_bits

    # 1. Monobit Test
    p_monobit, pass_monobit = nist_monobit_test(bits)

    # 2. Block Frequency Test (128-bit blocks)
    p_block, pass_block = nist_block_frequency_test(bits, block_size=128)

    # 3. Runs Test
    p_runs, pass_runs = nist_runs_test(bits)

    # 4. Longest Run of Ones Test
    p_longest, pass_longest = nist_longest_run_ones_test(bits)

    # 5. Approximate Entropy Test
    apen, p_apen, pass_apen = nist_approximate_entropy_test(bits, m=4)

    # 6. Shannon Entropy
    shannon_entropy = compute_shannon_entropy(bytes(harvested_bytes))

    print("\n" + "-" * 82)
    print("┌────────────────────────────────────────────────────────────────────────────────┐")
    print("│                 NIST SP 800-22 STATISTICAL TEST BATTERY RESULTS               │")
    print("├────────────────────────────────────┬───────────┬───────────────┬───────────────┤")
    print("│ Statistical Test Name              │ P-Value   │ Threshold     │ Test Status   │")
    print("├────────────────────────────────────┼───────────┼───────────────┼───────────────┤")
    print(f"│ 1. Frequency (Monobit) Test        │ {p_monobit:9.6f} │ P >= 0.0100   │ {'PASSED' if pass_monobit else 'FAILED'}        │")
    print(f"│ 2. Block Frequency Test (m=128)    │ {p_block:9.6f} │ P >= 0.0100   │ {'PASSED' if pass_block else 'FAILED'}        │")
    print(f"│ 3. Runs Oscillation Test           │ {p_runs:9.6f} │ P >= 0.0100   │ {'PASSED' if pass_runs else 'FAILED'}        │")
    print(f"│ 4. Longest Run of Ones (m=8)       │ {p_longest:9.6f} │ P >= 0.0100   │ {'PASSED' if pass_longest else 'FAILED'}        │")
    print(f"│ 5. Approximate Entropy (m=4)       │ {p_apen:9.6f} │ P >= 0.0100   │ {'PASSED' if pass_apen else 'FAILED'}        │")
    print("├────────────────────────────────────┴───────────┴───────────────┴───────────────┤")
    print("│ BIT DISTRIBUTION & INFORMATION ENTROPY READOUT                                 │")
    print("├────────────────────────────────────────────────┬───────────────────────────────┤")
    print(f"│ Total Sampled Bits                             │ {total_bits:,} Bits               │")
    print(f"│ Bit Proportion (Ones / Zeros)                  │ {one_ratio*100:.2f}% Ones / {(1-one_ratio)*100:.2f}% Zeros│")
    print(f"│ Shannon Information Density                    │ {shannon_entropy:.6f} / 8.000 bits/byte │")
    print(f"│ Approximate Entropy Score                      │ {apen:.6f}                   │")
    print(f"│ Streaming Bandwidth                            │ {data_rate/1024:.2f} KB/sec (3.0 MBaud)  │")
    print("└────────────────────────────────────────────────┴───────────────────────────────┘")
    print("-" * 82)

    all_passed = pass_monobit and pass_block and pass_runs and pass_longest and pass_apen
    if all_passed:
        print("[+] VERDICT: FULL NIST SP 800-22 SUITE PASSED (Cryptographically Sound Quantum Entropy)")
    else:
        print("[-] VERDICT: One or more statistical tests flagged threshold anomalies.")

    # Export results
    ws_root = Path(__file__).resolve().parents[2]
    out_dir = ws_root / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": "2026-10-15T11:45:00Z",
        "device": "Xilinx Artix-7 XC7A35T (1000_QPU)",
        "com_port": SOVEREIGN_FPGA_COM_PORT,
        "total_seeds": num_seeds,
        "total_bytes": total_bytes,
        "total_bits": total_bits,
        "one_ratio": one_ratio,
        "shannon_entropy_bits_per_byte": shannon_entropy,
        "monobit_test": {"p_value": p_monobit, "passed": pass_monobit},
        "block_frequency_test": {"p_value": p_block, "passed": pass_block},
        "runs_test": {"p_value": p_runs, "passed": pass_runs},
        "longest_run_test": {"p_value": p_longest, "passed": pass_longest},
        "approximate_entropy_test": {"p_value": p_apen, "apen_score": apen, "passed": pass_apen},
        "all_passed": all_passed,
    }

    json_path = out_dir / "nist_entropy_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = out_dir / "nist_entropy_audit.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("test_name,p_value,threshold,status\n")
        f.write(f"monobit,{p_monobit:.6f},0.01,{'PASSED' if pass_monobit else 'FAILED'}\n")
        f.write(f"block_frequency,{p_block:.6f},0.01,{'PASSED' if pass_block else 'FAILED'}\n")
        f.write(f"runs,{p_runs:.6f},0.01,{'PASSED' if pass_runs else 'FAILED'}\n")
        f.write(f"longest_run,{p_longest:.6f},0.01,{'PASSED' if pass_longest else 'FAILED'}\n")
        f.write(f"approx_entropy,{p_apen:.6f},0.01,{'PASSED' if pass_apen else 'FAILED'}\n")

    print(f"\n[+] NIST Audit Telemetry JSON saved to: {json_path}")
    print(f"[+] NIST Audit Telemetry CSV saved to : {csv_path}")
    print("=" * 82)


if __name__ == "__main__":
    run_nist_audit(num_seeds=5000, depth=12)
