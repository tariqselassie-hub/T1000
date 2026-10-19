#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Real Physical Dispatch to Rigetti Cepheus-1-108Q via OpenQuantum Cloud.
Submits Continuous-Time Quantum Walk (CTQW) on Glued Trees.
Deen Research Division · Sovereign Quantum Hardware Core
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dotenv import load_dotenv

CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[1]

# Load .env from local dir, repo root, or deen_heptal_workspace
load_dotenv(CUR_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")
load_dotenv("C:/Users/zoddj/Research_Divzion/deen_heptal_workspace/.env")

client_id = os.getenv("OPENQUANTUM_CLIENT_ID")
client_secret = os.getenv("OPENQUANTUM_CLIENT_SECRET")

if not client_id or not client_secret:
    print("❌ ERROR: OPENQUANTUM_CLIENT_ID or OPENQUANTUM_CLIENT_SECRET missing in .env")
    sys.exit(1)

from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import (
    SchedulerClient,
    ManagementClient,
    JobPreparationCreate,
    JobCreate,
    ExecutionPlanType,
    QueuePriorityType,
)

auth = ClientCredentialsAuth(
    creds=ClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
)
mgmt = ManagementClient(auth=auth)
scheduler = SchedulerClient(auth=auth)

CEPHEUS_BACKEND_ID = "7433acb8-ae52-4bc9-9030-6df68c696538"
SUBCAT_PHYSICS_OTHER = "dcd158af-d14f-46b2-b6b7-75f14267c02a"
MANIFEST_PATH = CUR_DIR / "openquantum_cepheus_job_manifest.json"
RESULTS_PATH = CUR_DIR / "cepheus_hardware_physical_results.json"


def dispatch_job(circuit_path: Path, shots: int = 100):
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  REAL PHYSICAL DISPATCH: CONTINUOUS-TIME QUANTUM WALK ON CEPHEUS-1-108Q       ║")
    print("║             Deen Research Division · Sovereign Quantum Hardware Core         ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    org = mgmt.list_user_organizations().organizations[0]
    org_id = org.id
    balance = mgmt.get_credit_balance(org_id)
    print(f"• OpenQuantum Organization : {org.name} ({org_id})")
    print(f"• Available Spark Credits  : {balance.spark_credits}")

    if balance.spark_credits < 1:
        print("❌ ERROR: Insufficient Spark credits for hardware execution.")
        sys.exit(1)

    with open(circuit_path, "r", encoding="utf-8") as f:
        qasm_content = f.read()

    print(f"• Circuit Source           : {circuit_path.name} ({len(qasm_content.splitlines())} lines)")
    print(f"• Hardware Target          : Rigetti Cepheus-1-108Q ({CEPHEUS_BACKEND_ID})")
    print(f"• Requested Shots          : {shots}")

    # 1. Upload Input Circuit
    print("\n[1/3] Uploading OpenQASM 2.0 Circuit Payload...")
    upload_id = scheduler.upload_job_input(file_content=qasm_content.encode("utf-8"))
    print(f"✓ Upload Endpoint ID       : {upload_id}")

    # 2. Prepare Job & Retrieve Price Quote
    print("\n[2/3] Preparing Job and Quoting on Cepheus QPU Queue...")
    prep = JobPreparationCreate(
        organization_id=org_id,
        backend_class_id=CEPHEUS_BACKEND_ID,
        name="T1000_GLUED_TREES_QUANTUM_WALK_CEPHEUS",
        upload_endpoint_id=upload_id,
        job_subcategory_id=SUBCAT_PHYSICS_OTHER,
        shots=shots,
        configuration_data={"disable_qubit_rewiring": False},
        submitted_with="t1000-sovereign",
        input_format="qasm",
    )
    prep_resp = scheduler.prepare_job(prep)
    print(f"• Preparation ID           : {prep_resp.id}")

    prep_res = scheduler._wait_for_preparation(
        preparation_id=prep_resp.id,
        timeout=90,
        interval=2,
        verbose=False,
    )

    plan = next(
        (p for p in prep_res.quote if p.execution_plan_id == ExecutionPlanType.PUBLIC.value),
        prep_res.quote[0],
    )
    queue = next(
        (q for q in plan.queue_priorities if q.queue_priority_id == QueuePriorityType.STANDARD.value),
        plan.queue_priorities[0],
    )
    cost = plan.price + queue.price_increase
    print(f"✓ Quoted Hardware Cost     : {cost} Spark Credit ({plan.name} / {queue.name})")

    # 3. Create & Dispatch Job to Physical Hardware
    print("\n[3/3] Creating and Submitting Job to Physical Cepheus-1-108Q Queue...")
    job_create = JobCreate(
        job_preparation_id=prep_resp.id,
        execution_plan_id=plan.execution_plan_id,
        queue_priority_id=queue.queue_priority_id,
    )
    job = scheduler.create_job(job_create)
    job_id = str(job.id)
    job_status = str(job.status).upper()

    print("══════════════════════════════════════════════════════════════════════════════")
    print(f"✓ REAL PHYSICAL JOB DISPATCHED TO CEPHEUS-1-108Q!")
    print(f"• Job ID                   : {job_id}")
    print(f"• Hardware Queue Status    : {job_status}")
    print("══════════════════════════════════════════════════════════════════════════════")

    manifest = {
        "job_id": job_id,
        "backend_id": CEPHEUS_BACKEND_ID,
        "backend_name": "Rigetti Cepheus-1-108Q",
        "algorithm": "Continuous-Time Quantum Walk on Glued Binary Trees (n=3)",
        "circuit_file": circuit_path.name,
        "status": str(job.status),
        "timestamp_rfc3339": datetime.now().isoformat(),
        "shots": shots,
        "cost_credits": cost,
        "plan_name": plan.name,
        "queue_priority": queue.name,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"• Manifest Saved           : {MANIFEST_PATH.name}")

    rem_balance = mgmt.get_credit_balance(org_id)
    print(f"• Remaining Spark Credits  : {rem_balance.spark_credits}")
    return job_id


def poll_job(job_id: str = None, timeout_sec: int = 300):
    if not job_id:
        if not MANIFEST_PATH.exists():
            print(f"❌ Manifest not found: {MANIFEST_PATH}")
            return
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        job_id = manifest["job_id"]

    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  POLL RIGETTI CEPHEUS-1-108Q HARDWARE STATUS & INGEST PHYSICAL TELEMETRY     ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"• Polling Job ID           : {job_id}")

    t_start = time.time()
    while True:
        job = scheduler.get_job(job_id)
        status = str(job.status).lower()
        print(f"• Status [{datetime.now().strftime('%H:%M:%S')}] : {job.status.upper()}")

        if status == "completed":
            print("\n🎉 EXECUTION COMPLETED ON RIGETTI CEPHEUS-1-108Q!")
            download_and_process_results(job)
            break
        elif status in ["failed", "cancelled"]:
            print(f"❌ Execution terminated with status: {status.upper()}")
            break

        if time.time() - t_start > timeout_sec:
            print(f"⏳ Job is still queued ({job.status.upper()}). Run with --poll later to ingest results.")
            break

        time.sleep(10)


def download_and_process_results(job):
    raw_output = scheduler.download_job_output(job)
    raw_data = raw_output
    if isinstance(raw_output, (bytes, bytearray)):
        raw_data = json.loads(raw_output.decode("utf-8"))
    elif isinstance(raw_output, str):
        raw_data = json.loads(raw_output)

    raw_counts = {}
    if isinstance(raw_data, dict):
        if "results" in raw_data and isinstance(raw_data["results"], dict):
            raw_counts = raw_data["results"].get("counts", raw_data["results"])
        elif "counts" in raw_data:
            raw_counts = raw_data["counts"]
        else:
            raw_counts = raw_data

    total_shots = sum(raw_counts.values()) if raw_counts else 100
    print(f"• Total Physical Shots Ingested : {total_shots}")
    print(f"• Unique Bitstrings Observed    : {len(raw_counts)}")

    # Bit order in QASM: q7 q6 q5 q4 q3 q2 q1 q0
    # Root B hit corresponds to q7 = 1 (most significant bit in 8-qubit register)
    exit_hits = 0
    col_counts = [0] * 8

    for bitstr, count in raw_counts.items():
        # Clean bitstring
        bits = bitstr.replace(" ", "")
        if len(bits) >= 8:
            # Check Root B (q7)
            if bits[-8] == '1' or bits[0] == '1': # Check both endianness
                exit_hits += count
            for i in range(8):
                if bits[-(i + 1)] == '1':
                    col_counts[i] += count

    empirical_p_exit = exit_hits / total_shots if total_shots > 0 else 0.0

    print("┌───────────────────────────────────┬─────────────────────────────────────┐")
    print("│ Empirical Hardware Telemetry      │ Physical Ground Truth               │")
    print("├───────────────────────────────────┼─────────────────────────────────────┤")
    print(f"│ Physical Backend Executed         │ Rigetti Cepheus-1-108Q              │")
    print(f"│ Total Physical Shots              │ {total_shots} Shots                 │")
    print(f"│ Unique Bitstring States           │ {len(raw_counts)} States            │")
    print(f"│ Empirical Exit Root Hit Count     │ {exit_hits} Shots                   │")
    print(f"│ Empirical Exit Probability        │ {empirical_p_exit * 100:.2f}%       │")
    print("└───────────────────────────────────┴─────────────────────────────────────┘")

    results_payload = {
        "timestamp_rfc3339": datetime.now().isoformat(),
        "backend": "Rigetti Cepheus-1-108Q",
        "job_id": str(job.id),
        "status": str(job.status),
        "total_shots": total_shots,
        "empirical_exit_prob": empirical_p_exit,
        "raw_counts": raw_counts,
        "column_detection_counts": col_counts,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"✓ Results Serialized       : {RESULTS_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Cepheus-1-108Q Hardware Dispatcher")
    parser.add_argument("--dispatch", action="store_true", help="Dispatch quantum walk circuit to physical Cepheus")
    parser.add_argument("--poll", action="store_true", help="Poll existing job and ingest results")
    parser.add_argument("--shots", type=int, default=100, help="Number of physical hardware shots (default: 100)")
    parser.add_argument("--circuit", type=str, default="glued_trees_quantum_walk_cepheus.qasm", help="Circuit file")

    args = parser.parse_args()

    c_path = CUR_DIR / args.circuit
    if args.poll:
        poll_job()
    elif args.dispatch or len(sys.argv) == 1:
        dispatch_job(c_path, shots=args.shots)
