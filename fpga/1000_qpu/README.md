# 1000_QPU

Fixed-point 14-qubit statevector engine for the Digilent Basys 3 (Artix-7 XC7A35T). The
specification, protocol, microarchitecture and results are in
[`docs/fpga_architecture.md`](../../docs/fpga_architecture.md); this file is the workflow.

```text
fpga/1000_qpu/
├── rtl/            qpu_top.v (controller, FIFO, UART), qpu_gate.v (pair pipeline),
│                   qpu_bank.v (parity bank), qpu_seven_seg.v, uart_rx.v, uart_tx.v
├── constraints/    basys3_1000_qpu.xdc
├── sim/            tb_qpu.v, run_sim.py, vectors/ (golden stream from the Rust model)
├── scripts/        build_bitstream.tcl, flash_bitstream.tcl, create_project.tcl, build_and_flash.bat
└── build/          bitstream and the timing and utilisation reports of the last build
```

## Workflow

1. Regenerate the golden vectors when the model or protocol changes:
   `cargo run --release -p t1000 --bin qpu_vectors -- fpga/1000_qpu/sim/vectors`
2. Simulate (Vivado xsim, set `VIVADO_DIR` if Vivado is not in `C:\AMDDesignTools\2026.1\Vivado`):
   `python fpga/1000_qpu/sim/run_sim.py` ends with `RESULT: PASS` when every reply byte equals the model's.
3. Build the bitstream: `vivado -mode batch -source scripts/build_bitstream.tcl` (from `fpga/1000_qpu`).
4. Program the board (volatile SRAM, a power cycle restores the flash): `scripts/flash_bitstream.tcl`.
5. Check the board against the same vectors:
   `.venv\Scripts\python.exe scripts\fpga\verify_1000_qpu.py --port COM6`
   For the full LiH circuit, generate a larger set outside the repository with
   `qpu_vectors <dir> --full` and pass it with `--vectors <dir>`.

The reset input is the centre button (U18, active high). The display shows `1000`; `led[15]` is a
heartbeat, `led[14]` busy, `led[13]` receive-FIFO overflow.
