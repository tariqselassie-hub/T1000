"""Cross-check headline README numbers against the committed result files (see docs/CLAIMS.md).

Run from anywhere:  python scripts/check_claims.py

For every claim, scan the numeric leaves of the named JSON file(s) for a value that matches the
number as printed in the README (to its printed precision; percentages may be stored as
fractions). A claim may carry a key hint that the matching JSON path must contain, which rules
out coincidental matches. Prints one line per claim, FOUND with the JSON path, or MISSING /
NO-FILE, and exits non-zero if any claim is not found.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# (README section, claim, file(s) relative to results/, number as printed, optional key hint)
CLAIMS = [
    ("4", "linear XEB", ["experiments/frontier_rcs_results.json"], "1.0045", "exact_xeb"),
    ("4", "KS D statistic", ["experiments/frontier_rcs_results.json"], "0.0021", "ks_statistic"),
    ("5", "MNIST 10-qubit VQC test accuracy %", ["experiments/mnist_qnn_training.json"], "84.50", "final_test_acc_pct"),
    ("5", "MNIST state coherence", ["experiments/mnist_qnn_training.json"], "350.94", "mean_coherence_l1"),
    ("5", "MNIST us per sample", ["experiments/mnist_qnn_training.json"], "34.38", "per_sample_latency_us"),
    ("14", "CHSH bridge value (exact bench)", ["experiments/chsh_qrack_ace_results.json"], "2.82842712", "chsh_s_value"),
    ("14", "36Q prefix ms", ["experiments/cut_bridge_36q_results.json"], "316.50", None),
    ("15", "Path A worst-group accuracy %", ["experiments/path_a_vs_c_comparison_report.json"], "61.91", "worst_group_accuracy_pct"),
    ("15", "Path A G2 accuracy %", ["experiments/path_a_vs_c_comparison_report.json"], "71.81", "g2_water_land"),
    ("16", "QMUX coarse accuracy %", ["experiments/ultrametric_tree_sorter_report.json"], "70.69", None),
    ("17", "glued trees classical hit %", ["experiments/glued_trees_report.json"], "49.40", "classical_success_rate_pct"),
    ("17", "glued trees quantum peak %", ["experiments/glued_trees_report.json"], "54.16", "quantum_peak_exit_prob"),
    ("18", "valve K=2 peak %", ["experiments/glued_tree_valve_report.json"], "77.16", "quantum_target_peak_prob"),
    ("19", "cascaded 2x2 peak %", ["experiments/cascaded_tree_valve_report.json"], "60.31", "quantum_target_peak_prob"),
    ("19", "cascaded contrast 2x2", ["experiments/cascaded_tree_valve_report.json"], "2489", "quantum_routing_contrast"),
    ("20", "topological walk joint accuracy %", ["topological_walk_sorter_telemetry.json"], "82.00", "joint_accuracy_pct"),
    ("20", "topological walk contrast", ["topological_walk_sorter_telemetry.json"], "2050", "mean_routing_contrast"),
    ("21", "Nighthawk SWAP fidelity", ["data/nighthawk_99_plaquette_telemetry.json"], "0.2119", "swap_fidelity"),
    ("21", "Nighthawk plaquette fidelity", ["data/nighthawk_99_plaquette_telemetry.json"], "0.6687", "teleport_fidelity"),
    ("21", "Cepheus seam purification before", ["data/cepheus_108q_telemetry.json"], "0.8588", "fid_swap_corner"),
    ("21", "Cepheus seam purification after", ["data/cepheus_108q_telemetry.json"], "0.9146", "fid_teleport_corner"),
    ("22", "Cepheus hardware exit probability", ["../experiments/cepheus_glued_trees_30q/cepheus_hardware_results.json"], "0.652", "empirical_exit_prob"),
    ("22", "Cepheus predictive accuracy %", ["../experiments/cepheus_glued_trees_30q/cepheus_hardware_results.json"], "84.43", "predictive_accuracy_pct"),
    ("22", "Cepheus simulator prediction", ["data/cepheus_glued_trees_telemetry.json"], "0.7722", "seam_noisy_peak_exit_prob"),
    ("27", "QEC threshold", ["data/vnni_qec_threshold_report.json"], "0.0814", "empirical_threshold"),
    ("27", "QEC suppression at p=0.001", ["data/vnni_qec_threshold_report.json"], "41.67", "suppression_ratio"),
    ("27", "QEC decodings per second (M)", ["data/vnni_qec_threshold_report.json"], "20.72", "throughput_mshots_sec"),
    ("28", "Coulomb molecules per second", ["data/molecular_eigensystem_report.json"], "14352.5", "throughput_mols_per_sec"),
    ("28", "QM9 ingestion molecules per second", ["data/molecular_eigensystem_report.json"], "4056.4", "throughput_mols_per_sec"),
    ("4", "noisy RCS fidelity gain 14Q % (best depth)", ["rcs_physical_noise_chaos.json"], "5147", "fidelity_gain_pct"),
    ("4", "noisy RCS fidelity gain 16Q % (best depth)", ["rcs_physical_noise_chaos.json"], "13543", "fidelity_gain_pct"),
    ("M158", "Szegedy quantum hit", ["data/directed_heptal_quantum_walk_report.json"], "0.7351", "peak_quantum_sink_prob"),
]


def leaves(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


def matches(value, printed):
    digits = len(printed.split(".")[1]) if "." in printed else 0
    target = float(printed)
    for scale in (1.0, 100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e-9):
        # The README rounds or truncates, so accept anything within one unit of the last digit.
        if abs(value * scale - target) < 10 ** (-digits):
            return True
    return False


def find(files, printed, hint):
    present = [f for f in files if (R / f).exists()]
    for f in present:
        data = json.loads((R / f).read_text(encoding="utf-8"))
        for path, v in leaves(data):
            if (hint is None or hint in path) and matches(v, printed):
                return "FOUND", f"{f} :: {path} = {v:g}"
    return ("MISSING" if present else "NO-FILE"), ", ".join(files)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    missing = 0
    for sec, claim, files, printed, hint in CLAIMS:
        status, where = find(files, printed, hint)
        missing += status != "FOUND"
        print(f"{status:8} S{sec:5} {claim:44} {printed:11} {where}")
    print(f"\n{len(CLAIMS) - missing}/{len(CLAIMS)} claims found in committed results")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
