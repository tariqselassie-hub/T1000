# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Charting: Pitt Quantum Kernel Ridge Regression (Q-KRR) & Stark-VQC benchmarks.

Every number is read from the recorded reports of `pitt_qkrr_train` and `pitt_vqc_train`:
  - results/data/pitt_{qkrr,vqc}_training_report*.json            (reference host)
  - results/hosts/<host>/data/pitt_{qkrr,vqc}_training_report*.json (other hosts)
When several reports share a configuration, the most recent one is used. Models flagged
`uses_target_as_input` (the Stark encodings on the dipole target) are drawn hatched in red and
labelled as a target leak, not as a prediction.

Panels:
  1. Dipole test R^2, all models
  2. Polarizability test R^2, all models
  3. ZZ kernel feature-map depth L=1 vs L=2 (test R^2, both targets)
  4. Train -> test generalization gap (test MSE / train MSE) on polarizability

Strictly adheres to Deen OS Sovereign Visual Specification:
  - Deep Obsidian (#05080d), Void Charcoal (#0f172a), Slate Grid (#1e293b)
  - Quantum Emerald (#00e676, #00ff88), Hardware Gold (#ffb300), Alert Crimson (#ef4444)
  - Zero cyan/purple, zero emojis.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme,
    style_card,
    save_deen_figure,
    BG_OBSIDIAN,
    CARD_CHARCOAL,
    GREEN_QUANTUM,
    GOLD_HARDWARE,
    GOLD_BRIGHT,
    RED_ALERT,
    TEXT_POLAR,
    TEXT_SLATE,
    TEXT_MUTED,
    GRID_SLATE,
)

WS_ROOT = Path(__file__).resolve().parents[2]

# Short display names for the model names written by the two binaries.
DISPLAY = {
    "Classical-Linear-Ridge": "Linear Ridge\n(19-D)",
    "Classical-RBF-Kernel": "RBF Kernel",
    "Quantum-Kernel-ZZ-8Q": "ZZ Q-KRR\n8Q",
    "Quantum-Stark-Kernel-8Q": "Stark Q-KRR\n8Q",
    "Classical-Ridge-19D": "Ridge 19-D\n(VQC run)",
    "Standard-VQC-8Q": "Standard VQC\n8Q",
    "Stark-Coupled-VQC-8Q": "Stark VQC\n8Q",
}
CLASSICAL = {"Classical-Linear-Ridge", "Classical-RBF-Kernel", "Classical-Ridge-19D"}


def _load(kind: str) -> list[dict]:
    """All recorded reports of one binary (`qkrr` or `vqc`), each tagged with its file path."""
    pattern = f"pitt_{kind}_training_report*.json"
    paths = list((WS_ROOT / "results" / "data").glob(pattern))
    paths += list((WS_ROOT / "results" / "hosts").glob(f"*/data/{pattern}"))
    reports = []
    for p in sorted(paths):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = p.relative_to(WS_ROOT).as_posix()
        reports.append(data)
    return reports


def _latest(reports: list[dict], **match) -> dict | None:
    """Most recent report whose fields equal `match`."""
    hits = [r for r in reports if all(r.get(k) == v for k, v in match.items())]
    return max(hits, key=lambda r: r.get("timestamp_utc", "")) if hits else None


def has_reports() -> bool:
    return bool(_load("qkrr") or _load("vqc"))


def _target_trials(qkrr: list[dict], vqc: list[dict], target: str) -> tuple[list[dict], list[str]]:
    """Trials for one target: Q-KRR at depth 1 followed by the VQC run, plus their sources."""
    trials, sources = [], []
    q = _latest(qkrr, target_property=target, depth=1)
    if q:
        trials += q["trials"]
        sources.append(f"{q['_path']} ({q['train_samples']}/{q['test_samples']})")
    v = _latest(vqc, target_property=target)
    if v:
        trials += v["trials"]
        sources.append(f"{v['_path']} ({v['train_samples']}/{v['test_samples']})")
    return trials, sources


def _bar_color(trial: dict) -> str:
    if trial.get("uses_target_as_input"):
        return RED_ALERT
    return GOLD_HARDWARE if trial["name"] in CLASSICAL else GREEN_QUANTUM


def _r2_panel(ax, trials: list[dict], sources: list[str], title: str, unit: str) -> None:
    style_card(ax, title=title, subtitle="Test R^2 (higher is better); red hatched = target leak")
    if not trials:
        ax.text(0.5, 0.5, "no recorded report", ha="center", color=TEXT_MUTED, transform=ax.transAxes)
        return
    x = np.arange(len(trials))
    r2 = [t["test_r2"] for t in trials]
    bars = ax.bar(
        x,
        r2,
        0.6,
        color=[_bar_color(t) for t in trials],
        edgecolor=TEXT_POLAR,
        linewidth=0.8,
        hatch=["//" if t.get("uses_target_as_input") else "" for t in trials],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(t["name"], t["name"]) for t in trials], fontsize=8, color=TEXT_SLATE)
    ax.set_ylabel("Test R^2", color=TEXT_POLAR, fontsize=9.5)
    ax.set_ylim(min(0.0, min(r2) - 0.1), 1.08)
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    for b, t in zip(bars, trials):
        label = f"{t['test_r2']:.3f}\nMSE {t['test_mse']:.2f} {unit}"
        if t.get("uses_target_as_input"):
            label = f"{t['test_r2']:.3f}\nTARGET LEAK"
        ax.text(
            b.get_x() + b.get_width() / 2,
            max(b.get_height(), 0) + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=RED_ALERT if t.get("uses_target_as_input") else TEXT_POLAR,
            fontweight="bold",
        )
    ax.text(
        0.01,
        -0.2,
        "source: " + "; ".join(sources),
        transform=ax.transAxes,
        fontsize=6.5,
        color=TEXT_MUTED,
    )


def _depth_panel(ax, qkrr: list[dict]) -> None:
    style_card(
        ax,
        title="PANEL 3: ZZ KERNEL FEATURE-MAP DEPTH L=1 VS L=2",
        subtitle="Test R^2 of the ZZ Q-KRR model; the Stark kernel does not depend on depth",
    )
    targets = ["dipole", "polarizability"]
    width = 0.35
    for i, depth in enumerate([1, 2]):
        vals = []
        for target in targets:
            rep = _latest(qkrr, target_property=target, depth=depth)
            zz = next((t for t in rep["trials"] if t["name"] == "Quantum-Kernel-ZZ-8Q"), None) if rep else None
            vals.append(zz["test_r2"] if zz else np.nan)
        pos = np.arange(len(targets)) + (i - 0.5) * width
        bars = ax.bar(
            pos,
            vals,
            width,
            label=f"Depth L={depth}",
            color=GREEN_QUANTUM if depth == 1 else GOLD_BRIGHT,
            edgecolor=TEXT_POLAR,
            linewidth=0.8,
        )
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=8.5, color=TEXT_POLAR, fontweight="bold")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(["Dipole (Debye)", "Polarizability (A^3)"], fontsize=9.5, color=TEXT_SLATE)
    ax.set_ylabel("Test R^2", color=TEXT_POLAR, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.85)


def _gap_panel(ax, trials: list[dict]) -> None:
    style_card(
        ax,
        title="PANEL 4: TRAIN -> TEST GENERALIZATION GAP (POLARIZABILITY)",
        subtitle="Test MSE / Train MSE (1.0 = no overfitting)",
    )
    if not trials:
        ax.text(0.5, 0.5, "no recorded report", ha="center", color=TEXT_MUTED, transform=ax.transAxes)
        return
    ratios = [t["test_mse"] / max(t["train_mse"], 1e-12) for t in trials]
    y = np.arange(len(trials))
    bars = ax.barh(y, ratios, height=0.5, color=[_bar_color(t) for t in trials],
                   edgecolor=TEXT_POLAR, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(t["name"], t["name"]).replace("\n", " ") for t in trials],
                       fontsize=8.5, color=TEXT_SLATE)
    ax.set_xscale("log")
    ax.axvline(1.0, color=TEXT_MUTED, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Test MSE / Train MSE (log scale)", color=TEXT_POLAR, fontsize=9.5)
    for b, r, t in zip(bars, ratios, trials):
        ax.text(r * 1.08, b.get_y() + b.get_height() / 2,
                f"{r:.1f}x  ({t['train_mse']:.2f} -> {t['test_mse']:.2f})",
                va="center", ha="left", fontsize=8, color=TEXT_POLAR)
    ax.set_xlim(0.5, max(ratios) * 8)


def main() -> None:
    apply_deen_theme()
    qkrr, vqc = _load("qkrr"), _load("vqc")
    if not (qkrr or vqc):
        raise FileNotFoundError("no pitt_qkrr / pitt_vqc training reports recorded")

    fig = plt.figure(figsize=(19, 13), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)

    dip, dip_src = _target_trials(qkrr, vqc, "dipole")
    pol, pol_src = _target_trials(qkrr, vqc, "polarizability")
    _r2_panel(fig.add_subplot(gs[0, 0]), dip, dip_src,
              "PANEL 1: DIPOLE MOMENT ||mu|| (DEBYE)", "D^2")
    _r2_panel(fig.add_subplot(gs[0, 1]), pol, pol_src,
              "PANEL 2: POLARIZABILITY alpha (A^3)", "A^6")
    _depth_panel(fig.add_subplot(gs[1, 0]), qkrr)
    _gap_panel(fig.add_subplot(gs[1, 1]), pol)

    fig.suptitle(
        "T-1000 SOVEREIGN TELEMETRY: PITT QUANTUM REPOSITORY KERNEL & VQC REGRESSION\n"
        "Recorded reports only - classical (gold), quantum (green), target leak (red, hatched)",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    fig.patch.set_facecolor(BG_OBSIDIAN)
    for ax in fig.axes:
        ax.set_facecolor(CARD_CHARCOAL)
        ax.grid(color=GRID_SLATE, linewidth=0.5, alpha=0.6)

    out_path = WS_ROOT / "results" / "charts" / "pitt_qkrr_depth_showdown.png"
    saved_path = save_deen_figure(fig, out_path, dpi=300)
    print(f"[CHART] Successfully generated sovereign figure: {saved_path}")


if __name__ == "__main__":
    main()
