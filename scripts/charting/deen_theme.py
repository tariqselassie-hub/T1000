# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Deen OS Sovereign Visual Specification & Matplotlib Theme Engine.
Strictly governs all visual artifacts, charts, telemetry, and figures.

Palette Directives:
  - Canvas / Background: Deep Obsidian (#05080d, #080c14)
  - Contrast / Cards:    Void Charcoal (#0f172a, #111827)
  - Quantum Telemetry:   Laser Emerald Green (#00e676, #00ff88, #10b981)
  - FPGA / Hardware:     Radiant Amber / Gold (#ffb300, #ffd700)
  - Alert / Critical:    Crimson Red (#ef4444, #dc2626)
  - Reticles / Grid:     Muted Slate Grid (#1e293b, #334155)
  - Monospace Labels:    Polar Slate / White (#f8fafc, #e2e8f0, #94a3b8)

STRICT ZERO TOLERANCE:
  - PROHIBITED: Cyan, teal, sky blue (#00ffff, #00f0ff, #38bdf8, #06b6d4, #0ea5e9).
  - PROHIBITED: Purple, violet, magenta (#a855f7, #8b5cf6, #9333ea, #d946ef).
  - PROHIBITED: Emojis anywhere in figures or text.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Sovereign Color Definitions
BG_OBSIDIAN = "#05080d"
BG_DARK = "#080c14"
CARD_CHARCOAL = "#0f172a"
CARD_DARK = "#111827"
GRID_SLATE = "#1e293b"
GRID_MUTED = "#334155"

GREEN_QUANTUM = "#00e676"
GREEN_EMERALD = "#00ff88"
GREEN_FLUX = "#10b981"

GOLD_HARDWARE = "#ffb300"
GOLD_BRIGHT = "#ffd700"

RED_ALERT = "#ef4444"
RED_CRIMSON = "#dc2626"

TEXT_POLAR = "#f8fafc"
TEXT_SLATE = "#e2e8f0"
TEXT_MUTED = "#94a3b8"

PROHIBITED_HEX = {
    "#00ffff", "#00f0ff", "#38bdf8", "#06b6d4", "#0ea5e9", "#0284c7",
    "#14b8a6", "#22d3ee", "#a855f7", "#8b5cf6", "#9333ea", "#d946ef",
    "#7c3aed", "#c084fc", "#6366f1", "#ec4899"
}

def apply_deen_theme() -> None:
    """Configures global Matplotlib rcParams for sovereign Deen OS aesthetics."""
    mpl.rcParams["figure.facecolor"] = BG_OBSIDIAN
    mpl.rcParams["axes.facecolor"] = BG_DARK
    mpl.rcParams["axes.edgecolor"] = GRID_SLATE
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["grid.color"] = GRID_SLATE
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.linewidth"] = 0.8
    mpl.rcParams["grid.alpha"] = 0.6
    mpl.rcParams["text.color"] = TEXT_POLAR
    mpl.rcParams["axes.labelcolor"] = TEXT_POLAR
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["xtick.color"] = TEXT_MUTED
    mpl.rcParams["ytick.color"] = TEXT_MUTED
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.facecolor"] = CARD_CHARCOAL
    mpl.rcParams["legend.edgecolor"] = GRID_SLATE
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial", "Helvetica"]

def style_card(ax: plt.Axes, title: str = "", subtitle: str = "") -> None:
    """Applies card boundary styling, title formatting, and grid configuration."""
    ax.set_facecolor(BG_DARK)
    for spine in ax.spines.values():
        spine.set_color(GRID_SLATE)
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle="--", linewidth=0.7, color=GRID_SLATE, alpha=0.7)
    if title:
        full_title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(full_title, color=TEXT_POLAR, fontsize=10, fontweight="bold", pad=10, loc="left")

def save_deen_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 300) -> Path:
    """Saves a figure adhering to sovereign background and resolution standards."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=BG_OBSIDIAN, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return path
