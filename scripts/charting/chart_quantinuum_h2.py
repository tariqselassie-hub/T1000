# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Quantinuum H2-1 Benchmark Visualizer for T-1000 Master Charting Suite.
Renders 4-panel sovereign publication diagram adhering to Deen OS Visual Specification:
- Deep Obsidian (#05080d)
- Laser Emerald Green (#00e676 / #00ff88)
- Radiant Gold (#ffb300 / #ffd700)
- Alert Crimson (#ef4444 / #dc2626)
- Polar Slate/White (#f8fafc)
Strictly ZERO cyan, ZERO purple, ZERO emojis.
"""

import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
repo_root = cur_dir.parents[1]
exp_dir = repo_root / "experiments" / "golden_quantinuum_h2"
sys.path.insert(0, str(exp_dir))

from chart_golden_quantinuum_h2 import chart_golden_quantinuum_h2


def chart_quantinuum_h2():
    return chart_golden_quantinuum_h2()


if __name__ == "__main__":
    chart_quantinuum_h2()
