# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Label-free Titanic features plus the label, for quantum_kernel_bench --csv (no group-survival features,
because those are built from labels and would leak across the benchmark's own folds)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

raw, n = features.load()
f = features.build(raw).iloc[:n]
out = f[["sex", "pclass", "age", "log_fare", "family", "ticket_size", "cabin_known"]].copy()
out["is_master"] = (f["title"] == "Master").astype(int)
out["survived"] = raw["Survived"].iloc[:n].astype(int)
out.to_csv(features.DATA_DIR / "kernel_input.csv", index=False)
print(out.shape)
