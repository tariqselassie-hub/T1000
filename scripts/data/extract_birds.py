# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Extracts and pre-processes the Waterbirds 11k dataset (waterbird_complete95_forest2water2)
from C:\\Users\\zoddj\\Downloads\\birds_dataset.zip into flat binary arrays for T-1000 QNN training.

Each image is converted to grayscale and resized to 32x32 (1,024 float32 values),
matching the 2^10 state dimension of a 10-qubit quantum state vector.
"""

import os
import sys
import zipfile
import io
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

def _find_zip_path():
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "birds_dataset.zip"),
        os.path.expanduser("~/Downloads/birds_dataset.zip"),
        r"C:\Users\zoddj\Downloads\birds_dataset.zip",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]

ZIP_PATH = _find_zip_path()
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "birds")
TARGET_SIZE = (32, 32)
EXPECTED_DIM = TARGET_SIZE[0] * TARGET_SIZE[1] # 1024

def process_dataset():
    if not os.path.isfile(ZIP_PATH):
        print(f"ERROR: Dataset zip not found at {ZIP_PATH}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Opening dataset archive: {ZIP_PATH}")
    t0 = time.time()

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        meta_entry = "waterbird_complete95_forest2water2/metadata.csv"
        with z.open(meta_entry) as f:
            df = pd.read_csv(f)

        print(f"Metadata rows loaded: {len(df)}")
        split_map = {0: "train", 1: "val", 2: "test"}

        manifest = {
            "dataset": "waterbirds_11k",
            "resolution": list(TARGET_SIZE),
            "state_dim": EXPECTED_DIM,
            "n_qubits": 10,
            "splits": {},
            "groups": {}
        }

        for split_id, split_name in split_map.items():
            sub_df = df[df['split'] == split_id].reset_index(drop=True)
            n_samples = len(sub_df)
            print(f"\nProcessing {split_name} split ({n_samples} samples)...")

            images_arr = np.zeros((n_samples, EXPECTED_DIM), dtype=np.float32)
            labels_arr = np.zeros(n_samples, dtype=np.uint8)
            places_arr = np.zeros(n_samples, dtype=np.uint8)

            t_split_start = time.time()
            for idx, row in sub_df.iterrows():
                fname = "waterbird_complete95_forest2water2/" + row['img_filename']
                data = z.read(fname)
                im = Image.open(io.BytesIO(data)).convert('L').resize(TARGET_SIZE, Image.Resampling.BILINEAR)
                pixels = np.asarray(im, dtype=np.float32) / 255.0
                images_arr[idx] = pixels.flatten()
                labels_arr[idx] = int(row['y'])
                places_arr[idx] = int(row['place'])

                if (idx + 1) % 1000 == 0 or (idx + 1) == n_samples:
                    elapsed = time.time() - t_split_start
                    rate = (idx + 1) / elapsed
                    print(f"  [{idx + 1}/{n_samples}] {rate:.1f} images/sec")

            img_file = os.path.join(OUT_DIR, f"{split_name}_images.bin")
            lbl_file = os.path.join(OUT_DIR, f"{split_name}_labels.bin")
            plc_file = os.path.join(OUT_DIR, f"{split_name}_places.bin")

            images_arr.tofile(img_file)
            labels_arr.tofile(lbl_file)
            places_arr.tofile(plc_file)

            print(f"  Saved {img_file} ({images_arr.nbytes / (1024*1024):.2f} MB)")
            print(f"  Saved {lbl_file}")
            print(f"  Saved {plc_file}")

            manifest["splits"][split_name] = {
                "count": n_samples,
                "y_0_count": int((labels_arr == 0).sum()),
                "y_1_count": int((labels_arr == 1).sum()),
                "place_0_count": int((places_arr == 0).sum()),
                "place_1_count": int((places_arr == 1).sum()),
                "g0_land_land": int(((labels_arr == 0) & (places_arr == 0)).sum()),
                "g1_land_water": int(((labels_arr == 0) & (places_arr == 1)).sum()),
                "g2_water_land": int(((labels_arr == 1) & (places_arr == 0)).sum()),
                "g3_water_water": int(((labels_arr == 1) & (places_arr == 1)).sum()),
            }

        manifest_path = os.path.join(OUT_DIR, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        total_time = time.time() - t0
        print(f"\nProcessing complete in {total_time:.2f} s. Manifest written to {manifest_path}")

if __name__ == "__main__":
    process_dataset()
