# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Extracts 10-dimensional semantic visual feature representations from the
Waterbirds dataset using a pre-trained MobileNetV3 visual backbone followed by
PCA projection into the 10-qubit quantum angle manifold [-pi, pi]^10.
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
import torch
import torchvision.transforms as T
import torchvision.models as models

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
N_QUBITS = 10
BATCH_SIZE = 64

def get_transforms():
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_all():
    if not os.path.isfile(ZIP_PATH):
        print(f"ERROR: Dataset zip not found at {ZIP_PATH}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Opening archive: {ZIP_PATH}")
    t0_total = time.time()

    # Load MobileNetV3-Small backbone
    print("Loading pre-trained MobileNetV3-Small visual backbone...")
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval()
    transform = get_transforms()

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        meta_entry = "waterbird_complete95_forest2water2/metadata.csv"
        with z.open(meta_entry) as f:
            df = pd.read_csv(f)

        print(f"Metadata rows loaded: {len(df)}")
        split_map = {0: "train", 1: "val", 2: "test"}

        raw_embeddings = {}
        labels_dict = {}
        places_dict = {}

        for split_id, split_name in split_map.items():
            sub_df = df[df['split'] == split_id].reset_index(drop=True)
            n_samples = len(sub_df)
            print(f"\n--- Extracting Backbone Embeddings for {split_name} ({n_samples} samples) ---")

            embed_list = []
            labels_list = []
            places_list = []

            t_split_start = time.time()
            batch_tensors = []

            for idx, row in sub_df.iterrows():
                fname = "waterbird_complete95_forest2water2/" + row['img_filename']
                data = z.read(fname)
                img = Image.open(io.BytesIO(data)).convert('RGB')
                tensor = transform(img)
                batch_tensors.append(tensor)
                labels_list.append(int(row['y']))
                places_list.append(int(row['place']))

                if len(batch_tensors) == BATCH_SIZE or (idx + 1) == n_samples:
                    batch = torch.stack(batch_tensors)
                    with torch.no_grad():
                        feats = backbone.avgpool(backbone.features(batch)).flatten(1)
                    embed_list.append(feats.numpy())
                    batch_tensors.clear()

                if (idx + 1) % 1000 == 0 or (idx + 1) == n_samples:
                    elapsed = time.time() - t_split_start
                    rate = (idx + 1) / elapsed
                    print(f"  [{idx + 1}/{n_samples}] {rate:.1f} images/sec (Elapsed: {elapsed:.1f}s)")

            raw_embeddings[split_name] = np.vstack(embed_list)
            labels_dict[split_name] = np.array(labels_list, dtype=np.uint8)
            places_dict[split_name] = np.array(places_list, dtype=np.uint8)

        # Fit PCA on Train Embeddings
        print("\nComputing Principal Component Analysis (PCA) on Train Embeddings...")
        train_embed = raw_embeddings["train"]
        mean_vec = train_embed.mean(axis=0)
        train_centered = train_embed - mean_vec

        # SVD: train_centered = U * S * Vt
        u, s, vt = np.linalg.svd(train_centered, full_matrices=False)
        components = vt[:N_QUBITS, :].T # Shape: (576, 10)
        var_explained = (s[:N_QUBITS] ** 2) / np.sum(s ** 2)
        print(f"Top {N_QUBITS} Components explain {np.sum(var_explained)*100:.2f}% of visual variance.")

        # Project and Scale to Quantum Angle Range [-pi, pi]
        # Standardize using training 3-sigma scaling
        train_proj = train_centered @ components
        std_proj = train_proj.std(axis=0, keepdims=True) + 1e-6

        # Target range: [-pi, pi] via tanh scaling or 2-sigma clipping
        manifest_meta = {
            "backbone": "mobilenet_v3_small_imagenet",
            "feature_dim": N_QUBITS,
            "variance_explained_ratio": var_explained.tolist(),
            "total_variance_explained": float(np.sum(var_explained)),
            "splits": {}
        }

        for split_name in ["train", "val", "test"]:
            centered = raw_embeddings[split_name] - mean_vec
            projected = centered @ components
            # Scale so 2.5 standard deviations maps to pi
            scaled = (projected / (2.5 * std_proj)) * np.pi
            scaled_clamped = np.clip(scaled, -np.pi, np.pi).astype(np.float32)

            out_feat_file = os.path.join(OUT_DIR, f"{split_name}_features_10d.bin")
            scaled_clamped.tofile(out_feat_file)
            print(f"Saved {split_name} features: {out_feat_file} ({scaled_clamped.shape})")

            manifest_meta["splits"][split_name] = {
                "count": len(scaled_clamped),
                "features_file": f"{split_name}_features_10d.bin",
                "mean_norm": float(np.mean(np.linalg.norm(scaled_clamped, axis=1))),
            }

        with open(os.path.join(OUT_DIR, "features_manifest.json"), "w") as f:
            json.dump(manifest_meta, f, indent=2)

        print(f"\nFeature extraction complete in {time.time() - t0_total:.2f} seconds.")

if __name__ == "__main__":
    extract_all()
