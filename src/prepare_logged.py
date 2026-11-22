"""
Step 0 for the leakage-safe experiment: build ONE un-filtered, un-balanced table
log2(TPM + 1) for all samples and all genes, from the cached data/raw/*_tpm.pkl files
(no download needed).

Put this file in:  src/prepare_logged.py
Run from the project root:   python src\\prepare_logged.py

Output:  experiments/leakage_safe/data/tcga_5cancers_logged.pkl   (label + ~59k genes, 3001 samples)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("experiments/leakage_safe/data")
CANCERS = ["BRCA", "LUAD", "THCA", "OV", "COAD"]


def load_raw_data() -> pd.DataFrame:
    frames = []
    for cancer in CANCERS:
        file_path = RAW / f"{cancer}_tpm.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        print(f"Loading {cancer}...")
        frames.append(pd.read_pickle(file_path))
    data = pd.concat(frames, join="inner")
    print("\nRaw dataset | samples:", data.shape[0], "| genes:", data.shape[1] - 1)
    print(data["label"].value_counts().to_dict())
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_raw_data()
    assert data.index.is_unique, "duplicate sample ids"

    labels = data["label"].copy()
    logged = np.log2(data.drop(columns="label") + 1)      # per-value transform, no leakage
    prepared = pd.concat([labels, logged], axis=1)

    out_file = OUT / "tcga_5cancers_logged.pkl"
    prepared.to_pickle(out_file)
    print("\nAfter log2:", logged.shape[0], "samples,", logged.shape[1], "genes")
    print("Saved to:", out_file)


if __name__ == "__main__":
    main()