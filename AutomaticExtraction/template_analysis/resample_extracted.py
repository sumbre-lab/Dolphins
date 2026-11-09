"""
Re-sample the extracted-whistle subset from the already-cached full filtered
pool (cache/extracted_whistles_full.parquet), without re-hitting the HF
dataset. Mirrors the sampling logic in fetch_extracted_whistles.py so the
resulting cache/extracted_whistles_sample.parquet is a drop-in replacement.
"""

import os

import numpy as np
import pandas as pd

from fetch_extracted_whistles import CACHE_DIR, N_PER_CATEGORY, SEED, SW_CATEGORY_IDS


def main():
    full_path = os.path.join(CACHE_DIR, "extracted_whistles_full.parquet")
    full = pd.read_parquet(full_path)
    print(f"Loaded {len(full)} rows from {full_path}")

    rng = np.random.default_rng(SEED)
    parts = []
    for sw_name in SW_CATEGORY_IDS:
        sub = full[full["sw_name"] == sw_name]
        n = min(N_PER_CATEGORY, len(sub))
        idx = rng.choice(sub.index.values, size=n, replace=False)
        parts.append(sub.loc[idx])
        print(f"{sw_name}: sampled {n} / {len(sub)}")

    sample = pd.concat(parts, ignore_index=True)
    sample_path = os.path.join(CACHE_DIR, "extracted_whistles_sample.parquet")
    sample.to_parquet(sample_path)
    print(f"Saved {len(sample)}-row sample to {sample_path}")


if __name__ == "__main__":
    main()
