"""
Fetch a random subset of extracted whistles for the 5 signature-whistle (SW)
categories from the dolphinteam/Whistle-Classification HF dataset.

Only reads the lightweight columns (skips `audio` and `f0_spectrogram`) via
column-pruned remote parquet reads, so this does not download the full
~3.4GB dataset.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = "dolphinteam/Whistle-Classification"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
SHARDS = [
    "data/train-00000-of-00006.parquet",
    "data/train-00001-of-00006.parquet",
    "data/train-00002-of-00006.parquet",
    "data/train-00003-of-00006.parquet",
    "data/train-00004-of-00006.parquet",
    "data/train-00005-of-00006.parquet",
    "data/validation-00000-of-00002.parquet",
    "data/validation-00001-of-00002.parquet",
    "data/test-00000-of-00002.parquet",
    "data/test-00001-of-00002.parquet",
]

# main_category ClassLabel names (from the parquet's embedded HF feature metadata):
# ["NSW_3", "NSW_2", "NSW_1", "SW_Dana", "SW_Luna", "SW_Nana", "SW_Neo", "SW_Nikita", "SW_Shy", "SW_Yosefa"]
SW_CATEGORY_IDS = {
    "SW_Luna": 4,
    "SW_Nana": 5,
    "SW_Neo": 6,
    "SW_Nikita": 7,
    "SW_Yosefa": 9,
}

COLUMNS = [
    "main_category", "name", "whistle_name",
    "onset", "offset", "duration", "snr_db",
    "f0_time", "f0_hz", "f0_ok",
]

N_PER_CATEGORY = 500
SEED = 0

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def fetch_shard(shard_path):
    url = f"{BASE_URL}/{shard_path}"
    t0 = time.time()
    fs = fsspec.filesystem("https")
    with fs.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read(columns=COLUMNS)
    df = table.to_pandas()
    df = df[df["main_category"].isin(SW_CATEGORY_IDS.values()) & (df["f0_ok"] == True)]  # noqa: E712
    df["shard"] = shard_path
    print(f"  {shard_path}: {len(df)} matching rows ({time.time()-t0:.1f}s)")
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Fetching {len(SHARDS)} shards (column-pruned, filtered to 5 SW categories, f0_ok=True)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        dfs = list(ex.map(fetch_shard, SHARDS))
    full = pd.concat(dfs, ignore_index=True)
    print(f"Total matching rows across all shards: {len(full)}  ({time.time()-t0:.1f}s)")

    id_to_name = {v: k for k, v in SW_CATEGORY_IDS.items()}
    full["sw_name"] = full["main_category"].map(id_to_name)
    print(full["sw_name"].value_counts())

    full_path = os.path.join(CACHE_DIR, "extracted_whistles_full.parquet")
    full.to_parquet(full_path)
    print(f"Saved full filtered pool to {full_path}")

    # Random subset per category
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
