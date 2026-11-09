"""
Fetch audio + f0_spectrogram for a handful of specific whistles (by
whistle_name) from the HF dataset, reading only the row group(s) that
contain them - avoids downloading full shards just for a few examples.
"""

import io
import os

import fsspec
import numpy as np
import pyarrow.parquet as pq

REPO = "dolphinteam/Whistle-Classification"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "cache")


def fetch_examples(targets):
    """
    targets: list of (whistle_name, shard_path)
    Returns dict whistle_name -> row dict (audio bytes/path, f0_time, f0_hz, onset, offset)
    """
    by_shard = {}
    for name, shard in targets:
        by_shard.setdefault(shard, []).append(name)

    results = {}
    for shard, names in by_shard.items():
        url = f"{BASE_URL}/{shard}"
        fs = fsspec.filesystem("https")
        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)

            # locate positions via the lightweight whistle_name column
            name_table = pf.read(columns=["whistle_name"])
            names_col = name_table.column("whistle_name").to_pylist()
            wanted_positions = {n: i for i, n in enumerate(names_col) if n in names}

            # map position -> row group index
            cum = 0
            group_bounds = []
            for g in range(pf.num_row_groups):
                n_rows = pf.metadata.row_group(g).num_rows
                group_bounds.append((cum, cum + n_rows, g))
                cum += n_rows

            needed_groups = set()
            for pos in wanted_positions.values():
                for lo, hi, g in group_bounds:
                    if lo <= pos < hi:
                        needed_groups.add(g)
                        break

            print(f"{shard}: fetching {len(needed_groups)} row group(s) for {len(names)} whistle(s)")
            for g in needed_groups:
                tbl = pf.read_row_group(g, columns=["whistle_name", "audio", "f0_time", "f0_hz", "onset", "offset"])
                df = tbl.to_pandas()
                sub = df[df["whistle_name"].isin(names)]
                for _, row in sub.iterrows():
                    results[row["whistle_name"]] = row

    return results


if __name__ == "__main__":
    targets = [
        ("Whistle6917", "data/validation-00000-of-00002.parquet"),
        ("Whistle9119", "data/train-00005-of-00006.parquet"),
        ("Whistle7766", "data/train-00005-of-00006.parquet"),
        ("Whistle6939", "data/train-00001-of-00006.parquet"),
        ("Whistle3593", "data/validation-00000-of-00002.parquet"),
        ("Whistle3654", "data/validation-00000-of-00002.parquet"),
        ("Whistle4110", "data/validation-00001-of-00002.parquet"),
        ("Whistle4086", "data/train-00003-of-00006.parquet"),
    ]
    res = fetch_examples(targets)
    print(f"\nFetched {len(res)} / {len(targets)} examples")
    for name, row in res.items():
        audio_bytes = row["audio"]["bytes"]
        print(f"  {name}: audio {len(audio_bytes)} bytes, path={row['audio']['path']}, "
              f"f0 points={len(row['f0_time'])}")

    import pickle
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "audio_examples.pkl"), "wb") as f:
        pickle.dump(res, f)
    print(f"Saved to {os.path.join(CACHE_DIR, 'audio_examples.pkl')}")
