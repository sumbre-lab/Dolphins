"""
Compares the single longest verified-continuous whistle found in each of
SW_Neo, SW_Luna and SW_Nana - the three signature-whistle categories that
have genuine >1s single whistles in the HF extracted-whistle pool. (See
plot_duration_examples.py for how "verified-continuous" is determined via
frame-to-frame F0 jump detection, and why SW_Nikita/SW_Yosefa don't qualify
under the same SNR/jump-threshold settings.)

The jump-based check only catches breaks bigger than JUMP_THRESH_HZ, so it
can still pass a track that hops onto a second, unrelated tonal source
mid-window via one single-frame jump smaller than that threshold - e.g. the
raw longest SW_Neo candidate, Whistle3175, jumps 1532 Hz at t=0.81s (under
the 3000 Hz cutoff) from a modulating ~7.3kHz whistle onto a dead-flat
~10.8kHz tone held for the next 400+ms - unlike every real whistle in this
set, which keeps modulating throughout, that plateau reads as a different
(possibly non-biological) source the tracker latched onto, not a
continuation of the same call. Caught by manual review, not automatically;
EXCLUDE_WHISTLES documents the ones found so far.
"""

import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from fetch_audio_examples import fetch_examples
from plot_duration_examples import (
    CACHE_DIR, OUTPUT_DIR, SNR_MIN, MIN_CLEAN_POINTS,
    INK, MUTED, compute_clean_segments, plot_one,
)

CATEGORIES = ["SW_Neo", "SW_Luna", "SW_Nana"]

# Longest-by-clean-duration candidates rejected after visual/frame-level
# review as two tonal sources bridged by one sub-threshold jump, not one
# continuous whistle (see module docstring).
EXCLUDE_WHISTLES = {"Whistle3175"}


def select_longest(full_df, category):
    sub = full_df[(full_df["sw_name"] == category) & (full_df["snr_db"] > SNR_MIN)]
    sub = compute_clean_segments(sub)
    cand = sub[sub["clean_n"] >= MIN_CLEAN_POINTS]
    cand = cand[~cand["whistle_name"].isin(EXCLUDE_WHISTLES)]
    return cand.loc[cand["clean_duration"].idxmax()]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_path = os.path.join(CACHE_DIR, "extracted_whistles_full.parquet")
    full_df = pd.read_parquet(full_path)
    chosen = {cat: select_longest(full_df, cat) for cat in CATEGORIES}

    print(f"Longest verified-continuous whistle per category (snr_db > {SNR_MIN}):")
    for cat, row in chosen.items():
        print(f"  {cat:10s} {row['whistle_name']:14s} "
              f"clean_duration={row['clean_duration']:.3f}s  "
              f"(raw window={row['duration']:.3f}s)  snr={row['snr_db']:.1f}dB")

    cache_path = os.path.join(CACHE_DIR, "duration_examples.pkl")
    audio = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            audio = pickle.load(f)

    missing = [row["whistle_name"] for row in chosen.values() if row["whistle_name"] not in audio]
    if missing:
        targets = [(row["whistle_name"], row["shard"]) for row in chosen.values()
                   if row["whistle_name"] in missing]
        fetched = fetch_examples(targets)
        audio.update(fetched)
        with open(cache_path, "wb") as f:
            pickle.dump(audio, f)
        print(f"Fetched and cached audio for {len(fetched)} whistle(s) -> {cache_path}")

    fig, axes = plt.subplots(1, len(chosen), figsize=(4.4 * len(chosen), 3.6))
    for ax, (cat, row) in zip(axes, chosen.items()):
        plot_one(ax, audio[row["whistle_name"]], row["clean_t"], row["clean_f"])
        ax.set_title(f"{cat}\n{row['whistle_name']}  ({row['clean_duration']:.2f} s)",
                     fontsize=10, color=INK)
        ax.set_xlabel("Time (s)", fontsize=9, color=INK)
    axes[0].set_ylabel("Freq (kHz)", fontsize=9, color=INK)

    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "long_whistle_examples.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
