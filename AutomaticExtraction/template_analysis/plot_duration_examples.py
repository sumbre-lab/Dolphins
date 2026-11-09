"""
Illustrates within-category duration variability in the automatically
extracted whistle set (HF dataset): plots real spectrograms (with the F0
track overlaid) for a "typical" and a "long" (>1s) SW_Luna example.

The dataset's `duration` field (offset - onset) is the extraction *window*,
not the whistle itself - many extractions have noisy, jittery F0 track
before/after the real whistle, or even jump mid-window onto a second,
unrelated tonal segment across a burst of trackless noise. So picking a
"long" example by raw duration alone is unsafe - e.g. Whistle3839 (SW_Nikita,
1.14s window) looks like a single smooth contour at a glance, but between
t=0.905-0.965s the F0 track jumps
10045 -> 17500 -> 11780 -> 21511 -> 17500 -> 5906 -> 11631 -> 6539 -> 16471 Hz
frame to frame - the tracker bouncing across noise/clicks onto a different
tonal band, not a real sweep.

To tell genuine long whistles from noise-bridged fragments, this script
finds the longest contiguous run of the F0 track with no frame-to-frame jump
bigger than JUMP_THRESH_HZ, and treats that run's span as the whistle's
"clean duration". The threshold has to be loose enough to allow real fast-
but-real transitions within one call (a quick onset chirp, a fast step
partway through can jump 1-2.5kHz between 5ms frames) while still rejecting
genuine tracker noise, which jumps by several kHz and keeps reversing
direction frame to frame (as in the Whistle3839 example above). 3000 Hz
was chosen by checking borderline cases by hand: at this threshold
Whistle10101 (SW_Luna) - a smooth single rise/plateau/fall arc from 8kHz to
14-16.5kHz and back down to 5kHz - is correctly kept whole at ~1.09s, while
Whistle3839's noise-bridged jump above is still correctly rejected as a
break (its jumps exceed 3000 Hz by 2-3x). Checked this way, verified >1s
single-whistle examples do exist in SW_Neo, SW_Nana and SW_Luna, but not in
SW_Nikita or SW_Yosefa in this cached sample. Each panel below is cropped
(spectrogram + audio) to the clean segment plus a little padding, instead of
the full extraction window.

Examples are picked from cache/extracted_whistles_full.parquet (the full
filtered SW pool from fetch_extracted_whistles.py), restricted to
snr_db > SNR_MIN. Audio for the chosen whistles is then fetched from the HF
dataset via targeted row-group reads (same approach as
fetch_audio_examples.py) and cached locally.
"""

import io
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram

from fetch_audio_examples import fetch_examples

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "cache")
OUTPUT_DIR = os.path.join(HERE, "output")

CATEGORY = "SW_Luna"
SNR_MIN = 15.0
TYPICAL_PERCENTILE = 0.50  # "long" is not a percentile - it's the single longest verified example

JUMP_THRESH_HZ = 3000.0   # frame-to-frame F0 jump above this = track left the real whistle
MIN_CLEAN_POINTS = 15     # >=75ms of contiguous track (at ~5ms F0 sampling) to be a valid example

CONTOUR_COLOR = "#eb6834"  # categorical slot 2 (orange) - pops against greyscale
INK = "#0b0b0b"
MUTED = "#898781"


def longest_clean_segment(t, f, jump_thresh_hz=JUMP_THRESH_HZ):
    """Longest contiguous run of (t, f) with no frame-to-frame |delta f|
    above jump_thresh_hz - i.e. the part of the F0 track that's actually
    following one continuous whistle, not noise."""
    order = np.argsort(t)
    t, f = t[order], f[order]
    if len(t) < 2:
        return t, f
    breaks = np.where(np.abs(np.diff(f)) > jump_thresh_hz)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(t)]])
    lo, hi = max(zip(bounds[:-1], bounds[1:]), key=lambda ab: t[ab[1] - 1] - t[ab[0]])
    return t[lo:hi], f[lo:hi]


def compute_clean_segments(sub):
    clean_t, clean_f, durs, ns = [], [], [], []
    for _, row in sub.iterrows():
        t = np.asarray(row["f0_time"], dtype=float)
        f = np.asarray(row["f0_hz"], dtype=float)
        mask = np.isfinite(t) & np.isfinite(f)
        ts, fs = longest_clean_segment(t[mask], f[mask])
        clean_t.append(ts)
        clean_f.append(fs)
        durs.append(ts[-1] - ts[0] if len(ts) else 0.0)
        ns.append(len(ts))
    sub = sub.copy()
    sub["clean_t"] = clean_t
    sub["clean_f"] = clean_f
    sub["clean_duration"] = durs
    sub["clean_n"] = ns
    return sub


def select_examples(full_df):
    """"typical": nearest to the median clean-duration. "long": the single
    longest verified-continuous whistle. Both among decent-SNR, well-tracked
    extractions of CATEGORY."""
    sub = full_df[(full_df["sw_name"] == CATEGORY) & (full_df["snr_db"] > SNR_MIN)]
    sub = compute_clean_segments(sub)
    cand = sub[sub["clean_n"] >= MIN_CLEAN_POINTS].copy()

    long_idx = cand["clean_duration"].idxmax()
    long_row = cand.loc[long_idx]
    cand = cand.drop(index=long_idx)

    typical_target = cand["clean_duration"].quantile(TYPICAL_PERCENTILE)
    typical_idx = (cand["clean_duration"] - typical_target).abs().idxmin()
    typical_row = cand.loc[typical_idx]

    return {"typical": typical_row, "long": long_row}


def plot_one(ax, row, clean_t, clean_f):
    audio_bytes = row["audio"]["bytes"]
    sr, data = wavfile.read(io.BytesIO(audio_bytes))
    data = data.astype(float)

    t0, t1 = clean_t[0], clean_t[-1]
    pad = min(0.15, max(0.03, 0.15 * (t1 - t0)))
    win_lo = max(0.0, t0 - pad)
    win_hi = min(len(data) / sr, t1 + pad)
    seg = data[int(win_lo * sr):int(win_hi * sr)]

    f, t, Sxx = spectrogram(seg, fs=sr, window="hann", nperseg=512, noverlap=460)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    ax.pcolormesh(t + win_lo, f / 1000, Sxx_db, shading="gouraud", cmap="Greys",
                  vmin=np.percentile(Sxx_db, 40), vmax=np.percentile(Sxx_db, 99.5))

    # Only overlay the trusted (longest-clean-run) F0 points, not the raw
    # track - the raw track can jitter into noise right at the window edges.
    ax.scatter(clean_t, np.asarray(clean_f) / 1000, color=CONTOUR_COLOR, s=10,
               linewidths=0, zorder=3)

    ax.set_xlim(win_lo, win_hi)
    ax.set_ylim(0, min(24, f.max() / 1000))
    ax.tick_params(labelsize=8, colors=MUTED)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_path = os.path.join(CACHE_DIR, "extracted_whistles_full.parquet")
    full_df = pd.read_parquet(full_path)
    chosen = select_examples(full_df)

    print(f"Selected {CATEGORY} examples spanning clean-track duration (snr_db > {SNR_MIN}):")
    for label, row in chosen.items():
        print(f"  {label:8s} {row['whistle_name']:14s} "
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

    fig, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen), 3.6))
    for ax, (label, row) in zip(axes, chosen.items()):
        plot_one(ax, audio[row["whistle_name"]], row["clean_t"], row["clean_f"])
        ax.set_title(f"{label}\n{row['whistle_name']}  ({row['clean_duration']:.2f} s)",
                     fontsize=10, color=INK)
        ax.set_xlabel("Time (s)", fontsize=9, color=INK)
    axes[0].set_ylabel("Freq (kHz)", fontsize=9, color=INK)

    fig.suptitle(
        f"Duration variability within {CATEGORY} (HF extracted whistles, SNR > {SNR_MIN:.0f} dB)",
        fontsize=12.5, color=INK, y=1.03)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"duration_variability_{CATEGORY}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
