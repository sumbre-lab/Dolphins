"""
Plot good SW_Luna / SW_Nikita examples (extracted whistles that matched the
template most closely) as real spectrograms with the F0 annotation overlaid.

Audio + F0 tracks come from cache/audio_examples.pkl (fetched by
fetch_audio_examples.py via targeted row-group reads from the HF dataset).
"""

import io
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "cache")
OUTPUT_DIR = os.path.join(HERE, "output")

EXAMPLES = {
    "SW_Luna": ["Whistle6917", "Whistle6939", "Whistle9119"],
    "SW_Nikita": ["Whistle3593", "Whistle4216", "Whistle3751"],
}

CONTOUR_COLOR = "#eb6834"  # categorical slot 2 (orange) - pops against greyscale
INK = "#0b0b0b"
MUTED = "#898781"


def plot_one(ax, row):
    audio_bytes = row["audio"]["bytes"]
    sr, data = wavfile.read(io.BytesIO(audio_bytes))
    data = data.astype(float)

    f, t, Sxx = spectrogram(data, fs=sr, window="hann", nperseg=512, noverlap=460)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    ax.pcolormesh(t, f / 1000, Sxx_db, shading="gouraud", cmap="Greys",
                  vmin=np.percentile(Sxx_db, 40), vmax=np.percentile(Sxx_db, 99.5))

    f0_time = np.asarray(row["f0_time"], dtype=float)
    f0_hz = np.asarray(row["f0_hz"], dtype=float)
    mask = np.isfinite(f0_time) & np.isfinite(f0_hz)
    ax.scatter(f0_time[mask], f0_hz[mask] / 1000, color=CONTOUR_COLOR, s=8,
               linewidths=0, zorder=3)

    ax.set_ylim(0, min(24, f.max() / 1000))
    ax.tick_params(labelsize=7, colors=MUTED)


def main():
    with open(os.path.join(CACHE_DIR, "audio_examples.pkl"), "rb") as f:
        examples = pickle.load(f)

    sw_names = list(EXAMPLES.keys())
    n_cols = max(len(v) for v in EXAMPLES.values())
    fig, axes = plt.subplots(len(sw_names), n_cols, figsize=(3.6 * n_cols, 3.4 * len(sw_names)))

    for row_i, sw_name in enumerate(sw_names):
        whistle_list = EXAMPLES[sw_name]
        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            if col_i < len(whistle_list):
                wname = whistle_list[col_i]
                plot_one(ax, examples[wname])
            else:
                ax.axis("off")
                continue
            if col_i == 0:
                ax.set_ylabel("Freq (kHz)", fontsize=9, color=INK)
            if row_i == len(sw_names) - 1:
                ax.set_xlabel("Time (s)", fontsize=8, color=INK)

    fig.tight_layout()

    # Bold row labels to the left of each row, outside the axes
    for row_i, sw_name in enumerate(sw_names):
        bbox = axes[row_i, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(bbox.x0 - 0.06, y_center, sw_name, rotation=90,
                  ha="center", va="center", fontsize=14, fontweight="bold", color=INK)

    out_path = os.path.join(OUTPUT_DIR, "spectrogram_examples_Luna_Nikita.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
