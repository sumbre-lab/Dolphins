"""
Plot the DTW template-validation distributions produced by run_analysis.py:
for each SW category, compares the natural pairwise variability among manual
annotations against how far the extraction template sits from real examples
(manual and automatically-extracted).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(HERE, "output")

SW_ORDER = ["SW_Neo", "SW_Luna", "SW_Nikita", "SW_Nana", "SW_Yosefa"]

COMPARISONS = [
    "background (manual-manual)",
    "template -> manual",
    "template -> extracted",
]
# Categorical palette slots 1/2/3 (blue/orange/aqua) - validated colorblind-safe
# ordering for a 3-series all-pairs form (references/palette.md).
COLORS = {
    "background (manual-manual)": "#2a78d6",
    "template -> manual": "#eb6834",
    "template -> extracted": "#1baf7a",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def violin(ax, data, position, color):
    if len(data) < 2:
        ax.scatter([position] * len(data), data, color=color, s=14, zorder=3)
        return
    parts = ax.violinplot([data], positions=[position], widths=0.8,
                           showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor(color)
        pc.set_linewidth(1)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    ax.vlines(position, q1, q3, color=color, lw=3, zorder=3)
    ax.scatter([position], [med], color="white", edgecolor=color, s=28, zorder=4, linewidth=1.5)
    # jittered raw points for smaller groups (extracted / template->manual), thin & translucent
    if len(data) <= 250:
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(data))
        ax.scatter(position + jitter, data, color=color, s=6, alpha=0.35, zorder=2, linewidths=0)


def main():
    dist_df = pd.read_csv(os.path.join(OUTPUT_DIR, "dtw_distances_long.csv"))
    summary_df = pd.read_csv(os.path.join(OUTPUT_DIR, "dtw_summary.csv")).set_index("sw_name")

    fig, axes = plt.subplots(1, len(SW_ORDER), figsize=(4.2 * len(SW_ORDER), 5.5), sharey=False)

    for ax, sw_name in zip(axes, SW_ORDER):
        sub = dist_df[dist_df["sw_name"] == sw_name]
        for i, comp in enumerate(COMPARISONS):
            data = sub.loc[sub["comparison"] == comp, "distance"].dropna().values
            violin(ax, data, i, COLORS[comp])

        ax.set_xlim(-0.6, 2.6)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["background\n(manual-manual)", "template\n→ manual", "template\n→ extracted"],
                            fontsize=8, color=INK)
        ax.set_title(sw_name, fontsize=13, color=INK, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        row = summary_df.loc[sw_name]
        note = (
            f"template→manual median: {row['tmpl_manual_median']:.3f}\n"
            f"({row['tmpl_manual_percentile_in_background']:.0f}th pct of background)\n"
            f"template→extracted median: {row['tmpl_extracted_median']:.3f}\n"
            f"({row['tmpl_extracted_percentile_in_background']:.0f}th pct of background)"
        )
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.3, color=MUTED)

    axes[0].set_ylabel("Normalized DTW distance", fontsize=10, color=INK)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[c],
                           markersize=9, alpha=0.8, label=c) for c in COMPARISONS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.04))

    fig.suptitle(
        "Where the extraction template sits relative to natural whistle variability",
        fontsize=13, color=INK, y=1.12,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUTPUT_DIR, "template_validation.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
