"""
Plot template-choice justification: for each SW, the distribution of "mean
DTW distance to a reference whistle population" across every manual whistle
that could have been picked as the template, with the actual template
highlighted (emphasis form - one series is the point, the rest are context).

The violin/strip shows candidate row-means - each already an average over
many comparisons, so its spread reflects genuine whistle-to-whistle
variability. The template is plotted as a single point at its own mean, on
that same scale - not with an error bar, since its point-to-point spread
(std across individual comparisons) is a different statistic than the
across-candidate spread the violin shows, and the two aren't visually
comparable.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(HERE, "output")

SW_ORDER = ["SW_Neo", "SW_Luna", "SW_Nikita", "SW_Nana", "SW_Yosefa"]

ACCENT = "#eb6834"     # categorical slot 2 (orange) - the template, emphasized
CONTEXT = "#898781"    # muted gray - the candidate pool, de-emphasized
CONTEXT_FILL = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def plot_ranking(cand_csv, summary_csv, out_name, y_label, suptitle):
    cand_df = pd.read_csv(cand_csv)
    summary_df = pd.read_csv(summary_csv).set_index("sw_name")

    fig, axes = plt.subplots(1, len(SW_ORDER), figsize=(3.6 * len(SW_ORDER), 5.6), sharey=False)

    for ax, sw_name in zip(axes, SW_ORDER):
        sub = cand_df[cand_df["sw_name"] == sw_name]
        candidates = sub[~sub["is_template"]]
        row_means = candidates["row_mean"].values

        # Distribution of candidate scores (context, gray) - each point is
        # itself a mean over many comparisons, so this is a distribution of means.
        parts = ax.violinplot([row_means], positions=[0], widths=0.7,
                               showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(CONTEXT_FILL)
            pc.set_alpha(0.5)
            pc.set_edgecolor(CONTEXT)
            pc.set_linewidth(1)

        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.18, 0.18, size=len(row_means))
        ax.scatter(jitter, row_means, color=CONTEXT, s=10, alpha=0.5, zorder=2, linewidths=0)

        row = summary_df.loc[sw_name]
        cand_mean, cand_std = row["candidate_mean_of_means"], row["candidate_std_of_means"]
        ax.hlines(cand_mean, -0.55, 0.55, color=CONTEXT, lw=1, linestyle="--", zorder=3)

        # Template: a single mean, on the same scale as the candidate means -
        # no error bar (its own std is a different, point-to-point statistic,
        # not comparable to the across-candidate spread the violin shows).
        tmpl_mean = row["template_mean"]
        ax.scatter([0.55], [tmpl_mean], marker="D", color=ACCENT, s=90, zorder=5,
                   edgecolor="white", linewidth=1)

        ax.set_xlim(-0.8, 1.0)
        ax.set_xticks([0, 0.55])
        ax.set_xticklabels(["manual\ncandidates", "chosen\ntemplate"], fontsize=8.5, color=INK)
        ax.set_title(sw_name, fontsize=13, color=INK, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        note = (
            f"template mean: {tmpl_mean:.3f}\n"
            f"candidates mean ± std: {cand_mean:.3f} ± {cand_std:.3f}\n"
            f"rank {int(row['template_rank'])} / {int(row['n_manual_candidates'])+1}  "
            f"(beats {row['template_pct_better_than']:.0f}%)\n"
            f"z = {row['template_z_score']:.2f}"
        )
        ax.text(0.5, -0.24, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color=MUTED)

    axes[0].set_ylabel(y_label, fontsize=9.5, color=INK)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CONTEXT, markersize=8,
                   alpha=0.7, label="manual whistle, used as a candidate template\n(mean over its comparisons)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=ACCENT, markersize=9,
                   label="template actually used for extraction\n(mean over its comparisons)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.07))

    fig.suptitle(suptitle, fontsize=12.5, color=INK, y=1.18)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_ranking(
        cand_csv=os.path.join(OUTPUT_DIR, "candidate_scores.csv"),
        summary_csv=os.path.join(OUTPUT_DIR, "template_ranking_summary.csv"),
        out_name="template_ranking.png",
        y_label="Mean normalized DTW distance to extracted whistles",
        suptitle=(
            "Is the chosen template a good candidate? Mean distance to the extracted whistle set,\n"
            "template vs. every other manual whistle that could have been picked instead"
        ),
    )
