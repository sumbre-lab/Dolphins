"""
Template choice justification, using the manual-annotation pool itself as
the reference population (instead of the automatically extracted whistles).

For each SW category, every manual whistle is scored by its mean DTW
distance to every OTHER manual whistle (leave-one-out - self-comparison is
excluded, since a whistle's distance to itself is trivially 0 and would
bias its score). This produces a distribution, across all manual
candidates, of "how representative is this whistle of the rest of the
manually-annotated corpus". The whistle actually chosen as the template is
then located within that distribution.

This is independent of the automatic extractor (whereas comparing against
extracted whistles carries the extractor's own DTW-to-template selection
bias), so it isolates "does the template look like a typical real whistle
of this dolphin" from "does the template look like what the pipeline
extracts".

distance matrix: D[i, j] = normalized DTW distance(manual_i, manual_j), i != j
candidate score: row_mean[i] = mean_{j != i} D[i, j]
template score:  tmpl_mean    = mean_j dtw(template, manual_j)   (all j, no exclusion needed)
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.extraction.dtw import compute_normalized_dtw_distance  # noqa: E402

CLUSTERS_DIR = "/media/zfnews31/DolphinAnalysis/Whistles_annotated_from_divers_video/Corrected_Clusters"
ANNOTATIONS_DIR = "/media/zfnews31/DolphinAnalysis/Annotations"
HERE = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(HERE, "output")

SW_CONFIG = {
    "SW_Neo": ("cluster_Neo", "SW_Neo_simple.mat"),
    "SW_Luna": ("cluster_Luna", "SW_Luna.mat"),
    "SW_Nikita": ("cluster_Nikita", "SW_Nikita_normal.mat"),
    "SW_Nana": ("cluster_Nana", "SW_Nana.mat"),
    "SW_Yosefa": ("cluster_Yosefa", "SW_Yosefa_long_wavy.mat"),
}

N_JOBS = -1


def load_mat_contour(mat_path):
    d = sio.loadmat(mat_path)
    wht = np.asarray(d["wht"]).squeeze().astype(float)
    whf = np.asarray(d["whf"]).squeeze().astype(float)
    return np.column_stack([wht, whf])


def load_manual_contours(cluster_folder):
    contours = {}
    for fname in sorted(os.listdir(cluster_folder)):
        if not fname.lower().endswith(".mat"):
            continue
        try:
            c = load_mat_contour(os.path.join(cluster_folder, fname))
            if len(c) >= 3:
                contours[fname] = c
        except Exception as e:
            print(f"  skipped {fname}: {e}")
    return contours


def full_pairwise_matrix(contours_dict):
    names = list(contours_dict.keys())
    n = len(names)
    pairs = list(itertools.combinations(range(n), 2))

    def _dist(pair):
        i, j = pair
        return compute_normalized_dtw_distance(contours_dict[names[i]], contours_dict[names[j]])

    upper = Parallel(n_jobs=N_JOBS)(delayed(_dist)(p) for p in pairs)

    D = np.zeros((n, n))
    for (i, j), d in zip(pairs, upper):
        D[i, j] = d
        D[j, i] = d
    return pd.DataFrame(D, index=names, columns=names)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidate_rows = []
    summary_rows = []

    for sw_name, (cluster_name, template_file) in SW_CONFIG.items():
        print(f"\n=== {sw_name} ===")
        cluster_folder = os.path.join(CLUSTERS_DIR, cluster_name)
        manual_contours = load_manual_contours(cluster_folder)
        n = len(manual_contours)
        print(f"  manual candidates: {n}")

        D = full_pairwise_matrix(manual_contours)
        # leave-one-out mean: row sum excludes the zero self-distance on the diagonal
        row_means = D.sum(axis=1) / (n - 1)
        # leave-one-out std, also excluding the self-distance
        row_stds = pd.Series(
            {name: np.std(D.loc[name].drop(name).values) for name in D.index},
            name="row_std",
        )

        template_contour = load_mat_contour(os.path.join(ANNOTATIONS_DIR, template_file))
        tmpl_dists = np.array([
            compute_normalized_dtw_distance(template_contour, manual_contours[name])
            for name in manual_contours
        ])
        tmpl_mean = tmpl_dists.mean()
        tmpl_std = tmpl_dists.std()

        n_beaten = int((row_means > tmpl_mean).sum())
        pct_better_than = 100.0 * n_beaten / len(row_means)
        rank = int((row_means < tmpl_mean).sum()) + 1
        z_score = (tmpl_mean - row_means.mean()) / row_means.std()

        print(f"  candidate row-means (leave-one-out): mean={row_means.mean():.4f} +/- {row_means.std():.4f} "
              f"(n={n})")
        print(f"  template mean distance to all manual whistles: {tmpl_mean:.4f} +/- {tmpl_std:.4f}")
        print(f"  template rank: {rank} / {n+1}  "
              f"(beats {pct_better_than:.1f}% of manual candidates, z={z_score:.2f})")

        best5 = row_means.sort_values().head(5)
        print("  top-5 best candidates (lowest leave-one-out mean distance):")
        for name, val in best5.items():
            marker = " <-- would beat template" if val < tmpl_mean else ""
            print(f"    {name}: {val:.4f}{marker}")

        for name in row_means.index:
            candidate_rows.append({
                "sw_name": sw_name,
                "candidate": name,
                "row_mean": row_means[name],
                "row_std": row_stds[name],
                "is_template": False,
            })
        candidate_rows.append({
            "sw_name": sw_name,
            "candidate": template_file,
            "row_mean": tmpl_mean,
            "row_std": tmpl_std,
            "is_template": True,
        })

        summary_rows.append({
            "sw_name": sw_name,
            "template_file": template_file,
            "n_manual_candidates": n,
            "candidate_mean_of_means": row_means.mean(),
            "candidate_std_of_means": row_means.std(),
            "candidate_min": row_means.min(),
            "candidate_best_file": row_means.idxmin(),
            "template_mean": tmpl_mean,
            "template_std": tmpl_std,
            "template_rank": rank,
            "template_pct_better_than": pct_better_than,
            "template_z_score": z_score,
        })

    cand_df = pd.DataFrame(candidate_rows)
    cand_path = os.path.join(OUTPUT_DIR, "candidate_scores_manual.csv")
    cand_df.to_csv(cand_path, index=False)
    print(f"\nSaved candidate scores to {cand_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "template_ranking_summary_manual.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ranking summary to {summary_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
