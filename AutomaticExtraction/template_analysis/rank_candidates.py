"""
Template choice justification.

For each SW category, treats every manually-annotated whistle as a
*candidate template* and scores it by its mean DTW distance to the whistles
the automatic extractor actually pulled out of recordings (a sample from the
HF dataset). This produces a distribution, across all manual candidates, of
"how well would this whistle have worked as the template". The whistle that
was actually chosen as the template is then located within that
distribution (rank / percentile / z-score) to show whether it was in fact a
good — or the best — choice.

distance matrix: D[i, j] = normalized DTW distance(manual_i, extracted_j)
candidate score: row_mean[i] = mean_j D[i, j],  row_std[i] = std_j D[i, j]
template score:  tmpl_mean    = mean_j dtw(template, extracted_j)
                  tmpl_std     = std_j dtw(template, extracted_j)
"""

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
CACHE_DIR = os.path.join(HERE, "cache")
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


def load_extracted_contours(sample_df, sw_name):
    sub = sample_df[sample_df["sw_name"] == sw_name]
    contours = {}
    for _, row in sub.iterrows():
        t = np.asarray(row["f0_time"], dtype=float)
        f = np.asarray(row["f0_hz"], dtype=float)
        mask = np.isfinite(t) & np.isfinite(f)
        t, f = t[mask], f[mask]
        if len(t) >= 3:
            contours[row["whistle_name"]] = np.column_stack([t, f])
    return contours


def cross_distance_matrix(manual_contours, extracted_contours):
    manual_names = list(manual_contours.keys())
    extracted_names = list(extracted_contours.keys())

    def _row(m_name):
        m = manual_contours[m_name]
        return [compute_normalized_dtw_distance(m, extracted_contours[e_name]) for e_name in extracted_names]

    rows = Parallel(n_jobs=N_JOBS)(delayed(_row)(m_name) for m_name in manual_names)
    D = pd.DataFrame(rows, index=manual_names, columns=extracted_names)
    return D


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_df = pd.read_parquet(os.path.join(CACHE_DIR, "extracted_whistles_sample.parquet"))
    print(f"Loaded {len(sample_df)} extracted-whistle samples")

    candidate_rows = []
    summary_rows = []

    for sw_name, (cluster_name, template_file) in SW_CONFIG.items():
        print(f"\n=== {sw_name} ===")
        cluster_folder = os.path.join(CLUSTERS_DIR, cluster_name)
        manual_contours = load_manual_contours(cluster_folder)
        extracted_contours = load_extracted_contours(sample_df, sw_name)
        print(f"  manual candidates: {len(manual_contours)}, extracted set: {len(extracted_contours)}")

        D = cross_distance_matrix(manual_contours, extracted_contours)
        row_means = D.mean(axis=1)
        row_stds = D.std(axis=1)

        template_contour = load_mat_contour(os.path.join(ANNOTATIONS_DIR, template_file))
        tmpl_dists = np.array([
            compute_normalized_dtw_distance(template_contour, extracted_contours[e_name])
            for e_name in extracted_contours
        ])
        tmpl_mean = tmpl_dists.mean()
        tmpl_std = tmpl_dists.std()

        n_beaten = int((row_means > tmpl_mean).sum())
        pct_better_than = 100.0 * n_beaten / len(row_means)
        rank = int((row_means < tmpl_mean).sum()) + 1  # 1 = best (lowest mean distance)
        z_score = (tmpl_mean - row_means.mean()) / row_means.std()

        print(f"  candidate row-means: mean={row_means.mean():.4f} +/- {row_means.std():.4f} "
              f"(n={len(row_means)})")
        print(f"  template mean distance to extracted: {tmpl_mean:.4f} +/- {tmpl_std:.4f}")
        print(f"  template rank: {rank} / {len(row_means)+1}  "
              f"(beats {pct_better_than:.1f}% of manual candidates, z={z_score:.2f})")

        best5 = row_means.sort_values().head(5)
        print("  top-5 best candidates (lowest mean distance to extracted):")
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
            "n_manual_candidates": len(manual_contours),
            "n_extracted": len(extracted_contours),
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
    cand_path = os.path.join(OUTPUT_DIR, "candidate_scores.csv")
    cand_df.to_csv(cand_path, index=False)
    print(f"\nSaved candidate scores to {cand_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "template_ranking_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ranking summary to {summary_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
