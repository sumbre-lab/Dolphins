"""
Template validation analysis.

For each of the 5 signature whistles (SW_Neo, SW_Luna, SW_Nikita, SW_Nana,
SW_Yosefa), compares the DTW-based extraction template against real whistle
examples, using the project's own normalized-DTW metric
(src/extraction/dtw.py:compute_normalized_dtw_distance):

  - background : pairwise DTW distance among manually-annotated examples only
                 (Corrected_Clusters) -> natural within-category variability,
                 independent of the extraction method.
  - tmpl_manual : DTW distance from the template to each manual example.
  - tmpl_extracted : DTW distance from the template to a random sample of
                 whistles the automatic extractor pulled out of recordings
                 (HF dataset) for that category.

Comparing tmpl_manual against the background distribution shows whether the
template is a "typical" member of the category. Comparing tmpl_extracted
against tmpl_manual can reveal selection bias introduced by the extraction
method itself (it used DTW-to-template matching to find these whistles in
the first place).
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


def pairwise_distances(contours_dict):
    names = list(contours_dict.keys())
    pairs = list(itertools.combinations(names, 2))

    def _dist(pair):
        a, b = pair
        return compute_normalized_dtw_distance(contours_dict[a], contours_dict[b])

    dists = Parallel(n_jobs=N_JOBS)(delayed(_dist)(p) for p in pairs)
    return dists


def template_distances(template_contour, contours_dict):
    names = list(contours_dict.keys())

    def _dist(name):
        return compute_normalized_dtw_distance(template_contour, contours_dict[name])

    dists = Parallel(n_jobs=N_JOBS)(delayed(_dist)(n) for n in names)
    return dict(zip(names, dists))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_path = os.path.join(CACHE_DIR, "extracted_whistles_sample.parquet")
    sample_df = pd.read_parquet(sample_path)
    print(f"Loaded {len(sample_df)} extracted-whistle samples from {sample_path}")

    rows = []
    summary_rows = []

    for sw_name, (cluster_name, template_file) in SW_CONFIG.items():
        print(f"\n=== {sw_name} ===")
        cluster_folder = os.path.join(CLUSTERS_DIR, cluster_name)
        manual_contours = load_manual_contours(cluster_folder)
        print(f"  manual examples: {len(manual_contours)}")

        extracted_contours = load_extracted_contours(sample_df, sw_name)
        print(f"  extracted examples: {len(extracted_contours)}")

        template_contour = load_mat_contour(os.path.join(ANNOTATIONS_DIR, template_file))

        background = pairwise_distances(manual_contours)
        print(f"  background pairwise (manual-manual): {len(background)} pairs, "
              f"median={np.median(background):.4f}")

        tmpl_manual = template_distances(template_contour, manual_contours)
        print(f"  template->manual: {len(tmpl_manual)} dists, "
              f"median={np.median(list(tmpl_manual.values())):.4f}")

        tmpl_extracted = template_distances(template_contour, extracted_contours)
        print(f"  template->extracted: {len(tmpl_extracted)} dists, "
              f"median={np.median(list(tmpl_extracted.values())):.4f}")

        for d in background:
            rows.append({"sw_name": sw_name, "comparison": "background (manual-manual)", "distance": d, "example": None})
        for name, d in tmpl_manual.items():
            rows.append({"sw_name": sw_name, "comparison": "template -> manual", "distance": d, "example": name})
        for name, d in tmpl_extracted.items():
            rows.append({"sw_name": sw_name, "comparison": "template -> extracted", "distance": d, "example": name})

        bg = np.asarray(background)
        tm = np.asarray(list(tmpl_manual.values()))
        te = np.asarray(list(tmpl_extracted.values()))
        pct_manual = 100.0 * np.mean(bg <= np.median(tm)) if len(bg) and len(tm) else np.nan
        pct_extracted = 100.0 * np.mean(bg <= np.median(te)) if len(bg) and len(te) else np.nan

        summary_rows.append({
            "sw_name": sw_name,
            "template_file": template_file,
            "n_manual": len(manual_contours),
            "n_extracted": len(extracted_contours),
            "n_background_pairs": len(bg),
            "background_median": np.median(bg),
            "background_p25": np.percentile(bg, 25),
            "background_p75": np.percentile(bg, 75),
            "tmpl_manual_median": np.median(tm),
            "tmpl_manual_percentile_in_background": pct_manual,
            "tmpl_extracted_median": np.median(te),
            "tmpl_extracted_percentile_in_background": pct_extracted,
        })

    dist_df = pd.DataFrame(rows)
    dist_path = os.path.join(OUTPUT_DIR, "dtw_distances_long.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"\nSaved long-format distances to {dist_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "dtw_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
