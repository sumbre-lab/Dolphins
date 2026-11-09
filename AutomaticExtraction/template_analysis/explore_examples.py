"""
Quick exploratory look at the manually-annotated whistle examples used for
template validation: plots a handful of contours per signature whistle (SW)
category from the Corrected_Clusters folder, plus the extraction template
for that category.
"""

import os
import random

import matplotlib.pyplot as plt
import scipy.io as sio

CLUSTERS_DIR = "/media/zfnews31/DolphinAnalysis/Whistles_annotated_from_divers_video/Corrected_Clusters"
ANNOTATIONS_DIR = "/media/zfnews31/DolphinAnalysis/Annotations"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Signature whistle -> (cluster folder, template .mat file used for extraction, per models.csv)
SW_CONFIG = {
    "SW_Neo": ("cluster_Neo", "SW_Neo_simple.mat"),
    "SW_Luna": ("cluster_Luna", "SW_Luna.mat"),
    "SW_Nikita": ("cluster_Nikita", "SW_Nikita_normal.mat"),
    "SW_Nana": ("cluster_Nana", "SW_Nana.mat"),
    "SW_Yosefa": ("cluster_Yosefa", "SW_Yosefa_long_wavy.mat"),
}

N_EXAMPLES = 4
SEED = 0


def load_contour(mat_path):
    d = sio.loadmat(mat_path)
    wht = d["wht"].squeeze()
    whf = d["whf"].squeeze()
    return wht, whf


def list_examples(cluster_folder):
    files = [
        f for f in os.listdir(cluster_folder)
        if f.lower().endswith(".mat")
    ]
    return sorted(files)


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        len(SW_CONFIG), N_EXAMPLES + 1,
        figsize=(4 * (N_EXAMPLES + 1), 3 * len(SW_CONFIG)),
    )

    for row, (sw_name, (cluster_name, template_file)) in enumerate(SW_CONFIG.items()):
        cluster_folder = os.path.join(CLUSTERS_DIR, cluster_name)
        example_files = list_examples(cluster_folder)
        print(f"{sw_name}: {len(example_files)} manual examples in {cluster_name}/")

        sample = random.sample(example_files, min(N_EXAMPLES, len(example_files)))

        # First column: the template used for extraction
        ax = axes[row, 0]
        template_path = os.path.join(ANNOTATIONS_DIR, template_file)
        t, f = load_contour(template_path)
        ax.plot(t - t[0], f / 1000, color="crimson", lw=1.5)
        ax.set_title(f"{sw_name}\nTEMPLATE ({template_file})", fontsize=8, color="crimson")
        ax.set_ylabel("Freq (kHz)")

        for col, fname in enumerate(sample, start=1):
            ax = axes[row, col]
            t, f = load_contour(os.path.join(cluster_folder, fname))
            ax.plot(t - t[0], f / 1000, color="steelblue", lw=1)
            ax.set_title(fname, fontsize=8)

        for col in range(len(sample) + 1, N_EXAMPLES + 1):
            axes[row, col].axis("off")

        for col in range(N_EXAMPLES + 1):
            axes[row, col].set_xlabel("Time (s)", fontsize=7)
            axes[row, col].tick_params(labelsize=7)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "example_contours_per_category.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
