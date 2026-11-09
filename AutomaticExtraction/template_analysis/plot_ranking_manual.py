"""
Same plot as plot_ranking.py, but scored against the manual-annotation pool
itself (leave-one-out) instead of the extracted whistles - see
rank_candidates_manual.py.
"""

import os

from plot_ranking import OUTPUT_DIR, plot_ranking

if __name__ == "__main__":
    plot_ranking(
        cand_csv=os.path.join(OUTPUT_DIR, "candidate_scores_manual.csv"),
        summary_csv=os.path.join(OUTPUT_DIR, "template_ranking_summary_manual.csv"),
        out_name="template_ranking_manual.png",
        y_label="Mean normalized DTW distance to other manual whistles",
        suptitle=(
            "Is the chosen template a good candidate? Mean distance to the rest of the manual\n"
            "annotation pool (leave-one-out), template vs. every whistle that could have been picked instead"
        ),
    )
