"""
Recreate panel C of DR_mothers_SW.pdf (DTW similarity of each individual's
whistle to the two candidate signature-whistle templates, NSW_1 / Dana and
NSW_2 / Shy) with larger text.

The original figure was hand-built in Inkscape and the underlying data/code
is not available. All values below were read directly off the source PDF
(AutomaticExtraction/template_analysis/output/DR_mothers_SW.pdf, panel C).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------
# Grayscale transfer function, measured directly off the original PDF.
#
# The cell colors in the original are NOT a plain linear grayscale of
# (value-vmin)/(vmax-vmin): the PDF (Inkscape/Cairo) applies a gamma-like
# lift to the underlying linear grayscale image, so mid-range values render
# noticeably lighter than a literal 'gray' colormap would produce. This LUT
# was extracted by rendering the source PDF at 600 dpi and sampling the
# colorbar gradient pixel-by-pixel against its (linearly-spaced) tick
# marks; both NSW_1 and NSW_2 colorbars gave the same normalized curve
# (max discrepancy < 0.5/255), confirming a single shared transfer function
# of normalized value -> gray level.
# ----------------------------------------------------------------------
_LUT_X = np.linspace(0.0, 1.0, 101)
_LUT_Y = np.array([
    0.0, 3.6, 19.9, 31.2, 40.2, 51.0, 60.7, 70.0, 75.5, 83.0,
    87.0, 93.0, 97.0, 102.0, 105.0, 109.0, 113.0, 118.0, 121.0, 124.0,
    128.0, 131.0, 134.0, 138.0, 141.0, 143.9, 147.0, 149.6, 153.0, 157.0,
    159.0, 162.8, 165.0, 169.0, 171.0, 174.0, 176.0, 180.0, 182.0, 185.0,
    189.0, 190.0, 192.0, 194.0, 195.0, 197.0, 199.0, 201.0, 202.0, 204.0,
    206.0, 208.0, 209.0, 211.0, 212.4, 214.0, 216.0, 217.0, 218.0, 220.0,
    221.0, 222.1, 223.0, 224.0, 226.0, 227.0, 228.0, 229.0, 231.0, 232.0,
    233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 240.0, 240.0, 241.0, 242.0,
    242.0, 243.0, 244.0, 245.0, 245.0, 246.0, 247.0, 247.0, 248.0, 248.0,
    249.0, 250.0, 250.0, 251.0, 252.0, 253.0, 253.0, 254.0, 254.0, 255.0,
    255.0,
]) / 255.0


def value_to_gray(value, vmin, vmax):
    norm = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    return float(np.interp(norm, _LUT_X, _LUT_Y))


PANEL_C_CMAP = ListedColormap(np.column_stack([
    np.interp(np.linspace(0, 1, 256), _LUT_X, _LUT_Y)
] * 3))

# ----------------------------------------------------------------------
# Data (read off the original figure)
# ----------------------------------------------------------------------
ROWS = ["Dana", "Shy", "Bar", "Cindy", "Dicky", "Lemon", "Nemesh",
        "Pashosh", "Rajah", "Saki", "Yassu", "Enzo", "Sheba"]

# name -> (DTW value, pB string as displayed, is significant/bold)
NSW1 = {
    "Dana":    (200.2,  "2.6e-67", True),
    "Shy":     (1748.2, "1.000",   False),
    "Bar":     (592.6,  "2.3e-07", True),
    "Cindy":   (730.1,  "0.020",   True),
    "Dicky":   (1424.1, "1.000",   False),
    "Lemon":   (992.4,  "1.000",   False),
    "Nemesh":  (272.9,  "3.2e-17", True),
    "Pashosh": (1617.7, "1.000",   False),
    "Rajah":   (445.7,  "3.1e-18", True),
    "Saki":    (804.1,  "0.209",   False),
    "Yassu":   (334.5,  "2.1e-12", True),
    "Enzo":    (1672.2, "1.000",   False),
    "Sheba":   (495.0,  "2.7e-04", True),
}

NSW2 = {
    "Dana":    (1285.7, "0.494",   False),
    "Shy":     (605.7,  "8.0e-41", True),
    "Bar":     (1795.8, "1.000",   False),
    "Cindy":   (2413.0, "1.000",   False),
    "Dicky":   (1052.9, "1.1e-08", True),
    "Lemon":   (917.0,  "5.2e-14", True),
    "Nemesh":  (680.0,  "2.3e-29", True),
    "Pashosh": (984.9,  "2.4e-09", True),
    "Rajah":   (1200.6, "0.018",   True),
    "Saki":    (1762.4, "1.000",   False),
    "Yassu":   (1606.3, "1.000",   False),
    "Enzo":    (3005.4, "1.000",   False),
    "Sheba":   (954.8,  "4.8e-08", True),
}

NSW1_VMIN, NSW1_VMAX = 200, 1000
NSW2_VMIN, NSW2_VMAX = 600, 1600

FOREST = [
    ("NSW_1 - Dana", 0.230, 0.195, 0.271, "2.56e-67", "black"),
    ("NSW_2 - Shy",  0.444, 0.395, 0.499, "8.05e-41", "0.45"),
]

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
mpl.rcParams["font.family"] = "sans-serif"
FS_ROWLABEL = 19
FS_CELL = 16
FS_TITLE = 22
FS_CBAR = 17
FS_CBAR_LABEL = 18
FS_FOREST_TICK = 20
FS_FOREST_TEXT = 18
FS_FOREST_AXLABEL = 20

WHITE_TEXT_THRESHOLD = 0.5  # gray level (0-1) below which text is drawn white


def draw_heatmap(ax, cax, data, vmin, vmax, title, show_row_labels):
    n = len(ROWS)
    ax.set_xlim(0, 1)
    ax.set_ylim(n, 0)
    ax.set_xticks([])

    for i, name in enumerate(ROWS):
        value, pb_str, bold = data[name]
        gray = value_to_gray(value, vmin, vmax)
        ax.add_patch(Rectangle((0, i), 1, 1, facecolor=(gray, gray, gray),
                                edgecolor="black", linewidth=1.4))
        txt_color = "white" if gray < WHITE_TEXT_THRESHOLD else "black"
        weight = "bold" if bold else "normal"
        ax.text(0.5, i + 0.32, f"{value:.1f}", ha="center", va="center",
                 fontsize=FS_CELL, color=txt_color)
        ax.text(0.5, i + 0.68, f"pB={pb_str}", ha="center", va="center",
                 fontsize=FS_CELL, color=txt_color, fontweight=weight)

    ax.set_yticks([i + 0.5 for i in range(n)])
    if show_row_labels:
        ax.set_yticklabels([f"SW_{name}" for name in ROWS], fontsize=FS_ROWLABEL)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=6, width=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=FS_TITLE, pad=14)

    # colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=PANEL_C_CMAP, norm=norm, orientation="vertical")
    cb.set_ticks(list(range(int(vmin), int(vmax) + 1, 200)))
    cb.ax.tick_params(labelsize=FS_CBAR, length=5, width=1.2)
    cb.set_label("DTW (a.u.)", fontsize=FS_CBAR_LABEL, rotation=270, labelpad=22)
    cb.outline.set_linewidth(1.2)


def draw_forest(ax):
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.6, zorder=1)

    ypos = list(range(len(FOREST)))[::-1]
    for y, (label, r, lo, hi, pb, color) in zip(ypos, FOREST):
        ax.axhline(y, color="0.85", linestyle="--", linewidth=1.2, zorder=0)
        ax.plot([lo, hi], [y, y], color=color, linewidth=4, solid_capstyle="round",
                 zorder=2)
        ax.plot(r, y, marker="o", markersize=15, color=color, zorder=3)
        ax.text(hi + 0.045, y, f"r={r:.3f} [{lo:.3f}–{hi:.3f}]; pB={pb}",
                 va="center", ha="left", fontsize=FS_FOREST_TEXT, color=color)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f[0] for f in FOREST], fontsize=FS_FOREST_TICK)
    ax.set_ylim(-0.7, len(FOREST) - 0.3)

    ax.set_xlim(0.14, 1.08)
    ax.set_xticks([round(0.1 * i, 1) for i in range(2, 11)])
    ax.tick_params(axis="x", labelsize=FS_FOREST_TICK, length=6, width=1.2)
    ax.set_xlabel("Expected DTW ratio", fontsize=FS_FOREST_AXLABEL, labelpad=10)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def main():
    fig = plt.figure(figsize=(14, 16))

    left1, w1 = 0.14, 0.22
    cb1_left, cb1_w = left1 + w1 + 0.02, 0.022
    left2, w2 = 0.53, 0.22
    cb2_left, cb2_w = left2 + w2 + 0.02, 0.022

    top, bottom_hm = 0.95, 0.28

    ax_hm1 = fig.add_axes([left1, bottom_hm, w1, top - bottom_hm])
    ax_cb1 = fig.add_axes([cb1_left, bottom_hm, cb1_w, top - bottom_hm])
    ax_hm2 = fig.add_axes([left2, bottom_hm, w2, top - bottom_hm])
    ax_cb2 = fig.add_axes([cb2_left, bottom_hm, cb2_w, top - bottom_hm])

    draw_heatmap(ax_hm1, ax_cb1, NSW1, NSW1_VMIN, NSW1_VMAX, "NSW_1", True)
    draw_heatmap(ax_hm2, ax_cb2, NSW2, NSW2_VMIN, NSW2_VMAX, "NSW_2", False)

    ax_forest = fig.add_axes([left1, 0.05, (cb2_left + cb2_w) - left1, 0.16])
    draw_forest(ax_forest)

    out_base = "/export/home1/users/zfne/mustun/Documents/GitHub/Dolphins/AutomaticExtraction/template_analysis/output/DR_mothers_SW_panelC_large_text"
    fig.savefig(out_base + ".png", dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_base + ".pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    print("Saved:", out_base + ".png", out_base + ".pdf")


if __name__ == "__main__":
    main()
