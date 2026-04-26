"""Figure 4 — v10 mechanism-screen render (FIX 3 de-risk pass).

Three changes from publication-ready_final_c:
  (1) Reorder/relabel so continuous gradients lead. Panels a and b are
      continuous (already true). Panel c and d are now labeled as derived
      from a–b: panel c title becomes "Derived classification (from a, b)";
      panel d title is unchanged but in-figure subtitle clarifies the
      derivation chain.
  (2) Annotation in panel d shows threshold sensitivity:
      focal class 17.4 % → 4.85 % under cropland-area-weighted thresholds
      (small text annotation in upper-right of the panel; does not crowd
      the regional stack).
  (3) Title hierarchy: figure suptitle drops the bare descriptive title and
      adds a subtitle "Mechanism-specific resilience screen". Any
      "vulnerability" phrasing was already absent and remains so.

Colorblind-safe palette:
  - Panel a (buffer): matplotlib `viridis` (perceptually uniform, cb-safe)
  - Panel b (exposure): matplotlib `magma` (perceptually uniform, cb-safe,
        reads warm/alarming at the high end)
  - Panel c (categorical): Okabe-Ito-derived 6-class palette
        (vermillion / reddish-purple / sky-blue / bluish-green / 2 grays)
  - Panel d (regions stack): tab10 (cb-acceptable for 9 categories;
        regional identity is secondary to the bar lengths)

Dimensions match prior Figure 4 (13.8 × 8.6 in, 2×2 grid).

Outputs:
  figures/Fig4_mechanism_screen_v10.png   (300 dpi)
  figures/Fig4_mechanism_screen_v10.pdf   (vector)
  figures/Fig4_mechanism_screen_v10.svg   (vector, editable)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import DIR_FIGURES, DIR_PROCESSED, DIR_RAW, REGION_GROUPS  # noqa: E402

mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   9.5,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Okabe-Ito-derived categorical palette for panel c (colorblind-safe)
#   reference: Okabe & Ito 2008 colorblind-safe 8-class set
# ---------------------------------------------------------------------------
PANEL_C_COLORS = {
    "low_buffer_high_exposure":  "#D55E00",   # vermillion (focal)
    "high_buffer_high_exposure": "#CC79A7",   # reddish-purple
    "low_buffer_low_exposure":   "#56B4E9",   # sky-blue (low stakes)
    "high_buffer_low_exposure":  "#009E73",   # bluish-green (resilient)
    "intermediate":              "#E8E8E8",   # very light gray
    "data_missing":              "#9E9E9E",   # medium gray
}
PANEL_C_LABELS = {
    "low_buffer_high_exposure":  "Low buffer / high exposure",
    "high_buffer_high_exposure": "High buffer / high exposure",
    "low_buffer_low_exposure":   "Low buffer / low exposure",
    "high_buffer_low_exposure":  "High buffer / low exposure",
    "intermediate":              "Intermediate",
    "data_missing":              "Data missing",
}
CLASS_ORDER = [
    "low_buffer_high_exposure",
    "high_buffer_high_exposure",
    "low_buffer_low_exposure",
    "high_buffer_low_exposure",
    "intermediate",
    "data_missing",
]
REGION_ORDER = list(REGION_GROUPS.keys())
EQUAL_EARTH = "+proj=eqearth +datum=WGS84 +units=m +no_defs"
STEM = "Fig4_mechanism_screen_v10"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def titled(ax, letter, title, pad=4, fontsize=9.5):
    ax.set_title(rf"$\mathbf{{{letter}}}$    {title}",
                 loc="left", fontsize=fontsize, pad=pad)


def save_triplet(fig, stem):
    out_paths = []
    for fmt in ("png", "pdf", "svg"):
        out = DIR_FIGURES / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=300 if fmt == "png" else None)
        out_paths.append(out)
        print(f"  wrote {out.name}  ({out.stat().st_size/1024:,.0f} KB)")
    return out_paths


def load_world():
    return gpd.read_file(
        DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
    )[["ADM0_A3", "ADMIN", "geometry"]] \
        .rename(columns={"ADM0_A3": "iso3", "ADMIN": "ne_name"}) \
        .to_crs(EQUAL_EARTH)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def draw_panel_a(ax, world, vul):
    """Continuous: cropland-weighted buffer proxy."""
    g = world.merge(vul[["iso3", "buffer_proxy_t_ha", "cropland_Mha"]],
                    on="iso3", how="left")
    g.loc[g["cropland_Mha"].fillna(0) < 0.1, "buffer_proxy_t_ha"] = np.nan
    g.plot(ax=ax, column="buffer_proxy_t_ha",
           cmap="viridis", vmin=2, vmax=15,
           edgecolor="white", linewidth=0.15,
           missing_kwds={"facecolor": "#F2F2F2", "edgecolor": "white",
                         "linewidth": 0.15},
           legend=True,
           legend_kwds={
               "label": "Cropland-weighted soil organic N stock, 0–30 cm "
                        "(t N ha⁻¹; truncated at 15)",
               "orientation": "horizontal", "shrink": 0.55,
               "pad": 0.01, "aspect": 32,
           })
    titled(ax, "a",
           "Country-level cropland-weighted soil organic N buffer proxy")
    ax.set_axis_off()


def draw_panel_b(ax, world, vul):
    """Continuous: exposure index."""
    g = world.merge(vul[["iso3", "exposure_combined"]], on="iso3", how="left")
    g.plot(ax=ax, column="exposure_combined",
           cmap="magma", vmin=0, vmax=1,
           edgecolor="white", linewidth=0.15,
           missing_kwds={"facecolor": "#F2F2F2", "edgecolor": "white",
                         "linewidth": 0.15},
           legend=True,
           legend_kwds={
               "label": "Exposure index (0–1)",
               "orientation": "horizontal", "shrink": 0.55,
               "pad": 0.01, "aspect": 32,
           })
    titled(ax, "b",
           "Country-level fertilizer-shock exposure index, 2018–2020 mean")
    ax.set_axis_off()


def draw_panel_c(ax, world, vul):
    """Categorical, derived from a, b."""
    g = world.merge(vul[["iso3", "panel_c_class"]], on="iso3", how="left")
    g["panel_c_class"] = g["panel_c_class"].fillna("data_missing")
    g["color"] = g["panel_c_class"].map(PANEL_C_COLORS)
    g.plot(ax=ax, color=g["color"], edgecolor="white", linewidth=0.15)
    titled(ax, "c",
           "Soil-buffer × exposure classification (derived from a, b)")
    ax.set_axis_off()
    handles = [mpatches.Patch(facecolor=PANEL_C_COLORS[k],
                              edgecolor="0.4", linewidth=0.4,
                              label=PANEL_C_LABELS[k])
               for k in CLASS_ORDER]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=8.0, frameon=False,
              handlelength=1.3, handleheight=1.0,
              columnspacing=0.8, labelspacing=0.4, borderpad=0.2)


def draw_panel_d(ax):
    """Horizontal stacked bar with threshold-sensitivity annotation."""
    summary = pd.read_csv(DIR_PROCESSED / "panel_d_summary.csv")
    pivot = (summary.pivot_table(index="panel_c_class", columns="region",
                                  values="cropland_Mha", aggfunc="sum")
                    .fillna(0).reindex(CLASS_ORDER))
    pivot = pivot[[r for r in REGION_ORDER if r in pivot.columns]]

    cmap = plt.get_cmap("tab10")
    y_labels = [PANEL_C_LABELS[c] for c in pivot.index]
    y_pos = np.arange(len(pivot))[::-1]
    lefts = np.zeros(len(pivot))
    for i, region in enumerate(pivot.columns):
        vals = pivot[region].values
        ax.barh(y_pos, vals, left=lefts, label=region,
                color=cmap(i % 10), edgecolor="white", linewidth=0.5,
                height=0.78)
        lefts = lefts + vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Cropland area (Mha)")
    titled(ax, "d", "FAOSTAT cropland area by derived class and region",
           pad=10)
    ax.tick_params(axis="x", direction="out", length=3)
    ax.tick_params(axis="y", direction="out", length=2)
    xmax = float(lefts.max())
    ax.set_xlim(0, xmax * 1.06)

    # ---- Threshold-sensitivity annotation (FIX 3) ----
    annotation = (
        "Threshold sensitivity (SI):\n"
        "country-tercile (locked):  17.4 % focal\n"
        "cropland-area-weighted:    4.85 % focal\n"
        "shift mostly explained by China"
    )
    ax.text(0.985, 0.18, annotation,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=6.8, color="#333333",
            bbox=dict(facecolor="white", edgecolor="0.7",
                      linewidth=0.5, boxstyle="round,pad=0.4"),
            family="monospace")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=6.2, title="Region", title_fontsize=6.8,
              frameon=False, handlelength=0.9, handleheight=0.8,
              columnspacing=0.5, labelspacing=0.35, borderpad=0.15)


# ---------------------------------------------------------------------------
# Assembled
# ---------------------------------------------------------------------------

def render_assembled(world, vul):
    print(f"Rendering {STEM} ...")
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 2,
                          left=0.03, right=0.84,
                          top=0.90, bottom=0.06,
                          hspace=0.28, wspace=0.10,
                          width_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0]); draw_panel_a(ax_a, world, vul)
    ax_b = fig.add_subplot(gs[0, 1]); draw_panel_b(ax_b, world, vul)
    ax_c = fig.add_subplot(gs[1, 0]); draw_panel_c(ax_c, world, vul)
    ax_d = fig.add_subplot(gs[1, 1]); draw_panel_d(ax_d)

    fig.text(0.5, 0.965,
             "Global geography of soil organic N buffering and "
             "fertilizer-shock exposure",
             ha="center", va="bottom",
             fontsize=11.5, weight="bold")
    fig.text(0.5, 0.945, "Mechanism-specific resilience screen",
             ha="center", va="bottom",
             fontsize=9.5, style="italic", color="#333333")

    save_triplet(fig, STEM)
    plt.close(fig)


def main() -> int:
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    world = load_world()
    render_assembled(world, vul)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
