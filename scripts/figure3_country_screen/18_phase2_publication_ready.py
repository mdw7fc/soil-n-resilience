"""Publication-ready render of Figure 4.

Visual / layout / caption-readiness revision of the Phase 2 figure. Analytical
content unchanged — reads vulnerability_country.csv and panel_d_summary.csv
as produced by the locked Phase 2 pipeline.

Differences from scripts/17_phase2_panels_production.py:
  - Shorter figure title; year window moved to panel b only
  - Panel titles explicitly note country-level
  - Panel letters (a/b/c/d) drawn separately from titles, anchored upper-left,
    bold lowercase
  - Panel b legend simplified ("Exposure index (0–1)" — drops the
    "combined-index used for visualization" phrase that conflicted with
    panel c's classification rule)
  - Panel c color palette tightened so red/purple are clearly distinct, and
    intermediate/data-missing are visually separable
  - Panel d as HORIZONTAL stacked bar chart with classes ordered top→bottom
    per spec (and a vertical short-label fallback rendered as a separate file)
  - Output triplet (PNG/PDF/SVG) with publication-ready filenames

Outputs:
  figures/fig4_publication_ready_final_c.{png,pdf,svg}
  figures/fig4_phase2_panel_d_horizontal.png
  figures/fig4_phase2_panel_d_vertical_short.png    (fallback)
  figures/fig4_caption_draft.txt
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

# ---------------------------------------------------------------------------
# Production typography
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   9.5,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,    # embed TrueType (Illustrator-editable)
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Panel-c categorical palette — tightened for distinguishability
#   - Strong red vs deep purple-blue: easily separable in print
#   - Green vs muted-blue-gray: separable (green = resilient buffer)
#   - Intermediate (very light cream) vs data missing (medium grey): separable
# ---------------------------------------------------------------------------
PANEL_C_COLORS = {
    "low_buffer_high_exposure":  "#C62828",   # strong saturated red (focal)
    "high_buffer_high_exposure": "#6A3D9A",   # deep blue-violet purple
    "low_buffer_low_exposure":   "#94A6B8",   # muted blue-gray
    "high_buffer_low_exposure":  "#2E7D32",   # forest green
    "intermediate":              "#EFEFEC",   # very light cream-grey
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
PANEL_C_LABELS_SHORT = {
    "low_buffer_high_exposure":  "LB / HE",
    "high_buffer_high_exposure": "HB / HE",
    "low_buffer_low_exposure":   "LB / LE",
    "high_buffer_low_exposure":  "HB / LE",
    "intermediate":              "Intermediate",
    "data_missing":              "Missing",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def titled(ax, letter: str, title: str, pad: int = 4, fontsize: float = 9.5):
    """Set a panel title with a bold lowercase letter prefix.

    Uses mathtext bold so the letter is clearly heavier than the title and
    sits with proper spacing — replaces the previous separate-panel-letter
    approach that rendered as e.g. 'aCountry-level' under tight bbox.
    """
    ax.set_title(rf"$\mathbf{{{letter}}}$    {title}",
                 loc="left", fontsize=fontsize, pad=pad)


def save_triplet(fig, stem: str):
    for fmt in ("png", "pdf", "svg"):
        out = DIR_FIGURES / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=300 if fmt == "png" else None)
        kb = out.stat().st_size / 1024
        print(f"  wrote {out.name}  ({kb:,.0f} KB)")


def load_world() -> gpd.GeoDataFrame:
    return gpd.read_file(
        DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
    )[["ADM0_A3", "ADMIN", "geometry"]] \
        .rename(columns={"ADM0_A3": "iso3", "ADMIN": "ne_name"}) \
        .to_crs(EQUAL_EARTH)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def draw_panel_a(ax, world, vul):
    """Country-level cropland-weighted soil organic N buffer proxy."""
    g = world.merge(vul[["iso3", "buffer_proxy_t_ha", "cropland_Mha"]],
                    on="iso3", how="left")
    g.loc[g["cropland_Mha"].fillna(0) < 0.1, "buffer_proxy_t_ha"] = np.nan
    g.plot(ax=ax, column="buffer_proxy_t_ha",
           cmap="YlGn", vmin=2, vmax=15,
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
    """Country-level fertilizer-shock exposure."""
    g = world.merge(vul[["iso3", "exposure_combined"]], on="iso3", how="left")
    g.plot(ax=ax, column="exposure_combined",
           cmap="YlOrRd", vmin=0, vmax=1,
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
           "Country-level fertilizer-shock exposure (2018–2020 mean)")
    ax.set_axis_off()


def draw_panel_c(ax, world, vul):
    """Country-level soil-buffer × exposure classification."""
    g = world.merge(vul[["iso3", "panel_c_class"]], on="iso3", how="left")
    g["panel_c_class"] = g["panel_c_class"].fillna("data_missing")
    g["color"] = g["panel_c_class"].map(PANEL_C_COLORS)
    g.plot(ax=ax, color=g["color"], edgecolor="white", linewidth=0.15)
    titled(ax, "c", "Soil-buffer × exposure classification")
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


def draw_panel_d_horizontal(ax):
    """Horizontal stacked bar — cropland Mha by class, stacked by region."""
    summary = pd.read_csv(DIR_PROCESSED / "panel_d_summary.csv")
    pivot = (summary.pivot_table(index="panel_c_class", columns="region",
                                  values="cropland_Mha", aggfunc="sum")
                    .fillna(0).reindex(CLASS_ORDER))
    pivot = pivot[[r for r in REGION_ORDER if r in pivot.columns]]

    cmap = plt.get_cmap("tab10")
    y_labels = [PANEL_C_LABELS[c] for c in pivot.index]
    y_pos = np.arange(len(pivot))[::-1]   # top-to-bottom: class 1 at top

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
    titled(ax, "d", "FAOSTAT cropland area by class and region", pad=10)
    ax.tick_params(axis="x", direction="out", length=3)
    ax.tick_params(axis="y", direction="out", length=2)
    # Right-side x-axis padding so the longest bar doesn't feel clipped
    xmax = float(lefts.max())
    ax.set_xlim(0, xmax * 1.06)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=6.2, title="Region", title_fontsize=6.8,
              frameon=False, handlelength=0.9, handleheight=0.8,
              columnspacing=0.5, labelspacing=0.35, borderpad=0.15)


def draw_panel_d_vertical_short(ax):
    """Vertical fallback with short labels."""
    summary = pd.read_csv(DIR_PROCESSED / "panel_d_summary.csv")
    pivot = (summary.pivot_table(index="panel_c_class", columns="region",
                                  values="cropland_Mha", aggfunc="sum")
                    .fillna(0).reindex(CLASS_ORDER))
    pivot = pivot[[r for r in REGION_ORDER if r in pivot.columns]]

    cmap = plt.get_cmap("tab10")
    x_labels = [PANEL_C_LABELS_SHORT[c] for c in pivot.index]
    bottoms = np.zeros(len(pivot))
    for i, region in enumerate(pivot.columns):
        vals = pivot[region].values
        ax.bar(x_labels, vals, bottom=bottoms, label=region,
               color=cmap(i % 10), edgecolor="white", linewidth=0.5)
        bottoms = bottoms + vals
    ax.set_ylabel("Cropland area (Mha)")
    titled(ax, "d", "FAOSTAT cropland area by class and region")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=6.2, title="Region", title_fontsize=6.8,
              frameon=False)


# ---------------------------------------------------------------------------
# Assembled figure (vector-native 2×2)
# ---------------------------------------------------------------------------

def render_assembled(world, vul):
    print("Rendering publication-ready 2×2 assembled figure ...")
    # Aspect ratio chosen for ~170 mm width at 2-column print: width 7.0 in
    fig = plt.figure(figsize=(13.8, 8.6))
    gs = fig.add_gridspec(2, 2,
                          left=0.03, right=0.84,    # leave room on right
                                                    # for panel-d region legend
                          top=0.92, bottom=0.06,
                          hspace=0.28, wspace=0.10,
                          width_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0]); draw_panel_a(ax_a, world, vul)
    ax_b = fig.add_subplot(gs[0, 1]); draw_panel_b(ax_b, world, vul)
    ax_c = fig.add_subplot(gs[1, 0]); draw_panel_c(ax_c, world, vul)
    ax_d = fig.add_subplot(gs[1, 1]); draw_panel_d_horizontal(ax_d)

    fig.suptitle(
        "Global geography of soil organic N buffering and "
        "fertilizer-shock exposure",
        fontsize=11.5, weight="bold", y=0.972,
    )
    save_triplet(fig, "fig4_publication_ready_final_c")
    plt.close(fig)


def render_panel_d_horizontal_only():
    print("Rendering standalone horizontal panel d ...")
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    draw_panel_d_horizontal(ax)
    fig.tight_layout()
    out = DIR_FIGURES / "fig4_phase2_panel_d_horizontal.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}  "
          f"({out.stat().st_size/1024:,.0f} KB)")


def render_panel_d_vertical_short_only():
    print("Rendering standalone vertical-short panel d (fallback) ...")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    draw_panel_d_vertical_short(ax)
    fig.tight_layout()
    out = DIR_FIGURES / "fig4_phase2_panel_d_vertical_short.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}  "
          f"({out.stat().st_size/1024:,.0f} KB)")


# ---------------------------------------------------------------------------
# Caption draft
# ---------------------------------------------------------------------------

CAPTION = """Figure 4. Global geography of soil organic N buffering and fertilizer-shock exposure. a, Country-level cropland-weighted soil organic N buffer proxy, calculated from SoilGrids SOC and C:N and weighted by MIRCA2000 cropped-area distribution. b, Fertilizer-shock exposure based on synthetic N application intensity and fertilizer import reliance, averaged over 2018–2020. The continuous exposure index is shown for visualization; classification in panel c uses the pre-specified high/low exposure rule. c, Country-level soil-buffer / exposure classification. Low buffer is defined as the bottom tercile of cropland-bearing countries by cropland-weighted buffer proxy; high exposure is defined as synthetic N intensity ≥ 150 kg N ha⁻¹ or import reliance ≥ 0.70 with N intensity ≥ 25 kg N ha⁻¹. d, FAOSTAT cropland area in each country class, stacked by region. The figure is a mechanism-specific resilience screen, not a food-security vulnerability index.
"""


def write_caption():
    out = DIR_FIGURES / "fig4_caption_draft.txt"
    out.write_text(CAPTION)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    world = load_world()
    render_assembled(world, vul)
    render_panel_d_horizontal_only()
    render_panel_d_vertical_short_only()
    write_caption()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
