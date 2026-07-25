"""Production-quality render of Figure 4 — 300 DPI PNG + PDF + SVG.

Differences from scripts/15_phase2_panels.py:
  - 300 DPI raster, plus PDF and SVG vector exports
  - Assembled figure built as native matplotlib subplots (not imread of PNGs),
    so vector export carries through
  - Consistent typography (Helvetica/Arial fallback to DejaVu Sans)
  - Panel letters drawn explicitly in figure-anchor positions
  - Re-export-hub caption footnote saved to figures/

Inputs:  data_processed/vulnerability_country.csv, panel_d_summary.csv,
         exposure_country.csv, ne_50m_admin_0_countries
Outputs (all in figures/):
  fig4_panel_a.{png,pdf,svg}
  fig4_panel_b.{png,pdf,svg}
  fig4_panel_c.{png,pdf,svg}
  fig4_panel_d.{png,pdf,svg}
  fig4_assembled.{png,pdf,svg}
  reexport_hubs_caption_footnote.txt
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
from _config import DIR_FIGURES, DIR_PROCESSED, DIR_RAW, REGION_GROUPS, YEAR_LABEL  # noqa: E402

# Production typography
mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,   # screen preview only
    "savefig.dpi":      300,   # production raster
    "pdf.fonttype":     42,    # embed TrueType (editable in Illustrator)
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
})

PANEL_C_COLORS = {
    "high_buffer_low_exposure":  "#5BAEDB",
    "high_buffer_high_exposure": "#7B5DA6",
    "low_buffer_low_exposure":   "#C7CDD3",
    "low_buffer_high_exposure":  "#D7301F",
    "intermediate":              "#E8E4DC",
    "data_missing":              "#FFFFFF",
}
PANEL_C_LABELS = {
    "high_buffer_low_exposure":  "High buffer / low exposure",
    "high_buffer_high_exposure": "High buffer / high exposure",
    "low_buffer_low_exposure":   "Low buffer / low exposure",
    "low_buffer_high_exposure":  "Low buffer / high exposure",
    "intermediate":              "Intermediate",
    "data_missing":              "Data missing",
}
EQUAL_EARTH = "+proj=eqearth +datum=WGS84 +units=m +no_defs"
EXPORT_FORMATS = ("png", "pdf", "svg")


def save_all(fig, stem: str) -> None:
    for fmt in EXPORT_FORMATS:
        out = DIR_FIGURES / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=300 if fmt == "png" else None,
                    metadata={"Title": stem} if fmt == "pdf" else None)
        size_kb = out.stat().st_size / 1024
        print(f"  wrote {out.name}  ({size_kb:,.0f} KB)")


def load_world() -> gpd.GeoDataFrame:
    return gpd.read_file(
        DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
    )[["ADM0_A3", "ADMIN", "geometry"]] \
        .rename(columns={"ADM0_A3": "iso3", "ADMIN": "ne_name"}) \
        .to_crs(EQUAL_EARTH)


def _draw_panel_a(ax, world, vul):
    g = world.merge(vul[["iso3", "buffer_proxy_t_ha", "cropland_Mha"]],
                    on="iso3", how="left")
    g.loc[g["cropland_Mha"].fillna(0) < 0.1, "buffer_proxy_t_ha"] = np.nan
    g.plot(ax=ax, column="buffer_proxy_t_ha",
           cmap="YlGn", vmin=2, vmax=15,
           edgecolor="white", linewidth=0.15,
           missing_kwds={"facecolor": "#F2F2F2", "edgecolor": "white",
                         "linewidth": 0.15},
           legend=True,
           legend_kwds={"label": "Cropland-weighted soil organic N stock 0–30 cm "
                                 "(t N ha⁻¹; truncated at 15)",
                        "orientation": "horizontal",
                        "shrink": 0.6, "pad": 0.04, "aspect": 30})
    ax.set_axis_off()


def _draw_panel_b(ax, world, vul):
    g = world.merge(vul[["iso3", "exposure_combined"]], on="iso3", how="left")
    g.plot(ax=ax, column="exposure_combined",
           cmap="YlOrRd", vmin=0, vmax=1,
           edgecolor="white", linewidth=0.15,
           missing_kwds={"facecolor": "#F2F2F2", "edgecolor": "white",
                         "linewidth": 0.15},
           legend=True,
           legend_kwds={"label": "Exposure index (0–1)",
                        "orientation": "horizontal",
                        "shrink": 0.6, "pad": 0.04, "aspect": 30})
    ax.set_axis_off()


def _draw_panel_c(ax, world, vul):
    g = world.merge(vul[["iso3", "panel_c_class"]], on="iso3", how="left")
    g["panel_c_class"] = g["panel_c_class"].fillna("data_missing")
    g["color"] = g["panel_c_class"].map(PANEL_C_COLORS)
    g.plot(ax=ax, color=g["color"], edgecolor="white", linewidth=0.15)
    handles = [mpatches.Patch(color=PANEL_C_COLORS[k], label=PANEL_C_LABELS[k])
               for k in ["low_buffer_high_exposure",
                         "high_buffer_high_exposure",
                         "low_buffer_low_exposure",
                         "high_buffer_low_exposure",
                         "intermediate", "data_missing"]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.05),
              fontsize=7, frameon=False, ncol=2,
              handlelength=1.0, handleheight=0.8, borderpad=0.2)
    ax.set_axis_off()


def _draw_panel_d(ax):
    summary = pd.read_csv(DIR_PROCESSED / "panel_d_summary.csv")
    class_order = ["low_buffer_high_exposure", "high_buffer_high_exposure",
                   "low_buffer_low_exposure", "high_buffer_low_exposure",
                   "intermediate", "data_missing"]
    pivot = (summary.pivot_table(index="panel_c_class", columns="region",
                                  values="cropland_Mha", aggfunc="sum")
                    .fillna(0).reindex(class_order))
    region_order = list(REGION_GROUPS.keys())
    pivot = pivot[[r for r in region_order if r in pivot.columns]]

    cmap = plt.get_cmap("tab10")
    bottoms = np.zeros(len(pivot))
    x_labels = [PANEL_C_LABELS[c] for c in pivot.index]
    for i, region in enumerate(pivot.columns):
        vals = pivot[region].values
        ax.bar(x_labels, vals, bottom=bottoms, label=region,
               color=cmap(i % 10), edgecolor="white", linewidth=0.5)
        bottoms = bottoms + vals
    ax.set_ylabel("Cropland area (Mha)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=7, title="Region", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def render_individual_panels(world, vul):
    print("Rendering individual panels at 300 DPI ...")
    # Panel a
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _draw_panel_a(ax, world, vul)
    ax.set_title("a   Cropland soil organic N buffer proxy",
                 fontsize=10, loc="left", weight="bold")
    save_all(fig, "fig4_panel_a"); plt.close(fig)
    # Panel b
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _draw_panel_b(ax, world, vul)
    ax.set_title(f"b   Fertilizer-shock exposure ({YEAR_LABEL})",
                 fontsize=10, loc="left", weight="bold")
    save_all(fig, "fig4_panel_b"); plt.close(fig)
    # Panel c
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    _draw_panel_c(ax, world, vul)
    ax.set_title("c   Combined buffer × exposure classification",
                 fontsize=10, loc="left", weight="bold")
    save_all(fig, "fig4_panel_c"); plt.close(fig)
    # Panel d
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    _draw_panel_d(ax)
    ax.set_title("d   Cropland area by class, stacked by region",
                 fontsize=10, loc="left", weight="bold")
    save_all(fig, "fig4_panel_d"); plt.close(fig)


def render_assembled(world, vul):
    print("Rendering assembled 4-panel figure (vector-native) ...")
    fig = plt.figure(figsize=(14, 8.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.03, right=0.97, top=0.93, bottom=0.06,
                          hspace=0.30, wspace=0.10)

    ax_a = fig.add_subplot(gs[0, 0]); _draw_panel_a(ax_a, world, vul)
    ax_a.set_title("a   Cropland soil organic N buffer proxy",
                   fontsize=10, loc="left", weight="bold")

    ax_b = fig.add_subplot(gs[0, 1]); _draw_panel_b(ax_b, world, vul)
    ax_b.set_title(f"b   Fertilizer-shock exposure ({YEAR_LABEL})",
                   fontsize=10, loc="left", weight="bold")

    ax_c = fig.add_subplot(gs[1, 0]); _draw_panel_c(ax_c, world, vul)
    ax_c.set_title("c   Combined buffer × exposure class",
                   fontsize=10, loc="left", weight="bold")

    ax_d = fig.add_subplot(gs[1, 1]); _draw_panel_d(ax_d)
    ax_d.set_title("d   Cropland area × class × region",
                   fontsize=10, loc="left", weight="bold")

    fig.suptitle(
        "Global geography of soil organic N buffering and fertilizer-shock "
        f"exposure  ({YEAR_LABEL})",
        fontsize=12, weight="bold", y=0.985,
    )
    save_all(fig, "fig4_assembled"); plt.close(fig)


def write_reexport_footnote():
    exp = pd.read_csv(DIR_PROCESSED / "exposure_country.csv")
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    flagged = vul[vul["reexport_flag"]].sort_values(
        "cropland_Mha", ascending=False
    )
    lines = []
    lines.append("Re-export hub footnote (for Figure 4 caption / SI)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Eleven countries are flagged as N-fertilizer re-export hubs under "
        "the tightened criteria (Imports + Exports / Apparent_consumption "
        "> 3, with both Imports + Exports > 50 kt N and Apparent_consumption "
        "> 10 kt N). They are shown un-hatched in the panels because none "
        "are classified low-buffer / high-exposure under the locked rule. "
        "The flagged countries (in descending order of apparent N "
        "consumption) are:"
    )
    lines.append("")
    for _, r in flagged.iterrows():
        lines.append(
            f"  • {r['fao_name']:35s}  intensity={r['n_intensity_raw']:.0f} "
            f"kg N/ha, reliance={r['import_reliance']:.2f}, class="
            f"{r['exposure_class']}"
        )
    out = DIR_FIGURES / "reexport_hubs_caption_footnote.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out.name}")


def main() -> int:
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    world = load_world()
    render_individual_panels(world, vul)
    render_assembled(world, vul)
    write_reexport_footnote()
    print("\nDone. PNG/PDF/SVG ready at production quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
