"""Render Phase 1 prototype panels b, c, d as choropleths + bar chart.

Inputs:
  - data_processed/vulnerability_country.csv
  - data_raw/ne_50m_admin_0_countries/*  (Natural Earth)

Outputs:
  - figures/fig4_phase1_panel_b_choropleth.png   exposure index choropleth
  - figures/fig4_phase1_panel_c_choropleth.png   5-class panel c choropleth
  - figures/fig4_phase1_panel_d_stacked.png      cropland Mha by class × region
  - figures/fig4_phase1_assembled.png            quick 2x2 assembly
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import (  # noqa: E402
    DIR_FIGURES,
    DIR_PROCESSED,
    DIR_RAW,
    REGION_GROUPS,
    YEAR_LABEL,
)


PANEL_C_COLORS = {
    "high_buffer_low_exposure":  "#5BAEDB",  # desaturated blue
    "high_buffer_high_exposure": "#7B5DA6",  # purple
    "low_buffer_low_exposure":   "#C7CDD3",  # light grey-blue
    "low_buffer_high_exposure":  "#D7301F",  # warm red (focal)
    "intermediate":              "#E8E4DC",  # neutral cream
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


def load_world() -> gpd.GeoDataFrame:
    ne = gpd.read_file(
        DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
    )[["ADM0_A3", "ADMIN", "geometry"]].rename(
        columns={"ADM0_A3": "iso3", "ADMIN": "ne_name"})
    return ne.to_crs(EQUAL_EARTH)


def panel_b(world: gpd.GeoDataFrame, vul: pd.DataFrame, out: Path) -> None:
    g = world.merge(vul[["iso3", "exposure_combined"]], on="iso3", how="left")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    g.plot(ax=ax, column="exposure_combined",
           cmap="YlOrRd", vmin=0, vmax=1,
           edgecolor="white", linewidth=0.2,
           missing_kwds={"facecolor": "#F2F2F2", "edgecolor": "white"},
           legend=True, legend_kwds={"label": "Exposure index (0–1)",
                                      "orientation": "horizontal",
                                      "shrink": 0.6, "pad": 0.04})
    ax.set_title(f"Panel b — Fertilizer-shock exposure  ({YEAR_LABEL})",
                 fontsize=11, loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def panel_c(world: gpd.GeoDataFrame, vul: pd.DataFrame, out: Path) -> None:
    g = world.merge(vul[["iso3", "panel_c_class"]], on="iso3", how="left")
    g["panel_c_class"] = g["panel_c_class"].fillna("data_missing")
    g["color"] = g["panel_c_class"].map(PANEL_C_COLORS)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    g.plot(ax=ax, color=g["color"], edgecolor="white", linewidth=0.2)

    handles = [mpatches.Patch(color=PANEL_C_COLORS[k], label=PANEL_C_LABELS[k])
               for k in ["low_buffer_high_exposure",
                         "high_buffer_high_exposure",
                         "low_buffer_low_exposure",
                         "high_buffer_low_exposure",
                         "intermediate", "data_missing"]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.02),
              fontsize=7, frameon=False)
    ax.set_title(f"Panel c — Combined buffer × exposure classification  "
                 f"({YEAR_LABEL})",
                 fontsize=11, loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def panel_d(vul: pd.DataFrame, out: Path) -> None:
    summary = pd.read_csv(DIR_PROCESSED / "panel_d_summary.csv")
    class_order = ["low_buffer_high_exposure", "high_buffer_high_exposure",
                   "low_buffer_low_exposure", "high_buffer_low_exposure",
                   "intermediate", "data_missing"]
    summary = summary[summary["panel_c_class"].isin(class_order)]
    pivot = (summary.pivot_table(index="panel_c_class", columns="region",
                                  values="cropland_Mha", aggfunc="sum")
                    .fillna(0).reindex(class_order))
    region_order = list(REGION_GROUPS.keys())
    pivot = pivot[[r for r in region_order if r in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    bottoms = np.zeros(len(pivot))
    x_labels = [PANEL_C_LABELS[c] for c in pivot.index]
    for i, region in enumerate(pivot.columns):
        vals = pivot[region].values
        ax.bar(x_labels, vals, bottom=bottoms,
               label=region, color=cmap(i % 10), edgecolor="white")
        bottoms = bottoms + vals
    ax.set_ylabel("Cropland area (Mha)")
    ax.set_title(f"Panel d — Cropland area × class × region  ({YEAR_LABEL})",
                 fontsize=11, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7,
              title="Region")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def assembled(out: Path) -> None:
    """Quick 2x2 assembly using existing panel PNGs (low-fi preview)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    files = [
        ("Panel a (deferred to Phase 1)", None),
        ("Panel b — exposure",      DIR_FIGURES / "fig4_phase1_panel_b_choropleth.png"),
        ("Panel c — combined",      DIR_FIGURES / "fig4_phase1_panel_c_choropleth.png"),
        ("Panel d — area × class × region", DIR_FIGURES / "fig4_phase1_panel_d_stacked.png"),
    ]
    for ax, (lbl, path) in zip(axes.ravel(), files):
        if path and path.exists():
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(lbl, fontsize=10, loc="left")
        else:
            ax.text(0.5, 0.5, lbl, ha="center", va="center", fontsize=12,
                    transform=ax.transAxes)
        ax.set_axis_off()
    fig.suptitle(f"Figure 4 (Phase 1 prototype, country-level only) — "
                 f"{YEAR_LABEL}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def main() -> int:
    print("Rendering Phase 1 prototype panels ...")
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    world = load_world()

    panel_b(world, vul, DIR_FIGURES / "fig4_phase1_panel_b_choropleth.png")
    panel_c(world, vul, DIR_FIGURES / "fig4_phase1_panel_c_choropleth.png")
    panel_d(vul,  DIR_FIGURES / "fig4_phase1_panel_d_stacked.png")
    assembled(DIR_FIGURES / "fig4_phase1_assembled.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
