"""Render Phase 2 4-panel Figure 4 (country-level).

Panels:
  a. Cropland-weighted soil organic N buffer proxy (choropleth, t N/ha)
  b. Fertilizer-shock exposure (choropleth, combined index 0–1)
  c. Combined buffer × exposure 5-class panel (choropleth)
  d. Cropland area by class, stacked by region (bar chart)

Outputs:
  figures/fig4_phase2_panel_a.png
  figures/fig4_phase2_panel_b.png
  figures/fig4_phase2_panel_c.png
  figures/fig4_phase2_panel_d.png
  figures/fig4_phase2_assembled.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from _config import DIR_FIGURES, DIR_PROCESSED, DIR_RAW, REGION_GROUPS, YEAR_LABEL  # noqa: E402


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


def load_world() -> gpd.GeoDataFrame:
    ne = gpd.read_file(
        DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
    )[["ADM0_A3", "ADMIN", "geometry"]].rename(
        columns={"ADM0_A3": "iso3", "ADMIN": "ne_name"})
    return ne.to_crs(EQUAL_EARTH)


def panel_a(world, vul, out: Path):
    """Cropland-weighted soil organic N buffer proxy."""
    g = world.merge(vul[["iso3", "buffer_proxy_t_ha", "cropland_Mha"]],
                    on="iso3", how="left")
    # Mask countries with negligible cropland to avoid noisy small-island colors
    g.loc[g["cropland_Mha"].fillna(0) < 0.1, "buffer_proxy_t_ha"] = np.nan

    fig, ax = plt.subplots(figsize=(11, 5.2))
    g.plot(ax=ax, column="buffer_proxy_t_ha",
           cmap="YlGn", vmin=2, vmax=15,
           edgecolor="white", linewidth=0.2,
           missing_kwds={"facecolor": "#F2F2F2", "edgecolor": "white"},
           legend=True,
           legend_kwds={"label": "Cropland-weighted soil N stock 0–30 cm "
                                 "(t N/ha; truncated at 15)",
                         "orientation": "horizontal", "shrink": 0.6,
                         "pad": 0.04})
    ax.set_title("a. Cropland soil organic N buffer proxy",
                 fontsize=11, loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def panel_b(world, vul, out: Path):
    g = world.merge(vul[["iso3", "exposure_combined"]], on="iso3", how="left")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    g.plot(ax=ax, column="exposure_combined",
           cmap="YlOrRd", vmin=0, vmax=1,
           edgecolor="white", linewidth=0.2,
           missing_kwds={"facecolor": "#F2F2F2", "edgecolor": "white"},
           legend=True,
           legend_kwds={"label": "Exposure index (0–1; combined-index used "
                                 "for visualization)",
                         "orientation": "horizontal", "shrink": 0.6,
                         "pad": 0.04})
    ax.set_title(f"b. Fertilizer-shock exposure  ({YEAR_LABEL})",
                 fontsize=11, loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def panel_c(world, vul, out: Path):
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
    ax.set_title("c. Combined buffer × exposure classification",
                 fontsize=11, loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def panel_d(out: Path):
    summary = pd.read_csv(DIR_PROCESSED / "panel_d_summary.csv")
    class_order = ["low_buffer_high_exposure", "high_buffer_high_exposure",
                   "low_buffer_low_exposure", "high_buffer_low_exposure",
                   "intermediate", "data_missing"]
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
    ax.set_title(f"d. Cropland area × class × region  ({YEAR_LABEL})",
                 fontsize=11, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7,
              title="Region")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def assemble(out: Path):
    """4-panel figure assembled into one image."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.10, wspace=0.05)
    files = [
        ("a", DIR_FIGURES / "fig4_phase2_panel_a.png"),
        ("b", DIR_FIGURES / "fig4_phase2_panel_b.png"),
        ("c", DIR_FIGURES / "fig4_phase2_panel_c.png"),
        ("d", DIR_FIGURES / "fig4_phase2_panel_d.png"),
    ]
    for (lbl, p), gsi in zip(files, gs):
        ax = fig.add_subplot(gsi)
        if p.exists():
            ax.imshow(plt.imread(p))
        ax.set_axis_off()
    fig.suptitle(
        "Figure 4 — Global geography of soil organic N buffering and "
        f"fertilizer-shock exposure  ({YEAR_LABEL})",
        fontsize=13, y=0.995,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def main() -> int:
    print("Rendering Phase 2 four-panel figure ...")
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    world = load_world()

    panel_a(world, vul, DIR_FIGURES / "fig4_phase2_panel_a.png")
    panel_b(world, vul, DIR_FIGURES / "fig4_phase2_panel_b.png")
    panel_c(world, vul, DIR_FIGURES / "fig4_phase2_panel_c.png")
    panel_d(DIR_FIGURES / "fig4_phase2_panel_d.png")
    assemble(DIR_FIGURES / "fig4_phase2_assembled.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
