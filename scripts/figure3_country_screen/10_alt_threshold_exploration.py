"""Phase 0 alternative-threshold exploration.

The locked spec (combined exposure index, threshold 0.66) leaves the focal
class (low_buffer_high_exposure) effectively empty. This script tests three
alternatives and reports cropland-area composition under each.

Alt 1: Lower the combined-index threshold (high ≥ 0.50; low < 0.25)
Alt 2: Max-rule — high if N_intensity ≥ 150 kg/ha OR import_reliance ≥ 0.7;
                  low  if N_intensity <  50 kg/ha AND import_reliance < 0.3
Alt 3: Tercile-based exposure (within cropland-bearing countries)

Tightens the re-export filter for all alternatives:
  reexport_flag_v2 = ((Imports + Exports) / Apparent_consumption > 3)
                     AND (Imports + Exports) > 50,000 t N

Output:
  data_processed/threshold_alternatives.csv   — class composition table
  figures/fig4_phase0_alt_panels.png          — three small-multiples maps

No locked file is overwritten; this is an exploration only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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


def buffer_class(s: pd.Series, low_cut: float, high_cut: float) -> pd.Series:
    out = pd.Series("data_missing", index=s.index, dtype=object)
    out[s < low_cut] = "low"
    out[s >= high_cut] = "high"
    out[(s >= low_cut) & (s < high_cut)] = "moderate"
    out[s.isna()] = "data_missing"
    return out


def alt1_combined(df: pd.DataFrame) -> pd.Series:
    out = pd.Series("data_missing", index=df.index, dtype=object)
    out[df["exposure_combined"] < 0.25] = "low"
    out[df["exposure_combined"] >= 0.50] = "high"
    mid = (df["exposure_combined"] >= 0.25) & (df["exposure_combined"] < 0.50)
    out[mid] = "moderate"
    out[df["exposure_combined"].isna()] = "data_missing"
    return out


def alt2_maxrule(df: pd.DataFrame) -> pd.Series:
    intensity = df["n_intensity_raw"]
    reliance  = df["import_reliance"]
    high = (intensity >= 150) | (reliance >= 0.7)
    low  = (intensity < 50)   & (reliance < 0.3)
    out = pd.Series("moderate", index=df.index, dtype=object)
    out[high] = "high"
    out[low]  = "low"
    out[intensity.isna() & reliance.isna()] = "data_missing"
    return out


def alt3_tercile(df: pd.DataFrame, cropland_min_Mha: float = 0.1) -> pd.Series:
    eligible = (df["exposure_combined"].notna()
                & (df["cropland_Mha"] >= cropland_min_Mha))
    cuts = df.loc[eligible, "exposure_combined"].quantile([1/3, 2/3]).values
    out = pd.Series("data_missing", index=df.index, dtype=object)
    out[df["exposure_combined"] < cuts[0]] = "low"
    out[df["exposure_combined"] >= cuts[1]] = "high"
    out[(df["exposure_combined"] >= cuts[0])
        & (df["exposure_combined"] < cuts[1])] = "moderate"
    out[df["exposure_combined"].isna()] = "data_missing"
    print(f"    alt3 exposure terciles (combined index): "
          f"low<{cuts[0]:.3f}, high≥{cuts[1]:.3f}")
    return out


CLASS_PANEL_C = {
    ("high", "low"):       "high_buffer_low_exposure",
    ("high", "high"):      "high_buffer_high_exposure",
    ("low",  "low"):       "low_buffer_low_exposure",
    ("low",  "high"):      "low_buffer_high_exposure",
}


def panel_c_class(b: pd.Series, e: pd.Series) -> pd.Series:
    out = []
    for bv, ev in zip(b, e):
        if bv == "data_missing" or ev == "data_missing":
            out.append("data_missing")
        elif (bv, ev) in CLASS_PANEL_C:
            out.append(CLASS_PANEL_C[(bv, ev)])
        else:
            out.append("intermediate")
    return pd.Series(out, index=b.index)


def composition(df: pd.DataFrame, cls_col: str) -> pd.DataFrame:
    return (df.groupby(cls_col)["cropland_Mha"].sum()
              .rename("cropland_Mha").reset_index()
              .assign(pct=lambda x: 100 * x["cropland_Mha"] / x["cropland_Mha"].sum()))


def render_alt_panels(df: pd.DataFrame, world: gpd.GeoDataFrame,
                      alts: dict[str, pd.Series], out: Path) -> None:
    fig, axes = plt.subplots(len(alts), 1, figsize=(11, 4.5*len(alts)))
    if len(alts) == 1:
        axes = [axes]
    for ax, (name, cls) in zip(axes, alts.items()):
        d = df[["iso3"]].assign(panel_c_class=cls)
        g = world.merge(d, on="iso3", how="left")
        g["panel_c_class"] = g["panel_c_class"].fillna("data_missing")
        g["color"] = g["panel_c_class"].map(PANEL_C_COLORS)
        g.plot(ax=ax, color=g["color"], edgecolor="white", linewidth=0.2)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_axis_off()
    handles = [mpatches.Patch(color=PANEL_C_COLORS[k], label=PANEL_C_LABELS[k])
               for k in ["low_buffer_high_exposure",
                         "high_buffer_high_exposure",
                         "low_buffer_low_exposure",
                         "high_buffer_low_exposure",
                         "intermediate", "data_missing"]]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(out.parent.parent)}")


def main() -> int:
    print("Phase 0 alternative-threshold exploration ...")
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")

    # Reload buffer + exposure inputs
    df = vul.copy()
    df["cropland_Mha"] = df["cropland_Mha"].fillna(0)

    # Buffer cuts (same as locked spec)
    eligible_b = (df["buffer_proxy_t_ha"].notna()
                  & (df["cropland_Mha"] >= 0.1))   # ≥ 100,000 ha
    bcuts = df.loc[eligible_b, "buffer_proxy_t_ha"].quantile([1/3, 2/3]).values
    print(f"  buffer terciles: low<{bcuts[0]:.2f}, high≥{bcuts[1]:.2f} t N/ha")
    df["buffer_class"] = buffer_class(df["buffer_proxy_t_ha"], bcuts[0], bcuts[1])

    # ----- Three alt exposure classifiers -----
    df["exp_alt1"] = alt1_combined(df)
    df["exp_alt2"] = alt2_maxrule(df)
    df["exp_alt3"] = alt3_tercile(df)

    df["pc_locked"] = vul["panel_c_class"]
    df["pc_alt1"] = panel_c_class(df["buffer_class"], df["exp_alt1"])
    df["pc_alt2"] = panel_c_class(df["buffer_class"], df["exp_alt2"])
    df["pc_alt3"] = panel_c_class(df["buffer_class"], df["exp_alt3"])

    rows = []
    for label, col in [
        ("locked (combined ≥0.66)", "pc_locked"),
        ("alt1 (combined ≥0.50)",   "pc_alt1"),
        ("alt2 (max-rule)",         "pc_alt2"),
        ("alt3 (exposure terciles)","pc_alt3"),
    ]:
        comp = (df.groupby(col)["cropland_Mha"].sum()
                  .rename("cropland_Mha").reset_index()
                  .rename(columns={col: "panel_c_class"})
                  .assign(rule=label,
                          pct=lambda x: 100 * x["cropland_Mha"]
                                          / x["cropland_Mha"].sum()))
        rows.append(comp)
    summary = (pd.concat(rows, ignore_index=True)
                 .pivot_table(index="rule", columns="panel_c_class",
                              values="pct", aggfunc="sum")
                 .fillna(0).round(1))
    print("\nClass composition (% of cropland) under each rule:")
    print(summary.to_string())
    summary.to_csv(DIR_PROCESSED / "threshold_alternatives.csv")

    # Top countries in focal class under alt2 (the most likely winner)
    print("\nTop 15 in low_buffer_high_exposure under alt2 (max-rule):")
    foc = (df[df["pc_alt2"] == "low_buffer_high_exposure"]
              .sort_values("cropland_Mha", ascending=False)
              .head(15))
    if len(foc):
        print(foc[["iso3", "fao_name", "region", "cropland_Mha",
                   "buffer_proxy_t_ha", "n_intensity_raw",
                   "import_reliance", "exposure_combined"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("  (still empty)")

    print("\nTop 15 in low_buffer_high_exposure under alt3 (terciles):")
    foc = (df[df["pc_alt3"] == "low_buffer_high_exposure"]
              .sort_values("cropland_Mha", ascending=False)
              .head(15))
    if len(foc):
        print(foc[["iso3", "fao_name", "region", "cropland_Mha",
                   "buffer_proxy_t_ha", "exposure_combined"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Render side-by-side maps
    world = gpd.read_file(
        DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
    )[["ADM0_A3", "geometry"]].rename(columns={"ADM0_A3": "iso3"}).to_crs(EQUAL_EARTH)

    df_iso = df[["iso3", "pc_locked", "pc_alt1", "pc_alt2", "pc_alt3"]].drop_duplicates("iso3")
    render_alt_panels(
        df_iso, world,
        {
            "Locked spec — combined index ≥ 0.66 (focal class effectively empty)":
                df.set_index("iso3").reindex(df_iso["iso3"])["pc_locked"].values,
            "Alt 1 — combined index ≥ 0.50":
                df.set_index("iso3").reindex(df_iso["iso3"])["pc_alt1"].values,
            "Alt 2 — max-rule (intensity ≥ 150 kg/ha OR reliance ≥ 0.7)":
                df.set_index("iso3").reindex(df_iso["iso3"])["pc_alt2"].values,
            "Alt 3 — exposure terciles":
                df.set_index("iso3").reindex(df_iso["iso3"])["pc_alt3"].values,
        },
        DIR_FIGURES / "fig4_phase0_alt_panels.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
