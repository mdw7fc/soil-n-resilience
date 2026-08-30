"""Phase 0 exposure-side diagnostics — sanity-checks before any soil work.

Outputs:
  data_processed/
    exposure_top_by_class.csv       # top 15 countries by cropland in each exposure class
    exposure_by_region.csv          # cropland area & exposure-class composition by region
    exposure_coverage.csv           # data-missing diagnostics by region
    exposure_sensitivity.csv        # sensitivity to threshold shifts ±20% and ±30%
  figures/
    fig4_phase0_panel_b.png         # exposure ranked horizontal bar chart, colored by class
    fig4_phase0_panel_d_exposure.png# stacked bar: cropland area × exposure_class × region
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import (  # noqa: E402
    DIR_FIGURES,
    DIR_PROCESSED,
    EXPOSURE_HIGH,
    EXPOSURE_LOW,
    REGION_GROUPS,
    YEAR_LABEL,
)


CLASS_COLORS = {
    "low":          "#9ECAE1",   # light blue
    "moderate":     "#BDBDBD",   # neutral grey
    "high":         "#D7301F",   # warm red
    "data_missing": "#EEEEEE",   # near-white
}
REGION_ORDER = list(REGION_GROUPS.keys())


def main() -> int:
    df = pd.read_csv(DIR_PROCESSED / "exposure_country.csv")
    df["cropland_Mha"] = df["cropland_ha"] / 1e6

    # ---------- Top countries by class ----------
    top_rows = []
    for cls in ["low", "moderate", "high", "data_missing"]:
        sub = (df[df["exposure_class"] == cls]
               .sort_values("cropland_Mha", ascending=False)
               .head(15))
        sub = sub.assign(rank_in_class=range(1, len(sub) + 1))
        top_rows.append(sub)
    top = pd.concat(top_rows, ignore_index=True)
    cols = ["exposure_class", "rank_in_class", "iso3", "fao_name", "region",
            "cropland_Mha", "n_intensity_raw", "import_reliance",
            "exposure_combined", "reexport_flag"]
    top[cols].to_csv(DIR_PROCESSED / "exposure_top_by_class.csv", index=False)

    # ---------- By region ----------
    df["region_grp"] = df["region"].where(df["region"].isin(REGION_ORDER), other="Unassigned")
    by_region = (df.groupby(["region_grp", "exposure_class"])
                   .agg(cropland_Mha=("cropland_Mha", "sum"),
                        n_countries=("iso3", "count"))
                   .reset_index())
    by_region.to_csv(DIR_PROCESSED / "exposure_by_region.csv", index=False)

    # ---------- Coverage ----------
    cov = (df.groupby("region_grp")
             .agg(n_countries=("iso3", "count"),
                  n_with_n_use=("data_missing_n_use", lambda s: int((~s).sum())),
                  n_with_cropland=("data_missing_cropland", lambda s: int((~s).sum())),
                  n_with_exposure=("exposure_combined",
                                   lambda s: int(s.notna().sum())),
                  cropland_Mha=("cropland_Mha", "sum"),
                  cropland_Mha_with_exposure=(
                      "cropland_Mha",
                      lambda s: float(df.loc[s.index]
                                      .loc[df["exposure_combined"].notna(),
                                           "cropland_Mha"].sum()))
                  )
             .reset_index())
    cov["pct_cropland_covered"] = (
        100 * cov["cropland_Mha_with_exposure"] / cov["cropland_Mha"].replace(0, np.nan)
    )
    cov.to_csv(DIR_PROCESSED / "exposure_coverage.csv", index=False)

    # ---------- Threshold sensitivity ----------
    rows = []
    for shift_pct in [-30, -20, -10, 0, 10, 20, 30]:
        lo = EXPOSURE_LOW * (1 + shift_pct / 100)
        hi = EXPOSURE_HIGH * (1 + shift_pct / 100)
        cls = pd.cut(df["exposure_combined"],
                     bins=[-np.inf, lo, hi, np.inf],
                     labels=["low", "moderate", "high"]).astype(str)
        cls = cls.where(df["exposure_combined"].notna(), other="data_missing")
        for c in ["low", "moderate", "high", "data_missing"]:
            mask = cls == c
            rows.append({
                "shift_pct": shift_pct,
                "low_threshold": round(lo, 4),
                "high_threshold": round(hi, 4),
                "exposure_class": c,
                "n_countries": int(mask.sum()),
                "cropland_Mha": float(df.loc[mask, "cropland_Mha"].sum()),
            })
    sens = pd.DataFrame(rows)

    # Reclassification rate vs baseline
    baseline = df["exposure_class"].copy()
    reclass_rates = []
    for shift_pct in [-30, -20, -10, 10, 20, 30]:
        lo = EXPOSURE_LOW * (1 + shift_pct / 100)
        hi = EXPOSURE_HIGH * (1 + shift_pct / 100)
        cls = pd.cut(df["exposure_combined"],
                     bins=[-np.inf, lo, hi, np.inf],
                     labels=["low", "moderate", "high"]).astype(str)
        cls = cls.where(df["exposure_combined"].notna(), other="data_missing")
        changed = (cls != baseline)
        reclass_rates.append({
            "shift_pct": shift_pct,
            "n_changed": int(changed.sum()),
            "frac_changed": float(changed.mean()),
            "cropland_Mha_changed": float(df.loc[changed, "cropland_Mha"].sum()),
            "cropland_Mha_total":   float(df["cropland_Mha"].sum()),
            "cropland_frac_changed": float(df.loc[changed, "cropland_Mha"].sum()
                                          / df["cropland_Mha"].sum()),
        })
    sens.to_csv(DIR_PROCESSED / "exposure_sensitivity.csv", index=False)
    rec_df = pd.DataFrame(reclass_rates)
    rec_df.to_csv(DIR_PROCESSED / "exposure_sensitivity_reclass.csv", index=False)

    # ---------- Figures ----------
    plot_panel_b(df)
    plot_panel_d_exposure_only(by_region)

    # ---------- Console summary ----------
    print("=" * 68)
    print("PHASE 0 EXPOSURE-SIDE DIAGNOSTICS")
    print(f"Year window: {YEAR_LABEL}")
    print("=" * 68)

    print(f"\nCountries: {len(df)}; with full exposure index: "
          f"{int(df['exposure_combined'].notna().sum())}")
    print(f"Total cropland represented: {df['cropland_Mha'].sum():.0f} Mha")

    print("\nExposure class composition (cropland-area-weighted):")
    cw = (df.groupby("exposure_class")["cropland_Mha"].sum()
              .reindex(["low", "moderate", "high", "data_missing"]))
    total = cw.sum()
    for k, v in cw.items():
        print(f"  {k:14s}  {v:8.1f} Mha   {100*v/total:5.1f} %")

    print("\nCoverage by region (% of cropland with exposure index):")
    print(cov[["region_grp", "n_countries", "n_with_exposure",
               "cropland_Mha", "pct_cropland_covered"]]
          .sort_values("cropland_Mha", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\nKill criterion #2 (data-missing > 20% of regional cropland):")
    flagged = cov[cov["pct_cropland_covered"] < 80][["region_grp", "pct_cropland_covered"]]
    if len(flagged) == 0:
        print("  PASS — all regions ≥ 80% cropland covered.")
    else:
        print("  FAIL — flagged regions:")
        print(flagged.to_string(index=False))

    print("\nKill criterion #4 (±20% threshold reclassifies > 25% of cropland):")
    crit = rec_df[rec_df["shift_pct"].isin([-20, 20])][
        ["shift_pct", "cropland_frac_changed"]]
    print(crit.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    worst = crit["cropland_frac_changed"].max()
    print(f"  worst-case ±20% reclassification: {100*worst:.1f}% of cropland")
    print("  PASS" if worst <= 0.25 else "  FAIL")

    print("\nRe-export hubs flagged:")
    print(df[df["reexport_flag"]][
        ["iso3", "fao_name", "n_intensity_raw", "import_reliance",
         "exposure_combined", "exposure_class", "cropland_Mha"]]
        .sort_values("cropland_Mha", ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return 0


def plot_panel_b(df: pd.DataFrame) -> None:
    """Horizontal bar chart of exposure_combined for top-50 cropland countries."""
    sub = (df.dropna(subset=["exposure_combined"])
              .sort_values("cropland_Mha", ascending=False)
              .head(50)
              .copy())
    sub = sub.iloc[::-1]  # so largest is at top

    fig, ax = plt.subplots(figsize=(8, 12))
    colors = [CLASS_COLORS[c] for c in sub["exposure_class"]]
    ax.barh(sub["iso3"] + "  " + sub["fao_name"].str[:25],
            sub["exposure_combined"],
            color=colors, edgecolor="white")
    ax.axvline(EXPOSURE_LOW, ls="--", lw=0.6, color="grey")
    ax.axvline(EXPOSURE_HIGH, ls="--", lw=0.6, color="grey")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Combined fertilizer-shock exposure index (0–1)")
    ax.set_title(f"Phase 0 — Panel b draft\n"
                 f"Top 50 countries by cropland area · {YEAR_LABEL}",
                 fontsize=10)
    for cls, color in CLASS_COLORS.items():
        ax.barh([], [], color=color, label=cls)
    ax.legend(title="Exposure class", loc="lower right", fontsize=8)
    plt.tight_layout()
    out = DIR_FIGURES / "fig4_phase0_panel_b.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.relative_to(DIR_FIGURES.parent)}")


def plot_panel_d_exposure_only(by_region: pd.DataFrame) -> None:
    """Stacked bar — cropland area by exposure class, stacked by region.

    This is panel d in 'exposure-only' form (no buffer dimension yet); the
    Phase 1 version will use the full 5-class panel c classification.
    """
    pivot = (by_region.pivot(index="exposure_class",
                             columns="region_grp",
                             values="cropland_Mha")
             .fillna(0)
             .reindex(["low", "moderate", "high", "data_missing"]))
    pivot = pivot[[r for r in REGION_ORDER if r in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(pivot))
    cmap = plt.get_cmap("tab10")
    for i, region in enumerate(pivot.columns):
        vals = pivot[region].values
        ax.bar(pivot.index, vals, bottom=bottoms,
               label=region, color=cmap(i % 10), edgecolor="white")
        bottoms = bottoms + vals
    ax.set_ylabel("Cropland area (Mha)")
    ax.set_title(f"Phase 0 — Panel d draft (exposure-only)\n"
                 f"Cropland area by exposure class · {YEAR_LABEL}", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
              title="Region")
    plt.tight_layout()
    out = DIR_FIGURES / "fig4_phase0_panel_d_exposure.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.relative_to(DIR_FIGURES.parent)}")


if __name__ == "__main__":
    sys.exit(main())
