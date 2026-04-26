"""Phase 2 sensitivity sweep + reclassification analysis.

Outputs:
  data_processed/buffer_aggregation_sensitivity.csv
      cwm vs all-land median: per-country buffer + class change
  data_processed/buffer_aggregation_summary.csv
      cropland area summary by aggregation method × class
  data_processed/threshold_sensitivity_summary.csv
      composition under each rule (with CWM buffer)
  data_processed/reclassification_phase1_to_phase2.csv
      which countries moved between Phase 1 (median) and Phase 2 (CWM) classes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import (  # noqa: E402
    DIR_PROCESSED,
    HIGH_INTENSITY_KG_HA,
    HIGH_RELIANCE,
    LOW_INTENSITY_KG_HA,
    LOW_RELIANCE,
    STAKE_FLOOR_SENSITIVITY,
)


CLASS_PANEL_C = {
    ("high", "low"):  "high_buffer_low_exposure",
    ("high", "high"): "high_buffer_high_exposure",
    ("low",  "low"):  "low_buffer_low_exposure",
    ("low",  "high"): "low_buffer_high_exposure",
}


def buffer_class(buf, low_cut, high_cut, eligible):
    out = pd.Series("data_missing", index=buf.index, dtype=object)
    out[buf < low_cut] = "low"
    out[buf >= high_cut] = "high"
    out[(buf >= low_cut) & (buf < high_cut)] = "moderate"
    out[~eligible | buf.isna()] = "data_missing"
    return out


def panel_c(b, e):
    out = []
    for bv, ev in zip(b, e):
        if bv == "data_missing" or ev == "data_missing":
            out.append("data_missing")
        elif (bv, ev) in CLASS_PANEL_C:
            out.append(CLASS_PANEL_C[(bv, ev)])
        else:
            out.append("intermediate")
    return pd.Series(out, index=b.index)


def exp_modified_maxrule(df, stake):
    intensity = df["n_intensity_raw"]
    reliance  = df["import_reliance"]
    high = ((intensity >= HIGH_INTENSITY_KG_HA) |
            ((reliance >= HIGH_RELIANCE) & (intensity >= stake)))
    low  = (intensity < LOW_INTENSITY_KG_HA) & (reliance < LOW_RELIANCE)
    out = pd.Series("moderate", index=df.index, dtype=object)
    out[high] = "high"
    out[low & ~high] = "low"
    out[intensity.isna() | reliance.isna()] = "data_missing"
    return out


def exp_raw_maxrule(df):
    intensity = df["n_intensity_raw"]; reliance = df["import_reliance"]
    high = (intensity >= 150) | (reliance >= 0.70)
    low  = (intensity < 50)   & (reliance < 0.30)
    out = pd.Series("moderate", index=df.index, dtype=object)
    out[high] = "high"; out[low & ~high] = "low"
    out[intensity.isna() | reliance.isna()] = "data_missing"
    return out


def exp_combined(df, hi, lo):
    e = df["exposure_combined"]
    out = pd.Series("data_missing", index=df.index, dtype=object)
    out[e < lo] = "low"; out[e >= hi] = "high"
    out[(e >= lo) & (e < hi)] = "moderate"; out[e.isna()] = "data_missing"
    return out


def exp_terciles(df):
    eligible = df["exposure_combined"].notna() & (df["cropland_Mha"] >= 0.1)
    cuts = df.loc[eligible, "exposure_combined"].quantile([1/3, 2/3]).values
    e = df["exposure_combined"]
    out = pd.Series("data_missing", index=df.index, dtype=object)
    out[e < cuts[0]] = "low"; out[e >= cuts[1]] = "high"
    out[(e >= cuts[0]) & (e < cuts[1])] = "moderate"
    out[e.isna()] = "data_missing"
    return out


def make_buffer_classes(buf_col: pd.Series, df: pd.DataFrame) -> tuple[pd.Series, tuple]:
    eligible = buf_col.notna() & (df["cropland_Mha"] >= 0.1)
    cuts = buf_col[eligible].quantile([1/3, 2/3]).values
    return buffer_class(buf_col, cuts[0], cuts[1], eligible), tuple(cuts)


def composition(df, cls):
    return (df.assign(_cls=cls).groupby("_cls")["cropland_Mha"].sum()
              .reset_index(name="cropland_Mha")
              .rename(columns={"_cls": "panel_c_class"})
              .assign(pct=lambda x: 100 * x["cropland_Mha"]
                                    / x["cropland_Mha"].sum()))


def main() -> int:
    print("=" * 72)
    print("Phase 2 sensitivity + reclassification analysis")
    print("=" * 72)

    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    df = vul.copy()

    has_median = "n_stock_median_allland_t_ha" in df.columns

    # -------- Buffer-aggregation sensitivity (CWM vs median) --------
    if has_median:
        bcls_cwm, cuts_cwm = make_buffer_classes(df["buffer_proxy_t_ha"], df)
        bcls_med, cuts_med = make_buffer_classes(
            df["n_stock_median_allland_t_ha"], df
        )
        ecls = exp_modified_maxrule(df, stake=25)
        cls_cwm = panel_c(bcls_cwm, ecls)
        cls_med = panel_c(bcls_med, ecls)

        comp_rows = []
        for label, cls in [("CWM (Phase 2 LOCKED)", cls_cwm),
                           ("median all-land (Phase 1)", cls_med)]:
            cw = composition(df, cls).assign(rule=label)
            comp_rows.append(cw)
        comp = (pd.concat(comp_rows, ignore_index=True)
                  .pivot_table(index="rule", columns="panel_c_class",
                               values="pct", aggfunc="sum")
                  .fillna(0).round(1))
        cols_order = ["low_buffer_high_exposure", "high_buffer_high_exposure",
                      "low_buffer_low_exposure", "high_buffer_low_exposure",
                      "intermediate", "data_missing"]
        comp = comp[[c for c in cols_order if c in comp.columns]]
        comp.to_csv(DIR_PROCESSED / "buffer_aggregation_summary.csv")
        print("\nBuffer aggregation: composition (% of cropland)")
        print(comp.to_string())
        print(f"\nBuffer terciles (CWM):    low<{cuts_cwm[0]:.2f}, "
              f"high≥{cuts_cwm[1]:.2f} t N/ha")
        print(f"Buffer terciles (median): low<{cuts_med[0]:.2f}, "
              f"high≥{cuts_med[1]:.2f} t N/ha")

        # Per-country reclassification
        reclass = (df[["iso3", "fao_name", "region", "cropland_Mha",
                       "buffer_proxy_t_ha", "n_stock_median_allland_t_ha",
                       "n_intensity_raw", "import_reliance"]]
                   .assign(class_phase1=cls_med.values,
                           class_phase2=cls_cwm.values,
                           buffer_class_phase1=bcls_med.values,
                           buffer_class_phase2=bcls_cwm.values,
                           changed=cls_cwm.values != cls_med.values))
        reclass.to_csv(DIR_PROCESSED / "reclassification_phase1_to_phase2.csv",
                       index=False)
        moved = reclass[reclass["changed"]]
        moved_Mha = moved["cropland_Mha"].sum()
        print(f"\nReclassification (Phase 1 median → Phase 2 CWM):")
        print(f"  countries that changed class: {len(moved)}")
        print(f"  cropland reclassified: {moved_Mha:.1f} Mha "
              f"({100*moved_Mha/df['cropland_Mha'].sum():.1f}%)")
        print(f"\n  Top 15 movers by cropland area:")
        print(moved.sort_values("cropland_Mha", ascending=False)
              .head(15)[["iso3", "fao_name", "cropland_Mha",
                          "buffer_proxy_t_ha", "n_stock_median_allland_t_ha",
                          "buffer_class_phase1", "buffer_class_phase2",
                          "class_phase1", "class_phase2"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # -------- Full threshold sensitivity (now with CWM buffer) --------
    bcls = bcls_cwm  # CWM is Phase 2 locked
    rules = {}
    for stake in STAKE_FLOOR_SENSITIVITY:
        label = f"modified max-rule (stake={stake:.0f})"
        if stake == 25: label += " *LOCKED*"
        rules[label] = panel_c(bcls, exp_modified_maxrule(df, stake))
    rules["raw max-rule (no stake floor)"]    = panel_c(bcls, exp_raw_maxrule(df))
    rules["combined index ≥ 0.66 (original)"] = panel_c(bcls, exp_combined(df, 0.66, 0.33))
    rules["combined index ≥ 0.50"]             = panel_c(bcls, exp_combined(df, 0.50, 0.25))
    rules["exposure terciles"]                 = panel_c(bcls, exp_terciles(df))

    rows = [composition(df, cls).assign(rule=label) for label, cls in rules.items()]
    pivot = (pd.concat(rows, ignore_index=True)
               .pivot_table(index="rule", columns="panel_c_class",
                            values="pct", aggfunc="sum")
               .fillna(0).round(1))
    pivot = pivot[[c for c in cols_order if c in pivot.columns]]
    pivot.to_csv(DIR_PROCESSED / "threshold_sensitivity_summary.csv")
    print("\nThreshold sensitivity (Phase 2 CWM buffer; % of cropland):")
    print(pivot.to_string())

    print("\nFocal-class cropland share by stake floor:")
    base = panel_c(bcls, exp_modified_maxrule(df, 25))
    base_focal = df.loc[base == "low_buffer_high_exposure", "cropland_Mha"].sum()
    print(f"  stake=25 (LOCKED): {base_focal:.1f} Mha "
          f"({100*base_focal/df['cropland_Mha'].sum():.1f}%)")
    for stake in STAKE_FLOOR_SENSITIVITY:
        if stake == 25: continue
        cls = panel_c(bcls, exp_modified_maxrule(df, stake))
        f = df.loc[cls == "low_buffer_high_exposure", "cropland_Mha"].sum()
        ch = (cls != base)
        ch_Mha = df.loc[ch, "cropland_Mha"].sum()
        print(f"  stake={stake:>3.0f}             : {f:6.1f} Mha "
              f"({100*f/df['cropland_Mha'].sum():5.1f}%)  | "
              f"reclassified vs LOCKED: {ch_Mha:5.1f} Mha "
              f"({100*ch_Mha/df['cropland_Mha'].sum():.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
