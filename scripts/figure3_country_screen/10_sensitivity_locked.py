"""Phase 1 sensitivity sweep — write the SI sensitivity table.

Compares cropland-area composition under:
  - Locked Phase-1 modified max-rule with stake floor 25 kg N/ha
  - Stake-floor variants: 10, 25, 50 kg N/ha (modified max-rule)
  - Original combined-index ≥ 0.66
  - Combined-index ≥ 0.50
  - Raw max-rule (no stake floor)
  - Exposure terciles (cropland-bearing countries)

Output:
  data_processed/threshold_sensitivity_summary.csv
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


def buffer_class(buf: pd.Series, low_cut: float, high_cut: float,
                 eligible: pd.Series) -> pd.Series:
    out = pd.Series("data_missing", index=buf.index, dtype=object)
    out[buf < low_cut] = "low"
    out[buf >= high_cut] = "high"
    out[(buf >= low_cut) & (buf < high_cut)] = "moderate"
    out[~eligible | buf.isna()] = "data_missing"
    return out


def panel_c(buf_cls: pd.Series, exp_cls: pd.Series) -> pd.Series:
    out = []
    for b, e in zip(buf_cls, exp_cls):
        if b == "data_missing" or e == "data_missing":
            out.append("data_missing")
        elif (b, e) in CLASS_PANEL_C:
            out.append(CLASS_PANEL_C[(b, e)])
        else:
            out.append("intermediate")
    return pd.Series(out, index=buf_cls.index)


def exp_modified_maxrule(df: pd.DataFrame, stake: float) -> pd.Series:
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


def exp_raw_maxrule(df: pd.DataFrame) -> pd.Series:
    """Raw max-rule, no material-stake floor."""
    intensity = df["n_intensity_raw"]
    reliance  = df["import_reliance"]
    high = (intensity >= 150) | (reliance >= 0.70)
    low  = (intensity < 50)   & (reliance < 0.30)
    out = pd.Series("moderate", index=df.index, dtype=object)
    out[high] = "high"
    out[low & ~high] = "low"
    out[intensity.isna() | reliance.isna()] = "data_missing"
    return out


def exp_combined(df: pd.DataFrame, hi: float, lo: float) -> pd.Series:
    e = df["exposure_combined"]
    out = pd.Series("data_missing", index=df.index, dtype=object)
    out[e < lo] = "low"
    out[e >= hi] = "high"
    out[(e >= lo) & (e < hi)] = "moderate"
    out[e.isna()] = "data_missing"
    return out


def exp_terciles(df: pd.DataFrame) -> pd.Series:
    eligible = df["exposure_combined"].notna() & (df["cropland_Mha"] >= 0.1)
    cuts = df.loc[eligible, "exposure_combined"].quantile([1/3, 2/3]).values
    e = df["exposure_combined"]
    out = pd.Series("data_missing", index=df.index, dtype=object)
    out[e < cuts[0]] = "low"
    out[e >= cuts[1]] = "high"
    out[(e >= cuts[0]) & (e < cuts[1])] = "moderate"
    out[e.isna()] = "data_missing"
    return out


def composition(df: pd.DataFrame, cls_col: pd.Series, label: str) -> pd.DataFrame:
    cw = df.assign(_cls=cls_col).groupby("_cls")["cropland_Mha"].sum()
    return (cw.rename("cropland_Mha")
              .reset_index()
              .rename(columns={"_cls": "panel_c_class"})
              .assign(rule=label,
                      pct=lambda x: 100 * x["cropland_Mha"]
                                    / x["cropland_Mha"].sum()))


def main() -> int:
    print("Phase 1 sensitivity sweep ...")
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    df = vul.copy()

    # Recompute buffer terciles (same as locked) and use that as the buffer
    # axis for every rule — only exposure varies across rules.
    eligible_b = (df["buffer_proxy_t_ha"].notna()
                  & (df["cropland_Mha"] >= 0.1))
    bcuts = df.loc[eligible_b, "buffer_proxy_t_ha"].quantile([1/3, 2/3]).values
    bcls = buffer_class(df["buffer_proxy_t_ha"], bcuts[0], bcuts[1], eligible_b)

    rules: dict[str, pd.Series] = {}
    # Stake-floor variants
    for stake in STAKE_FLOOR_SENSITIVITY:
        label = f"modified max-rule (stake={stake:.0f})"
        if stake == 25:
            label += " *LOCKED*"
        rules[label] = panel_c(bcls, exp_modified_maxrule(df, stake))
    # Other rules
    rules["raw max-rule (no stake floor)"]   = panel_c(bcls, exp_raw_maxrule(df))
    rules["combined index ≥ 0.66 (original)"] = panel_c(bcls, exp_combined(df, 0.66, 0.33))
    rules["combined index ≥ 0.50"]            = panel_c(bcls, exp_combined(df, 0.50, 0.25))
    rules["exposure terciles"]                = panel_c(bcls, exp_terciles(df))

    rows = [composition(df, cls, label) for label, cls in rules.items()]
    summary = pd.concat(rows, ignore_index=True)
    pivot = (summary.pivot_table(index="rule", columns="panel_c_class",
                                  values="pct", aggfunc="sum")
                    .fillna(0).round(1))
    cols_order = ["low_buffer_high_exposure", "high_buffer_high_exposure",
                  "low_buffer_low_exposure", "high_buffer_low_exposure",
                  "intermediate", "data_missing"]
    pivot = pivot[[c for c in cols_order if c in pivot.columns]]
    pivot.to_csv(DIR_PROCESSED / "threshold_sensitivity_summary.csv")
    print("\nClass composition (% of cropland) — Phase 1 sensitivity:")
    print(pivot.to_string())

    # Stake-floor stability check: how does focal-class cropland area
    # change between stake floors of 10 / 25 / 50?
    print("\nFocal-class (low_buffer_high_exposure) cropland area by stake floor:")
    for stake in STAKE_FLOOR_SENSITIVITY:
        cls = panel_c(bcls, exp_modified_maxrule(df, stake))
        focal_Mha = df.loc[cls == "low_buffer_high_exposure", "cropland_Mha"].sum()
        focal_pct = 100 * focal_Mha / df["cropland_Mha"].sum()
        print(f"  stake={stake:>3.0f} kg/ha   {focal_Mha:7.1f} Mha   "
              f"{focal_pct:5.1f}%")

    # Stability: % of cropland that changes class as stake floor moves 25 → 10 or 25 → 50
    base = panel_c(bcls, exp_modified_maxrule(df, 25))
    for stake in STAKE_FLOOR_SENSITIVITY:
        if stake == 25:
            continue
        alt = panel_c(bcls, exp_modified_maxrule(df, stake))
        changed = base != alt
        Mha_changed = df.loc[changed, "cropland_Mha"].sum()
        pct_changed = 100 * Mha_changed / df["cropland_Mha"].sum()
        print(f"  reclassification 25→{stake:.0f}: "
              f"{Mha_changed:6.1f} Mha changed  ({pct_changed:.1f}% of cropland)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
