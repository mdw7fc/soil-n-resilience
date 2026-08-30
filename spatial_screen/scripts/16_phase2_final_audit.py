"""Phase 2 final-audit suite — runs all four audits requested before
manuscript integration:

  1. Pearson + Spearman correlations between buffer (CWM) and exposure
     (combined index), weighted and unweighted by FAOSTAT cropland area.
  2. Buffer-threshold sensitivity:
       buffer metric: SOC/C:N (= N-stock) and SOC-only
       low-buffer cutoffs: bottom 30%, 33%, 40% (unweighted across countries)
       low-buffer cutoffs: bottom 30%, 33%, 40% (cropland-area-weighted)
       For each: focal-class cropland area, n focal countries, China's class,
       top 10 focal countries.
  3. Country QA: for China, Brazil, Australia, Thailand, South Africa,
     Bangladesh, Egypt, Uzbekistan, Paraguay, Zambia — buffer value, buffer
     percentile, N intensity, import reliance, exposure pathway, class under
     Phase 1, Phase 2, and each buffer-threshold variant.
  4. Area-accounting audit: confirm what panel d uses, and emit a one-paragraph
     methods note distinguishing MIRCA (buffer weight) from FAOSTAT (cropland
     area / N intensity / panel-d totals).

Outputs:
  data_processed/final_audit_summary.csv
  data_processed/buffer_threshold_sensitivity.csv
  data_processed/top_country_QA.csv
  data_processed/area_accounting_note.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _config import (  # noqa: E402
    DIR_PROCESSED,
    HIGH_INTENSITY_KG_HA,
    HIGH_RELIANCE,
    LOW_INTENSITY_KG_HA,
    LOW_RELIANCE,
    MATERIAL_STAKE_KG_HA,
)


CLASS_PANEL_C = {
    ("high", "low"):  "high_buffer_low_exposure",
    ("high", "high"): "high_buffer_high_exposure",
    ("low",  "low"):  "low_buffer_low_exposure",
    ("low",  "high"): "low_buffer_high_exposure",
}

QA_COUNTRIES = ["CHN", "BRA", "AUS", "THA", "ZAF", "BGD", "EGY",
                "UZB", "PRY", "ZMB"]


# ---------------------------------------------------------------------------
# Audit 1 — correlations
# ---------------------------------------------------------------------------

def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Pearson correlation with positive weights w."""
    w = w / w.sum()
    mx = (w * x).sum()
    my = (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    vx  = (w * (x - mx) ** 2).sum()
    vy  = (w * (y - my) ** 2).sum()
    return float(cov / np.sqrt(vx * vy))


def weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Spearman correlation: weighted Pearson on ranks."""
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return weighted_pearson(rx, ry, w)


def audit_correlations(vul: pd.DataFrame) -> pd.DataFrame:
    df = vul.dropna(subset=["buffer_proxy_t_ha", "exposure_combined",
                            "cropland_Mha"]).copy()
    df = df[df["cropland_Mha"] > 0]
    x = df["buffer_proxy_t_ha"].values
    y = df["exposure_combined"].values
    w = df["cropland_Mha"].values

    p_un,  pp_un = pearsonr(x, y)
    s_un,  ps_un = spearmanr(x, y)
    p_w   = weighted_pearson(x, y, w)
    s_w   = weighted_spearman(x, y, w)

    rows = [
        ("Pearson",  "unweighted",                p_un, pp_un),
        ("Pearson",  "cropland-area-weighted",    p_w,  np.nan),
        ("Spearman", "unweighted",                s_un, ps_un),
        ("Spearman", "cropland-area-weighted",    s_w,  np.nan),
    ]
    out = pd.DataFrame(rows, columns=["statistic", "weighting", "value",
                                      "p_value"])
    out["n_countries"] = len(df)
    out["cropland_Mha_total"] = w.sum()
    return out


# ---------------------------------------------------------------------------
# Audit 2 — buffer-threshold sensitivity
# ---------------------------------------------------------------------------

def cutoff_unweighted(values: np.ndarray, q: float) -> float:
    return float(np.nanquantile(values, q))


def cutoff_weighted(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted q-quantile."""
    ok = ~np.isnan(values) & (weights > 0)
    v = values[ok]
    w = weights[ok]
    order = np.argsort(v)
    v = v[order]; w = w[order]
    cw = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cw, q, side="left")
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def exposure_class_locked(df: pd.DataFrame) -> pd.Series:
    intensity = df["n_intensity_raw"]
    reliance  = df["import_reliance"]
    high = ((intensity >= HIGH_INTENSITY_KG_HA) |
            ((reliance >= HIGH_RELIANCE) & (intensity >= MATERIAL_STAKE_KG_HA)))
    low  = (intensity < LOW_INTENSITY_KG_HA) & (reliance < LOW_RELIANCE)
    out = pd.Series("moderate", index=df.index, dtype=object)
    out[high] = "high"
    out[low & ~high] = "low"
    out[intensity.isna() | reliance.isna()] = "data_missing"
    return out


def buffer_class_for(values: pd.Series, low_cut: float, high_cut: float,
                      eligible: pd.Series) -> pd.Series:
    out = pd.Series("data_missing", index=values.index, dtype=object)
    out[values < low_cut] = "low"
    out[values >= high_cut] = "high"
    out[(values >= low_cut) & (values < high_cut)] = "moderate"
    out[~eligible | values.isna()] = "data_missing"
    return out


def panel_c(b: pd.Series, e: pd.Series) -> pd.Series:
    out = []
    for bv, ev in zip(b, e):
        if bv == "data_missing" or ev == "data_missing":
            out.append("data_missing")
        elif (bv, ev) in CLASS_PANEL_C:
            out.append(CLASS_PANEL_C[(bv, ev)])
        else:
            out.append("intermediate")
    return pd.Series(out, index=b.index)


def audit_buffer_thresholds(vul: pd.DataFrame, buffer_field: pd.DataFrame
                              ) -> tuple[pd.DataFrame, dict]:
    eligible = (vul["cropland_Mha"] >= 0.1) & buffer_field["values"].notna()
    expcls = exposure_class_locked(vul)

    rows = []
    cls_columns: dict[str, pd.Series] = {}

    for low_q in (0.30, 1/3, 0.40):
        for weighting in ("unweighted", "cropland-weighted"):
            high_q = 1 - low_q   # symmetric: top fraction of same size
            v = buffer_field["values"].values
            w = vul["cropland_Mha"].fillna(0).values
            if weighting == "unweighted":
                low_cut  = cutoff_unweighted(v[eligible.values], low_q)
                high_cut = cutoff_unweighted(v[eligible.values], high_q)
            else:
                low_cut  = cutoff_weighted(v[eligible.values],
                                            w[eligible.values], low_q)
                high_cut = cutoff_weighted(v[eligible.values],
                                            w[eligible.values], high_q)

            bcls = buffer_class_for(buffer_field["values"], low_cut, high_cut,
                                    eligible)
            cls = panel_c(bcls, expcls)
            cls_columns[
                f"{buffer_field['name']}_{weighting}_low{int(low_q*100)}"
            ] = cls

            focal_mask = cls == "low_buffer_high_exposure"
            focal_Mha = vul.loc[focal_mask, "cropland_Mha"].sum()
            n_focal   = int(focal_mask.sum())
            chn_cls   = cls[vul["iso3"] == "CHN"].iloc[0] if (vul["iso3"] == "CHN").any() else "n/a"

            top = (vul.loc[focal_mask]
                       .sort_values("cropland_Mha", ascending=False)
                       .head(10)["iso3"]
                       .tolist())

            rows.append({
                "buffer_metric":     buffer_field["name"],
                "weighting":         weighting,
                "low_q":             low_q,
                "low_cutoff":        low_cut,
                "high_cutoff":       high_cut,
                "focal_cropland_Mha":focal_Mha,
                "focal_pct":         100 * focal_Mha
                                       / vul["cropland_Mha"].sum(),
                "n_focal_countries": n_focal,
                "China_class":       chn_cls,
                "top10_focal":       ";".join(top),
            })

    return pd.DataFrame(rows), cls_columns


# ---------------------------------------------------------------------------
# Audit 3 — country QA
# ---------------------------------------------------------------------------

def buffer_percentile(vul: pd.DataFrame, field: str) -> pd.Series:
    eligible = vul["cropland_Mha"] >= 0.1
    sub = vul.loc[eligible, field].dropna()
    rank = sub.rank(pct=True) * 100
    out = pd.Series(np.nan, index=vul.index)
    out.loc[rank.index] = rank.values
    return out


def exposure_pathway(row) -> str:
    intensity = row["n_intensity_raw"]
    reliance  = row["import_reliance"]
    if pd.isna(intensity) or pd.isna(reliance):
        return "data_missing"
    a = intensity >= HIGH_INTENSITY_KG_HA
    b = (reliance >= HIGH_RELIANCE) & (intensity >= MATERIAL_STAKE_KG_HA)
    low = (intensity < LOW_INTENSITY_KG_HA) & (reliance < LOW_RELIANCE)
    if a and b:  return "A+B"
    if a:        return "A (intensity)"
    if b:        return "B (reliance + stake)"
    if low:      return "low (both sub-thresholds)"
    return "moderate"


def audit_country_qa(vul: pd.DataFrame, recl: pd.DataFrame,
                     ns_cls: dict[str, pd.Series],
                     soc_cls: dict[str, pd.Series]) -> pd.DataFrame:
    nspct = buffer_percentile(vul, "buffer_proxy_t_ha")
    socpct = (buffer_percentile(vul, "soc_stock_cwm_t_ha")
              if "soc_stock_cwm_t_ha" in vul.columns
              else pd.Series(np.nan, index=vul.index))
    pathway = vul.apply(exposure_pathway, axis=1)

    rows = []
    for iso in QA_COUNTRIES:
        m = vul["iso3"] == iso
        if not m.any():
            continue
        i = vul.index[m][0]
        r = vul.loc[i]
        p1 = recl.loc[recl["iso3"] == iso, "class_phase1"]
        p2 = recl.loc[recl["iso3"] == iso, "class_phase2"]

        row = {
            "iso3":            iso,
            "country":         r["fao_name"],
            "cropland_Mha":    r["cropland_Mha"],
            "buffer_cwm":      r["buffer_proxy_t_ha"],
            "buffer_pctile":   nspct.iloc[i],
            "soc_cwm":         r.get("soc_stock_cwm_t_ha", np.nan),
            "soc_pctile":      socpct.iloc[i],
            "n_intensity":     r["n_intensity_raw"],
            "import_reliance": r["import_reliance"],
            "exposure_pathway":pathway.iloc[i],
            "exposure_class":  r["exposure_class"],
            "class_phase1":    p1.iloc[0] if len(p1) else "n/a",
            "class_phase2":    p2.iloc[0] if len(p2) else "n/a",
        }
        for k, s in {**ns_cls, **soc_cls}.items():
            row[f"class[{k}]"] = s.iloc[i]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Audit 4 — area-accounting note
# ---------------------------------------------------------------------------

def write_area_accounting_note() -> str:
    text = (
        "Area-accounting note (Phase 2)\n"
        "==============================\n\n"
        "Two distinct cropland datasets are used in this figure, and they "
        "answer different questions:\n\n"
        "  (1) FAOSTAT cropland area (Item: Cropland; Element: Area; "
        "Inputs/Land Use bulk CSV) is the country-level area used as the "
        "denominator for nitrogen application intensity (N_use_t × 1000 / "
        "cropland_ha) and as the bar height in panel d (cropland Mha × "
        "panel-c class × region). FAOSTAT cropland follows the World Census "
        "of Agriculture conventions and is reported as a single annual area "
        "per country, averaged here over 2018–2020.\n\n"
        "  (2) MIRCA2000 maximum total annual cropped area (Portmann et al. "
        "2010, Zenodo 7422506) is a 5-arcmin spatial grid of cropped area "
        "in hectares per cell. It is used ONLY as the spatial weight for "
        "aggregating the SoilGrids 2.0 soil organic N-stock proxy from 5 km "
        "pixel-level to country-level (cropland-weighted mean). MIRCA's "
        "global total (~1158 Mha) is not meant to match FAOSTAT's national "
        "cropland totals exactly, because MIRCA counts double-cropped cells "
        "twice and uses circa-2000 satellite-derived cropland extent.\n\n"
        "Implication: panel a is country-level cropland-weighted mean of "
        "soil organic N stock 0–30 cm, weighted by MIRCA. Panel b uses "
        "FAOSTAT cropland in the denominator of N intensity. Panels c and d "
        "use the panel-c classification (which depends on both the "
        "MIRCA-weighted buffer tercile and the FAOSTAT-derived intensity / "
        "reliance) and the FAOSTAT cropland area (panel d bar heights). The "
        "two datasets serve non-overlapping roles, so the slight mismatch "
        "in global cropland total (FAOSTAT ≈ 1567 Mha vs MIRCA cropped ≈ "
        "1158 Mha) is methodologically correct, not an inconsistency.\n"
    )
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Phase 2 final audit suite ...")
    vul = pd.read_csv(DIR_PROCESSED / "vulnerability_country.csv")
    buf = pd.read_csv(DIR_PROCESSED / "buffer_country.csv")
    recl = pd.read_csv(DIR_PROCESSED / "reclassification_phase1_to_phase2.csv")

    # SOC-only buffer is in buffer_country.csv as soc_stock_cwm_t_ha
    soc_field = buf[["iso3", "soc_stock_cwm_t_ha"]].rename(
        columns={"soc_stock_cwm_t_ha": "soc_only_cwm"})
    vul = vul.merge(soc_field, on="iso3", how="left")

    # ---- Audit 1 — correlations ----
    print("\n[1] Correlations buffer × exposure")
    cor = audit_correlations(vul)
    cor.to_csv(DIR_PROCESSED / "final_audit_summary.csv", index=False)
    print(cor.to_string(index=False))

    # ---- Audit 2 — buffer-threshold sensitivity ----
    print("\n[2] Buffer-threshold sensitivity")
    ns_summary, ns_cls_cols = audit_buffer_thresholds(
        vul, {"name": "n_stock_cwm",
              "values": vul["buffer_proxy_t_ha"]}
    )
    soc_summary, soc_cls_cols = audit_buffer_thresholds(
        vul, {"name": "soc_only_cwm",
              "values": vul["soc_only_cwm"]}
    )
    sens = pd.concat([ns_summary, soc_summary], ignore_index=True)
    sens.to_csv(DIR_PROCESSED / "buffer_threshold_sensitivity.csv", index=False)
    print(sens[["buffer_metric", "weighting", "low_q",
                "low_cutoff", "high_cutoff",
                "focal_cropland_Mha", "focal_pct",
                "n_focal_countries", "China_class"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---- Audit 3 — country QA ----
    print("\n[3] Country QA table")
    qa = audit_country_qa(vul, recl, ns_cls_cols, soc_cls_cols)
    qa.to_csv(DIR_PROCESSED / "top_country_QA.csv", index=False)
    show_cols = ["iso3", "country", "cropland_Mha", "buffer_cwm",
                 "buffer_pctile", "n_intensity", "import_reliance",
                 "exposure_pathway", "class_phase1", "class_phase2"]
    print(qa[show_cols].to_string(index=False,
                                   float_format=lambda x: f"{x:.2f}"))

    # ---- Audit 4 — area-accounting note ----
    print("\n[4] Area-accounting audit")
    note = write_area_accounting_note()
    (DIR_PROCESSED / "area_accounting_note.txt").write_text(note)
    fao_total = vul["cropland_Mha"].sum()
    print(f"  FAOSTAT cropland total used in panel d:    {fao_total:.0f} Mha")
    cropped_total = buf["cropped_ha_total"].sum() / 1e6 if "cropped_ha_total" in buf.columns else float("nan")
    print(f"  MIRCA cropped-area total used as weight:   {cropped_total:.0f} Mha")
    print(f"  Note written to data_processed/area_accounting_note.txt")

    # ---- Files written ----
    print("\nFiles written:")
    for f in ["final_audit_summary.csv",
              "buffer_threshold_sensitivity.csv",
              "top_country_QA.csv",
              "area_accounting_note.txt"]:
        path = DIR_PROCESSED / f
        size = path.stat().st_size
        print(f"  {f:42s}  ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
