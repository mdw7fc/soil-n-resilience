"""Build exposure_country.csv — fertilizer-shock exposure index per SPEC §2.2.

Inputs (data_raw/, via FAOSTAT bulk zips):
  - Inputs_FertilizersNutrient_E_All_Data_(Normalized).csv
      Item=Nutrient nitrogen N (total)
      Element ∈ {Agricultural Use, Production, Import quantity, Export quantity}
      Unit = t (tonnes N)
  - Inputs_LandUse_E_All_Data_(Normalized).csv
      Item=Cropland, Element=Area, Unit=1000 ha

Year handling: 2018–2020 mean (per SPEC §1).

Outputs:
  - data_processed/exposure_country.csv
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import (  # noqa: E402
    DIR_PROCESSED,
    EXPOSURE_HIGH,
    EXPOSURE_LOW,
    EXPOSURE_W_INTENSITY,
    EXPOSURE_W_RELIANCE,
    FAOSTAT_FERT_ZIP,
    FAOSTAT_LANDUSE_ZIP,
    FERT_ELEMENT_EXPORT,
    FERT_ELEMENT_IMPORT,
    FERT_ELEMENT_PRODUCTION,
    FERT_ELEMENT_USE,
    FERT_ITEM_N,
    LANDUSE_CROPLAND_UNIT_FACTOR,
    LANDUSE_ELEMENT_AREA,
    LANDUSE_ITEM_CROPLAND,
    N_INTENSITY_CAP,
    REEXPORT_RATIO_FLAG,
    YEARS,
)


def _read_zip_csv(zip_path: Path, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as fh:
            return pd.read_csv(fh, encoding="latin-1")


def load_fertilizer() -> pd.DataFrame:
    df = _read_zip_csv(
        FAOSTAT_FERT_ZIP,
        "Inputs_FertilizersNutrient_E_All_Data_(Normalized).csv",
    )
    df = df[df["Item"] == FERT_ITEM_N]
    df = df[df["Year"].isin(YEARS)]
    df = df[df["Element"].isin(
        [FERT_ELEMENT_USE, FERT_ELEMENT_PRODUCTION,
         FERT_ELEMENT_IMPORT, FERT_ELEMENT_EXPORT]
    )]
    # Sanity: tonnes N
    bad_units = set(df["Unit"].unique()) - {"t"}
    if bad_units:
        raise RuntimeError(f"Unexpected fertilizer units: {bad_units}")
    return df[["Area Code", "Area", "Element", "Year", "Value"]].copy()


def load_cropland() -> pd.DataFrame:
    df = _read_zip_csv(
        FAOSTAT_LANDUSE_ZIP,
        "Inputs_LandUse_E_All_Data_(Normalized).csv",
    )
    df = df[(df["Item"] == LANDUSE_ITEM_CROPLAND)
            & (df["Element"] == LANDUSE_ELEMENT_AREA)]
    df = df[df["Year"].isin(YEARS)]
    if not (df["Unit"].astype(str).str.strip() == "1000 ha").all():
        raise RuntimeError(f"Unexpected cropland units: {df['Unit'].unique()}")
    df["cropland_ha"] = df["Value"] * LANDUSE_CROPLAND_UNIT_FACTOR
    return df[["Area Code", "Area", "Year", "cropland_ha"]].copy()


def main() -> int:
    print("Building exposure_country.csv ...")

    countries = pd.read_csv(DIR_PROCESSED / "countries_master.csv")
    countries = countries[countries["iso3"].notna() & ~countries["historical"]]
    print(f"  countries in scope: {len(countries)}")

    fert = load_fertilizer()
    crop = load_cropland()

    # 3-yr mean per (country, element)
    fert_mean = (
        fert.groupby(["Area Code", "Area", "Element"], as_index=False)["Value"]
            .mean()
            .pivot(index=["Area Code", "Area"], columns="Element", values="Value")
            .reset_index()
            .rename_axis(columns=None)
    )
    fert_mean = fert_mean.rename(columns={
        FERT_ELEMENT_USE:        "n_use_t",
        FERT_ELEMENT_PRODUCTION: "n_prod_t",
        FERT_ELEMENT_IMPORT:     "n_imp_t",
        FERT_ELEMENT_EXPORT:     "n_exp_t",
    })

    crop_mean = (crop.groupby(["Area Code", "Area"], as_index=False)["cropland_ha"]
                     .mean())

    # Join: country master ← fertilizer ← cropland
    df = countries.merge(fert_mean, left_on="fao_area_code",
                         right_on="Area Code", how="left")
    df = df.merge(crop_mean[["Area Code", "cropland_ha"]],
                  left_on="fao_area_code", right_on="Area Code",
                  how="left", suffixes=("", "_lu"))

    # Coverage diagnostics before zero-fill
    df["data_missing_n_use"]   = df["n_use_t"].isna()
    df["data_missing_trade"]   = df["n_imp_t"].isna() & df["n_exp_t"].isna() & df["n_prod_t"].isna()
    df["data_missing_cropland"]= df["cropland_ha"].isna() | (df["cropland_ha"] <= 0)

    # Replace NaN with 0 for trade-side variables (FAO leaves blank when zero).
    # n_use stays as NA when missing — we need it explicitly for intensity.
    for col in ("n_prod_t", "n_imp_t", "n_exp_t"):
        df[col] = df[col].fillna(0.0)

    # Sub-index 1: N intensity (kg N / ha)
    #   Use is in tonnes N → ×1000 → kg N
    df["n_intensity_raw"] = np.where(
        (df["cropland_ha"] > 0) & df["n_use_t"].notna(),
        (df["n_use_t"] * 1000.0) / df["cropland_ha"],
        np.nan,
    )
    df["n_intensity_scaled"] = np.minimum(df["n_intensity_raw"], N_INTENSITY_CAP) / N_INTENSITY_CAP

    # Sub-index 2: Net import reliance
    df["apparent_consumption_t"] = df["n_prod_t"] + df["n_imp_t"] - df["n_exp_t"]
    df["net_imports_t"] = np.maximum(0.0, df["n_imp_t"] - df["n_exp_t"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ir = df["net_imports_t"] / df["apparent_consumption_t"]
    df["import_reliance"] = np.clip(ir, 0.0, 1.0)
    df.loc[df["apparent_consumption_t"] <= 0, "import_reliance"] = np.nan

    # Re-export hub flag
    with np.errstate(divide="ignore", invalid="ignore"):
        flow = (df["n_imp_t"] + df["n_exp_t"]) / df["apparent_consumption_t"].abs().replace(0, np.nan)
    df["reexport_flag"] = (flow > REEXPORT_RATIO_FLAG).fillna(False)

    # Combined exposure
    df["exposure_combined"] = (
        EXPOSURE_W_INTENSITY * df["n_intensity_scaled"]
        + EXPOSURE_W_RELIANCE * df["import_reliance"]
    )

    # Exposure class (3 levels for §2.3 panel c logic)
    def _exposure_class(x: float) -> str:
        if pd.isna(x):
            return "data_missing"
        if x < EXPOSURE_LOW:
            return "low"
        if x >= EXPOSURE_HIGH:
            return "high"
        return "moderate"

    df["exposure_class"] = df["exposure_combined"].apply(_exposure_class)

    out_cols = [
        "iso3", "fao_area_code", "fao_name", "region",
        "cropland_ha",
        "n_use_t", "n_prod_t", "n_imp_t", "n_exp_t",
        "apparent_consumption_t", "net_imports_t",
        "n_intensity_raw", "n_intensity_scaled",
        "import_reliance", "exposure_combined", "exposure_class",
        "reexport_flag",
        "data_missing_n_use", "data_missing_trade", "data_missing_cropland",
    ]
    out = df[out_cols].sort_values(["region", "iso3"]).reset_index(drop=True)
    out_path = DIR_PROCESSED / "exposure_country.csv"
    out.to_csv(out_path, index=False)
    print(f"  written: {out_path}")

    # Diagnostics
    n = len(out)
    n_use_present = int((~out["data_missing_n_use"]).sum())
    n_crop_present= int((~out["data_missing_cropland"]).sum())
    print(f"  countries: {n}")
    print(f"  with N-use data: {n_use_present}")
    print(f"  with cropland data: {n_crop_present}")
    print(f"  with exposure_combined: {int(out['exposure_combined'].notna().sum())}")
    print(f"  re-export hubs flagged: {int(out['reexport_flag'].sum())}")

    print("\n  Top 10 by N intensity (kg N/ha):")
    print(out.dropna(subset=["n_intensity_raw"])
              .sort_values("n_intensity_raw", ascending=False)
              .head(10)[["iso3", "fao_name", "n_intensity_raw",
                         "import_reliance", "exposure_combined",
                         "exposure_class", "reexport_flag"]]
              .to_string(index=False))

    print("\n  Top 10 by cropland area (Mha):")
    show = out.dropna(subset=["cropland_ha"]).copy()
    show["cropland_Mha"] = show["cropland_ha"] / 1e6
    print(show.sort_values("cropland_Mha", ascending=False)
              .head(10)[["iso3", "fao_name", "cropland_Mha",
                         "n_intensity_raw", "import_reliance",
                         "exposure_combined", "exposure_class"]]
              .to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
