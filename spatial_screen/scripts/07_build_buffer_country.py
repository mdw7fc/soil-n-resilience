"""Build buffer_country.csv — country-mean soil organic N stock 0–30 cm.

Phase 0 implementation:
  - SoilGrids 2.0 5 km aggregated rasters (Homolosine, ESRI:54052)
  - Natural Earth 1:50m admin-0 polygons (WGS84 → reprojected to Homolosine)
  - Per-country *unweighted* mean (cropland weighting deferred to Phase 1)

Computes:
  - SOC stock 0–30 cm     [t C / ha]   from `ocs_0-30cm` (already integrated)
  - Total N stock 0–30 cm [t N / ha]   integrated from N concentration × bulk
                                       density × depth slab thickness
  - C:N ratio = SOC / N_stock           [dimensionless]
  - buffer_proxy = SOC_stock / C:N      [t N / ha]
                 = N_stock_0_30cm       (mathematically equivalent)

Output:
  data_processed/buffer_country.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features

sys.path.insert(0, str(Path(__file__).parent))
from _config import DIR_PROCESSED, DIR_RAW  # noqa: E402

NE_SHP = DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"

# SoilGrids 2.0 native unit conversions to physical units.
# Verified empirically against raw raster medians:
#   nitrogen median raw = 262 → ÷100 → 2.62 g/kg ✓
#   bdod     median raw = 125 → ÷100 → 1.25 g/cm³ ✓
#   soc      median raw = 316 → ÷10  → 31.6 g/kg ✓
#   ocs      median raw =  45 → ×1   → 45 t/ha ✓ (stored in t/ha directly,
#                                                  not scaled despite docs)
SCALE = {
    "nitrogen": 0.01,   # cg/kg → g/kg
    "bdod":     0.01,   # cg/cm³ → g/cm³
    "soc":      0.1,    # dg/kg → g/kg
    "ocs":      1.0,    # already in t/ha
}

DEPTHS = [("0-5cm", 0.05), ("5-15cm", 0.10), ("15-30cm", 0.15)]


def open_layer(var: str, depth: str):
    path = DIR_RAW / f"soilgrids_{var}_{depth}_mean_5000.tif"
    return rasterio.open(path)


def read_array(src) -> np.ndarray:
    arr = src.read(1, masked=False).astype(np.float32)
    nodata = src.nodata if src.nodata is not None else -32768
    mask = arr == nodata
    arr[mask] = np.nan
    return arr


def main() -> int:
    print("Building buffer_country.csv ...")
    countries = pd.read_csv(DIR_PROCESSED / "countries_master.csv")
    countries = countries[countries["iso3"].notna() & ~countries["historical"]]

    # ---- Load Natural Earth ----
    ne = gpd.read_file(NE_SHP)[["ADM0_A3", "ADMIN", "geometry"]]
    ne = ne.rename(columns={"ADM0_A3": "iso3", "ADMIN": "ne_name"})

    # Use SoilGrids reference grid for rasterization
    ref_path = DIR_RAW / "soilgrids_ocs_0-30cm_mean_5000.tif"
    with rasterio.open(ref_path) as ref:
        target_crs = ref.crs
        target_transform = ref.transform
        target_shape = (ref.height, ref.width)

    print(f"  reprojecting Natural Earth to {target_crs} ...")
    ne_proj = ne.to_crs(target_crs)

    # Filter to countries present in our master
    ne_proj = ne_proj[ne_proj["iso3"].isin(countries["iso3"])].reset_index(drop=True)
    ne_proj["country_id"] = np.arange(1, len(ne_proj) + 1, dtype=np.int32)
    print(f"  countries with NE polygons: {len(ne_proj)}")

    # ---- Rasterize country IDs onto SoilGrids grid ----
    print("  rasterizing country IDs onto SoilGrids 5 km grid ...")
    shapes = ((geom, cid) for geom, cid in zip(ne_proj.geometry, ne_proj["country_id"]))
    country_id_raster = rasterio.features.rasterize(
        shapes,
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )
    n_assigned = int((country_id_raster > 0).sum())
    print(f"  pixels assigned to a country: {n_assigned:,} / {country_id_raster.size:,}")

    # ---- Read soil layers ----
    print("  reading SoilGrids layers ...")
    with open_layer("ocs", "0-30cm") as src:
        ocs = read_array(src) * SCALE["ocs"]   # t C / ha
    n_stack = []
    bd_stack = []
    for d, _ in DEPTHS:
        with open_layer("nitrogen", d) as src:
            n_stack.append(read_array(src) * SCALE["nitrogen"])  # g N / kg
        with open_layer("bdod", d) as src:
            bd_stack.append(read_array(src) * SCALE["bdod"])     # g / cm³

    # Total N stock 0-30 cm:
    #   stock_per_slab [t N / ha] = N_conc [g N / kg soil]
    #                              × BD [g / cm³] × 1e3 [kg / m³ per g/cm³ ratio?]
    # Derivation:
    #   1 ha = 10,000 m²
    #   slab volume = 10,000 m² × thickness [m] = thickness × 10,000 m³ /ha
    #   soil mass per ha = thickness × 10,000 × BD × 1e6 [g / m³ per g/cm³]
    #                    = thickness × 1e10 × BD  [g soil / ha]
    #                    = thickness × 1e7 × BD  [kg soil / ha]
    #   N mass per ha [kg] = N_conc [g/kg] × thickness × 1e7 × BD × 1e-3
    #                      = N_conc × thickness × 1e4 × BD  [kg N / ha]
    #   → divide by 1000 to get tonnes: N_stock = N_conc × BD × thickness × 10  [t N / ha]
    n_stock = np.zeros_like(ocs)
    valid_mask = np.zeros_like(ocs, dtype=bool)
    for (d_label, thickness), n_arr, bd_arr in zip(DEPTHS, n_stack, bd_stack):
        slab = n_arr * bd_arr * thickness * 10.0   # t N / ha
        slab_valid = ~np.isnan(slab)
        slab = np.where(slab_valid, slab, 0.0)
        n_stock = n_stock + slab
        valid_mask = valid_mask | slab_valid
    n_stock = np.where(valid_mask, n_stock, np.nan)

    # Cross-check SOC stock 0-30cm via SOC concentration × BD × thickness
    # (ocs is the published integrated value; we should match it within rounding)
    soc_stack = []
    for d, _ in DEPTHS:
        with open_layer("soc", d) as src:
            soc_stack.append(read_array(src) * SCALE["soc"])   # g C / kg
    soc_stock_check = np.zeros_like(ocs)
    soc_valid = np.zeros_like(ocs, dtype=bool)
    for (d_label, thickness), s_arr, bd_arr in zip(DEPTHS, soc_stack, bd_stack):
        slab = s_arr * bd_arr * thickness * 10.0
        slab_valid = ~np.isnan(slab)
        slab = np.where(slab_valid, slab, 0.0)
        soc_stock_check = soc_stock_check + slab
        soc_valid = soc_valid | slab_valid
    soc_stock_check = np.where(soc_valid, soc_stock_check, np.nan)

    # ---- Per-country zonal stats ----
    # Use len(ne_proj), not country_id_raster.max(), because some countries
    # (tiny islands) get 0 pixels and would otherwise be cut off.
    n_countries = int(len(ne_proj))

    def zonal_mean(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = country_id_raster.ravel()
        vals = arr.ravel()
        ok = (ids > 0) & ~np.isnan(vals)
        ids_ok = ids[ok]
        vals_ok = vals[ok]
        sums = np.bincount(ids_ok, weights=vals_ok, minlength=n_countries + 1)
        cnts = np.bincount(ids_ok, minlength=n_countries + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(cnts > 0, sums / cnts, np.nan)
        return means[1:], cnts[1:]   # skip id=0

    def zonal_median(arr: np.ndarray) -> np.ndarray:
        """Exact per-country median; O(N log N) per country.

        Robust to peat/forest outliers in countries with mixed land cover
        (Phase 1 substitution for cropland-weighted mean — see SPEC §9).
        """
        ids = country_id_raster.ravel()
        vals = arr.ravel()
        ok = (ids > 0) & ~np.isnan(vals)
        ids_ok = ids[ok]
        vals_ok = vals[ok]
        # Sort by country id, then group via np.split
        order = np.argsort(ids_ok, kind="stable")
        ids_sorted = ids_ok[order]
        vals_sorted = vals_ok[order]
        boundaries = np.searchsorted(ids_sorted,
                                      np.arange(1, n_countries + 1),
                                      side="left")
        boundaries_end = np.searchsorted(ids_sorted,
                                          np.arange(1, n_countries + 1),
                                          side="right")
        out = np.full(n_countries, np.nan, dtype=np.float32)
        for i, (s, e) in enumerate(zip(boundaries, boundaries_end)):
            if e > s:
                out[i] = np.median(vals_sorted[s:e])
        return out

    print("  computing zonal means + medians ...")
    ocs_mean,  ocs_cnt = zonal_mean(ocs)
    ocs_med            = zonal_median(ocs)
    nstock_mean, _     = zonal_mean(n_stock)
    nstock_med         = zonal_median(n_stock)
    socchk_mean, _     = zonal_mean(soc_stock_check)

    # ---- Assemble output ----
    out = pd.DataFrame({
        "country_id":         ne_proj["country_id"],
        "iso3":               ne_proj["iso3"],
        "ne_name":            ne_proj["ne_name"],
        "n_pixels":           ocs_cnt,
        # Mean-based fields (kept for cross-check / Phase 0 backwards-compat)
        "soc_stock_mean_t_ha":      ocs_mean,
        "soc_stock_chk_mean_t_ha":  socchk_mean,
        "n_stock_mean_t_ha":        nstock_mean,
        # Median-based fields (Phase 1 LOCKED aggregation method)
        "soc_stock_median_t_ha":    ocs_med,
        "n_stock_median_t_ha":      nstock_med,
    })
    # C:N ratio from medians (more robust than mean-based ratio in mixed land cover)
    out["c_to_n_ratio_median"] = (
        out["soc_stock_median_t_ha"] / out["n_stock_median_t_ha"]
    )
    # Phase 1 buffer proxy = median country N stock 0-30cm
    # (mathematically equivalent to SOC_med / C:N_med by construction)
    out["buffer_proxy_t_ha"] = out["n_stock_median_t_ha"]

    # Drop micro-territories with too few pixels (< 5 → noisy)
    out["enough_pixels"] = out["n_pixels"] >= 5

    out = out.sort_values("buffer_proxy_t_ha", ascending=False).reset_index(drop=True)

    out_path = DIR_PROCESSED / "buffer_country.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

    # Console diagnostics
    n_total = len(out)
    n_ok = int(out["enough_pixels"].sum())
    print(f"\n  countries with raster coverage: {n_total}")
    print(f"  with ≥ 5 pixels (usable): {n_ok}")
    print(f"  global mean SOC 0-30cm: {np.nanmean(ocs):.1f} t C/ha "
          f"(SOC-from-conc cross-check: {np.nanmean(soc_stock_check):.1f})")
    print(f"  global mean N stock 0-30cm: {np.nanmean(n_stock):.2f} t N/ha")
    print(f"  global mean C:N: {np.nanmean(ocs/np.where(n_stock>0, n_stock, np.nan)):.1f}")

    print("\n  Top 10 countries by buffer_proxy (median N stock, t/ha):")
    show = out[out["enough_pixels"]].head(10)
    print(show[["iso3", "ne_name", "soc_stock_median_t_ha",
                "n_stock_median_t_ha", "c_to_n_ratio_median",
                "buffer_proxy_t_ha"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n  Bottom 10 (low-buffer):")
    show = out[out["enough_pixels"]].tail(10)
    print(show[["iso3", "ne_name", "soc_stock_median_t_ha",
                "n_stock_median_t_ha", "c_to_n_ratio_median",
                "buffer_proxy_t_ha"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
