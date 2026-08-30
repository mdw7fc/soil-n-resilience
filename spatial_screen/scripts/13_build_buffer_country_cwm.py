"""Phase 2 — Build buffer_country.csv with CROPLAND-WEIGHTED MEAN aggregation.

Replaces the Phase 1 country-median approach with a proper cropland-weighted
mean using MIRCA2000 maximum total annual cropped area (Portmann et al. 2010).

Steps:
  1. Read MIRCA total cropped-area ASCII grid (5 arcmin, WGS84)
  2. Reproject MIRCA → SoilGrids 5 km Homolosine grid (nearest-neighbor; the
     5 arcmin cell is roughly the same as the 5 km cell, and we use MIRCA only
     as a weight so per-pixel proportional allocation is unnecessary).
  3. Read SoilGrids OCS, total-N stack
  4. Per country, compute:
        buffer_cwm = Σ(buffer_proxy × cropped_area) / Σ(cropped_area)
        soc_cwm    = Σ(SOC          × cropped_area) / Σ(cropped_area)
        n_stock_cwm= Σ(N_stock      × cropped_area) / Σ(cropped_area)
        cwm_pixels = number of cropland-bearing soil pixels in the country
        cropped_ha_total = Σ cropped_area
  5. Write buffer_country.csv with both Phase 1 (median, all-land) and
     Phase 2 (cropland-weighted mean) buffer columns. The buffer_proxy_t_ha
     used downstream is now the cropland-weighted mean.

Output: data_processed/buffer_country.csv
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).parent))
from _config import DIR_PROCESSED, DIR_RAW  # noqa: E402

NE_SHP = DIR_RAW / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
MIRCA_FILE = DIR_RAW / "mirca2000" / "maximum_cropped_area_grid" / "MAX_CROPPED_AREA_TOTAL_ANNUAL_HA.ASC.gz"

SCALE = {"nitrogen": 0.01, "bdod": 0.01, "soc": 0.1, "ocs": 1.0}
DEPTHS = [("0-5cm", 0.05), ("5-15cm", 0.10), ("15-30cm", 0.15)]


def read_mirca_ascii_grid(path: Path) -> tuple[np.ndarray, dict]:
    """Read gzipped ESRI-ASCII grid; return (array, metadata)."""
    with gzip.open(path, "rt") as f:
        header = {}
        for _ in range(6):
            line = f.readline().split()
            header[line[0].lower()] = float(line[1])
        ncols  = int(header["ncols"])
        nrows  = int(header["nrows"])
        xll    = header["xllcorner"]
        yll    = header["yllcorner"]
        cs     = header["cellsize"]
        nodata = header.get("nodata_value", -9999.0)

        # Parse remaining values
        data = np.loadtxt(f, dtype=np.float32)
        if data.size != nrows * ncols:
            raise RuntimeError(f"size mismatch: {data.size} vs {nrows*ncols}")
        arr = data.reshape((nrows, ncols))
        arr[arr == nodata] = 0.0  # zero cropland for missing cells
    transform = Affine(cs, 0, xll, 0, -cs, yll + nrows * cs)
    return arr, {
        "ncols": ncols, "nrows": nrows, "transform": transform,
        "cellsize": cs, "crs": "EPSG:4326", "nodata": nodata,
    }


def reproject_mirca_to_grid(mirca_arr: np.ndarray, mirca_meta: dict,
                             dst_crs: str, dst_transform, dst_shape) -> np.ndarray:
    """Reproject MIRCA cropped-area to target grid (Homolosine).

    Use 'sum' resampling so total cropped area is approximately preserved.
    rasterio supports Resampling.sum for upsampling/downsampling by area.
    """
    out = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=mirca_arr,
        destination=out,
        src_transform=mirca_meta["transform"],
        src_crs=mirca_meta["crs"],
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.sum,
        src_nodata=0,
        dst_nodata=0,
    )
    return out


def read_soilgrids_layer(var: str, depth: str) -> tuple[np.ndarray, object]:
    path = DIR_RAW / f"soilgrids_{var}_{depth}_mean_5000.tif"
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        meta = (src.crs, src.transform, src.shape)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr * SCALE[var], meta


def main() -> int:
    print("Phase 2 — building cropland-weighted buffer ...")

    countries = pd.read_csv(DIR_PROCESSED / "countries_master.csv")
    countries = countries[countries["iso3"].notna() & ~countries["historical"]]

    # --- Reference grid: SoilGrids 5km Homolosine ---
    ref_path = DIR_RAW / "soilgrids_ocs_0-30cm_mean_5000.tif"
    with rasterio.open(ref_path) as ref:
        target_crs = ref.crs
        target_transform = ref.transform
        target_shape = (ref.height, ref.width)
    print(f"  target grid: {target_shape}  CRS: {target_crs}")

    # --- MIRCA cropped area, reprojected ---
    print("  reading MIRCA2000 max cropped area grid (5 arcmin) ...")
    mirca_arr, mirca_meta = read_mirca_ascii_grid(MIRCA_FILE)
    print(f"    MIRCA shape: {mirca_arr.shape}, total cropped area = "
          f"{mirca_arr.sum()/1e6:.0f} Mha")
    print("  reprojecting MIRCA to Homolosine 5 km (Resampling.sum) ...")
    mirca_homolosine = reproject_mirca_to_grid(
        mirca_arr, mirca_meta,
        dst_crs=str(target_crs),
        dst_transform=target_transform,
        dst_shape=target_shape,
    )
    print(f"    after reprojection: {mirca_homolosine.shape}, total = "
          f"{mirca_homolosine.sum()/1e6:.0f} Mha "
          f"(should be close to original)")

    # --- Country boundaries on the same grid ---
    ne = gpd.read_file(NE_SHP)[["ADM0_A3", "ADMIN", "geometry"]] \
            .rename(columns={"ADM0_A3": "iso3", "ADMIN": "ne_name"})
    ne_proj = ne.to_crs(target_crs)
    ne_proj = ne_proj[ne_proj["iso3"].isin(countries["iso3"])].reset_index(drop=True)
    ne_proj["country_id"] = np.arange(1, len(ne_proj) + 1, dtype=np.int32)

    print("  rasterizing country IDs on Homolosine grid ...")
    shapes = ((g, c) for g, c in zip(ne_proj.geometry, ne_proj["country_id"]))
    country_id_raster = rasterio.features.rasterize(
        shapes, out_shape=target_shape, transform=target_transform,
        fill=0, dtype=np.int32, all_touched=False,
    )

    # --- SoilGrids layers ---
    print("  reading SoilGrids ...")
    ocs, _ = read_soilgrids_layer("ocs", "0-30cm")
    n_stack = []
    bd_stack = []
    for d, _t in DEPTHS:
        a, _ = read_soilgrids_layer("nitrogen", d); n_stack.append(a)
        a, _ = read_soilgrids_layer("bdod",     d); bd_stack.append(a)

    # Total N stock 0-30 cm
    n_stock = np.zeros_like(ocs)
    valid = np.zeros_like(ocs, dtype=bool)
    for (_, thickness), n_arr, bd_arr in zip(DEPTHS, n_stack, bd_stack):
        slab = n_arr * bd_arr * thickness * 10.0
        slab_valid = ~np.isnan(slab)
        slab = np.where(slab_valid, slab, 0.0)
        n_stock = n_stock + slab
        valid = valid | slab_valid
    n_stock = np.where(valid, n_stock, np.nan)

    # buffer_proxy = N_stock (= SOC / CN by construction)
    buffer_pix = n_stock

    # --- Country-level cropland-weighted mean ---
    print("  computing cropland-weighted means ...")
    n_countries = len(ne_proj)
    ids = country_id_raster.ravel()
    crop = mirca_homolosine.ravel()
    bvals = buffer_pix.ravel()
    socs  = ocs.ravel()
    nsts  = n_stock.ravel()

    base = (ids > 0) & (crop > 0)

    def cwm(values: np.ndarray) -> np.ndarray:
        v = base & ~np.isnan(values)
        sum_w  = np.bincount(ids[v], weights=crop[v],             minlength=n_countries+1)
        sum_wx = np.bincount(ids[v], weights=crop[v] * values[v], minlength=n_countries+1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sum_w > 0, sum_wx / sum_w, np.nan)

    b_cwm   = cwm(bvals)
    soc_cwm = cwm(socs)
    n_cwm   = cwm(nsts)
    sum_w   = np.bincount(ids[base & ~np.isnan(bvals)],
                           weights=crop[base & ~np.isnan(bvals)],
                           minlength=n_countries+1)
    cnt     = np.bincount(ids[base & ~np.isnan(bvals)], minlength=n_countries+1)

    # --- Read existing Phase 1 buffer for cross-comparison ---
    p1 = pd.read_csv(DIR_PROCESSED / "buffer_country.csv")[
        ["iso3", "n_pixels",
         "soc_stock_median_t_ha", "n_stock_median_t_ha",
         "c_to_n_ratio_median",
         "n_stock_mean_t_ha"]
    ].rename(columns={
        "n_stock_median_t_ha":  "n_stock_median_allland_t_ha",
        "soc_stock_median_t_ha":"soc_stock_median_allland_t_ha",
        "c_to_n_ratio_median":  "c_to_n_ratio_median_allland",
        "n_stock_mean_t_ha":    "n_stock_mean_allland_t_ha",
    })

    # --- Assemble output ---
    out = pd.DataFrame({
        "country_id":          ne_proj["country_id"],
        "iso3":                ne_proj["iso3"],
        "ne_name":             ne_proj["ne_name"],
        "n_pixels_cropland":   cnt[1:],
        "cropped_ha_total":    sum_w[1:],     # total MIRCA-allocated cropped ha
        "soc_stock_cwm_t_ha":  soc_cwm[1:],
        "n_stock_cwm_t_ha":    n_cwm[1:],
        "buffer_proxy_cwm_t_ha": b_cwm[1:],
    })
    out["c_to_n_ratio_cwm"] = out["soc_stock_cwm_t_ha"] / out["n_stock_cwm_t_ha"]

    out = out.merge(p1, on="iso3", how="left")
    out["enough_pixels"] = out["n_pixels_cropland"] >= 5
    # Phase 2 LOCKED buffer proxy = cropland-weighted mean
    out["buffer_proxy_t_ha"] = out["buffer_proxy_cwm_t_ha"]

    out = out.sort_values("buffer_proxy_t_ha", ascending=False).reset_index(drop=True)
    out_path = DIR_PROCESSED / "buffer_country.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

    # Diagnostics
    print(f"\n  countries with cropland coverage: "
          f"{int((out['n_pixels_cropland'] >= 5).sum())}")
    print(f"  countries with no cropland pixels: "
          f"{int((out['n_pixels_cropland'] < 5).sum())}")
    print(f"  global cropped area total: {out['cropped_ha_total'].sum()/1e6:.0f} Mha")

    print("\n  Top 10 countries by cropland-weighted buffer (t N/ha):")
    show = out[out["enough_pixels"]].head(10)
    print(show[["iso3", "ne_name", "soc_stock_cwm_t_ha",
                "n_stock_cwm_t_ha", "c_to_n_ratio_cwm",
                "buffer_proxy_cwm_t_ha", "n_stock_median_allland_t_ha"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n  Bottom 10 (low cropland-weighted buffer):")
    show = out[out["enough_pixels"]].tail(10)
    print(show[["iso3", "ne_name", "soc_stock_cwm_t_ha",
                "n_stock_cwm_t_ha", "c_to_n_ratio_cwm",
                "buffer_proxy_cwm_t_ha", "n_stock_median_allland_t_ha"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n  CWM vs all-land median for major-cropland countries:")
    big = out[(out["enough_pixels"]) & (out["cropped_ha_total"] > 5e6)]
    show = big.assign(
        delta=(big["buffer_proxy_cwm_t_ha"] - big["n_stock_median_allland_t_ha"])
    ).sort_values("delta")
    print(show[["iso3", "ne_name",
                "buffer_proxy_cwm_t_ha", "n_stock_median_allland_t_ha", "delta"]]
          .head(15)
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
