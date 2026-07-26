#!/usr/bin/env python3
"""Regenerate calibrated ceilings and simulated year-2 zero-synthetic-N yields.

Writes three artifacts:

  outputs/Table_S4_calibration_sol.csv     Supplementary table 4 as printed
  data/figS12_curves.json                  Figure S12's response curves
  data/crop_response_calibration_table.csv Figure S13's calibration column

The third was added when the build graph found it UNSOURCED: MANIFEST.md has
credited this script with regenerating it since the v14 deposit, and no script
in the deposit ever did, so Figure S13 was drawn against a frozen file of
unknown provenance. Its numeric columns are now derived here on every run. Its
one documentary column, `floor_source`, cannot be derived and is carried below
as a literal, which is what it has always been.
"""
from pathlib import Path
import csv
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

from monthly_model_v3 import (
    FAOSTAT_TARGETS, MonthlyNParams, apply_era5_climate_file, run_model)
from coupled_monthly import get_calibrated_ym
from soil_n_model import CropParams, get_default_regions

# Documentary provenance for `yield_min_regional`, transcribed from
# Supplementary table 4 and params.yaml. Not derivable from the model; if a
# floor moves in params.yaml its citation must move here in the same commit.
FLOOR_SOURCES = {
    "north_america": "Morrow Plots & Sanborn Field (Nafziger & Dunker 2011; Miles & Brown 2011)",
    "europe": "Rothamsted Broadbalk & Hoosfield (Poulton et al. 2018)",
    "east_asia": "China 1949 national avg (crop-mix blended)",
    "south_asia": "Pre-Green-Revolution wheat + ICRISAT Vertisol trials",
    "southeast_asia": "Wetland-rice BNF-supported floor (Ladha et al. 2016)",
    "latin_america": "Pampas/Cerrado blended pre-modern yields",
    "sub_saharan_africa": "TSBF unfertilized controls; AFDB traditional-yield synthesis",
    "fsu_central_asia": "Pryanishnikov Institute trials; Kazakhstan dryland wheat",
}

REGIONS = [
    "north_america", "europe", "east_asia", "south_asia",
    "southeast_asia", "latin_america", "sub_saharan_africa",
    "fsu_central_asia",
]


def patch_era5():
    apply_era5_climate_file(ROOT / "data/era5_regional_climates.json")


def main():
    patch_era5()
    mp = MonthlyNParams()
    regions = get_default_regions()
    rows = []
    calib_rows = []
    curves = {}
    mit_c = CropParams().mitscherlich_c
    for key in REGIONS:
        ym = get_calibrated_ym(key, mp)
        current = run_model(key, n_years=5, yield_max_override=ym, p=mp)
        no_n = run_model(key, synth_n=0.0, n_years=5,
                         yield_max_override=ym, p=mp)
        y0 = float(no_n["yield_tha"][2])
        ycur = float(current["yield_tha"][2])
        n_no = float(no_n["n_uptake"][2])
        n_cur = n_no + regions[key].synth_n_current
        x = [n_no + (400.0 - n_no) * i / 17 for i in range(18)]
        ys = []
        for total_n in x:
            sim = run_model(
                key, synth_n=max(0.0, total_n - n_no), n_years=5,
                yield_max_override=ym, p=mp)
            ys.append(float(sim["yield_tha"][2]))
        curves[key] = {
            "x": x, "y": ys, "Ncur": n_cur, "Nns": n_no,
            "fao": FAOSTAT_TARGETS[key],
            "floor": regions[key].yield_min_regional,
            "ym": ym, "y_at_cur": ycur, "y_nosynth": y0,
        }
        calib_rows.append({
            "region": key,
            "FAOSTAT_yobs_tha": FAOSTAT_TARGETS[key],
            "N_current_kgha": round(n_cur, 1),
            "N_no_synth_kgha": round(n_no, 1),
            "ymax_calibrated_tha": round(ym, 3),
            "mitscherlich_c": mit_c,
            "yield_floor_tha": regions[key].yield_min_regional,
            "y_no_synth_sim_tha": round(y0, 2),
            "floor_source": FLOOR_SOURCES[key],
        })
        rows.append({
            "region": key, "faostat_target_t_ha": FAOSTAT_TARGETS[key],
            "calibrated_y_max_t_ha": ym,
            "simulated_year2_no_synth_n_t_ha": y0,
            "empirical_floor_t_ha": regions[key].yield_min_regional,
        })

    (ROOT / "data/figS12_curves.json").write_text(
        json.dumps(curves, indent=1) + "\n")
    calib = ROOT / "data/crop_response_calibration_table.csv"
    with calib.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calib_rows[0]))
        writer.writeheader()
        writer.writerows(calib_rows)
    print(f"wrote {calib}")

    out = ROOT / "outputs/Table_S4_calibration_sol.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
