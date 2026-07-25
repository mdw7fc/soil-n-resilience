#!/usr/bin/env python3
"""Regenerate calibrated ceilings and simulated year-2 zero-synthetic-N yields."""
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
from soil_n_model import get_default_regions

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
    curves = {}
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
        rows.append({
            "region": key, "faostat_target_t_ha": FAOSTAT_TARGETS[key],
            "calibrated_y_max_t_ha": ym,
            "simulated_year2_no_synth_n_t_ha": y0,
            "empirical_floor_t_ha": regions[key].yield_min_regional,
        })

    (ROOT / "data/figS12_curves.json").write_text(
        json.dumps(curves, indent=1) + "\n")
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
