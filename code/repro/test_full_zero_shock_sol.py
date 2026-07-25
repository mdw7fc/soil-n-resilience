#!/usr/bin/env python3
"""Acceptance test: central S3 must be stationary when its shock is zero."""
from pathlib import Path
import copy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params
from soil_n_model import get_default_regions

REGIONS = [
    "north_america", "europe", "east_asia", "south_asia",
    "southeast_asia", "latin_america", "sub_saharan_africa",
    "fsu_central_asia",
]


def main():
    apply_era5_climate_file(ROOT / "data/era5_regional_climates.json")
    regions = get_default_regions()
    mp = MonthlyNParams()
    scenario = copy.deepcopy(get_scenario_params()["S3"])
    scenario.fert_price_shock = 0.0
    scenario.fert_supply_ceiling = 1.0
    worst = 0.0
    for key in REGIONS:
        frame = CoupledMonthlyModel(
            regions[key], scenario, region_key=key, t_max=30,
            yield_max_override=get_calibrated_ym(key, mp)).run()
        deviation = max(abs(frame.yield_fraction - 1.0))
        worst = max(worst, float(deviation))
        print(f"{key:22s} max 30-yr deviation {deviation:.3e}")
    assert worst < 1e-3, worst
    print(f"FULL S3 ZERO-SHOCK INVARIANCE: PASS ({worst:.3e})")


if __name__ == "__main__":
    main()
