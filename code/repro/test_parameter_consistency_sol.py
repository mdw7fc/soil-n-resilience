#!/usr/bin/env python3
"""Acceptance tests for corrected units, prices, and unique definitions."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

from parameter_registry import (
    WHC_MM_PER_SOC_PCT_30CM, REGIONAL_PRICES, nitrogen_cost_share)
from soil_n_model import get_default_regions
from monthly_model_v3 import MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES
from coupled_monthly import MonthlyBiophysicalEngine, get_calibrated_ym

EXPECTED_SHARES = {
    "sub_saharan_africa": 0.037,
    "south_asia": 0.153,
    "latin_america": 0.047,
    "north_america": 0.060,
}


def main():
    climate = json.loads((ROOT / "data/era5_regional_climates.json").read_text())
    for key, old in list(REGIONAL_CLIMATES.items()):
        new = climate[key]
        REGIONAL_CLIMATES[key] = MonthlyClimate(
            old.name, list(map(float, new["temp"])),
            list(map(float, new["precip"])), list(map(float, new["pet"])),
            old.planting_month, old.maturity_month)
    regions = get_default_regions()
    mp = MonthlyNParams()
    assert abs(WHC_MM_PER_SOC_PCT_30CM - 3.48) < 1e-12
    assert all(abs(r.whc_sensitivity - 3.48) < 1e-12
               for r in regions.values())

    for region, expected in EXPECTED_SHARES.items():
        ym = get_calibrated_ym(region, mp)
        baseline_yield = MonthlyBiophysicalEngine(
            regions[region], region_key=region, monthly_params=mp,
            yield_max_override=ym).step(
                regions[region].synth_n_current)["yield_tha"]
        got = nitrogen_cost_share(
            region, regions[region].synth_n_current,
            baseline_yield)
        assert abs(got - expected) < 1e-3, (region, got, expected)

    code_text = "\n".join(
        p.read_text(errors="ignore")
        for p in (ROOT / "code").rglob("*.py")
        if p.name != Path(__file__).name
    )
    assert "FERT_COST_FRAC" not in code_text
    assert "/ (300 * 0.01)" not in code_text
    assert "/(300 * 0.01)" not in code_text
    assert set(REGIONAL_PRICES) == set(EXPECTED_SHARES)

    print("PARAMETER CONSISTENCY: PASS")
    for region in EXPECTED_SHARES:
        ym = get_calibrated_ym(region, mp)
        baseline_yield = MonthlyBiophysicalEngine(
            regions[region], region_key=region, monthly_params=mp,
            yield_max_override=ym).step(
                regions[region].synth_n_current)["yield_tha"]
        share = nitrogen_cost_share(
            region, regions[region].synth_n_current,
            baseline_yield)
        print(f"  {region:22s} N-cost/revenue = {100*share:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
