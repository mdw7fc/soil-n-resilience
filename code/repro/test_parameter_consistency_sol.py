#!/usr/bin/env python3
"""Acceptance tests for corrected units, prices, and unique definitions."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

from parameter_registry import (
    WHC_MM_PER_SOC_PCT_30CM, REGIONAL_PRICES, nitrogen_cost_share)
from soil_n_model import get_default_regions
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import MonthlyBiophysicalEngine, get_calibrated_ym

# N cost share = N price x rate / (crop price x baseline yield). The
# denominator is a model output, so these are not free constants: F-002's
# production-path recalibration moves every baseline yield to its FAOSTAT
# target and therefore moves every share.
#
# OWED TO WP5. These four values are frozen at their pre-F-002 figures on
# purpose. Editing them here would retune a stale expectation inside a test
# and lose the evidence that a published number moved; they belong in
# docs/claims.yaml, where the claim register can mark them DRIFTED and carry
# the reason. This file therefore FAILS BY DESIGN until WP5 lands, and is
# excluded from mutation-coverage CATCH for that reason rather than as a
# broken test.
#
# CORRECTED 2026-07-25 (WP3). The note here previously read "the other three
# are within tolerance and do not move". That was wrong: the assertion below
# stopped at the first region and never evaluated the rest. Three of the four
# have drifted. Measured under production_path_v2 against the 1e-3 tolerance:
#
#   sub_saharan_africa  0.037 -> 0.035778   (-1.22e-3)  DRIFTED
#   south_asia          0.153 -> 0.147321   (-5.68e-3)  DRIFTED
#   latin_america       0.047 -> 0.049145   (+2.15e-3)  DRIFTED
#   north_america       0.060 -> 0.060800   (+8.00e-4)  within tolerance
#
# South Asia is the largest mover and the one that matters most: SI [163] and
# claims C-063/C-064 turn on which region carries the highest derived nitrogen
# cost share. WP5 should carry all three, not one.
EXPECTED_SHARES = {
    "sub_saharan_africa": 0.037,
    "south_asia": 0.153,
    "latin_america": 0.047,
    "north_america": 0.060,
}


def main():
    apply_era5_climate_file(ROOT / "data/era5_regional_climates.json")
    regions = get_default_regions()
    mp = MonthlyNParams()
    assert abs(WHC_MM_PER_SOC_PCT_30CM - 3.48) < 1e-12
    assert all(abs(r.whc_sensitivity - 3.48) < 1e-12
               for r in regions.values())

    # Every region is measured before anything is asserted. Failing on the
    # first drifted region hid two more for a full work package.
    drifted = []
    for region, expected in EXPECTED_SHARES.items():
        ym = get_calibrated_ym(region, mp)
        baseline_yield = MonthlyBiophysicalEngine(
            regions[region], region_key=region, monthly_params=mp,
            yield_max_override=ym).step(
                regions[region].synth_n_current)["yield_tha"]
        got = nitrogen_cost_share(
            region, regions[region].synth_n_current,
            baseline_yield)
        print(f"  {region:20s} expected {expected:.6f}  model {got:.6f}  "
              f"delta {got - expected:+.2e}"
              f"{'  DRIFTED' if abs(got - expected) >= 1e-3 else ''}")
        if abs(got - expected) >= 1e-3:
            drifted.append((region, got, expected))
    assert not drifted, (
        f"{len(drifted)} of {len(EXPECTED_SHARES)} derived cost shares have "
        f"drifted from their pre-F-002 figures: {drifted}. This is owed to "
        f"WP5's claim register, not repaired here -- see the note above."
    )

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
