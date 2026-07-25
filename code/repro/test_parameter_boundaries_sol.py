#!/usr/bin/env python3
"""Boundary, identity and calibration-target tests for all parameter groups."""
from pathlib import Path
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

from parameter_registry import (
    BASELINE_BNF_KG_N_HA_YR, BNF_COMPONENTS, REGIONAL_PRICES,
)
from soil_n_model import (
    CropParams, FeedbackParams, SOMPoolParams, get_default_regions,
    som_params_for_region,
)
from monthly_model_v3 import (
    FAOSTAT_TARGETS, MonthlyNParams, REGIONAL_CLIMATES,
    apply_era5_climate_file, run_model, century_dynamic_spinup,
)
from coupled_monthly import get_calibrated_ym
from coupled_econ_biophysical import (
    REGIONAL_ECON_PARAMS, get_scenario_params,
    get_supply_constrained_scenarios,
)


REGIONS = list(FAOSTAT_TARGETS)


def main():
    apply_era5_climate_file(ROOT / "data/era5_regional_climates.json")
    regions = get_default_regions()
    crop = CropParams()
    feedback = FeedbackParams()
    monthly = MonthlyNParams()

    # Pool and allocation identities.
    for key in REGIONS:
        som = som_params_for_region(key)
        assert abs(som.f_active + som.f_slow + som.f_passive - 1.0) < 1e-12
        assert 0 < som.k_active < 1
        assert 0 < som.k_slow < 1
        assert 0 < som.k_passive < 1
        assert 0 <= som.h_active_to_slow <= 1
        assert 0 <= som.h_slow_to_passive <= 1
    assert abs(feedback.cre_to_active + feedback.cre_to_slow - 1.0) < 1e-12

    # Crop and monthly-N boundaries.
    assert crop.residue_grain_ratio >= 0
    assert 0 < crop.residue_c_fraction < 1
    assert 0 < crop.grain_n_fraction < 1
    assert 0 < crop.harvest_index < 1
    assert monthly.q10 > 0
    assert monthly.t_min < monthly.t_ref
    assert 0 <= monthly.moist_min <= 1
    assert 0 <= monthly.moist_waterlog <= 1
    assert 0 <= monthly.immob_frac <= 1
    assert 0 <= monthly.max_uptake_frac <= 1
    assert monthly.min_n_pool >= 0

    # Regional and climate forcing boundaries.
    for key, region in regions.items():
        assert region.soc_initial > 0
        assert region.cn_bulk > 0
        assert region.cropland_mha > 0
        assert region.synth_n_current >= 0
        assert region.atm_n_deposition >= 0
        assert 0 <= region.residue_retention <= 1
        assert 0 <= region.cre_regional <= 1
        assert region.root_shoot_c_ratio >= 0
        assert region.yield_min_regional >= 0
        assert region.yield_max_regional == 0.0

        climate = REGIONAL_CLIMATES[key]
        assert len(climate.temp) == len(climate.precip) == len(climate.pet) == 12
        assert all(math.isfinite(x) for x in climate.temp)
        assert all(math.isfinite(x) and x >= 0 for x in climate.precip)
        assert all(math.isfinite(x) and x >= 0 for x in climate.pet)
        assert 1 <= climate.planting_month <= 12
        assert 1 <= climate.maturity_month <= 12

        component = BNF_COMPONENTS[key]
        assert 0 <= component["legume_frac"] < 1
        assert component["net_n_credit"] >= 0
        assert component["free_living_bnf"] >= 0
        expected_bnf = (
            component["legume_frac"] * component["net_n_credit"]
            / (1 - component["legume_frac"])
            + component["free_living_bnf"]
        )
        assert abs(expected_bnf - BASELINE_BNF_KG_N_HA_YR[key]) < 1e-12
        assert abs(region.bnf_potential - expected_bnf) < 1e-12

    # Economic sign/domain restrictions and non-singular land system.
    for key, values in REGIONAL_ECON_PARAMS.items():
        assert values["eta"] < 0
        assert values["eps_F_PF"] < 0
        assert values["eps_F_PY"] >= 0
        assert values["eps_F_N"] <= 0
        assert values["alpha"] >= 0
        assert values["eps_LD_PL"] < 0
        assert values["eps_LD_PY"] >= 0
        assert values["eps_LS_PL"] > 0
        assert abs(values["eps_LS_PL"] - values["eps_LD_PL"]) > 1e-10

    scenarios = {}
    scenarios.update(get_scenario_params())
    scenarios.update(get_supply_constrained_scenarios())
    for scenario in scenarios.values():
        assert scenario.fert_price_shock >= 0
        assert 0 <= scenario.fert_supply_ceiling <= 1
        assert scenario.fert_capacity_recovery_years >= 0

    # Calibration boundary: every runtime root lies within the solver bracket
    # and reproduces its FAOSTAT target on THE PATH THE PUBLISHED RUNS USE.
    #
    # F-002 (2026-07-25). This block used to root `run_model` and require the
    # year-2 yield to hit the target within 0.011 t/ha. It passed because
    # `get_calibrated_ym` was itself calibrated on `run_model` — the test and
    # the calibration shared a path, and neither was the path that produced a
    # published number. Under the production-path calibration the legacy path
    # now misses by up to 4.19%, which is the finding, not a regression: see
    # code/tests/test_calibration_fingerprint.py, which asserts that the gap
    # persists rather than papering over it.
    regions_all = get_default_regions()
    for key in REGIONS:
        ym = get_calibrated_ym(key, monthly)
        assert 1.0 <= ym <= 50.0
        actual = century_dynamic_spinup(
            key, p=monthly, synth_n=regions_all[key].synth_n_current,
            yield_max_override=ym, region_override=regions_all[key],
        )["yield_eq"]
        assert abs(actual - FAOSTAT_TARGETS[key]) <= 1e-3 * FAOSTAT_TARGETS[key], (
            key, actual, FAOSTAT_TARGETS[key]
        )

    # Price primitives and crop-equivalent baseline expenditure denominators.
    for key, price in REGIONAL_PRICES.items():
        assert price.nitrogen_usd_per_kg_n > 0
        assert price.crop_usd_per_t > 0
        region = regions[key]
        ym = get_calibrated_ym(key, monthly)
        baseline = run_model(
            key, n_years=5, yield_max_override=ym, p=monthly
        )["yield_tha"][2]
        net_crop_equivalent = (
            baseline - region.synth_n_current
            * price.nitrogen_usd_per_kg_n / price.crop_usd_per_t
        )
        assert net_crop_equivalent > 0

    print("PARAMETER BOUNDARIES AND CALIBRATION: PASS")
    print("  pool identities, physical/economic domains, climate vectors,")
    print("  BNF derivation, calibration roots and price denominators checked.")


if __name__ == "__main__":
    main()
