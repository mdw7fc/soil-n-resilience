#!/usr/bin/env python3
"""Deterministic lower/upper-bound and focal-mechanism acceptance tests."""
from pathlib import Path
import copy
import csv
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import (
    CoupledMonthlyModel, MonthlyBiophysicalEngine, get_calibrated_ym,
)
from coupled_econ_biophysical import get_scenario_params
from soil_n_model import get_default_regions
from parameter_registry import (
    SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY,
    WHC_MM_PER_SOC_PCT_30CM,
    WHC_MM_PER_SOC_PCT_HIGH,
    WHC_MM_PER_SOC_PCT_LOW,
)
from run_mc_ensemble import PRIORS, evaluate_one_draw


REGIONS = [
    "north_america", "europe", "east_asia", "south_asia",
    "southeast_asia", "latin_america", "sub_saharan_africa",
    "fsu_central_asia",
]
NONNEGATIVE = [
    "fert_applied_kgha", "yield_tha", "n_mineralized", "n_uptake",
    "n_leached", "n_denitrified", "n_immobilized", "soc_total",
]
TOL = 1e-9

# Columns that are diagnostics rather than model state. `ln_cap` (added by
# F-010 so the market-clearing test could be made from the DataFrame rather
# than from solver internals) is the log of the physical fertilizer ceiling and
# is undefined when the ceiling does not bind. NaN is the correct encoding of
# "not applicable", so a blanket finiteness assertion over every numeric column
# fails on a correct run. It is checked below where it is defined instead of
# being exempted outright.
DIAGNOSTIC = ["ln_cap"]


def assert_domain(frame, model):
    state = frame.drop(columns=DIAGNOSTIC, errors="ignore")
    values = state.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(values).all()
    if "ln_cap" in frame and "cap_binding" in frame:
        binding = frame["cap_binding"].astype(bool).to_numpy()
        ln_cap = frame["ln_cap"].to_numpy()
        # Defined exactly where the ceiling binds, and nowhere else. Both
        # directions matter: a ln_cap that goes finite while cap_binding is
        # false means the two disagree about whether the branch was entered.
        assert np.isfinite(ln_cap[binding]).all()
        assert np.isnan(ln_cap[~binding]).all()
    assert (frame[NONNEGATIVE] >= -TOL).all().all()
    assert model.bio.mineral_n >= -TOL
    assert min(model.bio.C_active, model.bio.C_slow, model.bio.C_passive) >= -TOL


def mc_oat_cases(regions, ym_table):
    central = {name: spec["mean"] for name, spec in PRIORS.items()}
    cases = [("MC_central", central)]
    for name, spec in PRIORS.items():
        for label in ("lo", "hi"):
            values = dict(central)
            values[name] = spec[label]
            cases.append((f"MC_{name}_{label}", values))

    records = []
    for case, values in cases:
        rows = evaluate_one_draw(pd.Series(values), regions, ym_table)
        frame = pd.DataFrame(rows)
        required = frame[[
            "soc_pct", "yield_pen", "y_base", "y_shock", "F_shocked",
            "PY_hat", "gamma_regional",
        ]].to_numpy()
        assert np.isfinite(required).all(), case
        priced_profit = frame.loc[frame["profit_chg"].notna(), "profit_chg"]
        assert np.isfinite(priced_profit).all(), case
        assert (frame[["y_base", "y_shock", "F_shocked"]] >= -TOL).all().all()
        for key in REGIONS:
            sub = frame[frame.region == key].set_index("soc_pct")
            low = float(sub.loc[50, "yield_pen"])
            high = float(sub.loc[150, "yield_pen"])
            passed = low + TOL >= high
            assert passed, (case, key, low, high)
            records.append({
                "test_family": "one-at-a-time MC prior bound",
                "case": case, "region": key, "horizon_year": 1,
                "low_soc_loss_pct": low, "high_soc_loss_pct": high,
                "monotone_pass": passed, "domain_pass": True,
            })
        print(f"  {case}: PASS")
    return records


def structural_cases(regions, ym_table):
    records = []
    mp = MonthlyNParams()
    equilibrium = {}
    for key in REGIONS:
        engine = MonthlyBiophysicalEngine(
            regions[key], region_key=key, monthly_params=mp,
            yield_max_override=ym_table[key],
        )
        equilibrium[key] = {
            "c_active": engine.C_active, "c_slow": engine.C_slow,
            "c_passive": engine.C_passive, "soc": engine.soc_initial,
            "mineral_n": engine.mineral_n, "yield_eq": engine.yield_baseline,
            "n_min_eq": engine.n_min_baseline,
        }

    whc_values = [
        WHC_MM_PER_SOC_PCT_LOW,
        WHC_MM_PER_SOC_PCT_30CM,
        WHC_MM_PER_SOC_PCT_HIGH,
    ]
    for whc in whc_values:
        for eps_n in SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY:
            case = f"WHC_{whc:g}_epsFN_{eps_n:g}"
            for key in REGIONS:
                region = copy.deepcopy(regions[key])
                region.whc_sensitivity = whc
                losses = {}
                for soc_pct in (50, 150):
                    scale = soc_pct / 100.0
                    scenario = copy.deepcopy(get_scenario_params()["S3"])
                    scenario.eps_F_N = eps_n
                    no_shock = copy.deepcopy(scenario)
                    no_shock.fert_price_shock = 0.0
                    outputs = []
                    for econ in (scenario, no_shock):
                        model = CoupledMonthlyModel(
                            region, econ, region_key=key, t_max=10,
                            yield_max_override=ym_table[key],
                            initial_pools=equilibrium[key],
                            monthly_params=mp,
                        )
                        model.bio.C_active *= scale
                        model.bio.C_slow *= scale
                        model.bio.C_passive *= scale
                        frame = model.run()
                        assert_domain(frame, model)
                        outputs.append(float(
                            frame.loc[frame.year == 10, "yield_tha"].iloc[0]
                        ))
                    shocked, control = outputs
                    assert control > 0
                    losses[soc_pct] = 100.0 * (1.0 - shocked / control)
                passed = losses[50] + TOL >= losses[150]
                records.append({
                    "test_family": "WHC x soil-N-response structural grid",
                    "case": case, "region": key, "horizon_year": 10,
                    "low_soc_loss_pct": losses[50],
                    "high_soc_loss_pct": losses[150],
                    "monotone_pass": passed, "domain_pass": True,
                })
            case_records = [r for r in records if r["case"] == case]
            failures = sum(not r["monotone_pass"] for r in case_records)
            print(f"  {case}: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return records


def main():
    apply_era5_climate_file(ROOT / "data/era5_regional_climates.json")
    regions = get_default_regions()
    mp = MonthlyNParams()
    ym_table = {key: get_calibrated_ym(key, mp) for key in REGIONS}

    print("One-at-a-time joint-prior bounds:")
    records = mc_oat_cases(regions, ym_table)
    print("WHC x soil-N-response structural grid:")
    records.extend(structural_cases(regions, ym_table))

    out = ROOT / "outputs/parameter_extreme_acceptance_sol.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    assert all(r["domain_pass"] for r in records)
    failures = [r for r in records if not r["monotone_pass"]]
    print(f"PARAMETER EXTREME AUDIT: COMPLETE ({len(records)} region-case checks)")
    if failures:
        print(f"  monotonic robustness threshold NOT MET in {len(failures)} checks")
        for record in failures:
            print(
                f"    {record['case']} {record['region']}: "
                f"{record['low_soc_loss_pct']:.6f} < "
                f"{record['high_soc_loss_pct']:.6f}"
            )
    else:
        print("  monotonic robustness threshold: PASS")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
