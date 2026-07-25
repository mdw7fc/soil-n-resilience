#!/usr/bin/env python3
"""Structural sensitivities for the two parameters lacking point estimates."""
from pathlib import Path
import copy
import csv
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

import numpy as np
from monthly_model_v3 import MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params
from soil_n_model import get_default_regions
from parameter_registry import (
    REGIONAL_PRICES, RegionalPrice, SOUTH_ASIA_FARMER_PAID_N_PRICE,
    SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY,
    WHC_MM_PER_SOC_PCT_LOW, WHC_MM_PER_SOC_PCT_30CM,
    WHC_MM_PER_SOC_PCT_HIGH)

REGIONS = [
    "north_america", "europe", "east_asia", "south_asia",
    "southeast_asia", "latin_america", "sub_saharan_africa",
    "fsu_central_asia",
]


def patch_era5():
    climate = json.loads((ROOT / "data/era5_regional_climates.json").read_text())
    for key, old in list(REGIONAL_CLIMATES.items()):
        new = climate[key]
        REGIONAL_CLIMATES[key] = MonthlyClimate(
            old.name, list(map(float, new["temp"])),
            list(map(float, new["precip"])), list(map(float, new["pet"])),
            old.planting_month, old.maturity_month)


def global_losses(regions, mp, eps_n, whc):
    scenario = copy.deepcopy(get_scenario_params()["S3"])
    scenario.eps_F_N = eps_n
    records = []
    for key in REGIONS:
        region = copy.deepcopy(regions[key])
        region.whc_sensitivity = whc
        ym = get_calibrated_ym(key, mp)
        frame = CoupledMonthlyModel(
            region=region, econ=scenario, region_key=key, t_max=30,
            yield_max_override=ym).run()
        records.append({
            "region": key, "weight": region.cropland_mha * float(
                frame.loc[frame.year == 0, "yield_tha"].iloc[0]),
            "loss10": 100 * (1 - float(
                frame.loc[frame.year == 10, "yield_fraction"].iloc[0])),
            "loss30": 100 * (1 - float(
                frame.loc[frame.year == 30, "yield_fraction"].iloc[0])),
        })
    weights = np.array([x["weight"] for x in records])
    weights /= weights.sum()
    return (
        float(np.dot(weights, [x["loss10"] for x in records])),
        float(np.dot(weights, [x["loss30"] for x in records])),
    )


def main():
    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()
    rows = []
    for whc in (WHC_MM_PER_SOC_PCT_LOW, WHC_MM_PER_SOC_PCT_30CM,
                WHC_MM_PER_SOC_PCT_HIGH):
        for eps_n in SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY:
            loss10, loss30 = global_losses(regions, mp, eps_n, whc)
            rows.append({
                "whc_mm_per_soc_pct": whc,
                "soil_n_demand_elasticity": eps_n,
                "global_loss_year10_pct": loss10,
                "global_loss_year30_pct": loss30,
            })
            print(whc, eps_n, round(loss10, 3), round(loss30, 3))
    out = ROOT / "outputs/structural_sensitivity_sol.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    # Price-convention sensitivity for the partial farm-budget outcome.
    from run_price_shock_analysis import farm_sweep_single
    price_rows = []
    price_cases = [
        ("south_asia", "market_replacement", 1.20),
        ("south_asia", "farmer_paid", SOUTH_ASIA_FARMER_PAID_N_PRICE),
        ("sub_saharan_africa", "audited_retail", 2.30),
        ("sub_saharan_africa", "old_SI_price", 1.40),
    ]
    for region_key, convention, n_price in price_cases:
        original = REGIONAL_PRICES[region_key]
        REGIONAL_PRICES[region_key] = RegionalPrice(
            n_price, original.crop_usd_per_t, convention)
        try:
            region = regions[region_key]
            ym = get_calibrated_ym(region_key, mp)
            for soc_pct in (50, 100):
                result = farm_sweep_single(
                    region, region_key, ym, mp, soc_pct, 1.0)
                price_rows.append({
                    "region": region_key, "price_convention": convention,
                    "nitrogen_usd_per_kg_n": n_price, "soc_pct": soc_pct,
                    "yield_change_pct": -result["yield_pen"],
                    "net_revenue_change_pct": result["margin_chg"],
                })
        finally:
            REGIONAL_PRICES[region_key] = original
    price_out = ROOT / "outputs/price_convention_sensitivity_sol.csv"
    with price_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(price_rows[0]))
        writer.writeheader()
        writer.writerows(price_rows)
    print(f"wrote {price_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
