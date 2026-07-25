#!/usr/bin/env python3
"""Write a machine-readable ledger of every live central model parameter."""
from dataclasses import fields
from pathlib import Path
import csv
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

from soil_n_model import CropParams, FeedbackParams, SOMPoolParams, get_default_regions
from monthly_model_v3 import MonthlyNParams
from coupled_econ_biophysical import (
    EconParams, REGIONAL_ECON_PARAMS, get_scenario_params,
    get_supply_constrained_scenarios)
from parameter_registry import (
    REGIONAL_PRICES, SOIL_N_RESPONSE_ELASTICITY_CENTRAL,
    SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY,
    WHC_MM_PER_SOC_PCT_30CM, WHC_MM_PER_SOC_PCT_LOW,
    WHC_MM_PER_SOC_PCT_HIGH)


def row(group, region, name, value, units, status, source, location, uncertainty):
    return dict(group=group, region=region, parameter=name, central_value=value,
                units=units, status=status, source=source,
                code_location=location, uncertainty_or_test=uncertainty)


def main():
    rows = []
    unit_map = {
        "q10": "ratio", "t_ref": "deg C", "t_min": "deg C",
        "leach_coeff": "per mm", "leach_base": "fraction month-1",
        "denitrif_base": "fraction month-1", "denitrif_wet_mult": "ratio",
        "immob_frac": "fraction", "max_uptake_frac": "fraction",
        "min_n_pool": "kg N ha-1", "yield_max": "t ha-1",
        "mitscherlich_c": "(kg N ha-1)-1", "yield_min": "t ha-1",
        "residue_grain_ratio": "ratio", "residue_c_fraction": "fraction",
        "residue_cn": "ratio", "grain_n_fraction": "fraction",
        "harvest_index": "fraction", "nue_apparent": "fraction",
    }
    for cls, group, location in (
        (MonthlyNParams, "monthly nitrogen", "code/model/monthly_model_v3.py"),
        (CropParams, "crop response", "code/model/soil_n_model.py"),
        (SOMPoolParams, "SOM turnover", "code/model/soil_n_model.py"),
        (FeedbackParams, "soil feedback", "code/model/soil_n_model.py"),
        (EconParams, "economic defaults", "code/model/coupled_econ_biophysical.py"),
    ):
        obj = cls()
        for f in fields(cls):
            rows.append(row(
                group, "global/default", f.name, getattr(obj, f.name),
                unit_map.get(f.name, "fraction or elasticity"),
                "literature/default or calibrated",
                "See manuscript/SI parameter citations and inline code notes",
                location, "MC prior, one-at-a-time test, or scenario comparison where listed"))

    regions = get_default_regions()
    regional_units = {
        "soc_initial": "t C ha-1", "cn_bulk": "ratio",
        "cropland_mha": "million ha", "synth_n_current": "kg N ha-1 yr-1",
        "pop_supported": "million people", "texture_class": "class",
        "whc_sensitivity": "mm per SOC percentage point",
        "water_stress_coeff": "fraction per mm",
        "baseline_water_deficit": "mm", "atm_n_deposition": "kg N ha-1 yr-1",
        "bnf_potential": "kg N ha-1 yr-1", "bnf_ramp_years": "yr",
        "residue_retention": "fraction", "yield_max_regional": "t ha-1",
        "mitscherlich_c_regional": "(kg N ha-1)-1",
        "yield_min_regional": "t ha-1", "root_shoot_c_ratio": "ratio",
        "cre_regional": "fraction",
    }
    skip = {"name"}
    for region, obj in regions.items():
        for f in fields(obj):
            if f.name in skip:
                continue
            status = "calibrated" if f.name in {
                "yield_max_regional", "cre_regional"} else "data/literature input"
            rows.append(row(
                "regional biophysical", region, f.name, getattr(obj, f.name),
                regional_units.get(f.name, "see code"), status,
                "Supplementary Table 1 and cited source family",
                "code/model/soil_n_model.py",
                "Regional comparison; selected terms included in MC"))
        for name, value in REGIONAL_ECON_PARAMS[region].items():
            rows.append(row(
                "regional economic", region, name, value, "elasticity",
                "literature-calibrated" if name != "eps_F_N" else "not estimated",
                "FDME/meta-elasticity sources; eps_F_N has no clean estimate",
                "code/model/coupled_econ_biophysical.py",
                "eps_F_PF halved sensitivity; eps_F_N structural range 0 to -1"))

    for region, price in REGIONAL_PRICES.items():
        rows.extend([
            row("regional price", region, "nitrogen_price",
                price.nitrogen_usd_per_kg_n, "USD kg N-1", "audited primitive",
                price.convention, "code/model/parameter_registry.py",
                "South Asia farmer-paid sensitivity; SSA retail evidence range"),
            row("regional price", region, "crop_price", price.crop_usd_per_t,
                "USD t-1", "audited primitive", "FAOSTAT producer-price convention",
                "code/model/parameter_registry.py",
                "Cost share is derived, never independently specified"),
        ])

    rows.extend([
        row("audited central", "global", "WHC_MM_PER_SOC_PCT_30CM",
            WHC_MM_PER_SOC_PCT_30CM, "mm per SOC percentage point",
            "literature-derived", "Minasny and McBratney 2018: 1.16 mm per 100 mm soil, scaled to 0-30 cm",
            "code/model/parameter_registry.py",
            f"{WHC_MM_PER_SOC_PCT_LOW} to {WHC_MM_PER_SOC_PCT_HIGH}"),
        row("audited central", "global", "eps_F_N",
            SOIL_N_RESPONSE_ELASTICITY_CENTRAL, "elasticity",
            "unsupported central value removed", "No clean empirical regional estimate",
            "code/model/parameter_registry.py",
            str(SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY)),
    ])

    for key, scenario in get_scenario_params().items():
        rows.append(row("scenario", key, "fert_price_shock",
                        scenario.fert_price_shock, "fraction",
                        "calibrated to global fertilizer reduction",
                        "model calibration", "code/model/coupled_econ_biophysical.py",
                        "severity sweep"))
    for key, scenario in get_supply_constrained_scenarios().items():
        rows.extend([
            row("scenario", key, "fert_supply_ceiling",
                scenario.fert_supply_ceiling, "fraction of baseline",
                "scenario assumption", "20 percent physical disruption",
                "code/model/coupled_econ_biophysical.py", "SC1 vs SC2"),
            row("scenario", key, "fert_capacity_recovery_years",
                scenario.fert_capacity_recovery_years, "yr",
                "scenario assumption", "capacity recovery scenario",
                "code/model/coupled_econ_biophysical.py", "SC1 vs SC2"),
        ])

    out_csv = ROOT / "PARAMETER_LEDGER_sol.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out_md = ROOT / "PARAMETER_LEDGER_sol.md"
    counts = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    lines = [
        "# Parameter ledger (SOL audit)",
        "",
        f"This ledger contains {len(rows)} live central, regional, price, and scenario entries.",
        "The CSV is authoritative; this page records the audit decisions.",
        "",
        "## Decisions",
        "",
        "- Regional nitrogen cost shares are derived from nitrogen price, crop price, modeled baseline yield, and nitrogen rate. No cost share is a primitive input.",
        "- The central SOC-to-water conversion is 3.48 mm per SOC percentage point for 0-30 cm; 2.32 and 8.40 are sensitivity bounds.",
        "- The fertilizer-demand response to soil N is zero centrally because no defensible regional estimate was identified; 0, -0.25, -0.50, and -1.0 are structural sensitivities.",
        "- Calibrated yield ceilings and year-2 no-synthetic-N yields are regenerated by `code/repro/make_table_s4_sol.py`.",
        "- Economic output is crop revenue net of nitrogen-fertilizer expenditure, not whole-farm gross margin.",
        "",
        "## Entry counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
