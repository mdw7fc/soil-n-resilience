#!/usr/bin/env python3
"""Build the exhaustive SOL parameter ledger and numeric-literal audit.

The semantic ledger contains every result-affecting primitive, forcing,
scenario control, calibration/solver setting, analysis design value,
uncertainty prior, random seed, and acceptance tolerance in the submitted
execution path. The companion literal audit lists every numeric literal in
model and reproduction code so omissions can be checked mechanically.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MODEL = HERE.parent / "model"
sys.path.insert(0, str(MODEL))

from parameter_registry import (
    BASELINE_BNF_KG_N_HA_YR,
    BNF_COMPONENTS,
    REGIONAL_PRICES,
    RESIDUE_C_FRACTION,
    SOC_T_C_HA_PER_PERCENT_30CM,
    SOIL_N_RESPONSE_ELASTICITY_CENTRAL,
    SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY,
    SOUTH_ASIA_FARMER_PAID_N_PRICE,
    WATER_STRESS_GAIN_SAT_SOC_PCT,
    WATER_STRESS_MIN_FACTOR,
    WATER_STRESS_SOFTPLUS_EPS_MM,
    WHC_MM_PER_SOC_PCT_30CM,
    WHC_MM_PER_SOC_PCT_HIGH,
    WHC_MM_PER_SOC_PCT_LOW,
)
from soil_n_model import (
    CropParams,
    FeedbackParams,
    SOMPoolParams,
    get_default_regions,
    som_params_for_region,
)
from monthly_model_v3 import (
    FAOSTAT_TARGETS,
    MonthlyNParams,
    REGIONAL_CLIMATES,
    apply_era5_climate_file,
)
from coupled_monthly import get_calibrated_ym
from som_4pool_monthly import FourPoolParams
from coupled_4pool import CLAY_SILT_DEFAULT
from coupled_econ_biophysical import (
    EconParams,
    REGIONAL_ECON_PARAMS,
    calibrate_price_shock,
    get_scenario_params,
    get_supply_constrained_scenarios,
)


REGIONS = [
    "north_america", "europe", "east_asia", "south_asia",
    "southeast_asia", "latin_america", "sub_saharan_africa",
    "fsu_central_asia",
]

FIELDNAMES = [
    "parameter_id", "group", "scope", "region_or_case", "parameter",
    "central_value", "units", "dependency", "calibration_status", "source",
    "code_location", "uncertainty_treatment", "evidence_role",
]


def encode(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def add(rows, group, scope, case, parameter, value, units, dependency,
        calibration_status, source, location, uncertainty, evidence_role):
    rows.append({
        "parameter_id": f"P{len(rows) + 1:04d}",
        "group": group,
        "scope": scope,
        "region_or_case": case,
        "parameter": parameter,
        "central_value": encode(value),
        "units": units,
        "dependency": dependency,
        "calibration_status": calibration_status,
        "source": source,
        "code_location": location,
        "uncertainty_treatment": uncertainty,
        "evidence_role": evidence_role,
    })


UNITS = {
    # SOM
    "f_active": "fraction of SOC", "f_slow": "fraction of SOC",
    "f_passive": "fraction of SOC", "k_active": "yr-1",
    "k_slow": "yr-1", "k_passive": "yr-1", "cn_active": "mass ratio",
    "cn_slow": "mass ratio", "cn_passive": "mass ratio",
    "h_active_to_slow": "fraction", "h_slow_to_passive": "fraction",
    "regime": "label",
    # Crop/soil feedback
    "yield_max": "t grain ha-1", "mitscherlich_c": "(kg N ha-1)-1",
    "yield_min": "t grain ha-1", "residue_grain_ratio": "mass ratio",
    "residue_c_fraction": "fraction", "residue_cn": "mass ratio",
    "grain_n_fraction": "fraction", "harvest_index": "fraction",
    "nue_apparent": "fraction", "residue_feedback": "boolean",
    "physical_feedback": "boolean", "physical_strength": "multiplier",
    "expansion_feedback": "boolean", "cn_coupling_feedback": "boolean",
    "cre_to_active": "fraction",
    "cre_to_slow": "fraction",
    # Monthly N
    "q10": "ratio per 10 deg C", "t_ref": "deg C", "t_min": "deg C",
    "moist_opt_lo": "precip/PET ratio", "moist_opt_hi": "precip/PET ratio",
    "moist_min": "multiplier", "moist_waterlog": "multiplier",
    "leach_coeff": "per 100 mm drainage", "leach_base": "fraction month-1",
    "denitrif_base": "fraction month-1", "denitrif_wet_mult": "multiplier",
    "immob_frac": "fraction", "max_uptake_frac": "fraction",
    "min_n_pool": "kg N ha-1",
    # Regional
    "soc_initial": "t C ha-1 (0-30 cm)", "cn_bulk": "mass ratio",
    "cropland_mha": "million ha", "synth_n_current": "kg N ha-1 yr-1",
    "pop_supported": "million people", "texture_class": "ordinal class",
    "whc_sensitivity": "mm per SOC percentage point",
    "water_stress_coeff": "fraction per mm",
    "baseline_water_deficit": "mm", "atm_n_deposition": "kg N ha-1 yr-1",
    "bnf_potential": "kg N ha-1 yr-1", "bnf_ramp_years": "yr",
    "residue_retention": "fraction", "yield_max_regional": "t grain ha-1",
    "mitscherlich_c_regional": "(kg N ha-1)-1",
    "yield_min_regional": "t grain ha-1", "root_shoot_c_ratio": "mass ratio",
    "cre_regional": "fraction",
    # Economic
    "fert_price_shock": "proportional change", "eps_F_PF": "elasticity",
    "eps_F_PY": "elasticity", "eps_F_N": "elasticity", "eta": "elasticity",
    "alpha": "elasticity", "eps_LD_PL": "elasticity",
    "eps_LD_PY": "elasticity", "eps_LS_PL": "elasticity",
    "fert_supply_ceiling": "fraction of baseline",
    "fert_capacity_recovery_years": "yr",
    "price_relaxes_with_recovery": "boolean",
}

FOURPOOL_SOURCES = {
    "f_pom": "Lavallee et al. 2020 (POM ~20-30% of SOC)",
    "f_maom": "Cotrufo et al. 2019 (MAOM 60-75%)",
    "f_mbc": "Anderson & Domsch 1989",
    "f_dom": "fast-cycling intermediate, documented inline",
    "cn_pom": "plant-derived C:N range 15-40, documented inline",
    "cn_dom": "documented inline",
    "cn_mbc": "microbial biomass C:N, documented inline",
    "cn_maom": "necromass+sorbed C:N 8-12, documented inline",
    "k_pom_to_dom": "Heckman et al. 2022 (POM MRT ~12 yr)",
    "k_maom_desorption": "Georgiou et al. 2022 (MAOM MRT ~140 yr)",
    "k_mbc_turnover": "~4 month MBC MRT, documented inline",
    "k_dom_uptake": "documented inline",
    "k_dom_sorption": "documented inline",
    "cue_max": "Wieder et al. 2015",
    "cue_min": "Manzoni et al. 2012",
    "cue_km_n": "half-saturation of CUE-N response, documented inline",
    "f_necro_to_maom": "Liang et al. 2017 (in-vivo pathway)",
    "f_necro_to_pom": "structural necromass fraction, documented inline",
    "f_necro_to_dom": "labile necromass fraction, documented inline",
    "qmax_per_claysilt": "Georgiou et al. 2022 (range 30-80 t C/ha)",
    "priming_sensitivity": "documented inline",
}

# F-031: per-parameter units and honest uncertainty treatment for the
# 4-pool rows (fifth audit round: units were the generic fallback and the
# treatment column implied these parameters were varied in the texture
# sensitivity, which varies clay_silt_fraction only).
FOURPOOL_UNITS = {
    "f_pom": "fraction of initial SOC",
    "f_maom": "fraction of initial SOC",
    "f_mbc": "fraction of initial SOC",
    "f_dom": "fraction of initial SOC",
    "cn_pom": "g C / g N (dimensionless C:N)",
    "cn_dom": "g C / g N (dimensionless C:N)",
    "cn_mbc": "g C / g N (dimensionless C:N)",
    "cn_maom": "g C / g N (dimensionless C:N)",
    "k_pom_to_dom": "yr^-1",
    "k_maom_desorption": "yr^-1",
    "k_mbc_turnover": "yr^-1",
    "k_dom_uptake": "yr^-1",
    "k_dom_sorption": "yr^-1",
    "cue_max": "g C assimilated / g C taken up (dimensionless)",
    "cue_min": "g C assimilated / g C taken up (dimensionless)",
    "cue_km_n": "fraction of baseline N availability (dimensionless)",
    "f_necro_to_maom": "fraction of MBC turnover flux",
    "f_necro_to_pom": "fraction of MBC turnover flux",
    "f_necro_to_dom": "fraction of MBC turnover flux",
    "qmax_per_claysilt": "t C/ha per unit clay+silt fraction",
    "priming_sensitivity": "dimensionless multiplier",
}

SOURCES = {
    "f_active": "Century/RothC structural defaults",
    "f_slow": "Century/RothC structural defaults",
    "f_passive": "Century/RothC structural defaults",
    "k_active": "Century/RothC-informed turnover",
    "k_slow": "equilibrium calibration; tropical multiplier from Laub et al. 2024",
    "k_passive": "equilibrium calibration; tropical multiplier from Laub et al. 2024",
    "cn_active": "Century/RothC pool stoichiometry",
    "cn_slow": "Century/RothC pool stoichiometry",
    "cn_passive": "Century/RothC pool stoichiometry",
    "h_active_to_slow": "Century/RothC humification structure",
    "h_slow_to_passive": "Century/RothC humification structure",
    "yield_max": "disabled fallback; canonical regional values solved at runtime",
    "mitscherlich_c": "fixed cross-region response curvature",
    "yield_min": "physiological nonnegative fallback",
    "residue_grain_ratio": "IPCC crop-residue relationship",
    "residue_c_fraction": "crop-residue carbon-content convention",
    "residue_cn": "production-weighted cereal-residue literature",
    "grain_n_fraction": "grain N-content literature",
    "harvest_index": "cereal harvest-index literature",
    "nue_apparent": "annual legacy-engine parameter; not used by monthly path",
    "q10": "standard decomposition-temperature response",
    "t_ref": "model reference temperature",
    "t_min": "model decomposition cutoff",
    "moist_opt_lo": "structural moisture-response assumption",
    "moist_opt_hi": "structural moisture-response assumption",
    "moist_min": "structural moisture-response assumption",
    "moist_waterlog": "structural moisture-response assumption",
    "leach_coeff": "monthly N-loss parameterization",
    "leach_base": "monthly N-loss parameterization",
    "denitrif_base": "monthly N-loss parameterization",
    "denitrif_wet_mult": "monthly N-loss parameterization",
    "immob_frac": "monthly N-balance parameterization",
    "max_uptake_frac": "calibrated in-season capture component",
    "min_n_pool": "numerical/biophysical mineral-N floor",
    "cre_to_active": "Century-style allocation assumption",
    "cre_to_slow": "Century-style allocation assumption",
}

REGIONAL_SOURCES = {
    "soc_initial": "ISRIC SoilGrids cropland regionalization",
    "cn_bulk": "regional soil C:N literature/assumption",
    "cropland_mha": "FAOSTAT land-use aggregation",
    "synth_n_current": "FAOSTAT/IFA regional N-use aggregation",
    "pop_supported": "crop-calorie allocation; legacy annual output only",
    "texture_class": "regional dominant-texture simplification",
    "whc_sensitivity": "derived 0-30 cm Minasny and McBratney 2018 scaling",
    "water_stress_coeff": "regional structural assumption",
    "baseline_water_deficit": "regional structural assumption",
    "atm_n_deposition": "Dentener/Vet/EMEP/NADP source family",
    "bnf_potential": "derived from BNF component primitives",
    "bnf_ramp_years": "legacy managed-transition assumption",
    "residue_retention": "regional residue-management literature",
    "yield_max_regional": "disabled sentinel; no independent ceiling",
    "mitscherlich_c_regional": "zero sentinel; global c used",
    "yield_min_regional": "long-term unfertilized experiments/historical records",
    "root_shoot_c_ratio": "Bolinder/Katterer/Johnson source family",
    "cre_regional": "calibrated to approximate regional SOC equilibrium",
}


def dataclass_rows(rows):
    classes = [
        (SOMPoolParams, "SOM default", "canonical and fallback",
         "code/model/soil_n_model.py::SOMPoolParams"),
        (CropParams, "crop default", "canonical and legacy",
         "code/model/soil_n_model.py::CropParams"),
        (FeedbackParams, "feedback default", "canonical and legacy",
         "code/model/soil_n_model.py::FeedbackParams"),
        (MonthlyNParams, "monthly N default", "canonical",
         "code/model/monthly_model_v3.py::MonthlyNParams"),
        (EconParams, "economic default", "scenario fallback",
         "code/model/coupled_econ_biophysical.py::EconParams"),
        # F-030: the microbially-explicit 4-pool scheme's parameters were
        # hardcoded outside the ledger (fourth audit round); every field is
        # now enumerated here, with the uniform texture assumption below.
        (FourPoolParams, "4-pool structural default", "structural sensitivity",
         "code/model/som_4pool_monthly.py::FourPoolParams"),
    ]
    prior_names = {
        "max_uptake_frac": "MC absolute prior 0.60-0.90 and NUE sweep 0.45-0.95",
        "mitscherlich_c": "MC multiplier 0.70-1.30",
        "k_slow": "MC multiplier 0.60-1.40",
    }
    for cls, group, scope, location in classes:
        obj = cls()
        for field in fields(cls):
            name = field.name
            value = getattr(obj, name)
            if name == "regime":
                status = "classification label"
            elif name in {"yield_max", "yield_min", "nue_apparent"}:
                status = "fallback/legacy"
            else:
                status = "externally specified"
            if cls is FourPoolParams:
                source = FOURPOOL_SOURCES.get(name, "documented inline and in SI")
                prior = ("fixed; not varied in submitted analyses (the "
                         "texture sensitivity varies clay_silt_fraction only, "
                         "Supplementary Note 2)")
                path = "structural sensitivity (Supplementary Note 2)"
                units = FOURPOOL_UNITS.get(name, "dimensionless")
            else:
                source = SOURCES.get(name, "documented inline and in SI")
                prior = prior_names.get(
                    name, "fixed; not varied in submitted analyses")
                path = ("central path" if scope.startswith("canonical")
                        else "fallback")
                units = UNITS.get(name, "fraction or elasticity")
            add(
                rows, group, scope, "global/default", name, value,
                units, "primitive",
                status, source,
                f"{location}.{name}",
                prior,
                path,
            )


def fourpool_extra_rows(rows):
    add(rows, "4-pool structural default", "structural sensitivity",
        "global/default", "clay_silt_fraction", CLAY_SILT_DEFAULT,
        "fraction", "assumption", "externally specified",
        "uniform texture assumption (RegionParams carries no texture "
        "fraction); MAOM sorption ceiling = qmax_per_claysilt x clay_silt; "
        "sensitivity 0.35-0.75 reported in Supplementary Note 2",
        "code/model/coupled_4pool.py::CLAY_SILT_DEFAULT",
        "varied 0.35-0.75 (Supplementary Note 2)",
        "structural sensitivity (Supplementary Note 2)")


def regional_rows(rows):
    regions = get_default_regions()
    for key in REGIONS:
        obj = regions[key]
        for field in fields(obj):
            name = field.name
            if name == "name":
                continue
            value = getattr(obj, name)
            dependency = "derived" if name == "bnf_potential" else "primitive"
            if name == "yield_max_regional":
                status = "disabled sentinel"
            elif name == "cre_regional":
                status = "calibrated"
            elif name == "bnf_potential":
                status = "derived, not independently specified"
            else:
                status = "externally specified"
            if name == "whc_sensitivity":
                uncertainty = (
                    f"deterministic bounds {WHC_MM_PER_SOC_PCT_LOW}-"
                    f"{WHC_MM_PER_SOC_PCT_HIGH}"
                )
            elif name == "cre_regional":
                uncertainty = "MC multiplier 0.40-1.80"
            elif name == "residue_retention":
                uncertainty = "MC multiplier 0.80-1.15"
            elif name == "cn_bulk":
                uncertainty = "one-at-a-time +/-20% mineralizable-N test"
            elif name == "yield_max_regional":
                uncertainty = "not applicable; runtime calibration is authoritative"
            else:
                uncertainty = "fixed; not varied in submitted analyses"
            add(
                rows, "regional biophysical", "canonical", key, name, value,
                UNITS.get(name, "see parameter definition"), dependency, status,
                REGIONAL_SOURCES.get(name, "regional source documented inline"),
                f"code/model/soil_n_model.py::get_default_regions[{key}].{name}",
                uncertainty, "central path",
            )

        for name, value in REGIONAL_ECON_PARAMS[key].items():
            if name == "eps_F_N":
                uncertainty = str(SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY)
                status = "unsupported estimate set to zero centrally"
            elif name == "eps_F_PF":
                uncertainty = "halved one-at-a-time test and MC multiplier 0.50-1.50"
                status = "literature-calibrated"
            elif name == "eta":
                uncertainty = "MC multiplier 0.60-1.40"
                status = "literature-calibrated"
            else:
                uncertainty = "fixed; not varied in submitted analyses"
                status = "literature-calibrated or structural"
            add(
                rows, "regional economic", "canonical", key, name, value,
                "elasticity", "primitive", status,
                "FDME/meta-elasticity and CGE land-response source families",
                f"code/model/coupled_econ_biophysical.py::REGIONAL_ECON_PARAMS[{key}].{name}",
                uncertainty, "central or scenario path",
            )


def shared_and_bnf_rows(rows):
    shared = [
        ("SOC_T_C_HA_PER_PERCENT_30CM", SOC_T_C_HA_PER_PERCENT_30CM,
         "t C ha-1 per SOC percentage point", "derived",
         "bulk density 1.3 g cm-3 x 0.30 m depth x 1% C",
         "single canonical conversion", "central path"),
        ("RESIDUE_C_FRACTION", RESIDUE_C_FRACTION, "fraction", "primitive",
         "crop-residue carbon-content convention", "fixed", "central path"),
        ("WATER_STRESS_GAIN_SAT_SOC_PCT", WATER_STRESS_GAIN_SAT_SOC_PCT,
         "SOC percentage points", "primitive",
         "structural saturation-scale assumption", "fixed", "central path"),
        ("WATER_STRESS_SOFTPLUS_EPS_MM", WATER_STRESS_SOFTPLUS_EPS_MM,
         "mm", "primitive", "seasonal/spatial deficit smoothing assumption",
         "fixed", "central path"),
        ("WATER_STRESS_MIN_FACTOR", WATER_STRESS_MIN_FACTOR, "fraction",
         "primitive", "numerical/physiological water-stress floor", "fixed",
         "central path"),
        ("WHC_MM_PER_SOC_PCT_30CM", WHC_MM_PER_SOC_PCT_30CM,
         "mm per SOC percentage point", "derived",
         "Minasny and McBratney 2018 scaled from 0-10 to 0-30 cm",
         f"bounds {WHC_MM_PER_SOC_PCT_LOW}-{WHC_MM_PER_SOC_PCT_HIGH}",
         "central path"),
        ("SOIL_N_RESPONSE_ELASTICITY_CENTRAL",
         SOIL_N_RESPONSE_ELASTICITY_CENTRAL, "elasticity", "primitive",
         "no clean empirical regional estimate; zero neutral central value",
         str(SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY), "central path"),
    ]
    for name, value, units, dependency, source, uncertainty, role in shared:
        add(
            rows, "shared physical/economic", "canonical", "global", name,
            value, units, dependency,
            "derived" if dependency == "derived" else "externally specified",
            source, f"code/model/parameter_registry.py::{name}",
            uncertainty, role,
        )

    for key in REGIONS:
        for name, value in BNF_COMPONENTS[key].items():
            units = {
                "legume_frac": "fraction", "net_n_credit": "kg N ha-1",
                "legume_yield_ceq": "t cereal-equivalent ha-1",
                "free_living_bnf": "kg N ha-1 yr-1",
            }[name]
            add(
                rows, "BNF primitive", "canonical", key, name, value, units,
                "primitive", "externally specified",
                "FAOSTAT crop-area shares and Herridge/Peoples/Ladha source family",
                f"code/model/parameter_registry.py::BNF_COMPONENTS[{key}].{name}",
                "fixed; BNF correlation is descriptive, not causal",
                "central N supply",
            )
        add(
            rows, "BNF derived", "canonical", key, "baseline_bnf",
            BASELINE_BNF_KG_N_HA_YR[key], "kg N ha-1 yr-1", "derived",
            "derived, not independently specified",
            "legume_frac*net_n_credit/(1-legume_frac)+free_living_bnf",
            f"code/model/parameter_registry.py::BASELINE_BNF_KG_N_HA_YR[{key}]",
            "inherits uncertainty of component assumptions",
            "central N supply",
        )


def climate_calibration_rows(rows):
    fallback = {
        k: {
            "temp": list(v.temp), "precip": list(v.precip), "pet": list(v.pet),
            "planting_month": v.planting_month,
            "maturity_month": v.maturity_month,
        }
        for k, v in REGIONAL_CLIMATES.items()
    }
    era5 = json.loads((ROOT / "data/era5_regional_climates.json").read_text())
    for key in REGIONS:
        for variable, units in (
            ("temp", "12 monthly deg C"),
            ("precip", "12 monthly mm"),
            ("pet", "12 monthly mm"),
        ):
            add(
                rows, "climate forcing", "canonical", key,
                f"ERA5_{variable}_monthly", era5[key][variable], units,
                "primitive forcing", "observed/reanalysis forcing",
                "ERA5 2001-2020 benchmark-location normals via Open-Meteo",
                f"data/era5_regional_climates.json::{key}.{variable}",
                "expert-profile climate swap sensitivity",
                "central forcing",
            )
            add(
                rows, "climate forcing", "sensitivity/fallback", key,
                f"expert_{variable}_monthly", fallback[key][variable], units,
                "primitive forcing", "expert representative profile",
                "original representative regional climatology",
                f"code/model/monthly_model_v3.py::REGIONAL_CLIMATES[{key}].{variable}",
                "compared against ERA5; not central",
                "Class B climate sensitivity",
            )
        for name in ("planting_month", "maturity_month"):
            add(
                rows, "crop calendar", "canonical", key, name,
                fallback[key][name], "calendar month (1-12)", "primitive",
                "externally specified", "representative dominant crop calendar",
                f"code/model/monthly_model_v3.py::REGIONAL_CLIMATES[{key}].{name}",
                "fixed; not varied", "central forcing",
            )
        add(
            rows, "yield calibration", "canonical", key, "FAOSTAT_target",
            FAOSTAT_TARGETS[key], "t grain ha-1", "primitive target",
            "observed calibration target",
            "production-weighted FAOSTAT cereal yield",
            f"code/model/monthly_model_v3.py::FAOSTAT_TARGETS[{key}]",
            "fixed target; model-structure sensitivity assessed separately",
            "calibration",
        )

    apply_era5_climate_file(ROOT / "data/era5_regional_climates.json")
    for key in REGIONS:
        ym = get_calibrated_ym(key, MonthlyNParams())
        add(
            rows, "yield calibration", "canonical", key,
            "calibrated_y_max", ym, "t grain ha-1", "derived/calibrated",
            "Brent root solution",
            "year-2 monthly-model yield matched to FAOSTAT target",
            "code/model/coupled_monthly.py::get_calibrated_ym",
            "recalibrated within declared NUE sweep; c varied in MC with y_max held",
            "central calibrated parameter",
        )


def scenario_price_analysis_rows(rows):
    scenarios = {}
    scenarios.update(get_scenario_params())
    scenarios.update(get_supply_constrained_scenarios())
    for key, obj in scenarios.items():
        for field in fields(EconParams):
            name = field.name
            value = getattr(obj, name)
            status = (
                "calibrated to 20% S1 global fertilizer reduction"
                if name == "fert_price_shock"
                else "scenario assumption"
            )
            uncertainty = {
                "fert_price_shock": "severity sweep 0-300%; MC multiplier 0.50-1.50",
                "eps_F_N": str(SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY),
                "fert_supply_ceiling": "SC1/SC2 conditional 20% physical cap",
                "fert_capacity_recovery_years": "SC1 zero vs SC2 20 yr",
            }.get(name, "scenario comparison or fixed")
            add(
                rows, "economic scenario", "canonical/conditional", key, name,
                value, UNITS.get(name, "see definition"), "primitive or derived",
                status, "scenario design",
                f"code/model/coupled_econ_biophysical.py::scenario[{key}].{name}",
                uncertainty, "Class A central S3 or Class B SC1/SC2",
            )

    for key, price in REGIONAL_PRICES.items():
        add(
            rows, "regional price", "four-region financial subset", key,
            "nitrogen_price", price.nitrogen_usd_per_kg_n, "USD kg N-1",
            "primitive", "externally specified", price.convention,
            f"code/model/parameter_registry.py::REGIONAL_PRICES[{key}].nitrogen",
            (
                "SSA old-SI 1.40 sensitivity" if key == "sub_saharan_africa"
                else (
                    f"South Asia farmer-paid {SOUTH_ASIA_FARMER_PAID_N_PRICE}"
                    if key == "south_asia" else "fixed"
                )
            ),
            "Class B partial-budget result",
        )
        add(
            rows, "regional price", "four-region financial subset", key,
            "crop_price", price.crop_usd_per_t, "USD t-1", "primitive",
            "externally specified", "FAOSTAT/producer-price convention",
            f"code/model/parameter_registry.py::REGIONAL_PRICES[{key}].crop",
            "fixed; financial ranking restricted to four audited pairs",
            "Class B partial-budget result",
        )
        add(
            rows, "regional price", "four-region financial subset", key,
            "nitrogen_price_in_yield_units",
            price.nitrogen_usd_per_kg_n / price.crop_usd_per_t,
            "t crop per kg N", "derived",
            "derived, not independently specified",
            "nitrogen USD kg-1 divided by crop USD t-1",
            "code/model/parameter_registry.py::nitrogen_price_in_yield_units",
            "inherits price-convention sensitivity",
            "Class B partial-budget result",
        )

    algorithms = [
        ("SOC layer depth", 0.30, "m", "physical conversion"),
        ("SOC conversion bulk density", 1.30, "Mg m-3", "physical conversion"),
        ("kg_per_t", 1000.0, "kg t-1", "unit conversion"),
        ("fertilizer split at planting", 0.33, "fraction", "fertilizer timing"),
        ("fertilizer split midseason", 0.67, "fraction", "fertilizer timing"),
        ("crop demand peak position", 0.60, "fraction of season", "demand profile"),
        ("crop demand profile width", 0.15, "fraction of season", "demand profile"),
        ("SOM input to active", 0.90, "fraction", "pool update"),
        ("SOM input to slow", 0.10, "fraction", "pool update"),
        ("initial mineral N", 12.0, "kg N ha-1", "initial condition"),
        ("spinup maximum", 2000, "yr", "solver"),
        ("spinup convergence window", 50, "yr", "solver"),
        ("spinup minimum before convergence", 100, "yr", "solver"),
        ("spinup fractional tolerance", 0.002, "fraction over 50 yr", "solver"),
        ("y_max Brent lower bound", 1.0, "t ha-1", "calibration solver"),
        ("y_max Brent upper bound", 50.0, "t ha-1", "calibration solver"),
        ("y_max Brent xtol", 0.01, "t ha-1", "calibration solver"),
        ("price-shock search lower bound", 0.01, "proportional change", "calibration solver"),
        ("price-shock search upper bound", 20.0, "proportional change", "calibration solver"),
        ("price-shock bisection iterations", 50, "iterations", "calibration solver"),
        ("global fertilizer-reduction target", 0.20, "fraction", "scenario calibration"),
        ("equilibrium denominator epsilon", 1e-10, "dimensionless", "numerical tolerance"),
        ("mineralization log floor", 1e-6, "kg N ha-1 yr-1", "numerical tolerance"),
        ("cap binding relative tolerance", 1e-9, "fraction", "numerical tolerance"),
    ]
    for name, value, units, role in algorithms:
        add(
            rows, "algorithm/solver", "canonical", "global", name, value,
            units, "primitive algorithm setting", "externally specified",
            "model algorithm or unit identity",
            "code/model/monthly_model_v3.py or coupled_monthly.py or coupled_econ_biophysical.py",
            "fixed; boundary/domain acceptance tests where applicable", role,
        )

    designs = [
        ("farm SOC grid", list(range(10, 205, 5)), "% regional equilibrium SOC",
         "Figure 1"),
        ("sustained SOC grid", list(range(10, 205, 10)), "% regional equilibrium SOC",
         "Figure 2a"),
        ("MC SOC levels", [50, 100, 150], "% regional equilibrium SOC", "MC"),
        ("farm price shock", 1.0, "proportional change", "Figure 1"),
        ("severity shocks", [0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3],
         "proportional change", "Figure S11"),
        ("N capture sweep", [.45, .55, .65, .75, .85, .95], "fraction",
         "Figure S10"),
        ("central run horizon", 30, "yr", "main trajectories"),
        ("farm sustained-gradient horizon", 10, "yr", "Figure 2a"),
        ("MC ensemble size", 1000, "draws", "Figure S9"),
        ("MC random seed", 20260424, "integer seed", "Figure S9"),
        ("summary percentiles", [5, 25, 50, 75, 95], "percentile", "MC"),
    ]
    for name, value, units, role in designs:
        add(
            rows, "analysis design", "canonical/sensitivity", role, name, value,
            units, "primitive analysis setting", "pre-specified analysis design",
            "figure/scenario design",
            "code/repro/run_price_shock_analysis.py, make_figure_s10.py, make_figure_s11.py, run_mc_ensemble.py",
            "explicit sweep or fixed reproducibility setting", role,
        )

    priors = {
        "max_uptake_frac": ("absolute", .75, .075, .60, .90),
        "mitscherlich_c_mult": ("multiplier", 1, .15, .70, 1.30),
        "k_slow_mult": ("multiplier", 1, .20, .60, 1.40),
        "cre_regional_mult": ("multiplier", 1, .30, .40, 1.80),
        "residue_retention_mult": ("multiplier", 1, .10, .80, 1.15),
        "eps_F_PF_mult": ("multiplier", 1, .30, .50, 1.50),
        "eta_mult": ("multiplier", 1, .25, .60, 1.40),
        "fert_price_shock_mult": ("multiplier", 1, .25, .50, 1.50),
    }
    for name, (mode, mean, sd, low, high) in priors.items():
        for part, value in (
            ("distribution", "truncated normal"), ("mode", mode),
            ("mean", mean), ("sd", sd), ("lower", low), ("upper", high),
        ):
            add(
                rows, "uncertainty prior", "MC sensitivity", name,
                f"{name}.{part}", value,
                "label" if isinstance(value, str) else "absolute or multiplier",
                "primitive prior setting", "pre-specified joint prior",
                "sensitivity range, not a posterior probability distribution",
                f"code/repro/run_mc_ensemble.py::PRIORS[{name}].{part}",
                "directly sampled in 1,000-draw ensemble", "Class B sensitivity",
            )

    thresholds = [
        ("cross-document numeric tolerance", .05, "percentage points"),
        ("zero-shock max drift", .001, "yield fraction over 30 yr"),
        ("market-clearing residual", 1e-10, "log points"),
        ("MC robustness frequency", .95, "fraction of relevant draws"),
        ("monotonic SOC low level", 50, "% regional mean SOC"),
        ("monotonic SOC high level", 150, "% regional mean SOC"),
    ]
    for name, value, units in thresholds:
        add(
            rows, "acceptance threshold", "test/evidence", "global", name,
            value, units, "primitive acceptance rule", "frozen prospectively",
            "EVIDENTIARY_STANDARD_sol.md",
            "EVIDENTIARY_STANDARD_sol.md::Prospective acceptance thresholds",
            "not varied", "accept/reject decision",
        )


def benchmark_and_spatial_rows(rows):
    """Record live design choices outside the regional coupled-model engine."""
    benchmark_values = [
        ("Broadbalk Nil observed yield", 1.00, "t grain ha-1",
         "Rothamsted 2000-2022 handout; approximate value",
         "code/repro/make_broadbalk_benchmark.py::YIELD", "Class C benchmark"),
        ("Broadbalk N3PKMg observed yield", 8.50, "t grain ha-1",
         "midpoint of reported 8-9 t ha-1 range",
         "code/repro/make_broadbalk_benchmark.py::YIELD", "Class C benchmark"),
        ("Broadbalk Nil modeled yield", 1.48, "t grain ha-1",
         "2000-2015 mean from documented Broadbalk benchmark run",
         "code/repro/make_broadbalk_benchmark.py::YIELD", "Class C benchmark"),
        ("Broadbalk N3PKMg modeled yield", 10.65, "t grain ha-1",
         "2000-2015 mean from documented Broadbalk benchmark run",
         "code/repro/make_broadbalk_benchmark.py::YIELD", "Class C benchmark"),
        ("hindcast target global N reduction", .15, "fraction",
         "2022 crisis scenario design",
         "code/repro/make_hindcast_benchmark.py::calibrate_price_shock",
         "Class C directional benchmark"),
        ("hindcast elasticity multipliers", [.5, 1.0, 2.0], "multipliers",
         "pre-specified one-at-a-time sensitivity",
         "code/repro/make_hindcast_benchmark.py::scenarios",
         "Class C directional benchmark"),
    ]
    for name, value, units, source, location, role in benchmark_values:
        add(
            rows, "benchmark design", "benchmark", "global", name, value,
            units, "primitive benchmark setting", "externally specified",
            source, location, "fixed or explicitly swept", role,
        )

    spatial_values = [
        ("year_start", 2018, "calendar year", "input averaging window"),
        ("year_end", 2020, "calendar year", "input averaging window"),
        ("n_intensity_cap", 300.0, "kg N ha-1", "continuous exposure scaling"),
        ("combined_weight_intensity", .5, "fraction", "sensitivity-only index"),
        ("combined_weight_reliance", .5, "fraction", "sensitivity-only index"),
        ("combined_low_threshold", .33, "index", "sensitivity-only index"),
        ("combined_high_threshold", .66, "index", "sensitivity-only index"),
        ("high_intensity_pathway", 150.0, "kg N ha-1", "default classification"),
        ("high_import_reliance", .70, "fraction", "default classification"),
        ("material_stake_floor", 25.0, "kg N ha-1", "default classification"),
        ("low_intensity_threshold", 50.0, "kg N ha-1", "default classification"),
        ("low_import_reliance", .30, "fraction", "default classification"),
        ("stake_floor_sensitivity", [10.0, 25.0, 50.0], "kg N ha-1",
         "threshold sensitivity"),
        ("reexport_ratio_flag", 3.0, "ratio", "re-export QA flag"),
        ("reexport_trade_floor", 50000.0, "t N", "re-export QA flag"),
        ("reexport_consumption_floor", 10000.0, "t N", "re-export QA flag"),
        ("FAOSTAT land-unit conversion", 1000.0, "ha per 1000 ha",
         "unit conversion"),
        ("buffer country minimum cropland", .1, "Mha", "country eligibility"),
        ("buffer quantile cutoffs", [1/3, 2/3], "country quantiles",
         "default classification"),
        ("buffer quantile sensitivity", [.30, 1/3, .40], "country quantiles",
         "threshold sensitivity"),
    ]
    for name, value, units, role in spatial_values:
        add(
            rows, "spatial-screen design", "main Figure 3", "global", name,
            value, units, "primitive screen setting", "pre-specified",
            "Figure 3 specification and final spatial audit",
            "spatial_screen/scripts/_config.py or 16_phase2_final_audit.py",
            "explicit threshold/rule sensitivities where listed",
            f"Class B mechanism screen: {role}",
        )


def write_literal_audit():
    """List every numeric literal in result-affecting source files."""
    files = sorted((ROOT / "code/model").glob("*.py"))
    files += sorted((ROOT / "code/repro").glob("*.py"))
    files += sorted((ROOT / "spatial_screen/scripts").glob("*.py"))
    visual_tokens = (
        "plt.", "ax.", "figsize", "fontsize", "dpi=", "color=", "alpha=",
        "linewidth", "lw=", "bbox", "legend", "subplots_adjust",
    )
    records = []
    for path in files:
        text = path.read_text()
        lines = text.splitlines()
        tree = ast.parse(text)
        per_line = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            ):
                per_line.setdefault(node.lineno, []).append(node.value)
        for line_no, values in sorted(per_line.items()):
            source_line = lines[line_no - 1].strip()
            if any(token in source_line for token in visual_tokens):
                disposition = "excluded visual formatting"
            elif path.name.startswith("make_figure_") and any(
                token in source_line for token in ("set_", "text(", "annotate(")
            ):
                disposition = "excluded visual formatting"
            else:
                disposition = "reviewed result-affecting/algebraic literal"
            records.append({
                "file": str(path.relative_to(ROOT)),
                "line": line_no,
                "values": json.dumps(values),
                "source_line": source_line,
                "disposition": disposition,
            })
    out = ROOT / "NUMERIC_LITERAL_AUDIT_sol.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main():
    rows = []
    shared_and_bnf_rows(rows)
    dataclass_rows(rows)
    fourpool_extra_rows(rows)
    regional_rows(rows)
    climate_calibration_rows(rows)
    scenario_price_analysis_rows(rows)
    benchmark_and_spatial_rows(rows)

    out_csv = ROOT / "PARAMETER_LEDGER_sol.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    literal_count = write_literal_audit()
    counts = {}
    for record in rows:
        counts[record["group"]] = counts.get(record["group"], 0) + 1
    derived = sum(r["dependency"].startswith("derived") for r in rows)
    fixed = sum("fixed; not varied" in r["uncertainty_treatment"] for r in rows)
    lines = [
        "# Parameter ledger (completed SOL audit)",
        "",
        f"The authoritative CSV contains **{len(rows)} semantic entries**. "
        f"The companion numeric-literal audit contains **{literal_count} code-line entries**.",
        "",
        "## Scope and completeness rule",
        "",
        "A live parameter is any primitive that can change a state, calibration, "
        "forcing, scenario, weight, statistic or acceptance decision without "
        "changing program structure. The ledger therefore includes climate "
        "vectors, crop calendars, calibration targets, solver settings, priors, "
        "seeds and test tolerances as well as named model coefficients.",
        "",
        "Plot colors, fonts, line widths, dimensions and label positions are "
        "excluded as non-scientific formatting. Every numeric literal in model "
        "and reproduction code is separately listed in "
        "`NUMERIC_LITERAL_AUDIT_sol.csv` with that disposition visible.",
        "",
        "## Duplicate-definition decisions",
        "",
        "- SOC stock-to-percentage conversion, residue carbon fraction and "
        "water-stress smoothing constants now have one source in "
        "`parameter_registry.py`.",
        "- Baseline BNF is derived once from legume fraction, net N credit and "
        "free-living fixation. `RegionParams.bnf_potential` is populated from "
        "that derived registry value; it is not a second primitive.",
        "- Regional `yield_max_regional` is a disabled zero sentinel. The only "
        "reported regional ceiling is the ERA5 runtime Brent calibration.",
        "- ERA5 JSON is the sole canonical climate forcing; repeated loader "
        "implementations delegate to one validated function.",
        "- Fertilizer cost share and N price in yield units are derived from "
        "audited nitrogen price, crop price, N rate and modeled yield.",
        "- The OFRA benchmark reads the generated Table S4 values; it no longer "
        "hardcodes the SSA ceiling or no-synthetic-N yield.",
        "",
        "## Audit summary",
        "",
        f"- Derived/calibrated entries explicitly marked: {derived}",
        f"- Entries explicitly fixed and not varied: {fixed}",
        "- Fixed-but-unvaried entries are limitations, not silently implied "
        "sources of certainty.",
        "",
        "## Entry counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    (ROOT / "PARAMETER_LEDGER_sol.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_csv} ({len(rows)} entries)")
    print(f"wrote {ROOT / 'PARAMETER_LEDGER_sol.md'}")
    print(f"wrote {ROOT / 'NUMERIC_LITERAL_AUDIT_sol.csv'} ({literal_count} entries)")


if __name__ == "__main__":
    main()
