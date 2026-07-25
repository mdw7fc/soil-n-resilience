"""Audited, single-source model parameters for the ERFS-100341 SOL freeze.

Only quantities used by more than one model or reproduction script belong
here. Derived quantities, including fertilizer cost shares, must be computed
from these primitive inputs rather than specified independently.
"""

from dataclasses import dataclass
from typing import Dict


# Physical conversions and shared response-shape constants. These are used by
# both the annual diagnostic engine and the canonical monthly coupled engine.
# Keeping them here prevents physically identical calculations from silently
# diverging.
SOC_T_C_HA_PER_PERCENT_30CM = 39.0
RESIDUE_C_FRACTION = 0.45
WATER_STRESS_GAIN_SAT_SOC_PCT = 1.0
WATER_STRESS_SOFTPLUS_EPS_MM = 3.0
WATER_STRESS_MIN_FACTOR = 0.30


# Minasny & McBratney (2018) report a mean increase in plant-available water
# of 1.16 mm per 100 mm soil for a one percentage-point increase in SOC.
# The model expresses SOC over 0-30 cm, giving 1.16 * 3 = 3.48 mm.
WHC_MM_PER_SOC_PCT_30CM = 3.48
WHC_MM_PER_SOC_PCT_LOW = 2.32
WHC_MM_PER_SOC_PCT_HIGH = 8.40


# No clean empirical estimate exists for fertilizer demand's response to
# changes in mineralized soil N. It is zero in the central run and examined
# only as a structural sensitivity.
SOIL_N_RESPONSE_ELASTICITY_CENTRAL = 0.0
SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY = (0.0, -0.25, -0.50, -1.0)


# Baseline landscape BNF is derived from three primitive quantities rather
# than specified a second time in RegionParams. The legume credit is expressed
# per cereal hectare because the monthly model simulates a cereal hectare:
#     BNF = legume_fraction * net_credit / (1 - legume_fraction)
#           + free_living_BNF.
# These values are scenario inputs, not fitted model outputs.
BNF_COMPONENTS = {
    "north_america": {
        "legume_frac": 0.35, "net_n_credit": 50.0,
        "legume_yield_ceq": 1.8, "free_living_bnf": 5.0,
    },
    "europe": {
        "legume_frac": 0.25, "net_n_credit": 40.0,
        "legume_yield_ceq": 1.3, "free_living_bnf": 5.0,
    },
    "east_asia": {
        "legume_frac": 0.20, "net_n_credit": 35.0,
        "legume_yield_ceq": 1.2, "free_living_bnf": 5.0,
    },
    "south_asia": {
        "legume_frac": 0.30, "net_n_credit": 40.0,
        "legume_yield_ceq": 1.0, "free_living_bnf": 5.0,
    },
    "southeast_asia": {
        "legume_frac": 0.25, "net_n_credit": 45.0,
        "legume_yield_ceq": 1.2, "free_living_bnf": 8.0,
    },
    "latin_america": {
        "legume_frac": 0.45, "net_n_credit": 40.0,
        "legume_yield_ceq": 1.5, "free_living_bnf": 5.0,
    },
    "sub_saharan_africa": {
        "legume_frac": 0.25, "net_n_credit": 30.0,
        "legume_yield_ceq": 0.8, "free_living_bnf": 5.0,
    },
    "fsu_central_asia": {
        "legume_frac": 0.20, "net_n_credit": 35.0,
        "legume_yield_ceq": 1.0, "free_living_bnf": 5.0,
    },
}


def baseline_bnf_kg_n_ha_yr(region_key: str) -> float:
    """Return derived landscape BNF for a canonical region."""

    p = BNF_COMPONENTS[region_key]
    frac = p["legume_frac"]
    return (
        frac * p["net_n_credit"] / (1.0 - frac)
        + p["free_living_bnf"]
    )


BASELINE_BNF_KG_N_HA_YR = {
    key: baseline_bnf_kg_n_ha_yr(key) for key in BNF_COMPONENTS
}


@dataclass(frozen=True)
class RegionalPrice:
    """Baseline prices used for the nitrogen-expenditure calculation."""

    nitrogen_usd_per_kg_n: float
    crop_usd_per_t: float
    convention: str


# Market/replacement-cost convention used for the disruption experiment.
# SSA uses the lower end of observed non-subsidized African retail urea prices
# expressed per kg N. South Asia is explicitly an import-parity convention.
REGIONAL_PRICES: Dict[str, RegionalPrice] = {
    "sub_saharan_africa": RegionalPrice(2.30, 300.0, "non-subsidized retail"),
    "south_asia": RegionalPrice(1.20, 280.0, "market replacement cost"),
    "latin_america": RegionalPrice(1.15, 260.0, "market replacement cost"),
    "north_america": RegionalPrice(1.10, 250.0, "market replacement cost"),
}


# Farmer-paid South Asian sensitivity. At the canonical modeled yield and
# fertilizer rate this produces an N-expenditure share close to 5%.
SOUTH_ASIA_FARMER_PAID_N_PRICE = 0.39


def nitrogen_price_in_yield_units(region_key: str) -> float:
    """Return tonnes of regional crop needed to purchase one kg of N."""

    p = REGIONAL_PRICES[region_key]
    return p.nitrogen_usd_per_kg_n / p.crop_usd_per_t


def nitrogen_cost_share(region_key: str, n_kg_ha: float, yield_t_ha: float) -> float:
    """Return nitrogen expenditure divided by gross crop revenue."""

    if yield_t_ha <= 0:
        raise ValueError("yield_t_ha must be positive")
    p = REGIONAL_PRICES[region_key]
    return p.nitrogen_usd_per_kg_n * n_kg_ha / (p.crop_usd_per_t * yield_t_ha)
