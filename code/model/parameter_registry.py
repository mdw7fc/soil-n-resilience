"""Audited, single-source model parameters for the ERFS-100341 SOL freeze.

Only quantities used by more than one model or reproduction script belong
here. Derived quantities, including fertilizer cost shares, must be computed
from these primitive inputs rather than specified independently.
"""

from dataclasses import dataclass
from typing import Dict


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
