"""prices.py -- nitrogen and crop prices, read from the registry.

v15 (F-011). New module. Before v15 the price primitives lived as literals in
``parameter_registry.py`` and the nitrogen cost share was, in one reported
place, a hardcoded dictionary rather than a derivation (F-013: the superseded
83.7 percent sub-Saharan Africa figure measured that dictionary). Prices are now
registered in ``params.yaml`` and derived here, once.

THE DERIVATION.

    n_price_usd_kg = n_benchmark_usd_kg * n_price_wedge

The benchmark is a world urea price expressed per kg N. The wedge is the
regional delivered-price premium over it. Splitting the two is what lets the
ensemble vary the quantity that matters for a regional comparison (the wedge)
while holding the common factor fixed, and it is what makes
``n_price_usd_kg`` a derived entry that cannot be edited independently.

RECONSTRUCTION GAP. The v15 eight-region wedge and crop-price tables did not
survive the crash, and the v14 deposit carries audited prices for four regions
only: north_america, south_asia, latin_america and sub_saharan_africa. Those
four are registered and reproduce the v14 values exactly. The other four are
absent from the registry, and asking for one raises rather than guessing. See
``v15/RECONSTRUCTION_GAPS.md``.

BASIS (F-005). A price is per tonne of grain; a cost share is dimensionless; a
wedge is a multiplier on a benchmark. Three different bases have already been
confused once in this project. Every function below names the basis it returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import registry as _reg

__all__ = [
    "N_BENCHMARK_USD_KG",
    "UREA_N_FRACTION",
    "PRICE_BENCHMARK_MAX_FACTOR",
    "COST_SHARE_BAND",
    "N_PRICE_WEDGE",
    "CROP_PRICE_USD_T",
    "SOUTH_ASIA_FARMER_PAID_N_PRICE",
    "PRICED_REGIONS",
    "RegionalPrice",
    "REGIONAL_PRICES",
    "n_price_usd_kg",
    "nitrogen_price_in_yield_units",
    "nitrogen_cost_share",
    "check_price_bounds",
    "check_cost_share_bounds",
]


# --- registered primitives -------------------------------------------------

N_BENCHMARK_USD_KG: float = float(_reg.value("n_benchmark_usd_kg"))
UREA_N_FRACTION: float = float(_reg.value("urea_n_fraction"))
PRICE_BENCHMARK_MAX_FACTOR: float = float(_reg.value("price_benchmark_max_factor"))
COST_SHARE_BAND: Tuple[float, float] = tuple(_reg.value("cost_share_band"))

N_PRICE_WEDGE: Dict[str, float] = dict(_reg.value("n_price_wedge"))
CROP_PRICE_USD_T: Dict[str, float] = dict(_reg.value("crop_price_usd_t"))

# Farmer-paid South Asian sensitivity. Subsidised; reported as a labelled
# one-region sensitivity in the SI, never in the main experiment.
SOUTH_ASIA_FARMER_PAID_N_PRICE: float = float(
    _reg.value("n_price_usd_kg_farmer_paid")["south_asia"]
)

# The regions the registry can price. Deliberately not all eight.
PRICED_REGIONS: List[str] = [
    rk for rk in _reg.REGIONS if rk in N_PRICE_WEDGE and rk in CROP_PRICE_USD_T
]

# Conventions carried forward from the v14 audit. A price without its
# convention is a number without its basis.
_PRICE_CONVENTIONS: Dict[str, str] = {
    "sub_saharan_africa": "non-subsidized retail",
    "south_asia": "market replacement cost",
    "latin_america": "market replacement cost",
    "north_america": "market replacement cost",
}


def n_price_usd_kg(region_key: str) -> float:
    """Delivered nitrogen price, USD per kg N. Derived, never registered."""
    try:
        wedge = N_PRICE_WEDGE[region_key]
    except KeyError:
        raise KeyError(
            f"no registered nitrogen wedge for {region_key!r}. The v15 "
            f"eight-region price table did not survive; see "
            f"v15/RECONSTRUCTION_GAPS.md. Priced regions: {PRICED_REGIONS}."
        ) from None
    return N_BENCHMARK_USD_KG * wedge


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
    rk: RegionalPrice(
        n_price_usd_kg(rk),
        float(CROP_PRICE_USD_T[rk]),
        _PRICE_CONVENTIONS[rk],
    )
    for rk in PRICED_REGIONS
}


def nitrogen_price_in_yield_units(region_key: str) -> float:
    """Tonnes of regional crop needed to purchase one kg of N.

    Basis: t_grain per kg_N. Not a cost share and not a price.
    """
    p = REGIONAL_PRICES[region_key]
    return p.nitrogen_usd_per_kg_n / p.crop_usd_per_t


def nitrogen_cost_share(region_key: str, n_kg_ha: float, yield_t_ha: float) -> float:
    """Nitrogen expenditure divided by gross crop revenue.

    Basis: dimensionless, per hectare, at the supplied application rate and
    yield. F-013: this derivation, not a stored dictionary, is what the paper's
    cost-share statements must be read off.
    """
    if yield_t_ha <= 0:
        raise ValueError("yield_t_ha must be positive")
    p = REGIONAL_PRICES[region_key]
    return p.nitrogen_usd_per_kg_n * n_kg_ha / (p.crop_usd_per_t * yield_t_ha)


# --- registered contract bounds --------------------------------------------

def check_price_bounds() -> List[str]:
    """Return a list of violations of ``price_benchmark_max_factor``.

    A regional delivered price more than this factor above the world benchmark
    is a data error rather than a market outcome.
    """
    out: List[str] = []
    for rk, wedge in N_PRICE_WEDGE.items():
        if wedge > PRICE_BENCHMARK_MAX_FACTOR:
            out.append(
                f"{rk}: wedge {wedge:.4g} exceeds price_benchmark_max_factor "
                f"{PRICE_BENCHMARK_MAX_FACTOR:g}"
            )
    return out


def check_cost_share_bounds(shares: Dict[str, float]) -> List[str]:
    """Return a list of cost shares outside the registered ``cost_share_band``."""
    lo, hi = COST_SHARE_BAND
    return [
        f"{rk}: nitrogen cost share {s:.4g} outside [{lo:g}, {hi:g}]"
        for rk, s in shares.items()
        if not (lo <= s <= hi)
    ]
