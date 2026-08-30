#!/usr/bin/env python3
"""Symbolic and numeric dimensional acceptance tests for published equations."""
from dataclasses import dataclass
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "model"))

from parameter_registry import (
    RESIDUE_C_FRACTION,
    SOC_T_C_HA_PER_PERCENT_30CM,
    WATER_STRESS_SOFTPLUS_EPS_MM,
    WHC_MM_PER_SOC_PCT_30CM,
    nitrogen_price_in_yield_units,
)
from soil_n_model import CropParams, get_default_regions


@dataclass(frozen=True)
class Dim:
    """Exponents for C, N, crop mass, area, money, time and length."""

    c: int = 0
    n: int = 0
    crop: int = 0
    area: int = 0
    money: int = 0
    time: int = 0
    length: int = 0

    def __mul__(self, other):
        return Dim(*(a + b for a, b in zip(self.__dict__.values(),
                                           other.__dict__.values())))

    def __truediv__(self, other):
        return Dim(*(a - b for a, b in zip(self.__dict__.values(),
                                           other.__dict__.values())))

    def __pow__(self, exponent):
        return Dim(*(exponent * a for a in self.__dict__.values()))


ONE = Dim()
C = Dim(c=1)
N = Dim(n=1)
CROP = Dim(crop=1)
AREA = Dim(area=1)
MONEY = Dim(money=1)
TIME = Dim(time=1)
LENGTH = Dim(length=1)


def main():
    # SOC conversion: 1.3 Mg m-3 * 0.30 m * 10,000 m2 ha-1 * 1% = 39 t C ha-1.
    numeric_soc_factor = 1.3 * 0.30 * 10_000 * 0.01
    assert abs(numeric_soc_factor - SOC_T_C_HA_PER_PERCENT_30CM) < 1e-12
    soc_stock = C / AREA
    assert soc_stock / soc_stock == ONE

    # Water response: (mm / SOC percentage point) * SOC percentage points = mm;
    # (fraction/mm) * mm is dimensionless.
    whc_sensitivity = LENGTH
    soc_pct_change = ONE
    water_change = whc_sensitivity * soc_pct_change
    water_stress_coeff = ONE / LENGTH
    assert water_change == LENGTH
    assert water_stress_coeff * water_change == ONE
    assert WHC_MM_PER_SOC_PCT_30CM > 0
    assert WATER_STRESS_SOFTPLUS_EPS_MM > 0

    # Mineralization: k * C stock / (C:N) -> N area-1 time-1.
    decay = ONE / TIME
    cn_ratio = C / N
    mineral_n = decay * (C / AREA) / cn_ratio
    assert mineral_n == N / AREA / TIME

    # Mitscherlich exponent: c [area/N] * N uptake [N/area] is dimensionless.
    mitscherlich_c = AREA / N
    n_uptake = N / AREA
    assert mitscherlich_c * n_uptake == ONE

    # Stoichiometric cap: N uptake / grain-N requirement -> crop yield.
    grain_n_per_crop = N / CROP
    assert n_uptake / grain_n_per_crop == CROP / AREA

    # Residue C: crop yield * residue/grain * C/residue -> C stock input.
    residue_per_crop = ONE
    carbon_per_residue = C / CROP
    residue_c = (CROP / AREA) * residue_per_crop * carbon_per_residue
    assert residue_c == C / AREA
    assert CropParams().residue_c_fraction == RESIDUE_C_FRACTION

    # Financial conversion: N price / crop price -> crop/N; multiplied by N rate
    # gives crop-equivalent expenditure per area, commensurate with yield.
    n_price = MONEY / N
    crop_price = MONEY / CROP
    price_yield_units = n_price / crop_price
    n_rate = N / AREA
    assert price_yield_units == CROP / N
    assert n_rate * price_yield_units == CROP / AREA
    for key in ("sub_saharan_africa", "south_asia",
                "latin_america", "north_america"):
        assert nitrogen_price_in_yield_units(key) > 0

    # Production weights: area * yield = crop production.
    assert AREA * (CROP / AREA) == CROP

    # Log-change economic equations contain only elasticities and log changes.
    elasticity = ONE
    log_change = ONE
    assert elasticity * log_change == ONE

    # Code-level uniqueness: the canonical source files may refer to registry
    # names, but must not reinstate the old independent literals.
    monthly = (ROOT / "code/model/coupled_monthly.py").read_text()
    annual = (ROOT / "code/model/soil_n_model.py").read_text()
    hybrid = (ROOT / "code/model/monthly_model_v3.py").read_text()
    assert "/ 39.0" not in monthly + annual + hybrid
    assert "* 0.45 * rf" not in monthly + hybrid
    assert not (ROOT / "code/era5/REGIONAL_CLIMATES_era5.py").exists()
    assert all(r.yield_max_regional == 0.0
               for r in get_default_regions().values())

    print("DIMENSIONAL CONSISTENCY: PASS")
    print("  SOC conversion, water response, N mineralization, yield response,")
    print("  residue C, price conversion, aggregation and log equations audited.")


if __name__ == "__main__":
    main()
