"""Audited, single-source model parameters for the ERFS-100341 SOL freeze.

v15 (F-011). THIS MODULE NO LONGER HOLDS VALUES. It is a compatibility shim.

Before v15 the constants below were literals here, and ``params.yaml`` restated
them with a mirror test comparing the two. The direction of authority has been
reversed: ``code/model/params.yaml`` is now the single source and
``code/model/registry.py`` is the loader. Prices moved to
``code/model/prices.py``. Everything this module exports is re-derived from
those two so that existing importers keep working without a second statement of
any number.

Do not add a value here. Add it to ``params.yaml``.
"""

from typing import Dict

import registry as _reg
from prices import (  # noqa: F401  (re-exported for backward compatibility)
    REGIONAL_PRICES,
    RegionalPrice,
    SOUTH_ASIA_FARMER_PAID_N_PRICE,
    nitrogen_cost_share,
    nitrogen_price_in_yield_units,
)


# Physical conversions and shared response-shape constants. These are used by
# both the annual diagnostic engine and the canonical monthly coupled engine.
SOC_T_C_HA_PER_PERCENT_30CM = _reg.soc_tha_per_pct()

# Not yet registered. Recorded here as owed: RESIDUE_C_FRACTION and the three
# water-stress shape constants are model constants that no params.yaml entry
# declares, so the registry cannot vouch for them and the mutation harness
# cannot reach them. Registering them is WP-later work, not a WP1 edit, because
# adding an entry changes the leaf count the harness is calibrated against.
RESIDUE_C_FRACTION = 0.45
WATER_STRESS_GAIN_SAT_SOC_PCT = 1.0
WATER_STRESS_SOFTPLUS_EPS_MM = 3.0
WATER_STRESS_MIN_FACTOR = 0.30


# Minasny & McBratney (2018) report a mean increase in plant-available water
# of 1.16 mm per 100 mm soil for a one percentage-point increase in SOC.
# The model expresses SOC over 0-30 cm, giving 1.16 * 3 = 3.48 mm.
WHC_MM_PER_SOC_PCT_30CM = _reg.value('whc_sensitivity')
_WHC_UNC = _reg.uncertainty('whc_sensitivity')
WHC_MM_PER_SOC_PCT_LOW, WHC_MM_PER_SOC_PCT_HIGH = _WHC_UNC['declared_absolute_bounds']


# No clean empirical estimate exists for fertilizer demand's response to
# changes in mineralized soil N. It is zero in the central run and examined
# only as a structural sensitivity. The registered eps_F_N (-0.5) is the S4
# setting; S1-S3 hold the elasticity at the central value below. F-011 scores
# eps_F_N DECLARED_NOT_WIRED for exactly this reason and that verdict is right:
# it is the scenario dial, not a model constant.
SOIL_N_RESPONSE_ELASTICITY_CENTRAL = 0.0
SOIL_N_RESPONSE_ELASTICITY_SENSITIVITY = (0.0, -0.25, -0.50, -1.0)


# Baseline landscape BNF is derived from three primitive quantities rather
# than specified a second time in RegionParams. The legume credit is expressed
# per cereal hectare because the monthly model simulates a cereal hectare:
#     BNF = legume_fraction * net_credit / (1 - legume_fraction)
#           + free_living_BNF.
# These values are scenario inputs, not fitted model outputs.
#
# NOT YET REGISTERED. F-007: fixation in the published run comes from
# get_regional_bnf, which reads these components; MANAGED_TRANSITION_PARAMS is
# neither registered nor drawn, so fixation carries no sampled uncertainty at
# all. Registering BNF_COMPONENTS is owed work.
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
