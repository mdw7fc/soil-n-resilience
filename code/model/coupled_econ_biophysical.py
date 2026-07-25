"""
Coupled Economic-Biophysical Model of Fertilizer Price Shocks
=============================================================

Integrates Dale Manning's partial equilibrium economic framework with
Wallenstein's 3-pool SOM biophysical model. The economic layer determines
fertilizer demand, food prices, and land allocation in response to a
fertilizer price shock. The biophysical layer tracks soil nitrogen dynamics,
crop yields, and SOM depletion using the full Century-informed 3-pool model.

Architecture:
    - At each annual timestep, the economic module resolves market equilibrium
      (fertilizer demand, food price, land allocation) given current soil state.
    - The biophysical module then updates SOM pools, computes N mineralization,
      and returns yield given the economically determined fertilizer application
      and land area.
    - Soil N dynamics feed back into the next period's economic decisions.

Scenarios (following Manning's framework):
    S1: No behavioral response (land fixed, fert fixed at shocked level)
    S2: Land expansion response (endogenous land market)
    S3: Full behavioral (land + fert response to food prices)
    SC1-SC2: Supply-constrained variants with S3 elasticities + soil-N
             soil-N demand feedback is off in the central case because no
             empirical regional estimate is available. SC1 = permanent 20%
             supply loss;
             SC2 = 20% loss with 20-year recovery. The S4 behavioral channel is
             reserved for supply-constrained scenarios where SOC decline
             is large enough for the feedback to matter.

Author: Matthew Wallenstein & Dale Manning
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Import the biophysical model components
from soil_n_model import (
    SOMPoolParams, CropParams, RegionParams, FeedbackParams,
    get_default_regions,
)
import registry as _reg
from parameter_registry import (
    SOC_T_C_HA_PER_PERCENT_30CM,
    SOIL_N_RESPONSE_ELASTICITY_CENTRAL,
)
# Declared aggregation bases (F-005). Every global mean in this module goes
# through one of these factories; there is no unlabelled weight vector.
from seams import (
    OUTCOME_COLS, INTENSITY_COLS,
    outcome_weights, intensity_weights, nitrogen_weights,
    assert_same_basis, basis_for_column,
)


# ============================================================
# ECONOMIC PARAMETERS
# ============================================================

@dataclass
class EconParams:
    """Economic parameters for the partial equilibrium model.

    All elasticities are in log-change form (percent change in response
    to percent change in driver). Literature sources noted where available.
    """
    # --- Fertilizer price shock ---
    fert_price_shock: float = 1.0  # Proportional increase (1.0 = 100% doubling)

    # --- Fertilizer demand elasticities ---
    # eps_F_PF: elasticity of fert demand w.r.t. fert price (negative)
    # Manning regional values from FDME framework + Ethiopia/SSA evidence.
    # NA: -0.20 (Roberts & Schlenker 2013 range); SSA: -0.50 (consensus;
    # Dale's -0.70 revised down -- see parameter-documentation.md).
    # NOTE: Region-specific values override via REGIONAL_ECON_PARAMS.
    eps_F_PF: float = -0.30

    # eps_F_PY: elasticity of fert demand w.r.t. food output price (positive)
    # Manning: 0.10 across regions (SSA: 0.03 consensus).
    # Lower than previous 0.30; reflects muted cross-price response in
    # empirical estimates (Huang & Khanna 2010).
    eps_F_PY: float = 0.10

    # eps_F_N: elasticity of fert demand w.r.t. soil N stock (negative).
    # No clean empirical estimate exists. The audited central run therefore
    # sets it to zero; negative values are structural sensitivities only.
    eps_F_N: float = 0.0

    # --- Food demand elasticity ---
    # eta: price elasticity of food demand (negative)
    # Manning regional values from FDME meta-elasticities framework.
    # Muhammad et al. (2011): -0.3 to -0.7 across income levels.
    # NOTE: Region-specific values override via REGIONAL_ECON_PARAMS.
    eta: float = -0.45

    # --- Production function elasticities ---
    # alpha: output elasticity w.r.t. land
    # Manning: calibrated as land cost share (growth-accounting logic).
    # Range 0.08-0.15 across regions. Previous 0.30 was too high;
    # growth accounting shows land contributes ~10% of output.
    # NOTE: Region-specific values override via REGIONAL_ECON_PARAMS.
    alpha: float = 0.10
    # beta and gamma computed from Mitscherlich function at each timestep

    # --- Land market elasticities ---
    # eps_LD_PL: land demand elasticity w.r.t. land price (negative)
    # Manning baseline -0.30; consistent with CGE land modeling practice.
    eps_LD_PL: float = -0.30

    # eps_LD_PY: land demand elasticity w.r.t. food price (positive)
    # Manning baseline 0.20. Frontier regions (LatAm, SSA) get higher
    # values via REGIONAL_ECON_PARAMS to reflect land availability.
    eps_LD_PY: float = 0.20

    # eps_LS_PL: land supply elasticity w.r.t. land price (positive)
    # Manning baseline 0.50. Differentiated regionally: higher for
    # frontier regions, lower for land-constrained regions.
    # Lubowski et al. (2006), Gurgel et al. (2007).
    eps_LS_PL: float = 0.50

    # --- Fertilizer supply constraint ---
    # Fraction of baseline fertilizer that is physically available post-shock.
    # 1.0 = no supply constraint (pure cost shock, any quantity at higher price)
    # 0.8 = 20% of global supply physically removed (e.g., trade disruption)
    # 0.0 = complete supply elimination (theoretical limit)
    # When binding, this caps F_level regardless of economic demand signals.
    fert_supply_ceiling: float = 1.0  # fraction of baseline; 1.0 = unconstrained

    # Whether the ceiling relaxes over time (new capacity built)
    # If > 0, ceiling ramps back toward 1.0 over this many years
    fert_capacity_recovery_years: float = 0.0  # 0 = no recovery

    # Whether price relaxes proportionally with supply recovery.
    # If True, PF_hat scales down as the ceiling recovers toward 1.0.
    price_relaxes_with_recovery: bool = True


# ============================================================
# REGIONAL ECONOMIC PARAMETERS
# ============================================================

# Region-specific overrides for economic elasticities.
# Consensus values from Manning (2026) regional calibration with
# independent verification against empirical literature.
# See paper4-coupled-manning/parameter-documentation.md for full justification.
#
# Sources by parameter:
#   eta: FDME meta-elasticities framework; Muhammad et al. 2011; Seale et al. 2003
#   alpha: Growth-accounting land cost shares (Manning calibration)
#   eps_F_PF: FDME + Ethiopia/SSA evidence; Roberts & Schlenker 2013 (US)
#   eps_F_PY: Manning calibration; Huang & Khanna 2010
#   eps_F_N: structural sensitivity only (no clean regional estimates)
#   Land elasticities: CGE practice (Lubowski et al. 2006; Gurgel et al. 2007)
#     with two-tier differentiation for frontier vs. constrained regions
# v15 (F-011): the eight regions' elasticities are read from
# code/model/params.yaml at import rather than restated here.
#
# eps_F_N is deliberately NOT read from the registry. The registered value
# (-0.5) is the S4 structural-sensitivity setting; the reported S1-S3 runs hold
# it at SOIL_N_RESPONSE_ELASTICITY_CENTRAL = 0.0, and eps_F_N is the dial that
# defines which scenario is running. Wiring it here would make the scenario
# definition a registry edit. F-011 scores it DECLARED_NOT_WIRED for this
# reason; the verdict is correct and is not a gap to close.
_ECON_REGISTRY_FIELDS = (
    'eta', 'alpha', 'eps_F_PF', 'eps_F_PY',
    'eps_LD_PL', 'eps_LD_PY', 'eps_LS_PL',
)

REGIONAL_ECON_PARAMS = {
    rk: dict(
        _reg.region_fields(rk, _ECON_REGISTRY_FIELDS),
        eps_F_N=SOIL_N_RESPONSE_ELASTICITY_CENTRAL,
    )
    for rk in _reg.REGIONS
}


# ============================================================
# BIOPHYSICAL SOM ENGINE (wraps Wallenstein model)
# ============================================================

class BiophysicalSOMEngine:
    """Wraps the 3-pool SOM model to provide N dynamics for the economic model.

    Updated to match corrected soil_n_model.py (v5): includes root C inputs,
    N immobilization from residue incorporation, regional CRE, and iterative
    yield-residue-immobilization convergence.

    At each timestep, accepts externally determined fertilizer application
    and returns:
        - N_mineralized (kg N/ha/yr)
        - yield (t/ha) given applied fert + mineralized N
        - updated SOM state
        - local elasticities (beta, gamma) from Mitscherlich function
    """

    def __init__(self, region: RegionParams, som_params: SOMPoolParams = None,
                 crop_params: CropParams = None, feedback_params: FeedbackParams = None):
        self.region = region
        self.som = som_params or SOMPoolParams()
        self.crop = crop_params or CropParams()
        self.fb = feedback_params or FeedbackParams()

        # Initialize SOM pools
        soc = region.soc_initial
        self.C_active = soc * self.som.f_active
        self.C_slow = soc * self.som.f_slow
        self.C_passive = soc * self.som.f_passive
        self.soc_initial = soc

        # Yield parameters
        self.y_max = region.yield_max_regional if region.yield_max_regional > 0 else crop_params.yield_max if crop_params else 5.0
        self.mit_c = region.mitscherlich_c_regional if region.mitscherlich_c_regional > 0 else crop_params.mitscherlich_c if crop_params else 0.015
        self.n_eff = self.crop.nue_apparent
        self.y_floor = region.yield_min_regional if region.yield_min_regional > 0 else 0.0

        # Regional CRE (use region-specific if available, else default)
        self.cre = region.cre_regional if region.cre_regional > 0 else self.fb.cre_base

        # Compute baseline yield (with full synthetic N, including immobilization)
        n_min_0 = self._total_n_mineralization()
        n_avail_0 = (
            n_min_0 + region.synth_n_current + region.bnf_potential
        )
        # Iterate to converge immobilization at baseline
        for _ in range(3):
            y_0 = self._mitscherlich(n_avail_0 * self.n_eff)
            res_c_0 = self._residue_c_input(y_0)
            n_immob_0 = self._n_immobilization(res_c_0)
            n_avail_0 = (
                (n_min_0 - n_immob_0) + region.synth_n_current
                + region.bnf_potential
            )
        self.yield_baseline = self._mitscherlich(n_avail_0 * self.n_eff)

        # Compute baseline N_mineralized for reference
        self.n_min_baseline = n_min_0

    def _soc_to_percent(self, soc_tha: float) -> float:
        return soc_tha / SOC_T_C_HA_PER_PERCENT_30CM

    def _total_n_mineralization(self) -> float:
        """Total N mineralized from all 3 pools (kg N/ha/yr)."""
        soc = self.C_active + self.C_slow + self.C_passive
        cn_factor = self._cn_coupling_factor(soc)
        n_min = (
            self.som.k_active * self.C_active / self.som.cn_active * 1000.0 +
            self.som.k_slow * self.C_slow / self.som.cn_slow * 1000.0 +
            self.som.k_passive * self.C_passive / self.som.cn_passive * 1000.0
        ) * cn_factor
        return n_min

    def _cn_coupling_factor(self, soc_current: float) -> float:
        if not self.fb.cn_coupling_feedback:
            return 1.0
        frac = soc_current / self.soc_initial
        if frac > 0.60:
            return 1.0
        elif frac < 0.30:
            return 0.6
        else:
            return 1.0 - 0.4 * (0.60 - frac) / 0.30

    def _water_stress(self) -> float:
        if not self.fb.physical_feedback:
            return 1.0
        soc = self.C_active + self.C_slow + self.C_passive
        soc_pct = self._soc_to_percent(soc)
        soc_pct_init = self._soc_to_percent(self.soc_initial)
        delta_soc_pct = soc_pct_init - soc_pct
        whc_loss_mm = delta_soc_pct * self.region.whc_sensitivity * self.fb.physical_strength
        total_deficit = self.region.baseline_water_deficit + max(0, whc_loss_mm)
        stress = 1.0 - self.region.water_stress_coeff * total_deficit
        return max(0.3, min(1.0, stress))

    def _mitscherlich(self, n_available: float, water_stress: float = None) -> float:
        """Yield from Mitscherlich function."""
        if water_stress is None:
            water_stress = self._water_stress()
        n_eff = max(0.0, n_available)
        y = self.y_max * (1.0 - np.exp(-self.mit_c * n_eff)) * water_stress
        return max(self.y_floor, y)

    def _residue_c_input(self, yield_actual: float) -> float:
        """Total carbon input from aboveground residue + root C (t C/ha/yr).

        Aboveground residue is subject to residue_retention.
        Root C stays in the soil regardless of retention.
        """
        above_ground = yield_actual * self.crop.residue_grain_ratio
        above_ground *= self.region.residue_retention

        root_c = yield_actual * self.crop.residue_grain_ratio * self.region.root_shoot_c_ratio

        return (above_ground + root_c) * self.crop.residue_c_fraction

    def _n_immobilization(self, residue_c: float) -> float:
        """Net N immobilized when residue C enters SOM pools (kg N/ha/yr).

        When high C:N residue (~60) is incorporated into lower C:N SOM pools
        (8-12), the stoichiometric mismatch draws mineral N from the soil
        solution, reducing plant-available N.
        """
        c_to_active = residue_c * self.cre * self.fb.cre_to_active
        c_to_slow = residue_c * self.cre * self.fb.cre_to_slow
        n_demand = (c_to_active / self.som.cn_active +
                    c_to_slow / self.som.cn_slow) * 1000  # kg N/ha

        n_supply = residue_c / self.crop.residue_cn * 1000  # kg N/ha

        return max(0.0, n_demand - n_supply)

    def step(self, fert_applied: float, bnf: float = None, dt: float = 1.0):
        """Advance one timestep with externally determined fertilizer.

        Args:
            fert_applied: kg N/ha/yr of synthetic fertilizer actually applied
            bnf: biological N fixation (kg N/ha/yr); defaults to the
                region's derived baseline landscape BNF
            dt: timestep in years

        Returns:
            dict with: yield_tha, n_mineralized, n_available, soc_total,
                       soc_fraction, water_stress, beta, gamma, n_immobilized
        """
        if bnf is None:
            bnf = self.region.bnf_potential

        # N mineralization
        n_min = self._total_n_mineralization()

        # Water stress
        ws = self._water_stress()

        # Iteratively solve yield-residue-immobilization feedback
        # (3 iterations is sufficient for convergence)
        n_immob = 0.0
        for _ in range(3):
            n_net = n_min - n_immob
            n_total = n_net + fert_applied + bnf
            n_available = n_total * self.n_eff
            y = self._mitscherlich(n_available, ws)
            res_c = self._residue_c_input(y)
            n_immob = self._n_immobilization(res_c)

        # Final values with converged immobilization
        n_net = n_min - n_immob
        n_total = n_net + fert_applied + bnf
        n_available = n_total * self.n_eff
        y = self._mitscherlich(n_available, ws)
        res_c = self._residue_c_input(y)

        # Compute LOCAL elasticities at current operating point
        # For Mitscherlich y = y_max * (1 - exp(-c * N)):
        #   dy/dN = y_max * c * exp(-c * N)
        #   elasticity = (dy/dN) * (N/y) = c * N * exp(-c*N) / (1 - exp(-c*N))
        # We need separate elasticities for soil N vs applied fert
        # since they enter through n_available = (n_net + F + bnf) * n_eff

        eps = 1e-10  # avoid division by zero
        exp_term = np.exp(-self.mit_c * max(n_available, eps))
        denom = max(1.0 - exp_term, eps)

        # d(yield)/d(n_available) * n_available / yield
        elasticity_n_total = self.mit_c * n_available * exp_term / denom

        # beta = elasticity w.r.t. soil N stock (through mineralization)
        # Partitioned by share of n_available coming from net mineralization
        n_min_share = (n_net * self.n_eff) / max(n_available, eps)
        beta = elasticity_n_total * max(0.0, n_min_share)

        # gamma = elasticity w.r.t. applied fertilizer
        fert_share = (fert_applied * self.n_eff) / max(n_available, eps)
        gamma = elasticity_n_total * fert_share

        # Update SOM pools (Euler step)
        # Decomposition
        d_active = self.som.k_active * self.C_active * dt
        d_slow = self.som.k_slow * self.C_slow * dt
        d_passive = self.som.k_passive * self.C_passive * dt

        # Humification
        h_a_to_s = d_active * self.som.h_active_to_slow
        h_s_to_p = d_slow * self.som.h_slow_to_passive

        # Residue input allocation (using regional CRE)
        if self.fb.residue_feedback:
            c_in_active = res_c * self.cre * self.fb.cre_to_active * dt
            c_in_slow = res_c * self.cre * self.fb.cre_to_slow * dt
        else:
            # Fixed residue at baseline yield (no feedback)
            fixed_res = self._residue_c_input(self.yield_baseline)
            c_in_active = fixed_res * self.cre * self.fb.cre_to_active * dt
            c_in_slow = fixed_res * self.cre * self.fb.cre_to_slow * dt

        # Pool updates
        self.C_active = max(0.0, self.C_active - d_active + c_in_active)
        self.C_slow = max(0.0, self.C_slow - d_slow + h_a_to_s + c_in_slow)
        self.C_passive = max(0.0, self.C_passive - d_passive + h_s_to_p)

        soc_new = self.C_active + self.C_slow + self.C_passive

        return {
            'yield_tha': y,
            'yield_fraction': y / self.yield_baseline if self.yield_baseline > 0 else 0,
            'n_mineralized': n_min,
            'n_net_mineralized': n_net,
            'n_immobilized': n_immob,
            'n_available': n_available,
            'fert_applied': fert_applied,
            'soc_total': soc_new,
            'soc_fraction': soc_new / self.soc_initial,
            'water_stress': ws,
            'beta': beta,
            'gamma': gamma,
            'residue_c': res_c,
        }


# ============================================================
# SCENARIO DEFINITIONS
# ============================================================

def calibrate_price_shock(target_reduction: float = 0.20) -> float:
    """Find the fert_price_shock that delivers a target global fert reduction.

    The log-linear demand model gives: F_hat = eps_F_PF * ln(1 + shock).
    With regional eps_F_PF, the **nitrogen-tonnage-weighted** average
    reduction must equal the target. We solve for the shock numerically.

    BASIS (F-005, 2026-07-25). Weights are ``cropland_mha * synth_n_current``
    — tonnes of synthetic N applied — supplied by ``seams.nitrogen_weights``.
    This is the only place in the model that uses that basis, and it is
    correct here for a reason that does not generalise: the scenario is
    *defined* as a reduction in nitrogen mass, so weighting it by anything
    other than nitrogen mass would mean the scenario does not deliver its own
    definition. Outcome shares take production tonnage
    (``seams.outcome_weights``); per-hectare rates take cropland area
    (``seams.intensity_weights``).

    THIS DOCSTRING SAID "area-weighted" UNTIL 2026-07-25. It was never
    area-weighted. The delivered year-1 reduction from the calibrated shock
    is 20.00% on this nitrogen basis, 21.42% on area and 19.56% on
    production, so any sentence pairing this 20% with a production-weighted
    yield loss owes the reader both bases.

    Args:
        target_reduction: desired fractional reduction in global fertilizer (0.20 = 20%)

    Returns:
        fert_price_shock: proportional price increase (e.g., 1.5 = 150% increase)
    """
    regions = get_default_regions()
    region_keys = tuple(regions.keys())
    W_n = nitrogen_weights(region_keys, regions)
    # Accumulate on the unnormalised tonnage and divide once — the arithmetic
    # this function has always used. The seam supplies the vector and the
    # label; it does not change the operation or its order, so the calibrated
    # shock is bit-identical to the published 1.0389792148114703.
    total_n = sum(W_n.raw)

    # Binary search for the price shock
    lo, hi = 0.01, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        PF_hat = np.log(1 + mid)
        weighted_reduction = 0.0
        for rn, region_n in zip(region_keys, W_n.raw):
            eps = REGIONAL_ECON_PARAMS[rn]['eps_F_PF']
            F_hat = eps * PF_hat
            # Reduction = 1 - exp(F_hat) for this region
            weighted_reduction += (1 - np.exp(F_hat)) * region_n
        global_reduction = weighted_reduction / total_n
        if global_reduction < target_reduction:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def get_scenario_params() -> Dict[str, EconParams]:
    """Return the three Manning scenarios with appropriate elasticities.

    The fert_price_shock is calibrated so that the **nitrogen-tonnage-
    weighted** global fertilizer reduction is 20% using regional price
    elasticities. This docstring said "area-weighted" until 2026-07-25
    (F-005); it inherited the error from ``calibrate_price_shock`` above,
    which is where the basis is actually chosen. On the area basis the same
    calibrated shock delivers 21.42%, and on the production basis 19.56%.

    S1: No behavioral response
        - All economic elasticities = 0 except eps_F_PF (regional)
        - Fertilizer drops by price-shock-implied amount and stays there
        - Land fixed

    S2: Land expansion response
        - Land market elasticities active
        - Rising food prices induce land conversion

    S3: Full behavioral response
        - Fertilizer demand responds to food price (eps_F_PY > 0)
        - Land market active
        - As food prices rise, farmers source more fert (other suppliers,
          organic sources, etc.)
        - The soil-N response elasticity is zero in the central case because
          no empirical regional estimate is available. Negative values are
          evaluated separately as structural sensitivities.
    """
    shock = calibrate_price_shock(0.20)

    # S1: Only price-mediated fertilizer reduction. All other response
    # channels disabled. Regional alpha, land, and other params will be
    # overridden by REGIONAL_ECON_PARAMS in the model __init__, but we
    # zero out the *response* channels here.
    s1 = EconParams(
        fert_price_shock=shock,
        eps_F_PY=0.0,    # No food price response
        eps_F_N=0.0,      # No soil N response
        eps_LD_PL=0.0,    # No land response
        eps_LD_PY=0.0,
        eps_LS_PL=0.0,
    )

    # S2: Add land expansion response. Land elasticities will be
    # picked up from REGIONAL_ECON_PARAMS (region-specific).
    # Set sentinel values that signal "use regional defaults."
    s2 = EconParams(
        fert_price_shock=shock,
        eps_F_PY=0.0,    # No food price fert response yet
        eps_F_N=0.0,
        # Land elasticities: use regional defaults (set nonzero so
        # __init__ doesn't override them to zero)
    )

    # S3: Observed-price and land-response channels active. eps_F_PY uses the
    # regional value (0.10 default, 0.03 SSA); eps_F_N is zero centrally.
    s3 = EconParams(
        fert_price_shock=shock,
        eps_F_N=SOIL_N_RESPONSE_ELASTICITY_CENTRAL,
        # eps_F_PY, alpha, land elasticities all from regional defaults
    )

    return {'S1': s1, 'S2': s2, 'S3': s3}


def get_supply_constrained_scenarios() -> Dict[str, EconParams]:
    """Scenarios where the price shock reflects a physical supply disruption.

    The key difference from get_scenario_params(): fertilizer supply is
    physically capped. Even if farmers want to buy more at higher prices,
    the molecules aren't there. This models scenarios like:
    - Major natural gas supply disruption (Hormuz, Russia pipeline cutoff)
    - Conflict destroying production capacity
    - Long-term fossil fuel depletion raising feedstock costs permanently

    Uses the S3 price and land responses plus a physical supply cap. The
    unsupported soil-N demand response is zero centrally and evaluated
    separately as a structural sensitivity.
    """
    shock = calibrate_price_shock(0.20)

    # Price and land response plus physical supply constraint.
    # eps_F_PY, alpha, and land elasticities from regional defaults.
    # eps_F_N remains zero in the central case.
    base_kwargs = dict(
        fert_price_shock=shock,
        eps_F_N=SOIL_N_RESPONSE_ELASTICITY_CENTRAL,
    )

    # SC1: 20% supply reduction, no recovery (permanent capacity loss)
    # Anchored to plausible disruption: loss of a major exporter (~Russia+Belarus
    # share of global N trade) with no alternative capacity built.
    sc1 = EconParams(**base_kwargs, fert_supply_ceiling=0.80,
                     fert_capacity_recovery_years=0.0)

    # SC2: 20% supply reduction with 20-year capacity recovery
    # Same initial disruption as SC1, but alternative production capacity
    # is gradually built (green ammonia, regional plants, etc.);
    # price relaxes proportionally as supply recovers.
    sc2 = EconParams(**base_kwargs, fert_supply_ceiling=0.80,
                     fert_capacity_recovery_years=20.0,
                     price_relaxes_with_recovery=True)

    return {'SC1_20pct': sc1, 'SC2_20pct_recovery': sc2}


# ============================================================
# COUPLED MODEL
# ============================================================

class CoupledEconBiophysicalModel:
    """
    Coupled economic-biophysical model.

    At each timestep:
    1. Economic module resolves equilibrium:
       - Fertilizer demand: F_hat = eps_F_PF * PF_hat + eps_F_PY * PY_hat + eps_F_N * N_hat
       - Food supply (Cobb-Douglas): Y_hat = alpha * L_hat + beta * N_hat + gamma * F_hat
       - Food demand: Q_hat = eta * PY_hat
       - Market clearing: Y_hat = Q_hat
       - Land market: L_hat = (eps_LD_PY * PY_hat) * eps_LS_PL / (eps_LS_PL - eps_LD_PL)
         (equating land demand and supply for the equilibrium land price)

    2. Biophysical module:
       - Takes economically determined F_t (kg N/ha) and L_t (Mha)
       - Updates 3-pool SOM, computes yield, returns state

    All variables are in log-change form relative to baseline (year 0).

    The system is solved simultaneously at each timestep: food price PY_hat
    is determined by market clearing, which depends on yield, which depends
    on fertilizer, which depends on food price. We solve the simultaneous
    system analytically.
    """

    def __init__(
        self,
        region: RegionParams,
        econ: EconParams,
        region_key: str = None,
        t_max: float = 100.0,
        dt: float = 1.0,
    ):
        self.region = region
        self.econ = econ
        self.region_key = region_key
        self.t_max = t_max
        self.dt = dt

        # Resolve elasticities: scenario (EconParams) takes precedence for
        # *response channel* parameters (eps_F_PY, eps_F_N, land elasticities)
        # because scenarios define which channels are active. Regional params
        # provide structural parameters (eta, alpha, eps_F_PF) that don't
        # change across scenarios.
        #
        # Logic: start with EconParams (scenario), then overlay regional
        # values for structural params. For response-channel params, the
        # scenario value wins IF it was explicitly set to zero (channel off).
        rp = {}
        if region_key and region_key in REGIONAL_ECON_PARAMS:
            rp = REGIONAL_ECON_PARAMS[region_key]

        # Structural params: always use regional if available
        self.eta = rp.get('eta', econ.eta)
        self.alpha = rp.get('alpha', econ.alpha)
        self.eps_F_PF = rp.get('eps_F_PF', econ.eps_F_PF)

        # Response-channel params: scenario value wins (allows zeroing out
        # channels), but if scenario uses the default (nonzero), use regional.
        # For eps_F_PY: scenario default is 0.10; if scenario sets 0.0, land off.
        self.eps_F_PY = econ.eps_F_PY if econ.eps_F_PY == 0.0 else rp.get('eps_F_PY', econ.eps_F_PY)
        self.eps_F_N = econ.eps_F_N  # Always from scenario (0 in S1-S3, nonzero in S4)

        # Land market: if scenario zeros these out, channels are off.
        # If scenario leaves at defaults, use regional values.
        if econ.eps_LD_PL == 0.0 and econ.eps_LD_PY == 0.0 and econ.eps_LS_PL == 0.0:
            # Land channel explicitly disabled by scenario
            self.eps_LD_PL = 0.0
            self.eps_LD_PY = 0.0
            self.eps_LS_PL = 0.0
        else:
            self.eps_LD_PL = rp.get('eps_LD_PL', econ.eps_LD_PL)
            self.eps_LD_PY = rp.get('eps_LD_PY', econ.eps_LD_PY)
            self.eps_LS_PL = rp.get('eps_LS_PL', econ.eps_LS_PL)

        # Initialize biophysical engine
        self.bio = BiophysicalSOMEngine(region)

        # Baseline values (year 0, pre-shock)
        self.F_baseline = region.synth_n_current  # kg N/ha/yr
        self.L_baseline = region.cropland_mha      # Mha
        self.Y_baseline = self.bio.yield_baseline   # t/ha
        self.N_min_baseline = self.bio.n_min_baseline  # kg N/ha/yr

        # Base price shock (may be relaxed over time in recovery scenarios)
        self.PF_hat_base = np.log(1 + econ.fert_price_shock)

        # Track cumulative log-changes
        self.PF_hat = self.PF_hat_base
        self.PY_hat = 0.0  # Food price (endogenous)
        self.F_hat = 0.0   # Fertilizer applied
        self.L_hat = 0.0   # Land
        self.N_hat = 0.0   # Soil N stock (from biophysical model)

    def _solve_equilibrium(self, beta: float, gamma: float):
        """Solve the simultaneous system for PY_hat, F_hat, L_hat.

        The system (in log-changes from current accumulated state):

        Food supply:  Y_hat = alpha * L_hat + beta * N_hat + gamma * F_hat
        Fert demand:  F_hat = eps_F_PF * PF_hat + eps_F_PY * PY_hat + eps_F_N * N_hat
        Food demand:  Q_hat = eta * PY_hat
        Mkt clearing: Y_hat = Q_hat
        Land market:  L_hat resolved from demand = supply

        Uses region-specific eps_F_PF and eta (set in __init__).

        PY = (beta * N_hat + gamma * (eps_F_PF * PF + eps_F_N * N_hat)) / (eta - alpha * lambda_L - gamma * eps_F_PY)
        """
        # All elasticities now resolved to instance level (regional overrides
        # applied in __init__), so use self.* consistently.

        # Land response coefficient: lambda_L = eps_LS_PL * eps_LD_PY / (eps_LS_PL - eps_LD_PL)
        if abs(self.eps_LS_PL - self.eps_LD_PL) > 1e-10:
            lambda_L = self.eps_LS_PL * self.eps_LD_PY / (self.eps_LS_PL - self.eps_LD_PL)
        else:
            lambda_L = 0.0

        # Solve for food price
        numerator = (beta * self.N_hat +
                     gamma * (self.eps_F_PF * self.PF_hat + self.eps_F_N * self.N_hat))
        denominator = self.eta - self.alpha * lambda_L - gamma * self.eps_F_PY

        if abs(denominator) > 1e-10:
            PY_hat = numerator / denominator
        else:
            PY_hat = 0.0

        # Fertilizer demand
        F_hat = self.eps_F_PF * self.PF_hat + self.eps_F_PY * PY_hat + self.eps_F_N * self.N_hat

        # Land
        L_hat = lambda_L * PY_hat

        return PY_hat, F_hat, L_hat

    def run(self) -> pd.DataFrame:
        """Run the coupled simulation."""
        n_steps = int(self.t_max / self.dt) + 1

        results = {
            'year': np.zeros(n_steps),
            # Economic variables (log-changes from baseline)
            'PF_hat': np.zeros(n_steps),
            'PY_hat': np.zeros(n_steps),
            'F_hat': np.zeros(n_steps),
            'L_hat': np.zeros(n_steps),
            'N_hat': np.zeros(n_steps),
            # Level variables
            'fert_applied_kgha': np.zeros(n_steps),
            'land_mha': np.zeros(n_steps),
            'food_price_index': np.zeros(n_steps),
            # Biophysical variables
            'yield_tha': np.zeros(n_steps),
            'yield_fraction': np.zeros(n_steps),
            'n_mineralized': np.zeros(n_steps),
            'soc_total': np.zeros(n_steps),
            'soc_fraction': np.zeros(n_steps),
            'water_stress': np.zeros(n_steps),
            'beta': np.zeros(n_steps),
            'gamma': np.zeros(n_steps),
            # Aggregate output
            'total_production_index': np.zeros(n_steps),
            'carrying_capacity_fraction': np.zeros(n_steps),
        }

        # Compute initial elasticities WITHOUT stepping SOM (fix year-0 mutation).
        # Query the biophysical engine for beta/gamma at the baseline operating point.
        _init_n_min = self.bio._total_n_mineralization()
        _init_ws = self.bio._water_stress()
        _init_n_avail = (
            _init_n_min + self.F_baseline + self.region.bnf_potential
        ) * self.bio.n_eff
        # Iterate immobilization at baseline
        _init_n_immob = 0.0
        for _ in range(3):
            _init_n_net = _init_n_min - _init_n_immob
            _init_n_avail = (
                _init_n_net + self.F_baseline + self.region.bnf_potential
            ) * self.bio.n_eff
            _init_y = self.bio._mitscherlich(_init_n_avail, _init_ws)
            _init_res_c = self.bio._residue_c_input(_init_y)
            _init_n_immob = self.bio._n_immobilization(_init_res_c)
        _init_n_net = _init_n_min - _init_n_immob
        _init_n_avail = (
            _init_n_net + self.F_baseline + self.region.bnf_potential
        ) * self.bio.n_eff
        eps = 1e-10
        exp_term = np.exp(-self.bio.mit_c * max(_init_n_avail, eps))
        denom_mit = max(1.0 - exp_term, eps)
        init_total_elast = self.bio.mit_c * _init_n_avail * exp_term / denom_mit
        fert_share = (self.F_baseline * self.bio.n_eff) / max(_init_n_avail, eps)
        n_min_share = (_init_n_net * self.bio.n_eff) / max(_init_n_avail, eps)
        init_beta = init_total_elast * max(0.0, n_min_share)
        init_gamma = init_total_elast * fert_share

        for i in range(n_steps):
            t = i * self.dt
            results['year'][i] = t

            if i == 0:
                # Baseline year: record initial state WITHOUT stepping SOM
                soc_0 = self.bio.C_active + self.bio.C_slow + self.bio.C_passive
                results['PF_hat'][i] = 0.0
                results['PY_hat'][i] = 0.0
                results['F_hat'][i] = 0.0
                results['L_hat'][i] = 0.0
                results['N_hat'][i] = 0.0
                results['fert_applied_kgha'][i] = self.F_baseline
                results['land_mha'][i] = self.L_baseline
                results['food_price_index'][i] = 1.0
                results['yield_tha'][i] = self.Y_baseline
                results['yield_fraction'][i] = 1.0
                results['n_mineralized'][i] = _init_n_min
                results['soc_total'][i] = soc_0
                results['soc_fraction'][i] = 1.0
                results['water_stress'][i] = _init_ws
                results['beta'][i] = init_beta
                results['gamma'][i] = init_gamma
                results['total_production_index'][i] = 1.0
                results['carrying_capacity_fraction'][i] = 1.0
                continue

            # Update N_hat from biophysical state
            current_n_min = self.bio._total_n_mineralization()
            if self.N_min_baseline > 0:
                self.N_hat = np.log(max(current_n_min, 1e-6) / self.N_min_baseline)
            else:
                self.N_hat = 0.0

            # Update PF_hat: relax price proportionally with supply recovery
            if (self.econ.price_relaxes_with_recovery and
                    self.econ.fert_capacity_recovery_years > 0 and t > 0):
                recovery_frac = min(1.0, t / self.econ.fert_capacity_recovery_years)
                self.PF_hat = self.PF_hat_base * (1.0 - recovery_frac)
            else:
                self.PF_hat = self.PF_hat_base

            # Get current elasticities from biophysical model
            # (use last step's values as approximation)
            beta = results['beta'][i-1]
            gamma = results['gamma'][i-1]

            # Solve economic equilibrium
            PY_hat, F_hat, L_hat = self._solve_equilibrium(beta, gamma)

            self.PY_hat = PY_hat
            self.F_hat = F_hat
            self.L_hat = L_hat

            # Convert log-changes to levels
            F_level = self.F_baseline * np.exp(F_hat)
            L_level = self.L_baseline * np.exp(L_hat)

            # Floor fertilizer at 0
            F_level = max(0.0, F_level)

            # Apply supply ceiling (physical constraint on fertilizer availability)
            # The ceiling constrains TOTAL regional N supply, not per-hectare rate.
            if self.econ.fert_supply_ceiling < 1.0:
                ceiling = self.econ.fert_supply_ceiling
                # Allow capacity recovery over time if specified
                if self.econ.fert_capacity_recovery_years > 0 and t > 0:
                    recovery_frac = min(1.0, t / self.econ.fert_capacity_recovery_years)
                    ceiling = ceiling + (1.0 - ceiling) * recovery_frac
                # Total N available = baseline total * ceiling
                total_n_available = self.F_baseline * self.L_baseline * ceiling
                # Per-hectare max given current land area
                F_max = total_n_available / max(L_level, 1e-6)
                F_level = min(F_level, F_max)

            # Advance biophysical model
            bio_state = self.bio.step(F_level)

            # Total production = yield * land (both as fractions of baseline)
            yield_frac = bio_state['yield_fraction']
            land_frac = L_level / self.L_baseline
            total_prod_index = yield_frac * land_frac

            # Store results
            results['PF_hat'][i] = self.PF_hat
            results['PY_hat'][i] = PY_hat
            results['F_hat'][i] = F_hat
            results['L_hat'][i] = L_hat
            results['N_hat'][i] = self.N_hat
            results['fert_applied_kgha'][i] = F_level
            results['land_mha'][i] = L_level
            results['food_price_index'][i] = np.exp(PY_hat)
            results['yield_tha'][i] = bio_state['yield_tha']
            results['yield_fraction'][i] = yield_frac
            results['n_mineralized'][i] = bio_state['n_mineralized']
            results['soc_total'][i] = bio_state['soc_total']
            results['soc_fraction'][i] = bio_state['soc_fraction']
            results['water_stress'][i] = bio_state['water_stress']
            results['beta'][i] = bio_state['beta']
            results['gamma'][i] = bio_state['gamma']
            results['total_production_index'][i] = total_prod_index
            results['carrying_capacity_fraction'][i] = total_prod_index

        return pd.DataFrame(results)


# ============================================================
# MULTI-REGION RUNNER
# ============================================================

def run_all_scenarios(
    regions: Dict[str, RegionParams] = None,
    t_max: float = 100.0,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run all scenarios for all regions.

    Returns: {scenario_name: {region_name: DataFrame}}
    """
    if regions is None:
        regions = get_default_regions()

    scenarios = get_scenario_params()
    results = {}

    for s_name, econ in scenarios.items():
        results[s_name] = {}
        for r_name, region in regions.items():
            model = CoupledEconBiophysicalModel(
                region=region, econ=econ, region_key=r_name, t_max=t_max,
            )
            results[s_name][r_name] = model.run()

    return results


def aggregate_global(
    scenario_results: Dict[str, pd.DataFrame],
    regions: Dict[str, RegionParams] = None,
) -> pd.DataFrame:
    """Aggregate across regions for one scenario on the declared bases.

    Every column is aggregated on the basis its quantity class declares in
    ``code/model/seams.py``, and the returned frame carries the two basis
    labels in ``agg.attrs`` so that no downstream consumer has to guess:

      * outcome shares (``yield_fraction``, ``soc_fraction``,
        ``food_price_index``) — **production tonnage**, ``cropland_mha`` times
        the region's own year-0 baseline yield;
      * per-hectare rates (``fert_applied_kgha``, ``n_mineralized``,
        ``water_stress``) — **cropland area**.

    Until 2026-07-25 (F-005) this function weighted all six columns by
    cropland area and its one-line docstring said so, which was at least
    honest; the error was that the paper's published headline came from a
    third, production-weighted vector normalised inline in
    ``code/repro/run_canonical.py``, and nothing in the repository compared
    the two. This function has exactly two call sites, both in this module's
    ``__main__`` demonstration, so switching the outcome columns to their
    declared basis moves no published number.
    """
    if regions is None:
        regions = get_default_regions()

    region_keys = tuple(scenario_results.keys())
    total_pop = sum(regions[k].pop_supported for k in region_keys)

    first_df = list(scenario_results.values())[0]
    agg = pd.DataFrame({'year': first_df['year'].values})

    # Year-0 baseline yield per region defines the production basis.
    y_base = [float(scenario_results[k]['yield_tha'].iloc[0]) for k in region_keys]
    W_out = outcome_weights(region_keys, y_base, regions, universe=region_keys)
    W_int = intensity_weights(region_keys, regions, universe=region_keys)

    for col in OUTCOME_COLS + INTENSITY_COLS:
        W = W_out if basis_for_column(col) == W_out.basis else W_int
        agg[col] = 0.0
        for j, r_name in enumerate(region_keys):
            agg[col] += scenario_results[r_name][col].values * W.weights[j]

    agg.attrs['basis_outcome'] = W_out.label()
    agg.attrs['basis_intensity'] = W_int.label()

    # Total production (sum of region production indices weighted by pop share)
    agg['pop_supported_millions'] = 0.0
    for r_name, df in scenario_results.items():
        agg['pop_supported_millions'] += (
            df['carrying_capacity_fraction'].values * regions[r_name].pop_supported
        )

    agg['carrying_capacity_fraction'] = agg['pop_supported_millions'] / total_pop

    # Land (sum of levels)
    agg['total_land_mha'] = 0.0
    for r_name, df in scenario_results.items():
        agg['total_land_mha'] += df['land_mha'].values

    return agg


# ============================================================
# VALIDATION & COMPARISON
# ============================================================

def validate_fert_reduction(t_max: float = 10.0):
    """Verify that the calibrated price shock delivers ~20% global fert reduction.

    Runs S1 (no behavioral response) for all regions and checks that the
    area-weighted fertilizer reduction in year 1 is close to 20%.
    """
    regions = get_default_regions()
    scenarios = get_scenario_params()
    s1 = scenarios['S1']

    total_n_baseline = 0.0
    total_n_shocked = 0.0

    results = {}
    for r_name, region in regions.items():
        model = CoupledEconBiophysicalModel(
            region=region, econ=s1, region_key=r_name, t_max=t_max,
        )
        df = model.run()
        fert_yr1 = df['fert_applied_kgha'].iloc[1]
        reduction = (1 - fert_yr1 / region.synth_n_current) * 100
        region_n = region.cropland_mha * region.synth_n_current
        total_n_baseline += region_n
        total_n_shocked += region.cropland_mha * fert_yr1
        results[r_name] = {
            'fert_baseline': region.synth_n_current,
            'fert_yr1': fert_yr1,
            'reduction_pct': reduction,
            'eps_F_PF': REGIONAL_ECON_PARAMS.get(r_name, {}).get('eps_F_PF', s1.eps_F_PF),
        }

    global_reduction = (1 - total_n_shocked / total_n_baseline) * 100
    return results, global_reduction


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_supply_constrained(
    regions: Dict[str, RegionParams] = None,
    t_max: float = 100.0,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run supply-constrained scenarios for all regions."""
    if regions is None:
        regions = get_default_regions()

    scenarios = get_supply_constrained_scenarios()
    results = {}

    for s_name, econ in scenarios.items():
        results[s_name] = {}
        for r_name, region in regions.items():
            model = CoupledEconBiophysicalModel(
                region=region, econ=econ, region_key=r_name, t_max=t_max,
            )
            results[s_name][r_name] = model.run()

    return results


if __name__ == '__main__':
    print("=" * 70)
    print("COUPLED ECONOMIC-BIOPHYSICAL MODEL")
    print("Fertilizer Price Shock Scenarios (Manning-Wallenstein)")
    print("=" * 70)

    regions = get_default_regions()

    # Calibrated price shock
    shock = calibrate_price_shock(0.20)
    print(f"\nCalibrated price shock for 20% global fert reduction: {shock:.1%}")

    # Validation: check regional reductions
    print(f"\n{'':=<70}")
    print("VALIDATION: Regional fert reductions under S1 (no behavioral response)")
    print(f"{'':=<70}")
    val_results, global_red = validate_fert_reduction()
    print(f"{'Region':<35s}  {'eps_F_PF':>8s}  {'Baseline':>8s}  {'Year 1':>8s}  {'Cut':>6s}")
    print("-" * 70)
    for r_name, region in regions.items():
        v = val_results[r_name]
        print(f"  {region.name:<33s}  {v['eps_F_PF']:>8.2f}  {v['fert_baseline']:>7.0f}  "
              f"{v['fert_yr1']:>7.1f}  {v['reduction_pct']:>5.1f}%")
    print(f"  {'GLOBAL WEIGHTED':>33s}  {'':>8s}  {'':>8s}  {'':>8s}  {global_red:>5.1f}%")

    # Run all scenarios
    all_results = run_all_scenarios(regions, t_max=100)

    print(f"\n{'':=<70}")
    print(f"GLOBAL SUMMARY: 100-Year Trajectories by Scenario")
    print(f"{'':=<70}")

    for s_name in ['S1', 'S2', 'S3']:
        agg = aggregate_global(all_results[s_name], regions)
        print(f"\n--- {s_name} ---")
        for yr in [0, 5, 10, 25, 50, 100]:
            if yr < len(agg):
                row = agg.iloc[yr]
                print(f"  Year {yr:3d}: Yield={row['yield_fraction']:.3f}  "
                      f"SOC={row['soc_fraction']:.3f}  "
                      f"FoodPrice={row['food_price_index']:.3f}  "
                      f"Fert={row['fert_applied_kgha']:.1f} kg/ha  "
                      f"Land={row['total_land_mha']:.0f} Mha  "
                      f"Pop={row['pop_supported_millions']:.0f} M")

    # Supply-constrained scenarios
    print(f"\n{'':=<70}")
    print(f"SUPPLY-CONSTRAINED SCENARIOS (full behavioral + soil-N feedback + ceiling)")
    print(f"{'':=<70}")

    sc_results = run_supply_constrained(regions, t_max=100)

    for s_name in ['SC1_20pct', 'SC2_20pct_recovery']:
        agg = aggregate_global(sc_results[s_name], regions)
        print(f"\n--- {s_name} ---")
        for yr in [0, 5, 10, 25, 50, 100]:
            if yr < len(agg):
                row = agg.iloc[yr]
                print(f"  Year {yr:3d}: Yield={row['yield_fraction']:.3f}  "
                      f"SOC={row['soc_fraction']:.3f}  "
                      f"FoodPrice={row['food_price_index']:.3f}  "
                      f"Fert={row['fert_applied_kgha']:.1f} kg/ha  "
                      f"Land={row['total_land_mha']:.0f} Mha  "
                      f"Pop={row['pop_supported_millions']:.0f} M")

    # Regional detail: S1 vs S3 at year 50
    print(f"\n{'':=<70}")
    print("REGIONAL DETAIL: S1 vs S3 at Year 50")
    print(f"{'':=<70}")
    print(f"{'Region':<25} {'S1 Yield':>10} {'S3 Yield':>10} {'S1 SOC':>8} {'S3 SOC':>8} "
          f"{'S1 Fert':>10} {'S3 Fert':>10} {'S3 Price':>10}")
    print("-" * 91)
    for r_name in regions.keys():
        s1 = all_results['S1'][r_name].iloc[50]
        s3 = all_results['S3'][r_name].iloc[50]
        print(f"{regions[r_name].name:<25} "
              f"{s1['yield_fraction']:>10.3f} {s3['yield_fraction']:>10.3f} "
              f"{s1['soc_fraction']:>8.3f} {s3['soc_fraction']:>8.3f} "
              f"{s1['fert_applied_kgha']:>10.1f} {s3['fert_applied_kgha']:>10.1f} "
              f"{s3['food_price_index']:>10.3f}")
