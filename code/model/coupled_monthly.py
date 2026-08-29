"""
Coupled Economic-Monthly-Biophysical Model
===========================================

Integrates Dale Manning's partial equilibrium economic framework with
Wallenstein's monthly N model (v3 hybrid: annual SOM + monthly N availability).

Architecture:
    At each annual timestep:
    1. Economic module resolves market equilibrium (fertilizer demand, food price,
       land allocation) given current soil state and price shock.
    2. Monthly biophysical module runs 12 monthly N balance steps with the
       economically determined fertilizer rate, yielding crop N uptake.
    3. Yield computed from crop N uptake (Mitscherlich + stoichiometric cap).
    4. Annual SOM pools updated from residue C inputs.
    5. Local elasticities computed for next period's economic solve.

Key difference from coupled_econ_biophysical.py:
    - N losses (leaching, denitrification) are resolved monthly with climate
    - Crop uptake is seasonally constrained (growing season only)
    - Stoichiometric cap on yield is inherent in the architecture
    - NUE emerges from process rather than being a tuning parameter

Author: Matthew Wallenstein & Dale Manning
"""

import copy

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

from soil_n_model import (
    SOMPoolParams, CropParams, RegionParams, FeedbackParams,
    get_default_regions, som_params_for_region,
)

# Import monthly N model components
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from monthly_model_v3 import (
    MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES, FAOSTAT_TARGETS,
    monthly_n_balance, temp_factor, moist_factor,
    growing_months, demand_profile, fert_profile,
    calibrate_ym, run_model, update_som_pools,
    century_dynamic_spinup,
    get_regional_bnf,
)
from scipy.optimize import brentq

# Re-export economic infrastructure unchanged from the original coupled model
from coupled_econ_biophysical import (
    EconParams, REGIONAL_ECON_PARAMS,
    calibrate_price_shock, get_scenario_params, get_supply_constrained_scenarios,
    supply_state,
)
from parameter_registry import (
    SOC_T_C_HA_PER_PERCENT_30CM,
    WATER_STRESS_GAIN_SAT_SOC_PCT,
    WATER_STRESS_MIN_FACTOR,
    WATER_STRESS_SOFTPLUS_EPS_MM,
)


# ============================================================
# MONTHLY BIOPHYSICAL ENGINE
# ============================================================

class MonthlyBiophysicalEngine:
    """Wraps the monthly N model to provide N/yield dynamics for the economic model.

    At each annual timestep, accepts externally determined fertilizer rate and
    returns yield, N mineralization, updated SOM state, and local elasticities.
    """

    def __init__(self, region: RegionParams, region_key: str = None,
                 som_params: SOMPoolParams = None, crop_params: CropParams = None,
                 feedback_params: FeedbackParams = None,
                 monthly_params: MonthlyNParams = None,
                 yield_max_override: float = None,
                 initial_pools: dict = None):
        self.region = region
        self.region_key = region_key
        # Regional SOM parameterization: tropical regions use Laub et al.
        # 2024 (Biogeosciences 21:3691–3716) Kenya-calibrated DayCent
        # posterior ratios applied to k_slow and k_passive. Temperate
        # regions retain Century/RothC defaults.
        # See tropical-reparam-2026-04-14/PARAMETERS.md for full mapping.
        self.som = som_params or som_params_for_region(region_key)
        self.crop = crop_params or CropParams()
        self.fb = feedback_params or FeedbackParams()
        self.mp = monthly_params or MonthlyNParams()

        # Climate data
        self.climate = REGIONAL_CLIMATES.get(region_key, REGIONAL_CLIMATES['north_america'])

        # Yield parameters
        self.mit_c = (region.mitscherlich_c_regional
                      if region.mitscherlich_c_regional > 0 else self.crop.mitscherlich_c)
        if yield_max_override is not None:
            self.y_max = yield_max_override
        elif region_key and region_key in FAOSTAT_TARGETS:
            # F-002: calibrate on the path this engine actually runs, not on
            # `run_model`. `calibrate_ym_production` is defined below in this
            # module; the late lookup avoids a forward reference at class-body
            # definition time and costs nothing at call time.
            self.y_max = calibrate_ym_production(
                region_key, FAOSTAT_TARGETS[region_key], self.mp,
                region=region, som_params=self.som, crop_params=self.crop)
        else:
            self.y_max = region.yield_max_regional
        self.y_floor = region.yield_min_regional if region.yield_min_regional > 0 else 0.0

        # Stoichiometric N cost per tonne grain
        self.n_cost_per_tonne = self.crop.grain_n_fraction * 1000  # 18 kg N / t grain

        # Baseline BNF
        self.bnf_baseline = get_regional_bnf(region_key or 'north_america')

        # Dynamic spinup: iterate to true steady state (eliminates
        # transient from fraction-based pool allocation). Pass through
        # the engine's SOM, crop, and region overrides so Monte Carlo
        # parameter perturbations propagate into the equilibrium pools.
        # `initial_pools` (optional) skips the spinup and seeds the
        # equilibrium directly from a previously computed dict — used by
        # MC ensembles that spin up once per (region, draw) and reuse
        # the equilibrium across SOC-scaling sub-runs.
        if initial_pools is not None:
            self.C_active = initial_pools['c_active']
            self.C_slow = initial_pools['c_slow']
            self.C_passive = initial_pools['c_passive']
            self.soc_initial = initial_pools['soc']
            self.mineral_n = initial_pools['mineral_n']
            self.yield_baseline = initial_pools['yield_eq']
            self.n_min_baseline = initial_pools['n_min_eq']
        else:
            eq = century_dynamic_spinup(
                region_key or 'north_america',
                p=self.mp,
                synth_n=region.synth_n_current,
                yield_max_override=self.y_max,
                som_params=self.som,
                crop_params=self.crop,
                region_override=region,
            )
            self.C_active = eq['c_active']
            self.C_slow = eq['c_slow']
            self.C_passive = eq['c_passive']
            self.soc_initial = eq['soc']
            self.mineral_n = eq['mineral_n']
            self.yield_baseline = eq['yield_eq']
            self.n_min_baseline = eq['n_min_eq']

    def _water_stress(self) -> float:
        """Water-stress factor (0-1) from a smooth, two-sided SOC-WHC response.

        Empirical anchor: Minasny & McBratney (2018), EJSS 69:39-47 — meta-
        analysis (~50 studies) finds an approximately linear WHC vs. SOC
        slope at low-to-moderate SOC, with diminishing returns at high SOC
        as porosity approaches its texture-determined ceiling. Hudson
        (1994), J Soil Water Conserv 49:189-194, reports a log-linear
        SOM-WHC relationship with the same qualitative saturation. We
        therefore use:

            • Loss side (SOC < baseline): linear with full whc_sensitivity
              (matches the meta-analytic slope; identical to the previous
              formulation, so calibration at SOC = baseline is preserved).
            • Gain side (SOC > baseline): exponential saturation toward a
              ceiling whc_gain_max_mm = whc_gain_sat_pct · whc_sensitivity,
              with the same initial slope at baseline. This gives C¹
              continuity at SOC = baseline (no kink) and a physically-
              motivated diminishing return.

        Replaces the previous one-sided ``max(0, soc_init - soc)`` clamp,
        which produced a hard slope discontinuity at SOC = baseline (see
        paper2-soil-resilience/figures_analysis/figure1b_diagnosis/).
        """
        if not self.fb.physical_feedback:
            return 1.0

        soc_current = self.C_active + self.C_slow + self.C_passive
        # 1 percentage point SOC in 0-30 cm at bulk density 1.3 g cm-3
        # corresponds to 39 t C ha-1.
        soc_pct = soc_current / SOC_T_C_HA_PER_PERCENT_30CM
        soc_pct_init = self.soc_initial / SOC_T_C_HA_PER_PERCENT_30CM
        delta = soc_pct_init - soc_pct  # >0 when degraded, <0 when accumulated
        whc_sens = self.region.whc_sensitivity * self.fb.physical_strength

        # Saturation scale for the gain side: characteristic SOC% at which
        # ~63% of the maximum WHC gain is realised. Anchored to the upper
        # end of the typical agronomic SOC range — i.e. an additional
        # 1% SOC above baseline captures most of the achievable WHC
        # benefit. (Minasny & McBratney 2018 Fig. 4 — slope flattens as
        # SOC approaches site-specific maxima ~3-5%.)
        whc_gain_sat_pct = WATER_STRESS_GAIN_SAT_SOC_PCT
        if delta >= 0.0:
            whc_change_mm = delta * whc_sens
        else:
            whc_gain_max_mm = whc_gain_sat_pct * whc_sens
            whc_change_mm = -whc_gain_max_mm * (
                1.0 - np.exp(delta / whc_gain_sat_pct)
            )

        # Smooth floor on total water deficit. Physical interpretation:
        # even regions with ~0 mean annual deficit experience seasonal and
        # within-region distributional deficit, so the hard ``max(0, …)``
        # clamp over-sharpens the transition. Soft-abs formulation:
        #   soft_pos(x, ε) = 0.5 · (x + √(x² + ε²))
        # is C¹-smooth, monotonically ≥ 0, soft_pos(x, ε) → x for x ≫ ε and
        # soft_pos(x, ε) → 0 for x ≪ -ε. ε = 3 mm represents seasonal /
        # spatial deficit variance that persists even at zero mean; large
        # deficits (≥10 mm) are perturbed by <5%, so degraded-soil
        # behaviour is essentially unchanged.
        raw = self.region.baseline_water_deficit + whc_change_mm
        eps_mm = WATER_STRESS_SOFTPLUS_EPS_MM
        total_deficit = 0.5 * (raw + np.sqrt(raw * raw + eps_mm * eps_mm))
        stress = 1.0 - self.region.water_stress_coeff * total_deficit
        return max(WATER_STRESS_MIN_FACTOR, min(1.0, stress))

    def step(self, fert_applied: float, bnf: float = None) -> Dict:
        """Advance one year with externally determined fertilizer.

        Args:
            fert_applied: kg N/ha/yr of synthetic fertilizer
            bnf: biological N fixation (kg N/ha/yr), defaults to self.bnf_baseline

        Returns:
            dict with yield, N dynamics, SOM state, elasticities
        """
        if bnf is None:
            bnf = self.bnf_baseline

        # Run 12-month N balance — pass perturbed SOM so MC k_slow_mult
        # propagates into mineralization (otherwise the function would
        # silently fall back to default SOMPoolParams).
        nb = monthly_n_balance(
            self.C_active, self.C_slow, self.C_passive,
            self.region.cn_bulk, fert_applied, bnf,
            self.region.atm_n_deposition, self.climate,
            self.mineral_n, self.mp,
            som_params=self.som,
        )
        self.mineral_n = nb['mineral_n_end']
        n_uptake = nb['uptake']
        n_min_annual = nb['min']

        # Water stress from SOC state
        ws = self._water_stress()

        # Yield from crop N uptake (Mitscherlich + stoichiometric cap + water stress)
        y_mit = self.y_max * (1.0 - np.exp(-self.mit_c * n_uptake)) * ws
        y_stoich = n_uptake / self.n_cost_per_tonne if self.n_cost_per_tonne > 0 else y_mit
        y = min(y_mit, y_stoich)
        y = max(self.y_floor, y)

        # Residue C input
        hi = self.crop.harvest_index
        rf = (1 - hi) / hi
        rr = self.region.residue_retention
        shoot_c = y * self.crop.residue_c_fraction * rf * rr
        root_c = (
            y * self.crop.residue_c_fraction * rf
            * self.region.root_shoot_c_ratio
        )
        res_c = (shoot_c + root_c) * self.region.cre_regional

        # Annual SOM pool update
        self.C_active, self.C_slow, self.C_passive = update_som_pools(
            self.C_active, self.C_slow, self.C_passive, res_c, self.som
        )
        soc_new = self.C_active + self.C_slow + self.C_passive

        # Compute local elasticities for economic module
        # The Mitscherlich elasticity w.r.t. n_uptake:
        #   dy/dN = y_max * mit_c * exp(-mit_c * N) * ws
        #   elasticity = (dy/dN) * (N/y) = mit_c * N * exp(-cN) / (1 - exp(-cN))
        # But if stoichiometric cap binds, dy/dN = 1/n_cost_per_tonne
        eps = 1e-10
        n_up = max(n_uptake, eps)

        if y_stoich < y_mit:
            # Stoichiometric cap binds: linear regime
            elasticity_n_total = 1.0  # unit elasticity (proportional)
        else:
            # Mitscherlich governs
            exp_term = np.exp(-self.mit_c * n_up)
            denom = max(1.0 - exp_term, eps)
            elasticity_n_total = self.mit_c * n_up * exp_term / denom

        # Partition elasticity between soil N and applied fertilizer.
        # Use gross input shares (consistent basis for all sources).
        total_n_input = n_min_annual + fert_applied + bnf + self.region.atm_n_deposition
        if total_n_input > eps:
            soil_share = n_min_annual / total_n_input
            fert_share = fert_applied / total_n_input
        else:
            soil_share = 0.5
            fert_share = 0.5

        beta = elasticity_n_total * max(0.0, soil_share)
        gamma = elasticity_n_total * fert_share

        return {
            'yield_tha': y,
            'yield_fraction': y / self.yield_baseline if self.yield_baseline > 0 else 0,
            'n_mineralized': n_min_annual,
            'n_uptake': n_uptake,
            'n_leached': nb['leach'],
            'n_denitrified': nb['den'],
            'n_immobilized': nb['immob'],
            'soc_total': soc_new,
            'soc_fraction': soc_new / self.soc_initial if self.soc_initial > 0 else 1,
            'water_stress': ws,
            'beta': beta,
            'gamma': gamma,
        }


# ============================================================
# COUPLED MODEL
# ============================================================

class CoupledMonthlyModel:
    """
    Coupled economic-biophysical model using monthly N resolution.

    Same economic equilibrium framework as CoupledEconBiophysicalModel
    but with MonthlyBiophysicalEngine replacing the annual BiophysicalSOMEngine.

    At each timestep:
    1. Economic module solves for PY_hat, F_hat, L_hat
    2. Monthly biophysical module runs 12 months with economically determined F
    3. Yield, SOC, and elasticities feed back to next period
    """

    def __init__(
        self,
        region: RegionParams,
        econ: EconParams,
        region_key: str = None,
        t_max: float = 100.0,
        dt: float = 1.0,
        yield_max_override: float = None,
        initial_pools: dict = None,
        som_params: SOMPoolParams = None,
        crop_params: CropParams = None,
        feedback_params: FeedbackParams = None,
        monthly_params: MonthlyNParams = None,
    ):
        self.region = region
        self.econ = econ
        self.region_key = region_key
        self.t_max = t_max
        self.dt = dt

        # Resolve elasticities (same logic as original coupled model)
        rp = {}
        if region_key and region_key in REGIONAL_ECON_PARAMS:
            rp = REGIONAL_ECON_PARAMS[region_key]

        # Structural params: always use regional if available
        self.eta = rp.get('eta', econ.eta)
        self.alpha = rp.get('alpha', econ.alpha)
        self.eps_F_PF = rp.get('eps_F_PF', econ.eps_F_PF)

        # Response-channel params
        self.eps_F_PY = econ.eps_F_PY if econ.eps_F_PY == 0.0 else rp.get('eps_F_PY', econ.eps_F_PY)
        self.eps_F_N = econ.eps_F_N

        # Land market
        if econ.eps_LD_PL == 0.0 and econ.eps_LD_PY == 0.0 and econ.eps_LS_PL == 0.0:
            self.eps_LD_PL = 0.0
            self.eps_LD_PY = 0.0
            self.eps_LS_PL = 0.0
        else:
            self.eps_LD_PL = rp.get('eps_LD_PL', econ.eps_LD_PL)
            self.eps_LD_PY = rp.get('eps_LD_PY', econ.eps_LD_PY)
            self.eps_LS_PL = rp.get('eps_LS_PL', econ.eps_LS_PL)

        # Initialize biophysical engine
        self.bio = MonthlyBiophysicalEngine(
            region, region_key=region_key,
            som_params=som_params,
            crop_params=crop_params,
            feedback_params=feedback_params,
            monthly_params=monthly_params,
            yield_max_override=yield_max_override,
            initial_pools=initial_pools,
        )

        # Baseline values
        self.F_baseline = region.synth_n_current
        self.L_baseline = region.cropland_mha
        self.Y_baseline = self.bio.yield_baseline
        self.N_min_baseline = self.bio.n_min_baseline

        # Price shock
        self.PF_hat_base = np.log(1 + econ.fert_price_shock)

        # Track log-changes
        self.PF_hat = self.PF_hat_base
        self.PY_hat = 0.0
        self.F_hat = 0.0
        self.L_hat = 0.0
        self.N_hat = 0.0

    def _lambda_L(self) -> float:
        """Land-market reduction coefficient, L_hat = lambda_L * PY_hat."""
        if abs(self.eps_LS_PL - self.eps_LD_PL) > 1e-10:
            return self.eps_LS_PL * self.eps_LD_PY / (self.eps_LS_PL - self.eps_LD_PL)
        return 0.0

    def _solve_equilibrium(self, beta: float, gamma: float) -> Tuple[float, float, float]:
        """Solve the simultaneous system for PY_hat, F_hat, L_hat.

        Identical to the original coupled model's equilibrium solver.
        """
        if abs(self.eps_LS_PL - self.eps_LD_PL) > 1e-10:
            lambda_L = self.eps_LS_PL * self.eps_LD_PY / (self.eps_LS_PL - self.eps_LD_PL)
        else:
            lambda_L = 0.0

        numerator = (beta * self.N_hat +
                     gamma * (self.eps_F_PF * self.PF_hat + self.eps_F_N * self.N_hat))
        denominator = self.eta - self.alpha * lambda_L - gamma * self.eps_F_PY

        if abs(denominator) > 1e-10:
            PY_hat = numerator / denominator
        else:
            PY_hat = 0.0

        F_hat = self.eps_F_PF * self.PF_hat + self.eps_F_PY * PY_hat + self.eps_F_N * self.N_hat
        L_hat = lambda_L * PY_hat

        return PY_hat, F_hat, L_hat

    def _solve_equilibrium_capped(self, beta: float, gamma: float, ln_c: float):
        """Constrained equilibrium when the physical fertilizer cap binds.

        When supply is rationed, fertilizer quantity is set by physical
        availability, not price: F_hat = ln(c) - L_hat = ln(c) - lambda_L*PY_hat
        (exactly the per-hectare cap F0*L0*c / L_level in log-changes).
        Substituting into supply Y_hat = alpha*L_hat + beta*N_hat + gamma*F_hat
        and imposing market clearing Y_hat = eta*PY_hat gives

            PY_hat = (beta*N_hat + gamma*ln(c)) / (eta - (alpha - gamma)*lambda_L).

        Food price and land are then consistent with the fertilizer actually
        available; the fertilizer-price term drops out because price no longer
        rations demand once the physical cap binds (constrained-cap fix).
        """
        if abs(self.eps_LS_PL - self.eps_LD_PL) > 1e-10:
            lambda_L = self.eps_LS_PL * self.eps_LD_PY / (self.eps_LS_PL - self.eps_LD_PL)
        else:
            lambda_L = 0.0

        num = beta * self.N_hat + gamma * ln_c
        den = self.eta - (self.alpha - gamma) * lambda_L
        PY_hat = num / den if abs(den) > 1e-10 else 0.0
        L_hat = lambda_L * PY_hat
        F_hat = ln_c - L_hat
        return PY_hat, F_hat, L_hat

    def _clear_market_realized(self, supply):
        """Clear the food market on the realized biogeochemical yield (F-024).

        Until v15 the equilibrium closed on the log-linear supply relation
        Y_hat = alpha*L_hat + beta*N_hat + gamma*F_hat, so the reported food
        price cleared a first-order expansion of the production response while
        the reported production was the nonlinear one the biogeochemistry
        delivered. F-022 measured the gap (1.54 pp worst, ~0.22 pp chronic) and
        F-023 traced the price bias to about -1 pp at year 10. This method is
        Dale Manning's remedy: root-find the food price at which demand equals
        the production change the biophysical model actually delivers,

            eta*PY = ln(yield_frac(F_level(PY))) + alpha*lambda_L*PY,

        evaluating each candidate price by running the monthly biophysical
        step for that price's fertilizer level from a snapshot of the soil
        state. beta and gamma no longer enter the clearing at all; they are
        recorded as diagnostics only. The physical supply ceiling is a
        quantity constraint inside the residual, so no separate capped solver
        exists to drift from this one.

        The prototype behind this wiring reproduced the linear model to
        0.0e+00 before its realized mode was believed, and the realized mode
        converged in 8 brentq evaluations per step over 240 clearings
        (logs/run_228_proto.log). Returns the achieved residual so that an
        external check can fail if convergence ever degrades.
        """
        snap = copy.deepcopy(self.bio)

        def implied(PY):
            F_hat = (self.eps_F_PF * self.PF_hat + self.eps_F_PY * PY
                     + self.eps_F_N * self.N_hat)
            L_hat = self._lambda_L() * PY
            F_level = max(0.0, self.F_baseline * np.exp(F_hat))
            L_level = self.L_baseline * np.exp(L_hat)
            capped = False
            if supply.ceiling < 1.0:
                F_max = (self.F_baseline * self.L_baseline * supply.ceiling
                         / max(L_level, 1e-6))
                if F_level > F_max:
                    F_level, capped = F_max, True
            return F_hat, L_hat, F_level, L_level, capped

        def residual(PY):
            _, L_hat, F_level, _, _ = implied(PY)
            trial = copy.deepcopy(snap)
            yf = max(trial.step(F_level)['yield_fraction'], 1e-9)
            return self.eta * PY - (np.log(yf) + self.alpha * L_hat)

        lo, hi = self.PY_hat - 0.10, self.PY_hat + 0.10
        for _ in range(12):
            if residual(lo) * residual(hi) < 0:
                break
            lo -= 0.10
            hi += 0.10
        else:
            raise RuntimeError(
                'realized clearing found no bracket for region %r; the food '
                'market did not clear' % (self.region_key,))
        PY_hat = brentq(residual, lo, hi, xtol=1e-12)
        F_hat, L_hat, F_level, L_level, capped = implied(PY_hat)
        self.bio = snap
        bio_state = self.bio.step(F_level)
        resid = self.eta * PY_hat - (
            np.log(max(bio_state['yield_fraction'], 1e-9))
            + self.alpha * L_hat)
        return PY_hat, F_hat, L_hat, F_level, L_level, capped, bio_state, resid

    def run(self) -> pd.DataFrame:
        """Run the coupled simulation."""
        n_steps = int(self.t_max / self.dt) + 1

        results = {
            'year': np.zeros(n_steps),
            'PF_hat': np.zeros(n_steps),
            'PY_hat': np.zeros(n_steps),
            'F_hat': np.zeros(n_steps),
            'L_hat': np.zeros(n_steps),
            'N_hat': np.zeros(n_steps),
            'fert_applied_kgha': np.zeros(n_steps),
            'land_mha': np.zeros(n_steps),
            'food_price_index': np.zeros(n_steps),
            'yield_tha': np.zeros(n_steps),
            'yield_fraction': np.zeros(n_steps),
            'n_mineralized': np.zeros(n_steps),
            'n_uptake': np.zeros(n_steps),
            'n_leached': np.zeros(n_steps),
            'n_denitrified': np.zeros(n_steps),
            'n_immobilized': np.zeros(n_steps),
            'soc_total': np.zeros(n_steps),
            'soc_fraction': np.zeros(n_steps),
            'water_stress': np.zeros(n_steps),
            'beta': np.zeros(n_steps),
            'gamma': np.zeros(n_steps),
            'total_production_index': np.zeros(n_steps),
            'carrying_capacity_fraction': np.zeros(n_steps),
            'cap_binding': np.zeros(n_steps),
            'clearing_residual': np.zeros(n_steps),
            # F-010 diagnostic. ln(ceiling) for the step, after any capacity
            # recovery, so that an external check can re-solve the constrained
            # equilibrium from the DataFrame alone rather than reaching into
            # solver internals. NaN in steps where the cap does not bind.
            'ln_cap': np.full(n_steps, np.nan),
        }

        # Compute initial elasticities without mutating state
        # Run a diagnostic step to get beta/gamma at baseline
        bnf_base = self.bio.bnf_baseline
        init_nb = monthly_n_balance(
            self.bio.C_active, self.bio.C_slow, self.bio.C_passive,
            self.region.cn_bulk, self.F_baseline, bnf_base,
            self.region.atm_n_deposition, self.bio.climate,
            self.bio.mineral_n, self.bio.mp,
            som_params=self.bio.som,
        )
        n_up_init = init_nb['uptake']
        n_min_init = init_nb['min']
        eps = 1e-10
        n_up_safe = max(n_up_init, eps)

        # Check if stoichiometric cap would bind
        y_mit_init = self.bio.y_max * (1.0 - np.exp(-self.bio.mit_c * n_up_safe))
        y_stoich_init = n_up_safe / self.bio.n_cost_per_tonne

        if y_stoich_init < y_mit_init:
            init_elast = 1.0
        else:
            exp_term = np.exp(-self.bio.mit_c * n_up_safe)
            denom = max(1.0 - exp_term, eps)
            init_elast = self.bio.mit_c * n_up_safe * exp_term / denom

        # Gross input shares (consistent with step())
        total_n_input = n_min_init + self.F_baseline + bnf_base + self.region.atm_n_deposition
        soil_share = n_min_init / max(total_n_input, eps)
        fert_share = self.F_baseline / max(total_n_input, eps)
        init_beta = init_elast * max(0.0, soil_share)
        init_gamma = init_elast * fert_share

        # Normalize N_hat to the year-0 recorded mineralization so that
        # N_hat = 0 at the true baseline flux (stationary-baseline fix, second
        # discontinuity). Previously N_min_baseline was the spin-up value
        # n_min_eq, which differed from the year-0 recorded flux and made
        # N_hat nonzero in year 1 before the shock affected the soil.
        self.N_min_baseline = n_min_init

        for i in range(n_steps):
            t = i * self.dt
            results['year'][i] = t

            if i == 0:
                # Baseline year
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
                results['n_mineralized'][i] = n_min_init
                results['n_uptake'][i] = n_up_init
                results['n_leached'][i] = init_nb['leach']
                results['n_denitrified'][i] = init_nb['den']
                results['n_immobilized'][i] = init_nb['immob']
                results['soc_total'][i] = soc_0
                results['soc_fraction'][i] = 1.0
                results['water_stress'][i] = self.bio._water_stress()
                results['beta'][i] = init_beta
                results['gamma'][i] = init_gamma
                results['total_production_index'][i] = 1.0
                results['carrying_capacity_fraction'][i] = 1.0
                results['cap_binding'][i] = 0.0
                results['clearing_residual'][i] = 0.0
                continue

            # Update N_hat from biophysical state (use previous step's mineralization)
            current_n_min = results['n_mineralized'][i-1]
            if self.N_min_baseline > 0:
                self.N_hat = np.log(max(current_n_min, 1e-6) / self.N_min_baseline)
            else:
                self.N_hat = 0.0

            # Disruption timeline. One definition, in soil-side
            # coupled_econ_biophysical.supply_state; see its docstring.
            supply = supply_state(self.econ, t)
            self.PF_hat = self.PF_hat_base * supply.price_frac

            # The food market clears on the realized biogeochemical yield;
            # see _clear_market_realized. The linearized solve survives only
            # as the bracket guess, and beta/gamma are diagnostics.
            beta = results['beta'][i-1]
            gamma = results['gamma'][i-1]
            self.PY_hat = self._solve_equilibrium(beta, gamma)[0]
            (PY_hat, F_hat, L_hat, F_level, L_level, cap_binding,
             bio_state, clearing_residual) = self._clear_market_realized(supply)
            ln_cap_i = np.log(supply.ceiling) if cap_binding else np.nan

            self.PY_hat = PY_hat
            self.F_hat = F_hat
            self.L_hat = L_hat

            # Total production index
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
            results['n_uptake'][i] = bio_state['n_uptake']
            results['n_leached'][i] = bio_state['n_leached']
            results['n_denitrified'][i] = bio_state['n_denitrified']
            results['n_immobilized'][i] = bio_state['n_immobilized']
            results['soc_total'][i] = bio_state['soc_total']
            results['soc_fraction'][i] = bio_state['soc_fraction']
            results['water_stress'][i] = bio_state['water_stress']
            results['beta'][i] = bio_state['beta']
            results['gamma'][i] = bio_state['gamma']
            results['total_production_index'][i] = total_prod_index
            results['carrying_capacity_fraction'][i] = total_prod_index
            results['cap_binding'][i] = 1.0 if cap_binding else 0.0
            results['clearing_residual'][i] = clearing_residual
            results['ln_cap'][i] = ln_cap_i

        return pd.DataFrame(results)


# ============================================================
# CALIBRATION — PRODUCTION PATH
# ============================================================
#
# Finding F-002 (2026-07-25). `monthly_model_v3.calibrate_ym` roots on
# `run_model`, which uses the global `CropParams.mitscherlich_c` and applies
# no baseline water-stress multiplier. Every published run goes through
# `century_dynamic_spinup` plus `MonthlyBiophysicalEngine`, which use
# `region.mitscherlich_c_regional` and do apply water stress. The manuscript's
# statement that yields are calibrated to FAOSTAT was true of a path that was
# never run: measured under the published ERA5 forcing, production baseline
# yields missed their FAOSTAT targets by -3.87% (South Asia) to +4.19% (Latin
# America).
#
# No test caught it because every test compared the model to itself.
#
# `calibrate_ym_production` roots the production path itself. The legacy
# `calibrate_ym` is left importable on purpose: the test measures the gap
# rather than deleting the evidence.

#: Identifies the code path a cached `yield_max` was fitted on. It is the
#: first element of `calibration_fingerprint`, so every `yield_max` cached on
#: disk under the old scheme is stale by construction and cannot be reused.
CALIBRATION_SCHEME = 'production_path_v2'

#: RegionParams fields that enter the calibration objective and therefore
#: must be hashed into the fingerprint. The first nine are the fields the
#: legacy `run_model` objective touched. `production_path_v2` added the last
#: four: the production path uses the regional Mitscherlich curvature rather
#: than the global one, and applies the baseline water-stress multiplier.
#: A field that moves `yield_max` and is not in this tuple is a silent
#: cache-poisoning bug; `test_calibration_fingerprint` perturbs all 19
#: RegionParams fields and fails if an unregistered one moves the answer.
YM_REGION_FIELDS = (
    'soc_initial',
    'cn_bulk',
    'synth_n_current',
    'atm_n_deposition',
    'residue_retention',
    'root_shoot_c_ratio',
    'cre_regional',
    'yield_min_regional',
    'yield_max_regional',
    # added by production_path_v2 (F-002)
    'mitscherlich_c_regional',
    'baseline_water_deficit',
    'water_stress_coeff',
    'whc_sensitivity',
)

#: Convergence tolerance on the calibrated baseline yield, as a fraction of
#: the FAOSTAT target. The published requirement is 1e-3 relative; the solver
#: is run tighter than that so the assertion has headroom.
YM_CALIBRATION_RTOL = 1e-5

_YM_CACHE = {}

def _mp_cache_key(mp: MonthlyNParams) -> tuple:
    """Hash MonthlyNParams fields for cache keying."""
    return (mp.q10, mp.t_ref, mp.t_min, mp.moist_opt_lo, mp.moist_opt_hi,
            mp.moist_min, mp.moist_waterlog, mp.leach_coeff, mp.leach_base,
            mp.denitrif_base, mp.denitrif_wet_mult, mp.immob_frac,
            mp.max_uptake_frac, mp.min_n_pool)


def _region_cache_key(region: RegionParams) -> tuple:
    """Hash the RegionParams fields the calibration objective depends on."""
    return tuple(float(getattr(region, f)) for f in YM_REGION_FIELDS)


def calibration_fingerprint(region_key: str,
                            mp: MonthlyNParams = None,
                            region: RegionParams = None) -> tuple:
    """Everything a cached `yield_max` is a function of.

    The scheme name comes first, so a cache written by an earlier scheme can
    never be mistaken for one written by this scheme, whatever else agrees.

    `som_params` is deliberately absent. F-003: a first-order pool at steady
    state passes its input through unchanged (`c_slow* = 0.46 c_in / k_slow`),
    so the mineralization flux `k_slow * c_slow*` is invariant to `k_slow` and
    SOM kinetics do not reach the calibrated baseline. Measured span of the
    baseline yield across the full `k_slow` prior is 0.098%, against a
    calibration tolerance of 0.20%.
    """
    if mp is None:
        mp = MonthlyNParams()
    if region is None:
        region = get_default_regions()[region_key]
    return (CALIBRATION_SCHEME, region_key,
            _mp_cache_key(mp), _region_cache_key(region))


def calibrate_ym_production(region_key: str,
                            target: float,
                            mp: MonthlyNParams = None,
                            region: RegionParams = None,
                            som_params: SOMPoolParams = None,
                            crop_params: CropParams = None,
                            rtol: float = YM_CALIBRATION_RTOL) -> float:
    """Calibrate `yield_max` on the code path the published runs use.

    Roots the equilibrium yield returned by `century_dynamic_spinup` — the
    same call `MonthlyBiophysicalEngine.__init__` makes, with the same
    regional Mitscherlich curvature, the same baseline water-stress
    multiplier and the same residue-C feedback into the SOM pools — against
    the region's FAOSTAT target.

    The objective is not monotone-trivial: raising `yield_max` raises yield,
    which raises residue C, which raises SOC and hence mineralization, which
    raises uptake. The fixed point is what is being solved, which is why this
    cannot be done on a five-year `run_model` call.

    Returns
    -------
    float
        `yield_max` (t/ha) such that the production-path equilibrium yield
        equals `target` to within `rtol` relative.
    """
    if mp is None:
        mp = MonthlyNParams()
    if region is None:
        region = get_default_regions()[region_key]

    def y_prod(ym: float) -> float:
        eq = century_dynamic_spinup(
            region_key,
            p=mp,
            synth_n=region.synth_n_current,
            yield_max_override=ym,
            som_params=som_params,
            crop_params=crop_params,
            region_override=region,
        )
        return eq['yield_eq']

    def obj(ym: float) -> float:
        return y_prod(ym) - target

    # Bracket outward rather than assuming [1, 50] contains a sign change.
    lo, hi = 1.0, 50.0
    f_lo, f_hi = obj(lo), obj(hi)
    expand = 0
    while f_lo * f_hi > 0.0 and expand < 8:
        if abs(f_lo) < abs(f_hi):
            lo = max(1e-3, lo / 4.0)
            f_lo = obj(lo)
        else:
            hi = hi * 4.0
            f_hi = obj(hi)
        expand += 1

    if f_lo * f_hi <= 0.0:
        ym = brentq(obj, lo, hi, xtol=1e-10, rtol=1e-14, maxiter=200)
        return float(ym)

    # No sign change even after expansion: the target is unreachable on this
    # path (a stoichiometric or water-stress ceiling below it). Report the
    # closest attainable value rather than a silently wrong root.
    best, best_e = lo, abs(f_lo)
    for ym in np.linspace(lo, hi, 80):
        e = abs(obj(ym))
        if e < best_e:
            best, best_e = float(ym), e
    return float(best)


def get_calibrated_ym(region_key: str, mp: MonthlyNParams = None,
                      region: RegionParams = None) -> float:
    """Get calibrated yield_max for a region, cached for performance.

    Calibrates on the production path (F-002). The cache key is the full
    `calibration_fingerprint`, whose first element is `CALIBRATION_SCHEME`,
    so a value fitted under the legacy `run_model` objective can never be
    served here.
    """
    if mp is None:
        mp = MonthlyNParams()
    if region is None:
        region = get_default_regions()[region_key]
    cache_key = calibration_fingerprint(region_key, mp, region)
    if cache_key not in _YM_CACHE:
        target = FAOSTAT_TARGETS[region_key]
        _YM_CACHE[cache_key] = calibrate_ym_production(
            region_key, target, mp, region)
    return _YM_CACHE[cache_key]


def clear_ym_cache():
    """Clear calibration cache (e.g., after changing MonthlyNParams)."""
    _YM_CACHE.clear()


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == '__main__':
    print('=' * 70)
    print('COUPLED MONTHLY MODEL — Smoke Test')
    print('=' * 70)

    regions = get_default_regions()
    mp = MonthlyNParams()

    # Calibrate all regions
    print('\nCalibrating yield_max for each region...')
    ym_cal = {}
    for rk in REGIONAL_CLIMATES:
        ym_cal[rk] = get_calibrated_ym(rk, mp)
        print(f'  {rk}: ym = {ym_cal[rk]:.3f}')

    # Run S3 for all regions
    scenarios = get_scenario_params()
    s3 = scenarios['S3']
    shock = calibrate_price_shock(0.20)
    print(f'\nCalibrated price shock: {shock:.4f} ({shock*100:.1f}%)')

    print(f'\n{"Region":<25} {"Y_base":>7} {"Y_yr10":>7} {"Loss%":>7} '
          f'{"SOC_0":>7} {"SOC_10":>7} {"Nup_0":>7} {"Nup_10":>7}')
    print('-' * 90)

    for rk, r in regions.items():
        ym = ym_cal[rk]
        model = CoupledMonthlyModel(
            region=r, econ=s3, region_key=rk, t_max=30.0,
            yield_max_override=ym,
        )
        df = model.run()
        yr0 = df[df['year'] == 0].iloc[0]
        yr10 = df[df['year'] == 10].iloc[0]
        loss = (1 - yr10['yield_fraction']) * 100
        print(f'  {r.name:<23} {yr0["yield_tha"]:>7.2f} {yr10["yield_tha"]:>7.2f} '
              f'{loss:>6.1f}% {yr0["soc_total"]:>7.1f} {yr10["soc_total"]:>7.1f} '
              f'{yr0["n_uptake"]:>7.1f} {yr10["n_uptake"]:>7.1f}')

    print('\nDone.')
