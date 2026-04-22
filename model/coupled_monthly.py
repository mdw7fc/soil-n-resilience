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

# Re-export economic infrastructure unchanged from the original coupled model
from coupled_econ_biophysical import (
    EconParams, REGIONAL_ECON_PARAMS,
    calibrate_price_shock, get_scenario_params, get_supply_constrained_scenarios,
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
                 yield_max_override: float = None):
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
            self.y_max = calibrate_ym(region_key, FAOSTAT_TARGETS[region_key], self.mp)
        else:
            self.y_max = region.yield_max_regional
        self.y_floor = region.yield_min_regional if region.yield_min_regional > 0 else 0.0

        # Stoichiometric N cost per tonne grain
        self.n_cost_per_tonne = self.crop.grain_n_fraction * 1000  # 18 kg N / t grain

        # Baseline BNF
        self.bnf_baseline = get_regional_bnf(region_key or 'north_america')

        # Dynamic spinup: iterate to true steady state (eliminates
        # transient from fraction-based pool allocation)
        eq = century_dynamic_spinup(
            region_key or 'north_america',
            p=self.mp,
            synth_n=region.synth_n_current,
            yield_max_override=self.y_max,
        )
        self.C_active = eq['c_active']
        self.C_slow = eq['c_slow']
        self.C_passive = eq['c_passive']
        self.soc_initial = eq['soc']

        # Mineral N pool at equilibrium
        self.mineral_n = eq['mineral_n']

        # Baseline yield and N mineralization from equilibrium
        self.yield_baseline = eq['yield_eq']
        self.n_min_baseline = eq['n_min_eq']

    def _water_stress(self) -> float:
        """Water stress factor (0-1) based on SOC-driven water holding capacity loss."""
        if not self.fb.physical_feedback:
            return 1.0

        soc_current = self.C_active + self.C_slow + self.C_passive
        soc_pct = soc_current / (300 * 0.01)  # 30cm depth, bulk density ~1.0
        soc_pct_init = self.soc_initial / (300 * 0.01)
        delta = max(0, soc_pct_init - soc_pct)
        whc_loss_mm = delta * self.region.whc_sensitivity * self.fb.physical_strength
        total_deficit = self.region.baseline_water_deficit + max(0, whc_loss_mm)
        stress = 1.0 - self.region.water_stress_coeff * total_deficit
        return max(0.3, min(1.0, stress))

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

        # Run 12-month N balance
        nb = monthly_n_balance(
            self.C_active, self.C_slow, self.C_passive,
            self.region.cn_bulk, fert_applied, bnf,
            self.region.atm_n_deposition, self.climate,
            self.mineral_n, self.mp
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
        shoot_c = y * 1000 * 0.45 * rf * rr / 1000  # t C/ha
        root_c = y * 1000 * 0.45 * rf * self.region.root_shoot_c_ratio / 1000
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
            yield_max_override=yield_max_override,
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
        }

        # Compute initial elasticities without mutating state
        # Run a diagnostic step to get beta/gamma at baseline
        bnf_base = self.bio.bnf_baseline
        init_nb = monthly_n_balance(
            self.bio.C_active, self.bio.C_slow, self.bio.C_passive,
            self.region.cn_bulk, self.F_baseline, bnf_base,
            self.region.atm_n_deposition, self.bio.climate,
            self.bio.mineral_n, self.bio.mp
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
                continue

            # Update N_hat from biophysical state (use previous step's mineralization)
            current_n_min = results['n_mineralized'][i-1]
            if self.N_min_baseline > 0:
                self.N_hat = np.log(max(current_n_min, 1e-6) / self.N_min_baseline)
            else:
                self.N_hat = 0.0

            # Update PF_hat with recovery
            if (self.econ.price_relaxes_with_recovery and
                    self.econ.fert_capacity_recovery_years > 0 and t > 0):
                recovery_frac = min(1.0, t / self.econ.fert_capacity_recovery_years)
                self.PF_hat = self.PF_hat_base * (1.0 - recovery_frac)
            else:
                self.PF_hat = self.PF_hat_base

            # Get elasticities from previous step
            beta = results['beta'][i-1]
            gamma = results['gamma'][i-1]

            # Solve equilibrium
            PY_hat, F_hat, L_hat = self._solve_equilibrium(beta, gamma)
            self.PY_hat = PY_hat
            self.F_hat = F_hat
            self.L_hat = L_hat

            # Convert to levels
            F_level = self.F_baseline * np.exp(F_hat)
            L_level = self.L_baseline * np.exp(L_hat)
            F_level = max(0.0, F_level)

            # Supply ceiling
            if self.econ.fert_supply_ceiling < 1.0:
                ceiling = self.econ.fert_supply_ceiling
                if self.econ.fert_capacity_recovery_years > 0 and t > 0:
                    recovery_frac = min(1.0, t / self.econ.fert_capacity_recovery_years)
                    ceiling = ceiling + (1.0 - ceiling) * recovery_frac
                total_n_available = self.F_baseline * self.L_baseline * ceiling
                F_max = total_n_available / max(L_level, 1e-6)
                F_level = min(F_level, F_max)

            # Advance biophysical model
            bio_state = self.bio.step(F_level)

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

        return pd.DataFrame(results)


# ============================================================
# CALIBRATION CACHE
# ============================================================

_YM_CACHE = {}

def _mp_cache_key(mp: MonthlyNParams) -> tuple:
    """Hash MonthlyNParams fields for cache keying."""
    return (mp.q10, mp.t_ref, mp.t_min, mp.moist_opt_lo, mp.moist_opt_hi,
            mp.moist_min, mp.moist_waterlog, mp.leach_coeff, mp.leach_base,
            mp.denitrif_base, mp.denitrif_wet_mult, mp.immob_frac,
            mp.max_uptake_frac, mp.min_n_pool)


def get_calibrated_ym(region_key: str, mp: MonthlyNParams = None) -> float:
    """Get calibrated yield_max for a region, cached for performance."""
    if mp is None:
        mp = MonthlyNParams()
    cache_key = (region_key, _mp_cache_key(mp))
    if cache_key not in _YM_CACHE:
        target = FAOSTAT_TARGETS[region_key]
        _YM_CACHE[cache_key] = calibrate_ym(region_key, target, mp)
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
