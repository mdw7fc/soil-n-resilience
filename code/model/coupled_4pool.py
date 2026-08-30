#!/usr/bin/env python3
"""Coupled economic model on the microbially-explicit 4-pool SOM scheme.

Structural-sensitivity companion to `coupled_monthly.py` (Supplementary
Note 2): the economic layer, disruption timeline, realized-yield market
clearing (F-025) and the two-sided SOC->WHC water-stress response are
identical to the Century/RothC coupled model; only the SOM engine differs
(`som_4pool_monthly.py`). Adopted into the deposit under F-029 from the
project's April four-pool line, with four corrections that bring it onto
the released core so the engine comparison isolates the stabilization
mechanism:

  P1. ERA5 forcing (the caller applies apply_era5_climate_file before
      constructing engines, as every deposited generator does).
  P2. Stationary spin-up: the equilibrium loop applies the same baseline
      water-stress multiplier the simulation applies (the v1.3 Century
      correction; the April spin-up lacked it).
  P3. Production-path y_max calibration: brentq roots the 4-pool dynamic
      spin-up's own equilibrium yield against the FAOSTAT target, the
      analogue of calibrate_ym_production (F-002/D3). Where the
      stoichiometric N bound keeps the equilibrium yield below the target
      at any ceiling (Sub-Saharan Africa under ERA5), the smallest
      plateau-achieving ceiling is used and the shortfall is recorded in
      CALIBRATION_SHORTFALL — disclosed, not retuned (F-008 rule).
  P4. The current smooth two-sided water-stress response shared with the
      Century engine, replacing the April one-sided clamp.
  P5/P6. Realized clearing and the scenario-supplied eps_F_N central
      (-0.50, F-026) enter exactly as in CoupledMonthlyModel.

`cue_fixed=True` holds CUE at its N-replete baseline so the
CUE-downregulation contribution can be isolated by difference
(Supplementary table 2). Generator: code/repro/run_4pool_comparison.py.
"""
import copy

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from som_4pool_monthly import (
    FourPoolParams, fourpool_annual_step, fourpool_analytic_init,
    monthly_n_balance_4pool,
)
from monthly_model_v3 import (  # noqa: E402
    MonthlyNParams, REGIONAL_CLIMATES, FAOSTAT_TARGETS,
    get_regional_bnf, baseline_water_stress, apply_era5_climate_file,
)
from soil_n_model import (  # noqa: E402
    CropParams, FeedbackParams, RegionParams, get_default_regions,
)
from coupled_econ_biophysical import (  # noqa: E402
    EconParams, REGIONAL_ECON_PARAMS, calibrate_price_shock,
    get_scenario_params, get_supply_constrained_scenarios, supply_state,
)
from parameter_registry import (  # noqa: E402
    SOC_T_C_HA_PER_PERCENT_30CM,
    WATER_STRESS_GAIN_SAT_SOC_PCT,
    WATER_STRESS_MIN_FACTOR,
    WATER_STRESS_SOFTPLUS_EPS_MM,
)

# April value: qmax_per_claysilt * 0.55 for every region (RegionParams has no
# clay_silt_fraction; the April code's getattr default did the same, so this
# is continuity, not a new assumption).
CLAY_SILT_DEFAULT = 0.55


# ---------------------------------------------------------------------------
# Corrected 4-pool dynamic spin-up (P2) and calibration (P3)
# ---------------------------------------------------------------------------

def fourpool_dynamic_spinup(region_key, ym, mp=None, fourpool_p=None,
                            synth_n=None, n_spinup=2000, tol=0.002,
                            clay_silt=CLAY_SILT_DEFAULT):
    mp = mp or MonthlyNParams()
    fourpool_p = fourpool_p or FourPoolParams()
    region = get_default_regions()[region_key]
    crop = CropParams()
    climate = REGIONAL_CLIMATES[region_key]
    if synth_n is None:
        synth_n = region.synth_n_current
    bnf_annual = get_regional_bnf(region_key)
    qmax = fourpool_p.qmax_per_claysilt * clay_silt

    pools = fourpool_analytic_init(region.soc_initial, qmax, fourpool_p)
    c_pom, c_dom = pools['c_pom'], pools['c_dom']
    c_mbc, c_maom = pools['c_mbc'], pools['c_maom']
    pom_baseline = c_pom

    mit_c = (region.mitscherlich_c_regional
             if region.mitscherlich_c_regional > 0 else crop.mitscherlich_c)
    n_grain_t = crop.grain_n_fraction * 1000
    hi = crop.harvest_index
    rf = (1 - hi) / hi
    rr = region.residue_retention
    res_cf = getattr(crop, 'residue_c_fraction', 0.45)

    # P2: the spin-up yield carries the baseline water-stress factor, exactly
    # as the simulation's step() does at delta_SOC = 0.
    ws0 = baseline_water_stress(region)

    target_y = FAOSTAT_TARGETS.get(region_key, 3.0)
    shoot_c = target_y * res_cf * rf * rr
    root_c = target_y * res_cf * rf * region.root_shoot_c_ratio
    lagged_c_input = (shoot_c + root_c) * region.cre_regional

    mineral_n = 12.0
    baseline_n_total = synth_n + bnf_annual + 50.0
    prev_n_total = baseline_n_total
    converged, years = False, n_spinup
    hist = []
    y = target_y
    nb = None
    for yr in range(n_spinup):
        n_frac = 1.0 if yr == 0 else max(
            0.01, prev_n_total / max(baseline_n_total, 1.0))
        step = fourpool_annual_step(
            c_pom, c_dom, c_mbc, c_maom,
            c_input=lagged_c_input, qmax=qmax, mems=fourpool_p,
            n_available_frac=n_frac, pom_baseline=pom_baseline)
        annual_n_min = max(0, step['net_n_mineralized'])
        c_pom, c_dom = step['c_pom'], step['c_dom']
        c_mbc, c_maom = step['c_mbc'], step['c_maom']

        nb = monthly_n_balance_4pool(
            annual_n_min, synth_n, bnf_annual,
            region.atm_n_deposition, climate, mineral_n, mp)
        mineral_n = nb['mineral_n_end']

        n_eff = nb['uptake']
        y_mit = ym * (1 - np.exp(-mit_c * n_eff)) * ws0
        y_st = n_eff / n_grain_t if n_grain_t > 0 else y_mit
        y = min(y_mit, y_st)
        y = max(region.yield_min_regional
                if region.yield_min_regional > 0 else 0.0, y)

        shoot_c = y * res_cf * rf * rr
        root_c = y * res_cf * rf * region.root_shoot_c_ratio
        lagged_c_input = (shoot_c + root_c) * region.cre_regional

        prev_n_total = synth_n + bnf_annual + annual_n_min
        if yr == 0:
            baseline_n_total = prev_n_total

        soc = c_pom + c_dom + c_mbc + c_maom
        hist.append(soc)
        if yr >= 50:
            drift = abs(soc - hist[yr - 50]) / max(hist[yr - 50], 0.1)
            if drift < tol and yr >= 100:
                converged, years = True, yr + 1
                break

    return {
        'c_pom': c_pom, 'c_dom': c_dom, 'c_mbc': c_mbc, 'c_maom': c_maom,
        'soc': c_pom + c_dom + c_mbc + c_maom,
        'mineral_n': mineral_n,
        'yield_eq': y,
        'n_min_eq': nb['min'],
        'c_input_eq': lagged_c_input,
        'pom_baseline': pom_baseline,
        'converged': converged, 'years': years,
        'qmax': qmax,
    }


_YM_CACHE = {}
CALIBRATION_SHORTFALL = {}


def calibrate_ym_fourpool(region_key, mp=None, fourpool_p=None,
                          clay_silt=CLAY_SILT_DEFAULT):
    """P3: production-path calibration against the FAOSTAT target."""
    cache_key = (region_key, round(clay_silt, 4))
    if cache_key in _YM_CACHE:
        return _YM_CACHE[cache_key]
    target = FAOSTAT_TARGETS[region_key]

    def yeq(ym):
        return fourpool_dynamic_spinup(region_key, ym, mp, fourpool_p,
                                       clay_silt=clay_silt)['yield_eq']

    lo, hi = target * 1.01, target * 6.0
    y_lo, y_hi = yeq(lo), yeq(hi)
    if y_lo > target:        # even a barely-above-target ceiling overshoots
        ym = lo
    elif y_hi < target:
        # Stoichiometric bound: the 4-pool equilibrium N supply cannot carry
        # yield to the FAOSTAT target at any ceiling (SSA under ERA5). Take
        # the smallest ceiling that achieves the plateau and record the
        # shortfall; the run is normalized to its own baseline yield.
        plateau = y_hi
        ym = hi
        for cand in np.linspace(lo, hi, 25):
            if yeq(cand) >= plateau * 0.999:
                ym = cand
                break
        CALIBRATION_SHORTFALL[(region_key, round(clay_silt, 4))] = (plateau, target)
    else:
        ym = brentq(lambda x: yeq(x) - target, lo, hi, rtol=1e-4)
    _YM_CACHE[cache_key] = ym
    return ym


# ---------------------------------------------------------------------------
# Engine (April pool scheme, current stress function)
# ---------------------------------------------------------------------------

class FourPoolBiophysicalEngine:
    def __init__(self, region: RegionParams, region_key: str,
                 crop_params: CropParams = None,
                 feedback_params: FeedbackParams = None,
                 monthly_params: MonthlyNParams = None,
                 fourpool_params: FourPoolParams = None,
                 yield_max_override: float = None,
                 cue_fixed: bool = False,
                 clay_silt: float = CLAY_SILT_DEFAULT):
        self.region = region
        self.region_key = region_key
        self.crop = crop_params or CropParams()
        self.fb = feedback_params or FeedbackParams()
        self.mp = monthly_params or MonthlyNParams()
        self.fp4 = fourpool_params or FourPoolParams()
        # cue_fixed: hold CUE at its N-replete baseline (n_frac = 1) so the
        # CUE-downregulation contribution can be isolated by difference
        # (Supplementary table 2).
        self.cue_fixed = cue_fixed
        self.clay_silt = clay_silt
        self.climate = REGIONAL_CLIMATES[region_key]

        self.mit_c = (region.mitscherlich_c_regional
                      if region.mitscherlich_c_regional > 0
                      else self.crop.mitscherlich_c)
        self.y_max = (yield_max_override if yield_max_override is not None
                      else calibrate_ym_fourpool(region_key, self.mp, self.fp4,
                                                 clay_silt=clay_silt))
        self.y_floor = (region.yield_min_regional
                        if region.yield_min_regional > 0 else 0.0)
        self.n_cost_per_tonne = self.crop.grain_n_fraction * 1000
        self.bnf_baseline = get_regional_bnf(region_key)

        eq = fourpool_dynamic_spinup(region_key, self.y_max, self.mp, self.fp4,
                                     synth_n=region.synth_n_current,
                                     clay_silt=clay_silt)
        self.qmax = eq['qmax']
        self.c_pom, self.c_dom = eq['c_pom'], eq['c_dom']
        self.c_mbc, self.c_maom = eq['c_mbc'], eq['c_maom']
        self.pom_baseline = eq['pom_baseline']
        self.soc_initial = eq['soc']
        self.mineral_n = eq['mineral_n']
        self.baseline_c_input = eq['c_input_eq']
        self.lagged_c_input = eq['c_input_eq']
        self.yield_baseline = eq['yield_eq']
        self.n_min_baseline = eq['n_min_eq']
        self.baseline_n_total = (region.synth_n_current + self.bnf_baseline
                                 + eq['n_min_eq'])
        self.prev_n_total = self.baseline_n_total
        self.spinup_converged = eq['converged']
        self.spinup_years = eq['years']

    def _water_stress(self) -> float:
        """P4: the current two-sided smooth response, on 4-pool SOC."""
        if not self.fb.physical_feedback:
            return 1.0
        soc_current = self.c_pom + self.c_dom + self.c_mbc + self.c_maom
        soc_pct = soc_current / SOC_T_C_HA_PER_PERCENT_30CM
        soc_pct_init = self.soc_initial / SOC_T_C_HA_PER_PERCENT_30CM
        delta = soc_pct_init - soc_pct
        whc_sens = self.region.whc_sensitivity * self.fb.physical_strength
        if delta >= 0.0:
            whc_change_mm = delta * whc_sens
        else:
            gain_max = WATER_STRESS_GAIN_SAT_SOC_PCT * whc_sens
            whc_change_mm = -gain_max * (
                1.0 - np.exp(delta / WATER_STRESS_GAIN_SAT_SOC_PCT))
        raw = self.region.baseline_water_deficit + whc_change_mm
        eps = WATER_STRESS_SOFTPLUS_EPS_MM
        total_deficit = 0.5 * (raw + np.sqrt(raw * raw + eps * eps))
        stress = 1.0 - self.region.water_stress_coeff * total_deficit
        return max(WATER_STRESS_MIN_FACTOR, min(1.0, stress))

    def step(self, fert_applied: float, bnf: float = None) -> dict:
        if bnf is None:
            bnf = self.bnf_baseline
        if self.cue_fixed:
            n_frac = 1.0
        else:
            n_frac = max(0.01, self.prev_n_total
                         / max(self.baseline_n_total, 1.0))

        c_input_used = self.lagged_c_input
        st = fourpool_annual_step(
            self.c_pom, self.c_dom, self.c_mbc, self.c_maom,
            c_input=c_input_used, qmax=self.qmax, mems=self.fp4,
            n_available_frac=n_frac, pom_baseline=self.pom_baseline)
        annual_n_min = max(0, st['net_n_mineralized'])
        self.c_pom, self.c_dom = st['c_pom'], st['c_dom']
        self.c_mbc, self.c_maom = st['c_mbc'], st['c_maom']
        soc_new = self.c_pom + self.c_dom + self.c_mbc + self.c_maom

        nb = monthly_n_balance_4pool(
            annual_n_min, fert_applied, bnf,
            self.region.atm_n_deposition, self.climate, self.mineral_n, self.mp)
        self.mineral_n = nb['mineral_n_end']
        n_uptake, n_min_annual = nb['uptake'], nb['min']

        ws = self._water_stress()
        y_mit = self.y_max * (1.0 - np.exp(-self.mit_c * n_uptake)) * ws
        y_st = (n_uptake / self.n_cost_per_tonne
                if self.n_cost_per_tonne > 0 else y_mit)
        y = max(self.y_floor, min(y_mit, y_st))

        res_cf = getattr(self.crop, 'residue_c_fraction', 0.45)
        hi = self.crop.harvest_index
        rf = (1 - hi) / hi
        shoot_c = y * res_cf * rf * self.region.residue_retention
        root_c = y * res_cf * rf * self.region.root_shoot_c_ratio
        self.lagged_c_input = (shoot_c + root_c) * self.region.cre_regional
        self.prev_n_total = fert_applied + bnf + annual_n_min

        eps = 1e-10
        n_up = max(n_uptake, eps)
        if y_st < y_mit:
            elast = 1.0
        else:
            e = np.exp(-self.mit_c * n_up)
            elast = self.mit_c * n_up * e / max(1.0 - e, eps)
        tot = n_min_annual + fert_applied + bnf + self.region.atm_n_deposition
        soil_share = n_min_annual / tot if tot > eps else 0.5
        fert_share = fert_applied / tot if tot > eps else 0.5

        return {
            'yield_tha': y,
            'yield_fraction': (y / self.yield_baseline
                               if self.yield_baseline > 0 else 0),
            'n_mineralized': n_min_annual,
            'n_uptake': n_uptake,
            'n_leached': nb['leach'],
            'n_denitrified': nb['den'],
            'n_immobilized': nb['immob'],
            'soc_total': soc_new,
            'soc_fraction': (soc_new / self.soc_initial
                             if self.soc_initial > 0 else 1),
            'water_stress': ws,
            'beta': elast * max(0.0, soil_share),
            'gamma': elast * fert_share,
            'cue': st['cue'], 'maom_sat': st['maom_sat'],
            'priming': st['priming'],
            'total_respired': st['total_respired'],
            'resp_cue': st['resp_cue'],
            'resp_necro': st['resp_necro'],
            'c_input': c_input_used,
            'c_pom': self.c_pom, 'c_dom': self.c_dom,
            'c_mbc': self.c_mbc, 'c_maom': self.c_maom,
        }


# ---------------------------------------------------------------------------
# Coupled model (current economic layer + realized clearing, P5/P6)
# ---------------------------------------------------------------------------

class Coupled4PoolModel:
    def __init__(self, region, econ: EconParams, region_key,
                 t_max=30.0, dt=1.0, yield_max_override=None,
                 fourpool_params=None, cue_fixed=False,
                 clay_silt=CLAY_SILT_DEFAULT):
        self.region, self.econ, self.region_key = region, econ, region_key
        self.t_max, self.dt = t_max, dt
        rp = REGIONAL_ECON_PARAMS.get(region_key, {})
        self.eta = rp.get('eta', econ.eta)
        self.alpha = rp.get('alpha', econ.alpha)
        self.eps_F_PF = rp.get('eps_F_PF', econ.eps_F_PF)
        self.eps_F_PY = (econ.eps_F_PY if econ.eps_F_PY == 0.0
                         else rp.get('eps_F_PY', econ.eps_F_PY))
        self.eps_F_N = econ.eps_F_N
        if (econ.eps_LD_PL == 0.0 and econ.eps_LD_PY == 0.0
                and econ.eps_LS_PL == 0.0):
            self.eps_LD_PL = self.eps_LD_PY = self.eps_LS_PL = 0.0
        else:
            self.eps_LD_PL = rp.get('eps_LD_PL', econ.eps_LD_PL)
            self.eps_LD_PY = rp.get('eps_LD_PY', econ.eps_LD_PY)
            self.eps_LS_PL = rp.get('eps_LS_PL', econ.eps_LS_PL)

        self.bio = FourPoolBiophysicalEngine(
            region, region_key, yield_max_override=yield_max_override,
            fourpool_params=fourpool_params, cue_fixed=cue_fixed,
            clay_silt=clay_silt)

        self.F_baseline = region.synth_n_current
        self.L_baseline = region.cropland_mha
        self.Y_baseline = self.bio.yield_baseline
        self.N_min_baseline = self.bio.n_min_baseline
        self.PF_hat_base = np.log(1 + econ.fert_price_shock)
        self.PF_hat, self.PY_hat = self.PF_hat_base, 0.0
        self.F_hat = self.L_hat = self.N_hat = 0.0

    def _lambda_L(self):
        if abs(self.eps_LS_PL - self.eps_LD_PL) > 1e-10:
            return (self.eps_LS_PL * self.eps_LD_PY
                    / (self.eps_LS_PL - self.eps_LD_PL))
        return 0.0

    def _solve_equilibrium(self, beta, gamma):
        lam = self._lambda_L()
        num = (beta * self.N_hat
               + gamma * (self.eps_F_PF * self.PF_hat
                          + self.eps_F_N * self.N_hat))
        den = self.eta - self.alpha * lam - gamma * self.eps_F_PY
        PY = num / den if abs(den) > 1e-10 else 0.0
        F = (self.eps_F_PF * self.PF_hat + self.eps_F_PY * PY
             + self.eps_F_N * self.N_hat)
        return PY, F, lam * PY

    def _clear_market_realized(self, supply):
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

        step = 0.10
        lo, hi = self.PY_hat - step, self.PY_hat + step
        for _ in range(24):
            if residual(lo) * residual(hi) < 0:
                break
            step *= 1.6
            lo -= step
            hi += step
        else:
            raise RuntimeError('4-pool realized clearing found no bracket '
                               'for %r' % (self.region_key,))
        PY_hat = brentq(residual, lo, hi, xtol=1e-12)
        F_hat, L_hat, F_level, L_level, capped = implied(PY_hat)
        self.bio = snap
        bio_state = self.bio.step(F_level)
        resid = self.eta * PY_hat - (
            np.log(max(bio_state['yield_fraction'], 1e-9))
            + self.alpha * L_hat)
        return PY_hat, F_hat, L_hat, F_level, L_level, capped, bio_state, resid

    def run(self) -> pd.DataFrame:
        n_steps = int(self.t_max / self.dt) + 1
        cols = ['year', 'PF_hat', 'PY_hat', 'F_hat', 'L_hat', 'N_hat',
                'fert_applied_kgha', 'land_mha', 'food_price_index',
                'yield_tha', 'yield_fraction', 'n_mineralized', 'n_uptake',
                'n_leached', 'n_denitrified', 'n_immobilized', 'soc_total',
                'soc_fraction', 'water_stress', 'beta', 'gamma',
                'total_production_index', 'cap_binding', 'clearing_residual',
                'cue', 'maom_sat', 'priming', 'resp_cue', 'resp_necro',
                'c_input', 'c_pom', 'c_dom', 'c_mbc', 'c_maom']
        res = {c: np.zeros(n_steps) for c in cols}

        # Baseline-year diagnostics without mutating engine state
        b = self.bio
        init_step = fourpool_annual_step(
            b.c_pom, b.c_dom, b.c_mbc, b.c_maom,
            c_input=b.baseline_c_input, qmax=b.qmax, mems=b.fp4,
            n_available_frac=1.0, pom_baseline=b.pom_baseline)
        init_n_min = max(0, init_step['net_n_mineralized'])
        init_nb = monthly_n_balance_4pool(
            init_n_min, self.F_baseline, b.bnf_baseline,
            self.region.atm_n_deposition, b.climate, b.mineral_n, b.mp)
        eps = 1e-10
        n_up = max(init_nb['uptake'], eps)
        y_mit = b.y_max * (1.0 - np.exp(-b.mit_c * n_up))
        y_st = n_up / b.n_cost_per_tonne
        if y_st < y_mit:
            elast = 1.0
        else:
            e = np.exp(-b.mit_c * n_up)
            elast = b.mit_c * n_up * e / max(1.0 - e, eps)
        tot = init_nb['min'] + self.F_baseline + b.bnf_baseline \
            + self.region.atm_n_deposition
        init_beta = elast * max(0.0, init_nb['min'] / max(tot, eps))
        init_gamma = elast * (self.F_baseline / max(tot, eps))
        # Stationary-baseline fix: N_hat = 0 at the year-0 recorded flux.
        self.N_min_baseline = init_nb['min']

        for i in range(n_steps):
            t = i * self.dt
            res['year'][i] = t
            if i == 0:
                res['fert_applied_kgha'][i] = self.F_baseline
                res['land_mha'][i] = self.L_baseline
                res['food_price_index'][i] = 1.0
                res['yield_tha'][i] = self.Y_baseline
                res['yield_fraction'][i] = 1.0
                res['n_mineralized'][i] = init_nb['min']
                res['n_uptake'][i] = init_nb['uptake']
                res['n_leached'][i] = init_nb['leach']
                res['n_denitrified'][i] = init_nb['den']
                res['n_immobilized'][i] = init_nb['immob']
                res['soc_total'][i] = b.soc_initial
                res['soc_fraction'][i] = 1.0
                res['water_stress'][i] = b._water_stress()
                res['beta'][i], res['gamma'][i] = init_beta, init_gamma
                res['total_production_index'][i] = 1.0
                res['cue'][i] = init_step['cue']
                res['maom_sat'][i] = init_step['maom_sat']
                res['priming'][i] = init_step['priming']
                res['resp_cue'][i] = init_step['resp_cue']
                res['resp_necro'][i] = init_step['resp_necro']
                res['c_input'][i] = b.baseline_c_input
                res['c_pom'][i], res['c_dom'][i] = b.c_pom, b.c_dom
                res['c_mbc'][i], res['c_maom'][i] = b.c_mbc, b.c_maom
                continue

            cur = res['n_mineralized'][i - 1]
            self.N_hat = (np.log(max(cur, 1e-6) / self.N_min_baseline)
                          if self.N_min_baseline > 0 else 0.0)
            supply = supply_state(self.econ, t)
            self.PF_hat = self.PF_hat_base * supply.price_frac

            beta, gamma = res['beta'][i - 1], res['gamma'][i - 1]
            self.PY_hat = self._solve_equilibrium(beta, gamma)[0]
            (PY_hat, F_hat, L_hat, F_level, L_level, capped,
             s, resid) = self._clear_market_realized(supply)
            self.PY_hat, self.F_hat, self.L_hat = PY_hat, F_hat, L_hat

            res['PF_hat'][i] = self.PF_hat
            res['PY_hat'][i] = PY_hat
            res['F_hat'][i] = F_hat
            res['L_hat'][i] = L_hat
            res['N_hat'][i] = self.N_hat
            res['fert_applied_kgha'][i] = F_level
            res['land_mha'][i] = L_level
            res['food_price_index'][i] = np.exp(PY_hat)
            for k in ('yield_tha', 'yield_fraction', 'n_mineralized',
                      'n_uptake', 'n_leached', 'n_denitrified',
                      'n_immobilized', 'soc_total', 'soc_fraction',
                      'water_stress', 'beta', 'gamma', 'cue', 'maom_sat',
                      'priming', 'resp_cue', 'resp_necro', 'c_input',
                      'c_pom', 'c_dom', 'c_mbc', 'c_maom'):
                res[k][i] = s[k]
            res['total_production_index'][i] = (
                s['yield_fraction'] * L_level / self.L_baseline)
            res['cap_binding'][i] = 1.0 if capped else 0.0
            res['clearing_residual'][i] = resid

        return pd.DataFrame(res)
