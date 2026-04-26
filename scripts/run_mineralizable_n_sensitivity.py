#!/usr/bin/env python3
"""
Mineralizable-N sensitivity for the SOC-buffering result.

One-at-a-time sensitivity around the Century/RothC SOM engine,
testing whether the within-region SOC gradient (yield change vs SOC%)
under a 100% fertilizer-price shock is preserved across reasonable
perturbations of the three SOM parameters that govern mineralizable-N
supply in the monthly pipeline:

  - active pool fraction (f_active): 2%, 4% (baseline), 8%
        Slow pool held at 38%; passive pool absorbs the change
        (60%, 58%, 54%).
  - slow-pool decay rate (k_slow):   x0.75, x1.0 (baseline), x1.25
        Multiplies the regional baseline k_slow.
  - bulk soil C:N (region.cn_bulk):  x0.8, x1.0 (baseline), x1.2
        The monthly pipeline uses a single bulk C:N in the
        mineralization formula
            (k_a*C_a + k_s*C_s + k_p*C_p) / cn_bulk * abiotic
        so this parameter governs the SOC-to-mineralizable-N
        stoichiometric assumption.

For each parameter perturbation we run the SOC gradient (10-200%,
step 10) under a 100% fertilizer-price shock for two regions:
Sub-Saharan Africa (tropical, high-leverage) and North America
(temperate, well-buffered).

Outputs:
  data/mineralizable_n_sensitivity.pkl
"""

import sys, os, pickle, copy
import numpy as np
from pathlib import Path
from dataclasses import replace

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_DIR = REPO_ROOT  # alias for legacy refs
ROOT_DIR = REPO_ROOT  # alias for legacy refs
MODEL_DIR = ROOT_DIR / 'model'
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / 'scripts'))

DATA_DIR = PROJECT_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

import soil_n_model as snm
import monthly_model_v3 as mmv3
from coupled_monthly import (
    CoupledMonthlyModel, MonthlyBiophysicalEngine, get_calibrated_ym,
    clear_ym_cache,
)
from coupled_econ_biophysical import (
    EconParams, REGIONAL_ECON_PARAMS,
)
from soil_n_model import (
    get_default_regions, SOMPoolParams, som_params_for_region, TROPICAL_REGIONS,
)


# ---- Configuration ----
REGIONS_TO_RUN = ['sub_saharan_africa', 'north_america']
SOC_PCTS = list(range(10, 205, 10))   # 10% to 200%, step 10
PRICE_SHOCK_FRAC = 1.0                # 100% price spike

# Cost shares as in the main analysis
FERT_COST_FRAC = {
    'sub_saharan_africa': 0.25,
    'south_asia': 0.20,
    'latin_america': 0.12,
    'north_america': 0.08,
}


def build_perturbed_som(region_key, f_active=None, k_slow_mult=1.0):
    """Return a SOMPoolParams for the region with optional perturbations.

    Holds f_passive constant when f_active is varied, by absorbing the
    delta into f_slow (so the slow:active ratio shifts but the passive
    long-residence pool stays at its calibrated share).
    """
    base = som_params_for_region(region_key)
    new_f_active = base.f_active if f_active is None else f_active
    # Hold f_passive constant; absorb f_active change in f_slow.
    new_f_slow = 1.0 - new_f_active - base.f_passive
    return replace(
        base,
        f_active=new_f_active,
        f_slow=new_f_slow,
        k_slow=base.k_slow * k_slow_mult,
    )


def patched_som_for_region(custom_som):
    """Return a function with the same signature as som_params_for_region
    that ignores region_key and returns the supplied custom_som."""
    def _f(region_key):
        return custom_som
    return _f


def run_soc_gradient(region_key, custom_som, cn_bulk_mult=1.0,
                    soc_pcts=SOC_PCTS, price_shock=PRICE_SHOCK_FRAC):
    """Run the SOC gradient under a single perturbed parameterization.

    Patches som_params_for_region to return custom_som during spinup,
    and scales region.cn_bulk by cn_bulk_mult.

    Returns dict: soc_pct, yield_pen, profit_chg, fert_red, base_yield.
    """
    regions = get_default_regions()
    region = regions[region_key]
    if cn_bulk_mult != 1.0:
        region = replace(region, cn_bulk=region.cn_bulk * cn_bulk_mult)

    # Patch the spinup's view of SOM params
    orig_snm = snm.som_params_for_region
    orig_mmv3 = mmv3.som_params_for_region
    snm.som_params_for_region = patched_som_for_region(custom_som)
    mmv3.som_params_for_region = patched_som_for_region(custom_som)

    try:
        clear_ym_cache()
        ym = get_calibrated_ym(region_key)

        rp = REGIONAL_ECON_PARAMS.get(region_key, {})
        fcf = FERT_COST_FRAC.get(region_key, 0.15)

        # Equilibrium pools (under this parameterization)
        engine_eq = MonthlyBiophysicalEngine(
            region, region_key=region_key, som_params=custom_som,
            yield_max_override=ym,
        )
        C_a_eq, C_s_eq, C_p_eq = engine_eq.C_active, engine_eq.C_slow, engine_eq.C_passive
        base_fert = region.synth_n_current

        # Regional baseline yield + gamma at SOC=100% (used for the
        # output-price recovery — a market-level property that does not
        # vary with farm SOC)
        eng_regional = MonthlyBiophysicalEngine(
            region, region_key=region_key, som_params=custom_som,
            yield_max_override=ym,
        )
        state_regional = eng_regional.step(base_fert)
        y_regional_baseline = state_regional['yield_tha']
        gamma_regional = state_regional['gamma']

        eps_F_PF = rp.get('eps_F_PF', -0.20)
        eps_F_PY = rp.get('eps_F_PY', 0.10)
        eta = rp.get('eta', -0.30)
        PF_hat = np.log(1 + price_shock)
        denom = eta - gamma_regional * eps_F_PY
        PY_hat = (gamma_regional * eps_F_PF * PF_hat / denom
                  if abs(denom) > 1e-10 else 0.0)
        F_hat = eps_F_PF * PF_hat + eps_F_PY * PY_hat
        F_shocked = max(0.0, base_fert * np.exp(F_hat))

        out = {'soc_pct': [], 'yield_pen': [], 'profit_chg': [],
               'fert_red': [], 'base_yield': []}

        for soc_pct in soc_pcts:
            scale = soc_pct / 100.0

            # Baseline yield at this farm's SOC, no shock
            eng_b = MonthlyBiophysicalEngine(
                region, region_key=region_key, som_params=custom_som,
                yield_max_override=ym,
            )
            eng_b.C_active = C_a_eq * scale
            eng_b.C_slow = C_s_eq * scale
            eng_b.C_passive = C_p_eq * scale
            state_b = eng_b.step(base_fert)
            y_base_soc = state_b['yield_tha']

            # Shocked yield at this farm's SOC
            eng_s = MonthlyBiophysicalEngine(
                region, region_key=region_key, som_params=custom_som,
                yield_max_override=ym,
            )
            eng_s.C_active = C_a_eq * scale
            eng_s.C_slow = C_s_eq * scale
            eng_s.C_passive = C_p_eq * scale
            state_s = eng_s.step(F_shocked)
            y_shock = state_s['yield_tha']

            yield_pen = (1 - y_shock / y_base_soc) * 100 if y_base_soc > 0 else 0.0
            fert_red = (1 - F_shocked / base_fert) * 100 if base_fert > 0 else 0.0

            pf_per_unit = fcf * y_regional_baseline / base_fert if base_fert > 0 else 0
            profit_b = y_base_soc - base_fert * pf_per_unit
            profit_s = (y_shock * np.exp(PY_hat)
                        - F_shocked * pf_per_unit * (1 + price_shock))
            profit_chg = ((profit_s / profit_b - 1) * 100
                          if abs(profit_b) > 1e-10 else 0.0)

            out['soc_pct'].append(soc_pct)
            out['yield_pen'].append(yield_pen)
            out['profit_chg'].append(profit_chg)
            out['fert_red'].append(fert_red)
            out['base_yield'].append(y_base_soc)

        return out

    finally:
        snm.som_params_for_region = orig_snm
        mmv3.som_params_for_region = orig_mmv3


# ---- Define the sensitivity grid ----

def define_scenarios():
    """Return list of (label, scenario_kwargs) tuples."""
    scenarios = [
        # Baseline
        ('baseline', {'f_active': 0.04, 'k_slow_mult': 1.0, 'cn_bulk_mult': 1.0}),
        # f_active dimension
        ('f_active=0.02', {'f_active': 0.02, 'k_slow_mult': 1.0, 'cn_bulk_mult': 1.0}),
        ('f_active=0.08', {'f_active': 0.08, 'k_slow_mult': 1.0, 'cn_bulk_mult': 1.0}),
        # k_slow dimension
        ('k_slow x0.75', {'f_active': 0.04, 'k_slow_mult': 0.75, 'cn_bulk_mult': 1.0}),
        ('k_slow x1.25', {'f_active': 0.04, 'k_slow_mult': 1.25, 'cn_bulk_mult': 1.0}),
        # cn_bulk dimension
        ('cn_bulk x0.8', {'f_active': 0.04, 'k_slow_mult': 1.0, 'cn_bulk_mult': 0.8}),
        ('cn_bulk x1.2', {'f_active': 0.04, 'k_slow_mult': 1.0, 'cn_bulk_mult': 1.2}),
    ]
    return scenarios


def main():
    scenarios = define_scenarios()
    results = {}

    for region_key in REGIONS_TO_RUN:
        print(f'\n=== {region_key} ===')
        results[region_key] = {}
        for label, kw in scenarios:
            print(f'  Running {label}...')
            custom_som = build_perturbed_som(
                region_key, f_active=kw['f_active'], k_slow_mult=kw['k_slow_mult'],
            )
            r = run_soc_gradient(
                region_key, custom_som, cn_bulk_mult=kw['cn_bulk_mult'],
            )
            results[region_key][label] = {
                'soc_pct': r['soc_pct'],
                'yield_pen': r['yield_pen'],
                'profit_chg': r['profit_chg'],
                'fert_red': r['fert_red'],
                'base_yield': r['base_yield'],
                'kw': kw,
            }

            # Quick summary: gap between SOC=50% and SOC=100%
            soc_arr = np.array(r['soc_pct'])
            yp = np.array(r['yield_pen'])
            i50 = int(np.where(soc_arr == 50)[0][0]) if 50 in soc_arr else None
            i100 = int(np.where(soc_arr == 100)[0][0]) if 100 in soc_arr else None
            i150 = int(np.where(soc_arr == 150)[0][0]) if 150 in soc_arr else None
            if i50 is not None and i100 is not None:
                gap_50_100 = yp[i50] - yp[i100]
                gap_100_150 = (yp[i100] - yp[i150]) if i150 is not None else float('nan')
                print(f'    yield_pen at SOC=50%: {yp[i50]:.2f}%; '
                      f'at SOC=100%: {yp[i100]:.2f}%; '
                      f'gap 50->100: {gap_50_100:.2f} pp; '
                      f'gap 100->150: {gap_100_150:.2f} pp')

    out_path = DATA_DIR / 'mineralizable_n_sensitivity.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({'results': results, 'scenarios': scenarios,
                     'soc_pcts': SOC_PCTS, 'price_shock': PRICE_SHOCK_FRAC},
                    f)
    print(f'\nSaved: {out_path} ({out_path.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
