#!/usr/bin/env python3
"""Canonical source computation for main-text Figures 1 and 2a.

A "farm" in this paper is a soil condition, not a georeferenced landholding.
Every farm in a region faces the same regional prices, economic parameters and
climate; farms differ only in the size of their soil organic carbon stock and
the mineralized nitrogen it supplies. The gradient is therefore built by
spinning the Century three-pool scheme to its regional equilibrium once, then
rescaling the three carbon pools by a common factor to place the farm at that
fraction of the regional mean SOC.

Part 1 (Figure 1) is a single-season farm-level calculation under a 100%
fertilizer price spike. The output-price recovery P_Y_hat is a *market*
property: it depends on the aggregate regional supply response, evaluated at
the regional-mean SOC, not on any individual farm's soil. A single high-SOC
farm does not depress regional supply and so receives the same price cushion
as everyone else in its region.

Part 2 (Figure 2a) is the year-10 gradient under the canonical S3 scenario
(sustained 20% price-mediated fertilizer supply reduction), run through the
full coupled model. Each SOC level is paired with a no-shock control at the
same SOC that carries the *same* behavioural elasticities, including
eps_F_N, so the shocked/control ratio isolates the price-shock effect rather
than the structural SOC effect. Without matching the eps_F_N channel the
depletion-feedback compensation in S3 overcorrects at very low SOC and
produces non-physical apparent yield gains.

Writes data/figure1_farm_gradient.json and data/figure2_soc_gradient.json.

Added to the deposit in v1.3: Figures 1 and 2 were previously produced from
working scripts held outside the archive and could not be regenerated from it.
They now can.
"""
import os
import sys
import json
import copy
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))

import numpy as np

from monthly_model_v3 import MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES
from coupled_monthly import (CoupledMonthlyModel, MonthlyBiophysicalEngine,
                             get_calibrated_ym)
from coupled_econ_biophysical import (get_scenario_params, REGIONAL_ECON_PARAMS,
                                      EconParams)
from soil_n_model import get_default_regions
from parameter_registry import (
    REGIONAL_PRICES,
    nitrogen_cost_share,
    nitrogen_price_in_yield_units,
)

DATA = os.path.join(HERE, '..', '..', 'data')

ALL_REGIONS = ['north_america', 'europe', 'east_asia', 'south_asia',
               'southeast_asia', 'latin_america', 'sub_saharan_africa',
               'fsu_central_asia']

# The four regions with farm-gate cost-structure data, used for Figure 1.
KEY4 = ['sub_saharan_africa', 'south_asia', 'latin_america', 'north_america']

ABBR = {'north_america': 'NA', 'europe': 'EU', 'east_asia': 'EA',
        'south_asia': 'SA', 'southeast_asia': 'SEA', 'latin_america': 'LATAM',
        'sub_saharan_africa': 'SSA', 'fsu_central_asia': 'FSU'}
LABEL = {'north_america': 'North America', 'europe': 'Europe',
         'east_asia': 'East Asia', 'south_asia': 'South Asia',
         'southeast_asia': 'Southeast Asia', 'latin_america': 'Latin America',
         'sub_saharan_africa': 'Sub-Saharan Africa',
         'fsu_central_asia': 'FSU/Central Asia'}

FINE_SOC_PCTS = list(range(10, 205, 5))       # Figure 1, step 5
GRADIENT_SOC_PCTS = list(range(10, 205, 10))  # Figure 2a, step 10
PRICE_SHOCK_FINE = 1.0                        # 100% fertilizer price spike
T_MAX_GRADIENT = 10


def patch_era5():
    """Replace the built-in climatologies with the ERA5-derived ones."""
    clim = json.load(open(os.path.join(DATA, 'era5_regional_climates.json')))
    for k, c in list(REGIONAL_CLIMATES.items()):
        n = clim[k]
        REGIONAL_CLIMATES[k] = MonthlyClimate(
            c.name, list(map(float, n['temp'])), list(map(float, n['precip'])),
            list(map(float, n['pet'])), c.planting_month, c.maturity_month)


# =====================================================================
# Part 1: farm-level gradient under a 100% price spike (Figure 1)
# =====================================================================

def farm_sweep_single(region, rn, ym, mp, soc_pct, price_shock_frac):
    """One farm-level (SOC level, price shock) combination.

    Returns yield penalty, fertilizer reduction and change in revenue net of
    nitrogen-fertilizer expenditure,
    all in percent relative to the same farm with no shock.
    """
    rp = REGIONAL_ECON_PARAMS.get(rn, {})
    if rn not in REGIONAL_PRICES:
        raise KeyError(f"No audited nitrogen/crop price pair for {rn}")

    eq = MonthlyBiophysicalEngine(region, region_key=rn, monthly_params=mp,
                                  yield_max_override=ym)
    C_a_eq, C_s_eq, C_p_eq = eq.C_active, eq.C_slow, eq.C_passive
    base_fert = region.synth_n_current

    # Regional-mean references: the market-clearing output-price recovery
    # depends on the aggregate regional supply elasticity, not the farm's.
    state_regional = MonthlyBiophysicalEngine(
        region, region_key=rn, monthly_params=mp,
        yield_max_override=ym).step(base_fert)
    y_regional_baseline = state_regional['yield_tha']
    gamma_regional = state_regional['gamma']

    scale = soc_pct / 100.0

    def engine_at_soc():
        e = MonthlyBiophysicalEngine(region, region_key=rn, monthly_params=mp,
                                     yield_max_override=ym)
        e.C_active = C_a_eq * scale
        e.C_slow = C_s_eq * scale
        e.C_passive = C_p_eq * scale
        return e

    state_base = engine_at_soc().step(base_fert)
    y_base_soc = state_base['yield_tha']
    n_min_base = state_base['n_mineralized']

    if price_shock_frac <= 0:
        return dict(yield_pen=0.0, fert_red=0.0, margin_chg=0.0,
                    base_yield=y_base_soc, base_fert=base_fert,
                    base_nmin=n_min_base, F_shocked=base_fert,
                    y_shock=y_base_soc, PY_hat=0.0)

    eps_F_PF = rp.get('eps_F_PF', -0.20)
    eps_F_PY = rp.get('eps_F_PY', 0.10)
    eta = rp.get('eta', -0.30)
    PF_hat = np.log(1 + price_shock_frac)

    denom = eta - gamma_regional * eps_F_PY
    PY_hat = (gamma_regional * eps_F_PF * PF_hat / denom
              if abs(denom) > 1e-10 else 0.0)
    F_hat = eps_F_PF * PF_hat + eps_F_PY * PY_hat
    F_shocked = max(0.0, base_fert * np.exp(F_hat))

    y_shock = engine_at_soc().step(F_shocked)['yield_tha']

    yield_pen = (1 - y_shock / y_base_soc) * 100 if y_base_soc > 0 else 0.0
    fert_red = (1 - F_shocked / base_fert) * 100 if base_fert > 0 else 0.0

    # Revenue net of nitrogen-fertilizer expenditure. Nitrogen and crop prices
    # are primitive inputs; cost share is derived only as a diagnostic.
    pf_per_unit = nitrogen_price_in_yield_units(rn)
    margin_b = y_base_soc - base_fert * pf_per_unit
    margin_s = (y_shock * np.exp(PY_hat)
                - F_shocked * pf_per_unit * (1 + price_shock_frac))
    margin_chg = ((margin_s / margin_b - 1) * 100
                  if abs(margin_b) > 1e-10 else 0.0)

    return dict(yield_pen=yield_pen, fert_red=fert_red, margin_chg=margin_chg,
                base_yield=y_base_soc, base_fert=base_fert,
                base_nmin=n_min_base, F_shocked=F_shocked, y_shock=y_shock,
                PY_hat=float(PY_hat))


def build_figure1(regions, mp):
    out = {}
    for rn in KEY4:
        r = regions[rn]
        ym = get_calibrated_ym(rn, mp)
        rec = dict(soc_pct=[], yield_pen=[], fert_red=[], margin_chg=[])
        for soc_pct in FINE_SOC_PCTS:
            res = farm_sweep_single(r, rn, ym, mp, soc_pct, PRICE_SHOCK_FINE)
            rec['soc_pct'].append(soc_pct)
            rec['yield_pen'].append(float(res['yield_pen']))
            rec['fert_red'].append(float(res['fert_red']))
            rec['margin_chg'].append(float(res['margin_chg']))
        out[rn] = rec
        i50 = FINE_SOC_PCTS.index(50)
        i100 = FINE_SOC_PCTS.index(100)
        i200 = FINE_SOC_PCTS.index(200)
        print('  %-20s yield %6.2f %6.2f %6.2f   margin %7.2f %7.2f %7.2f'
              % (rn, -rec['yield_pen'][i50], -rec['yield_pen'][i100],
                 -rec['yield_pen'][i200], rec['margin_chg'][i50],
                 rec['margin_chg'][i100], rec['margin_chg'][i200]))
    return out


# =====================================================================
# Part 2: year-10 SOC gradient under S3 (Figure 2a)
# =====================================================================

def build_figure2a(regions, mp):
    s3 = get_scenario_params()['S3']
    out, ref_yields = {}, {}

    for rn in ALL_REGIONS:
        r = regions[rn]
        ym = get_calibrated_ym(rn, mp)

        eq = MonthlyBiophysicalEngine(r, region_key=rn, monthly_params=mp,
                                      yield_max_override=ym)
        C_a_eq, C_s_eq, C_p_eq = eq.C_active, eq.C_slow, eq.C_passive

        # Reference: regional-mean SOC, no shock, no behavioural response.
        econ_ref = EconParams(fert_price_shock=0.0, eps_F_PY=0.0, eps_F_N=0.0,
                              eps_LD_PL=0.0, eps_LD_PY=0.0, eps_LS_PL=0.0)
        df_ref = CoupledMonthlyModel(r, econ_ref, region_key=rn,
                                     t_max=T_MAX_GRADIENT,
                                     yield_max_override=ym).run()
        ref_yield = df_ref.loc[df_ref['year'] == T_MAX_GRADIENT,
                               'yield_tha'].iloc[0]
        ref_yields[rn] = float(ref_yield)

        rec = dict(soc_pct=[], total_penalty=[], ctrl_penalty=[],
                   yield_shock=[], yield_noshock=[])

        for soc_pct in GRADIENT_SOC_PCTS:
            scale = soc_pct / 100.0

            m_shock = CoupledMonthlyModel(r, copy.deepcopy(s3), region_key=rn,
                                          t_max=T_MAX_GRADIENT,
                                          yield_max_override=ym)
            m_shock.bio.C_active = C_a_eq * scale
            m_shock.bio.C_slow = C_s_eq * scale
            m_shock.bio.C_passive = C_p_eq * scale
            df_s = m_shock.run()
            y_s = df_s.loc[df_s['year'] == T_MAX_GRADIENT, 'yield_tha'].iloc[0]

            econ_ns = copy.deepcopy(s3)
            econ_ns.fert_price_shock = 0.0
            m_ns = CoupledMonthlyModel(r, econ_ns, region_key=rn,
                                       t_max=T_MAX_GRADIENT,
                                       yield_max_override=ym)
            m_ns.bio.C_active = C_a_eq * scale
            m_ns.bio.C_slow = C_s_eq * scale
            m_ns.bio.C_passive = C_p_eq * scale
            df_n = m_ns.run()
            y_n = df_n.loc[df_n['year'] == T_MAX_GRADIENT, 'yield_tha'].iloc[0]

            rec['soc_pct'].append(soc_pct)
            rec['total_penalty'].append(
                float((1 - y_s / ref_yield) * 100) if ref_yield > 0 else 0.0)
            rec['ctrl_penalty'].append(
                float((1 - y_s / y_n) * 100) if y_n > 0 else 0.0)
            rec['yield_shock'].append(float(y_s))
            rec['yield_noshock'].append(float(y_n))

        out[rn] = rec
        print('  %-20s shock-induced yr10  10%%/100%%/200%%: %6.2f %6.2f %6.2f'
              % (rn, -rec['ctrl_penalty'][0],
                 -rec['ctrl_penalty'][GRADIENT_SOC_PCTS.index(100)],
                 -rec['ctrl_penalty'][-1]))
    return out, ref_yields


def main():
    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()

    print('Figure 1: farm-level gradient, 100% fertilizer price spike')
    print('  %-20s %-22s %s' % ('', 'yield change @50/100/200',
                                'margin change @50/100/200'))
    f1 = build_figure1(regions, mp)
    derived_shares = {}
    for rn in KEY4:
        r = regions[rn]
        ym = get_calibrated_ym(rn, mp)
        y = MonthlyBiophysicalEngine(
            r, region_key=rn, monthly_params=mp,
            yield_max_override=ym).step(r.synth_n_current)['yield_tha']
        derived_shares[rn] = nitrogen_cost_share(
            rn, r.synth_n_current, y)
    json.dump(dict(regions=f1, soc_pcts=FINE_SOC_PCTS,
                   price_shock=PRICE_SHOCK_FINE,
                   regional_prices={
                       k: dict(
                           nitrogen_usd_per_kg_n=v.nitrogen_usd_per_kg_n,
                           crop_usd_per_t=v.crop_usd_per_t,
                           convention=v.convention)
                       for k, v in REGIONAL_PRICES.items()
                   },
                   derived_n_cost_share=derived_shares),
              open(os.path.join(DATA, 'figure1_farm_gradient.json'), 'w'),
              indent=1)
    print('  wrote data/figure1_farm_gradient.json')

    print('\nFigure 2a: year-10 SOC gradient under S3')
    f2, refs = build_figure2a(regions, mp)
    json.dump(dict(regions=f2, ref_yields=refs, soc_pcts=GRADIENT_SOC_PCTS),
              open(os.path.join(DATA, 'figure2_soc_gradient.json'), 'w'),
              indent=1)
    print('  wrote data/figure2_soc_gradient.json')


if __name__ == '__main__':
    main()
