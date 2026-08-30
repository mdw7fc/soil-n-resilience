#!/usr/bin/env python3
"""How far apart are the economic model's yield and the biogeochemical one?

Dale Manning's question, 2026-08-07, on the 7_22 draft. Two things are being
asked and they have different answers, so this script measures both.

(1) THE REPORTED YIELD. There is one. `yield_fraction` comes from the
    Mitscherlich response in the monthly biogeochemical engine and nothing
    else writes it. The economic block does not produce a yield; it produces a
    fertilizer rate, a land area and a food price.

(2) THE IMPLIED YIELD. There is a second one, and it is implicit. The
    equilibrium closes on market clearing, Y_hat = eta * PY_hat, with the
    log-linear supply relation Y_hat = alpha*L_hat + beta*N_hat + gamma*F_hat.
    So the food price the model reports is the price that would clear a market
    for the LOG-LINEAR production change, while the production the model
    reports is the NONLINEAR one the biogeochemistry actually delivers. Those
    two are not forced to agree. Nothing in the model reconciles them and
    nothing until now measured the gap.

    beta and gamma are recomputed every step as the local elasticities of the
    Mitscherlich curve (coupled_monthly, the `elasticity_n_total` block), so
    the log-linear supply relation is a first-order expansion of the
    biogeochemical response re-anchored at each step. The gap is therefore
    second-order in the step-to-step move and should be small. "Should be" is
    the assumption; this measures it.

Reports, per region and scenario, over 30 years:
    Y_realized  = ln(yield_frac * land_frac)   what the biogeochemistry gave
    Y_demand    = eta * PY_hat                  what the reported price clears
    Y_supply    = alpha*L_hat + beta*N_hat + gamma*F_hat
                                                the log-linear supply relation

Y_demand and Y_supply agree by construction, but only if the comparison uses
the elasticities the solver actually used, which are the PREVIOUS step's: the
loop reads `results['beta'][i-1]` and `results['gamma'][i-1]` because the
current step's elasticities are not known until the biogeochemistry has been
advanced, which happens after the equilibrium is solved. A first version of
this script compared against the current step's stored beta and gamma and got a
0.948 pp residual, which is the size of that one-step lag and not a defect in
the solver. The lag is a real property of the coupling and is reported
separately below. Y_realized against Y_demand is the quantity Dale asked about.

Writes results/econ_biophysical_yield_gap.csv. Reports nothing not written.
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))

import numpy as np  # noqa: E402
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file  # noqa: E402
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym  # noqa: E402
from coupled_econ_biophysical import (  # noqa: E402
    get_scenario_params, calibrate_price_shock, get_supply_constrained_scenarios,
    REGIONAL_ECON_PARAMS,
)
from soil_n_model import get_default_regions  # noqa: E402

DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']

apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))
reg = get_default_regions()
mp = MonthlyNParams()


def scenarios():
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(0.20)
    sc = get_supply_constrained_scenarios()
    return [('S3', s3), ('SC1_20pct', sc['SC1_20pct']),
            ('SC2_20pct_recovery', sc['SC2_20pct_recovery'])]


def region_econ(econ, key):
    """The elasticities actually in force for this region.

    Regional overrides are applied inside the model, so reading the scenario
    object alone would compare against the wrong alpha and eta for six of the
    eight regions.
    """
    over = REGIONAL_ECON_PARAMS.get(key, {})
    return (float(over.get('alpha', econ.alpha)),
            float(over.get('eta', econ.eta)))


def main():
    rows = []
    for sname, econ in scenarios():
        for key in RO:
            m = CoupledMonthlyModel(region=reg[key], econ=econ, region_key=key,
                                    t_max=30.0,
                                    yield_max_override=get_calibrated_ym(key, mp))
            df = m.run()
            alpha, eta = region_econ(econ, key)
            L0 = float(df['land_mha'].iloc[0])
            yr = df['year'].to_numpy()
            # First row of each whole year, so the annual snapshot is a real
            # step of the monthly loop and i-1 is a real previous step.
            want = {}
            for i in range(1, len(df)):
                y = int(round(float(yr[i])))
                if abs(float(yr[i]) - y) < 1e-9 and 1 <= y <= 30 and y not in want:
                    want[y] = i
            for y in sorted(want):
                i = want[y]
                r = df.iloc[i]
                # The elasticities the solver used at this step are the
                # previous step's; see this file's docstring.
                beta_used = float(df['beta'].iloc[i - 1])
                gamma_used = float(df['gamma'].iloc[i - 1])
                land_frac = float(r['land_mha']) / L0
                y_real = np.log(max(float(r['yield_fraction']) * land_frac, 1e-12))
                y_dem = eta * float(r['PY_hat'])
                y_sup = (alpha * float(r['L_hat']) + beta_used * float(r['N_hat'])
                         + gamma_used * float(r['F_hat']))
                lag_pp = 100.0 * abs(float(r['beta']) - beta_used
                                     + float(r['gamma']) - gamma_used)
                rows.append(dict(
                    scenario=sname, region=key, year=y,
                    cap_binding=int(r['cap_binding']),
                    y_realized_pct=100.0 * (np.exp(y_real) - 1.0),
                    y_demand_pct=100.0 * (np.exp(y_dem) - 1.0),
                    y_supply_pct=100.0 * (np.exp(y_sup) - 1.0),
                    solver_residual_pp=100.0 * (np.exp(y_dem) - np.exp(y_sup)),
                    realized_minus_econ_pp=100.0 * (np.exp(y_real) - np.exp(y_dem)),
                    elasticity_lag_pp=lag_pp,
                    # What the food price would be if it cleared the production
                    # the biogeochemistry actually delivered, against what the
                    # model reports. This is where the gap lands on a published
                    # number: C-060 quotes regional food price indices.
                    price_index_reported=float(np.exp(float(r['PY_hat']))),
                    price_index_on_realized=float(np.exp(y_real / eta)),
                    price_index_error_pp=100.0 * (np.exp(float(r['PY_hat']))
                                                  - np.exp(y_real / eta)),
                    eta=eta,
                    clearing_residual=float(r['clearing_residual']),
                ))

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, 'econ_biophysical_yield_gap.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    solver = max(abs(r['solver_residual_pp']) for r in rows)
    gap = [abs(r['realized_minus_econ_pp']) for r in rows]
    worst = sorted(rows, key=lambda r: -abs(r['realized_minus_econ_pp']))[:3]
    capped = [r for r in rows if r['cap_binding']]
    clearing = max([abs(r['clearing_residual']) for r in capped] or [0.0])

    print('rows %d over %d scenarios x 8 regions x 30 years' % (len(rows), 3))
    print('solver residual (demand vs log-linear supply)  max %.2e pp' % solver)
    print('realized minus econ-implied production         max %.4f pp   mean %.4f pp'
          % (max(gap), sum(gap) / len(gap)))
    for r in worst:
        print('  worst %-18s %-18s yr%-3d realized %+7.3f%%  econ-implied %+7.3f%%  gap %+.4f pp'
              % (r['scenario'], r['region'], r['year'], r['y_realized_pct'],
                 r['y_demand_pct'], r['realized_minus_econ_pp']))
    print('cap-binding steps %d of %d; max clearing_residual %.2e'
          % (len(capped), len(rows), clearing))
    print('elasticity lag |d(beta+gamma)| per step         max %.4f pp'
          % max(r['elasticity_lag_pp'] for r in rows))
    pe = [r['price_index_error_pp'] for r in rows]
    wp = sorted(rows, key=lambda r: -abs(r['price_index_error_pp']))[:3]
    y10 = [abs(r['price_index_error_pp']) for r in rows if r['year'] == 10]
    print('food price index error (reported minus clears-realized)')
    print('  max %+.3f pp   mean |err| %.3f pp   year-10 mean |err| %.3f pp'
          % (max(pe, key=abs), sum(abs(x) for x in pe) / len(pe),
             sum(y10) / len(y10)))
    for r in wp:
        print('  worst %-18s %-18s yr%-3d reported %.4f  clears-realized %.4f  %+.3f pp'
              % (r['scenario'], r['region'], r['year'], r['price_index_reported'],
                 r['price_index_on_realized'], r['price_index_error_pp']))
    print('wrote %s' % os.path.relpath(out, os.path.join(HERE, '..', '..')))


if __name__ == '__main__':
    main()
