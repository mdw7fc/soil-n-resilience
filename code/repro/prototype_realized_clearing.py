#!/usr/bin/env python3
"""Prototype: clear the food market on the realized biogeochemical yield.

Dale Manning's proposed remedy for F-022/F-023 (2026-08-29): instead of
clearing eta*PY_hat against the log-linear supply relation, root-find the food
price at which demand equals the production change the biogeochemistry
actually delivers at that price's fertilizer level, with land at alpha:

    eta*PY = ln(yield_frac(F_level(PY))) + alpha*lambda_L*PY

Each residual evaluation runs the monthly biophysical step for the candidate
fertilizer level from a snapshot of the soil state, so beta and gamma drop out
of the clearing entirely; the elasticities remain diagnostics.

HISTORICAL NOTE: F-025 wired realized clearing into CoupledMonthlyModel
itself, so this prototype's linear-mode gate was passed against the
PRE-CHANGE model at commit 6c2cf9b and will no longer reproduce run(). It is
kept as the evidence trail for the adoption decision, not as a living test.

This file was a PROTOTYPE, not a wiring change: it reimplements the annual loop
of CoupledMonthlyModel compactly, in two modes.

  linear   -- must reproduce CoupledMonthlyModel.run() exactly. This is the
              gate: if the compact loop cannot reproduce the model it copies,
              nothing its realized mode says can be trusted.
  realized -- Dale's clearing.

Writes results/realized_clearing_comparison.csv and prints only summaries.
"""
import copy
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file, monthly_n_balance  # noqa: E402
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym  # noqa: E402
from coupled_econ_biophysical import (  # noqa: E402
    get_scenario_params, calibrate_price_shock, supply_state,
)
from soil_n_model import get_default_regions  # noqa: E402

DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']


class RealizedClearingModel(CoupledMonthlyModel):
    """CoupledMonthlyModel with a selectable clearing mode."""

    def __init__(self, *a, clearing='linear', **kw):
        super().__init__(*a, **kw)
        self.clearing = clearing
        self.solver_evals = []

    def _init_elasticities(self):
        bnf_base = self.bio.bnf_baseline
        nb = monthly_n_balance(
            self.bio.C_active, self.bio.C_slow, self.bio.C_passive,
            self.region.cn_bulk, self.F_baseline, bnf_base,
            self.region.atm_n_deposition, self.bio.climate,
            self.bio.mineral_n, self.bio.mp, som_params=self.bio.som)
        eps = 1e-10
        n_up = max(nb['uptake'], eps)
        y_mit = self.bio.y_max * (1.0 - np.exp(-self.bio.mit_c * n_up))
        y_st = n_up / self.bio.n_cost_per_tonne
        if y_st < y_mit:
            el = 1.0
        else:
            e = np.exp(-self.bio.mit_c * n_up)
            el = self.bio.mit_c * n_up * e / max(1.0 - e, eps)
        tot = nb['min'] + self.F_baseline + bnf_base + self.region.atm_n_deposition
        b = el * max(0.0, nb['min'] / max(tot, eps))
        g = el * (self.F_baseline / max(tot, eps))
        return nb, b, g

    def _levels(self, PY, ceiling):
        """Fertilizer and land levels implied by a candidate food price."""
        F_hat = self.eps_F_PF * self.PF_hat + self.eps_F_PY * PY + self.eps_F_N * self.N_hat
        L_hat = self._lambda_L() * PY
        F_level = max(0.0, self.F_baseline * np.exp(F_hat))
        L_level = self.L_baseline * np.exp(L_hat)
        capped = False
        if ceiling < 1.0:
            F_max = self.F_baseline * self.L_baseline * ceiling / max(L_level, 1e-6)
            if F_level > F_max:
                F_level, capped = F_max, True
        return F_hat, L_hat, F_level, L_level, capped

    def run(self):
        n_steps = int(self.t_max / self.dt) + 1
        cols = ['year', 'PY_hat', 'F_hat', 'L_hat', 'N_hat', 'fert_applied_kgha',
                'land_mha', 'food_price_index', 'yield_fraction', 'n_mineralized',
                'soc_fraction', 'cap_binding']
        res = {c: np.zeros(n_steps) for c in cols}
        nb0, beta, gamma = self._init_elasticities()
        self.N_min_baseline = nb0['min']
        prev_n_min = nb0['min']

        for i in range(n_steps):
            t = i * self.dt
            res['year'][i] = t
            if i == 0:
                res['fert_applied_kgha'][i] = self.F_baseline
                res['land_mha'][i] = self.L_baseline
                res['food_price_index'][i] = 1.0
                res['yield_fraction'][i] = 1.0
                res['n_mineralized'][i] = prev_n_min
                res['soc_fraction'][i] = 1.0
                continue

            self.N_hat = (np.log(max(prev_n_min, 1e-6) / self.N_min_baseline)
                          if self.N_min_baseline > 0 else 0.0)
            supply = supply_state(self.econ, t)
            self.PF_hat = self.PF_hat_base * supply.price_frac

            if self.clearing == 'linear':
                PY, F_hat, L_hat = self._solve_equilibrium(beta, gamma)
                F_level = max(0.0, self.F_baseline * np.exp(F_hat))
                L_level = self.L_baseline * np.exp(L_hat)
                capped = False
                if supply.ceiling < 1.0:
                    F_max = (self.F_baseline * self.L_baseline * supply.ceiling
                             / max(L_level, 1e-6))
                    if F_level > F_max * (1.0 + 1e-9):
                        PY, F_hat, L_hat = self._solve_equilibrium_capped(
                            beta, gamma, np.log(supply.ceiling))
                        F_level = max(0.0, self.F_baseline * np.exp(F_hat))
                        L_level = self.L_baseline * np.exp(L_hat)
                        capped = True
                bio_state = self.bio.step(F_level)
            else:
                snap = copy.deepcopy(self.bio)
                evals = [0]

                def residual(PY):
                    evals[0] += 1
                    _, L_hat, F_level, _, _ = self._levels(PY, supply.ceiling)
                    trial = copy.deepcopy(snap)
                    yf = max(trial.step(F_level)['yield_fraction'], 1e-9)
                    return self.eta * PY - (np.log(yf) + self.alpha * L_hat)

                g0, _, _ = self._solve_equilibrium(beta, gamma)
                lo, hi = g0 - 0.10, g0 + 0.10
                for _ in range(12):
                    if residual(lo) * residual(hi) < 0:
                        break
                    lo -= 0.10
                    hi += 0.10
                else:
                    raise RuntimeError('no bracket at t=%r' % t)
                PY = brentq(residual, lo, hi, xtol=1e-12)
                self.solver_evals.append(evals[0])
                F_hat, L_hat, F_level, L_level, capped = self._levels(PY, supply.ceiling)
                self.bio = snap
                bio_state = self.bio.step(F_level)

            self.PY_hat, self.F_hat, self.L_hat = PY, F_hat, L_hat
            prev_n_min = bio_state['n_mineralized']
            beta, gamma = bio_state['beta'], bio_state['gamma']
            res['PY_hat'][i] = PY
            res['F_hat'][i] = F_hat
            res['L_hat'][i] = L_hat
            res['N_hat'][i] = self.N_hat
            res['fert_applied_kgha'][i] = F_level
            res['land_mha'][i] = L_level
            res['food_price_index'][i] = np.exp(PY)
            res['yield_fraction'][i] = bio_state['yield_fraction']
            res['n_mineralized'][i] = prev_n_min
            res['soc_fraction'][i] = bio_state['soc_fraction']
            res['cap_binding'][i] = 1.0 if capped else 0.0
        import pandas as pd
        return pd.DataFrame(res)


def main():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))
    reg = get_default_regions()
    mp = MonthlyNParams()
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(0.20)

    # Gate: the compact linear loop must reproduce the real model.
    worst = 0.0
    for key in ('north_america', 'sub_saharan_africa'):
        kw = dict(region=reg[key], econ=s3, region_key=key, t_max=30.0,
                  yield_max_override=get_calibrated_ym(key, mp))
        ref = CoupledMonthlyModel(**kw).run()
        mine = RealizedClearingModel(clearing='linear', **kw).run()
        for c in ('PY_hat', 'yield_fraction', 'fert_applied_kgha'):
            worst = max(worst, float(np.nanmax(np.abs(
                ref[c].to_numpy() - mine[c].to_numpy()))))
    print('linear-mode reproduction gate: worst |diff| %.3e %s'
          % (worst, 'PASS' if worst < 1e-9 else 'FAIL'))
    if worst >= 1e-9:
        sys.exit(1)

    rows = []
    ev_all = []
    for key in RO:
        kw = dict(region=reg[key], econ=s3, region_key=key, t_max=30.0,
                  yield_max_override=get_calibrated_ym(key, mp))
        ref = CoupledMonthlyModel(**kw).run()
        m = RealizedClearingModel(clearing='realized', **kw)
        new = m.run()
        ev_all += m.solver_evals
        for y in (1, 2, 5, 10, 20, 30):
            i = int(y)
            rows.append(dict(
                region=key, year=y,
                price_linear=float(np.exp(ref['PY_hat'].iloc[i])),
                price_realized=float(np.exp(new['PY_hat'].iloc[i])),
                dprice_pp=100 * float(np.exp(new['PY_hat'].iloc[i])
                                      - np.exp(ref['PY_hat'].iloc[i])),
                yloss_linear_pct=100 * (1 - float(ref['yield_fraction'].iloc[i])),
                yloss_realized_pct=100 * (1 - float(new['yield_fraction'].iloc[i])),
                dyield_pp=100 * (float(new['yield_fraction'].iloc[i])
                                 - float(ref['yield_fraction'].iloc[i])),
                dfert_pct=100 * (float(new['fert_applied_kgha'].iloc[i])
                                 / float(ref['fert_applied_kgha'].iloc[i]) - 1),
            ))
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, 'realized_clearing_comparison.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dp = [r['dprice_pp'] for r in rows]
    dy = [r['dyield_pp'] for r in rows]
    dfz = [r['dfert_pct'] for r in rows]
    print('solver: %d clearings, evals/step mean %.1f max %d'
          % (len(ev_all), sum(ev_all) / len(ev_all), max(ev_all)))
    print('price change (realized minus linear): mean %+0.3f pp  range [%+0.3f, %+0.3f]'
          % (sum(dp) / len(dp), min(dp), max(dp)))
    print('yield change: mean %+0.4f pp  max|.| %.4f pp' % (sum(dy) / len(dy), max(abs(v) for v in dy)))
    print('fertilizer change: max|.| %.3f %%' % max(abs(v) for v in dfz))
    print('year 10:')
    for r in rows:
        if r['year'] == 10:
            print('  %-20s price %.4f -> %.4f (%+0.3f pp)   yloss %.3f -> %.3f'
                  % (r['region'], r['price_linear'], r['price_realized'],
                     r['dprice_pp'], r['yloss_linear_pct'], r['yloss_realized_pct']))
    print('wrote %s' % os.path.relpath(out, os.path.join(HERE, '..', '..')))


if __name__ == '__main__':
    main()
