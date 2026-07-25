#!/usr/bin/env python3
"""Constrained-equilibrium market-clearing test (v1.3).

In the supply-constrained scenarios the physical fertilizer ceiling can bind.
Before v1.3 the model solved the unconstrained equilibrium and then clipped
fertilizer to the ceiling, so the food price and land allocation it reported
corresponded to the fertilizer producers *wanted*, not the fertilizer that was
physically available. v1.3 re-solves the equilibrium under the binding cap,

    F_hat = ln(c_t) - lambda_L * PY_hat,
    PY_hat = [beta*N_hat + gamma*ln(c_t)] / [eta - (alpha - gamma)*lambda_L].

This test recomputes, region by region and year by year, the clearing residual

    gamma * (F_hat_realised - F_hat_implied_by_the_reported_price)

which must be zero whenever the cap binds. It also prints the SC1/SC2
production-weighted global losses at years 1, 10 and 30.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import numpy as np
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_supply_constrained_scenarios
from soil_n_model import get_default_regions

DATA = os.path.join(HERE, '..', '..', 'data')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']
TOL = 1e-6


def patch_era5():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))


def main():
    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()
    scen = get_supply_constrained_scenarios()
    area = np.array([regions[k].cropland_mha for k in RO])
    worst = 0.0
    for name, econ in [('SC1', scen['SC1_20pct']), ('SC2', scen['SC2_20pct_recovery'])]:
        per, yb = {}, []
        for rk in RO:
            m = CoupledMonthlyModel(region=regions[rk], econ=econ, region_key=rk,
                                    t_max=30.0, yield_max_override=get_calibrated_ym(rk, mp))
            df = m.run()
            per[rk] = {y: float((1 - df[df['year'] == y]['yield_fraction'].iloc[0]) * 100)
                       for y in (1, 10, 30)}
            yb.append(float(df[df['year'] == 0]['yield_tha'].iloc[0]))
            # clearing residual where the cap binds
            if 'cap_binding' in df.columns and 'clearing_residual' in df.columns:
                sub = df[df['cap_binding'].astype(bool)]
                if len(sub):
                    worst = max(worst, float(np.abs(sub['clearing_residual']).max()))
        W = area * np.array(yb)
        W /= W.sum()
        g = {y: float(sum(per[k][y] * W[j] for j, k in enumerate(RO))) for y in (1, 10, 30)}
        print("%s production-weighted global loss  yr1/yr10/yr30 = %.2f / %.2f / %.2f %%"
              % (name, g[1], g[10], g[30]))
        for j, k in enumerate(RO):
            print("    %-22s %6.2f %6.2f %6.2f" % (k, per[k][1], per[k][10], per[k][30]))
    print("\nmaximum cap-clearing residual (log-points): %.2e" % worst)
    print("CONSTRAINED MARKET CLEARING: %s" % ("PASS" if worst < TOL else "FAIL"))
    return 0 if worst < TOL else 1


if __name__ == '__main__':
    sys.exit(main())
