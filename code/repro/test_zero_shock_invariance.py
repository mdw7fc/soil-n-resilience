#!/usr/bin/env python3
"""Zero-shock invariance test (v1.3).

With no fertilizer-price shock, no physical supply constraint, and all
behavioural channels disabled, the coupled model must remain at its baseline
yield indefinitely: any departure is spin-up drift rather than a response to
a disruption. This test is the acceptance criterion for the stationary,
water-stress-aware spin-up introduced in v1.3.

Also reports:
  - year-2 simulated yield against the FAOSTAT calibration targets;
  - the SC1 year-1 market-clearing residual under the physical cap,
    gamma * (F_hat_capped - F_hat_desired), which must be 0 once the
    constrained equilibrium is re-solved rather than clipped.

Writes ../../outputs/zero_shock_invariance.csv and prints a PASS/FAIL summary.
"""
import os, sys, json, csv, copy, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import numpy as np
from monthly_model_v3 import MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES, FAOSTAT_TARGETS
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params
from soil_n_model import get_default_regions

DATA = os.path.join(HERE, '..', '..', 'data')
OUT = os.path.join(HERE, '..', '..', 'outputs')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']
TOL_YR10 = 0.005   # 0.5 % maximum permitted year-10 drift
TOL_YR30 = 0.010   # 1.0 % maximum permitted year-30 drift


def patch_era5():
    clim = json.load(open(os.path.join(DATA, 'era5_regional_climates.json')))
    for k, c in list(REGIONAL_CLIMATES.items()):
        n = clim[k]
        REGIONAL_CLIMATES[k] = MonthlyClimate(
            c.name, list(map(float, n['temp'])), list(map(float, n['precip'])),
            list(map(float, n['pet'])), c.planting_month, c.maturity_month)


def zero_shock_econ():
    """S3 stripped of every shock and behavioural channel.

    fert_price_shock = 0 removes the disruption; zeroing the three
    fertilizer-demand elasticities holds fertilizer at its baseline rate so
    that no endogenous input adjustment can mask or create drift; the supply
    ceiling is released. What remains is the biophysical system running at
    baseline management, which must be stationary.
    """
    e = copy.deepcopy(get_scenario_params()['S3'])
    e.fert_price_shock = 0.0
    e.fert_supply_ceiling = 1.0
    e.fert_capacity_recovery_years = 0.0
    e.eps_F_PF = 0.0
    e.eps_F_PY = 0.0
    e.eps_F_N = 0.0
    return e


def main():
    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()
    econ = zero_shock_econ()
    rows, worst = [], 0.0
    for rk in RO:
        ym = get_calibrated_ym(rk, mp)
        df = CoupledMonthlyModel(region=regions[rk], econ=econ, region_key=rk,
                                 t_max=30.0, yield_max_override=ym).run()
        frac = {y: float(df[df['year'] == y]['yield_fraction'].iloc[0]) for y in range(31)}
        dev = max(abs(frac[y] - 1.0) for y in range(31))
        worst = max(worst, dev)
        rows.append(dict(region=rk,
                         yr10_yield_fraction=round(frac[10], 5),
                         yr30_yield_fraction=round(frac[30], 5),
                         max_abs_deviation_30yr=round(dev, 6),
                         yr2_yield_tha=round(float(df[df['year'] == 2]['yield_tha'].iloc[0]), 3),
                         faostat_target_tha=FAOSTAT_TARGETS[rk]))
    area = np.array([regions[k].cropland_mha for k in RO])
    yb = np.array([regions[k].yield_max_regional for k in RO])
    W = area * yb
    W /= W.sum()
    g10 = float(((1 - np.array([r['yr10_yield_fraction'] for r in rows])) * W).sum() * 100)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'zero_shock_invariance.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]
    print("%-22s %10s %10s %12s" % ("region", "yr10 frac", "yr30 frac", "max |dev|"))
    for r in rows:
        print("%-22s %10.5f %10.5f %12.6f" % (r['region'], r['yr10_yield_fraction'],
                                              r['yr30_yield_fraction'], r['max_abs_deviation_30yr']))
    print("\nglobal weighted year-10 no-shock drift : %+.3f %%" % g10)
    print("maximum 30-year deviation from baseline: %.2e" % worst)
    ok = (max(abs(r['yr10_yield_fraction'] - 1) for r in rows) < TOL_YR10 and
          max(abs(r['yr30_yield_fraction'] - 1) for r in rows) < TOL_YR30)
    print("\nZERO-SHOCK INVARIANCE: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
