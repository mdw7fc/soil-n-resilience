#!/usr/bin/env python3
"""Generate results/s3_shock_calibration.csv, retiring the last unsourced input.

This file existed only as a deposited artifact from the lost v15 tree: the
Makefile carried `--allow-unsourced results/s3_shock_calibration.csv` as a
named debt line (F-009/F-015), and the second external audit correctly
observed that a deposit nothing regenerates can go stale invisibly. It had:
the deposited copy was computed under the zero eps_F_N central and the linear
clearing, while C-050 read its columns as if they were current.

What the file records, per region: the calibrated global price shock
(nitrogen-tonnage-weighted to a 20 percent S1 reduction, uniform across
regions by construction), the region's S1 year-1 fertilizer-use reduction,
and the S3 fertilizer-use reduction at years 1, 5, 10 and 30 plus the mean
over the sustained-disruption years. All reductions are fractions of the
baseline synthetic-N rate, matching the deposited schema so C-050's
resolvers read it unchanged.
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
    get_scenario_params, calibrate_price_shock,
)
from soil_n_model import get_default_regions  # noqa: E402

DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']
TARGET = 0.20


def fert_path(region, econ, key, ym):
    df = CoupledMonthlyModel(region=region, econ=econ, region_key=key,
                             t_max=30.0, yield_max_override=ym).run()
    base = float(df['fert_applied_kgha'].iloc[0])
    red = {}
    series = []
    for y in range(1, 31):
        f = float(df.loc[df['year'] == y, 'fert_applied_kgha'].iloc[0])
        r = 1.0 - f / base
        series.append(r)
        if y in (1, 5, 10, 30):
            red[y] = r
    red['mean'] = float(np.mean(series))
    return base, red


def main():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))
    reg = get_default_regions()
    mp = MonthlyNParams()
    shock = calibrate_price_shock(TARGET)
    scen = get_scenario_params()
    s1, s3 = scen['S1'], scen['S3']
    s1.fert_price_shock = shock
    s3.fert_price_shock = shock

    rows = []
    for key in RO:
        ym = get_calibrated_ym(key, mp)
        base, r1 = fert_path(reg[key], s1, key, ym)
        _, r3 = fert_path(reg[key], s3, key, ym)
        rows.append(dict(
            region=key,
            synth_n_baseline_kgha=round(base, 2),
            solved_shock_pct=round(100.0 * shock, 1),
            target_reduction=TARGET,
            s1_reduction_yr1=round(r1[1], 5),
            s3_reduction_mean=round(r3['mean'], 5),
            s3_reduction_yr1=round(r3[1], 5),
            s3_reduction_yr5=round(r3[5], 5),
            s3_reduction_yr10=round(r3[10], 5),
            s3_reduction_yr30=round(r3[30], 5),
        ))
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, 's3_shock_calibration.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote %s  shock %.1f%%' % (os.path.relpath(out), 100 * shock))
    for r in rows:
        print('  %-20s s1_yr1 %.3f  s3_mean %.3f' % (
            r['region'], r['s1_reduction_yr1'], r['s3_reduction_mean']))


if __name__ == '__main__':
    main()
