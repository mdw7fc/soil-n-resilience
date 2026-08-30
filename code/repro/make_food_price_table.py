#!/usr/bin/env python3
"""Regional food-price response under the sustained S3 price shock.

Writes data/food_price_response.csv: the regional output-price index
exp(P_Y_hat) at years 1, 10 and 30 for every region, plus the
production-weighted global aggregate. These are the values quoted in the
Supplementary Information ("Food-price impacts").

P_Y_hat is the reduced-form regional output-price response of the linearized
market-clearing system described in the Methods. In unconstrained runs it is a
reduced-form price index conditional on the assumed elasticities, not a
calibrated food-price forecast.

New in deposit v1.3: this table was previously reported without a generator.
"""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'model'))

import numpy as np
from run_canonical import patch_era5_climate, REGIONS, ABBR, DATA
from monthly_model_v3 import MonthlyNParams
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params, calibrate_price_shock
from soil_n_model import get_default_regions


def main():
    patch_era5_climate()
    regions = get_default_regions()
    mp = MonthlyNParams()
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(0.20)

    rows = []
    for rk in REGIONS:
        r = regions[rk]
        ym = get_calibrated_ym(rk, mp)
        df = CoupledMonthlyModel(region=r, econ=s3, region_key=rk,
                                 t_max=30.0, yield_max_override=ym).run()
        y0 = df[df['year'] == 0].iloc[0]

        def pct(yr):
            v = float(df[df['year'] == yr]['food_price_index'].iloc[0])
            return round((v - 1.0) * 100, 2)

        rows.append(dict(region=rk, abbr=ABBR[rk],
                         cropland_mha=float(r.cropland_mha),
                         y_base=float(y0['yield_tha']),
                         food_price_pct_yr1=pct(1),
                         food_price_pct_yr10=pct(10),
                         food_price_pct_yr30=pct(30)))

    w = np.array([x['cropland_mha'] * x['y_base'] for x in rows])
    w = w / w.sum()
    glob = {k: round(float(np.dot(w, [x[k] for x in rows])), 2)
            for k in ('food_price_pct_yr1', 'food_price_pct_yr10', 'food_price_pct_yr30')}
    rows.append(dict(region='GLOBAL (production-weighted)', abbr='GLOBAL',
                     cropland_mha='', y_base='', **glob))

    out = os.path.join(DATA, 'food_price_response.csv')
    with open(out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        [wr.writerow(x) for x in rows]

    for x in rows:
        print('%-30s yr1=%+6.2f%%  yr10=%+6.2f%%  yr30=%+6.2f%%'
              % (x['region'], x['food_price_pct_yr1'],
                 x['food_price_pct_yr10'], x['food_price_pct_yr30']))
    print('wrote data/food_price_response.csv')


if __name__ == '__main__':
    main()
