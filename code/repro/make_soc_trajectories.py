#!/usr/bin/env python3
"""Deposit the 30-year soil organic carbon trajectory the paper reports a number from.

Claim C-010 states that under 30 years of sustained disruption sub-Saharan
Africa's SOC declines by 2.5%. The canonical run has always deposited yield
losses and never the carbon they are supposed to follow from, so the half of
that sentence about soil was the half nothing could check. For a paper whose
argument is that soil organic matter buffers a fertilizer disruption, an
undeposited SOC trajectory is the wrong thing to be missing.

This runs the same scenario as `run_canonical.py`: S3, ERA5 climate, the
calibrated shock, 30 years, region-calibrated yield maxima. It is a separate
script rather than three more columns in the canonical CSV because canonical is
the root of the build graph and widening it restales every downstream node for
a change that alters no existing number.

Running the same configuration twice in two files is a real risk, and it is the
risk this repository has been paying for elsewhere: a quantity computed in two
places is a quantity that can drift apart. So the duplication is checked rather
than trusted. This script also writes each region's year-1, year-10 and year-30
yield loss, and `code/tests/test_soc_trajectories.py` fails if any of them
differs from `data/canonical_ERA5_y30.json`. If the two configurations ever
diverge, the loss columns diverge first and the test says so.

Writes: data/soc_trajectories.csv  (region x year, 0-30)
        data/soc_trajectories.json (per-region summary + the series)
"""
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
DATA = os.path.join(HERE, '..', '..', 'data')

from monthly_model_v3 import (MonthlyClimate, MonthlyNParams,   # noqa: E402
                              REGIONAL_CLIMATES)
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym  # noqa: E402
from coupled_econ_biophysical import (get_scenario_params,       # noqa: E402
                                      calibrate_price_shock)
from soil_n_model import get_default_regions                     # noqa: E402

REGIONS = ['north_america', 'europe', 'east_asia', 'south_asia',
           'southeast_asia', 'latin_america', 'sub_saharan_africa',
           'fsu_central_asia']

#: The scenario definition, registered as fert_reduction_target. Passed to the
#: same calibration the canonical run uses so the two cannot be calibrated to
#: different targets without the loss columns disagreeing.
TARGET_REDUCTION = 0.20

HORIZON_YEARS = 30


def patch_era5_climate():
    clim = json.load(open(os.path.join(DATA, 'era5_regional_climates.json')))
    for k, c in list(REGIONAL_CLIMATES.items()):
        n = clim[k]
        REGIONAL_CLIMATES[k] = MonthlyClimate(
            c.name, list(map(float, n['temp'])), list(map(float, n['precip'])),
            list(map(float, n['pet'])), c.planting_month, c.maturity_month)


def run():
    patch_era5_climate()
    regions = get_default_regions()
    mp = MonthlyNParams()
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(TARGET_REDUCTION)

    series, summary = {}, []
    for rk in REGIONS:
        r = regions[rk]
        ym = get_calibrated_ym(rk, mp)
        df = CoupledMonthlyModel(region=r, econ=s3, region_key=rk,
                                 t_max=float(HORIZON_YEARS),
                                 yield_max_override=ym).run()
        soc = [float(df[df['year'] == y]['soc_total'].iloc[0])
               for y in range(HORIZON_YEARS + 1)]
        series[rk] = soc
        loss = {y: float((1 - df[df['year'] == y]['yield_fraction'].iloc[0])
                         * 100) for y in (1, 10, 30)}
        # Percent decline from the year-0 stock, positive for a loss. Expressed
        # against the model's own year-0 rather than params.yaml's soc_initial:
        # the two agree at initialisation, and a divergence between them is a
        # spin-up defect that belongs in its own test, not silently folded into
        # a decline percentage.
        decline = 100.0 * (soc[0] - soc[HORIZON_YEARS]) / soc[0]
        summary.append(dict(
            region=rk,
            soc_yr0_tha=round(soc[0], 4),
            soc_yr30_tha=round(soc[HORIZON_YEARS], 4),
            soc_decline_pct_yr30=round(decline, 4),
            soc_decline_pct_yr10=round(
                100.0 * (soc[0] - soc[10]) / soc[0], 4),
            loss_yr1=round(loss[1], 4),
            loss_yr10=round(loss[10], 4),
            loss_yr30=round(loss[30], 4)))
    return summary, series


def main():
    summary, series = run()
    with open(os.path.join(DATA, 'soc_trajectories.csv'), 'w',
              newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['year'] + REGIONS)
        for y in range(HORIZON_YEARS + 1):
            w.writerow([y] + [round(series[rk][y], 4) for rk in REGIONS])
    with open(os.path.join(DATA, 'soc_trajectories.json'), 'w') as fh:
        json.dump({'regions': summary, 'series': series,
                   'horizon_years': HORIZON_YEARS}, fh, indent=1)
        fh.write('\n')
    for s in summary:
        print('%-20s SOC %.2f -> %.2f t/ha  decline yr10/yr30 = '
              '%.2f%% / %.2f%%'
              % (s['region'], s['soc_yr0_tha'], s['soc_yr30_tha'],
                 s['soc_decline_pct_yr10'], s['soc_decline_pct_yr30']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
