#!/usr/bin/env python3
"""Regenerate data/figS8_curves.json (Figure S8 elasticity-sensitivity source).

Runs the canonical S3 scenario under the ERA5 climate twice: once at the
baseline fertilizer-demand price elasticity eps_F_PF, once at half that value
in every region. Writes per-region and production-weighted global yield-loss
trajectories for years 0-30.

Added in v1.3: previously figS8_curves.json was produced ad hoc and could not
be regenerated from the deposit, so Figure S8 did not track changes to the
model. It now does.
"""
import os, sys, json, copy, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import numpy as np
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params, calibrate_price_shock, REGIONAL_ECON_PARAMS
from soil_n_model import get_default_regions

DATA = os.path.join(HERE, '..', '..', 'data')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']


def patch_era5():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))


def trajectories(econ, regions, mp, halve=False):
    """Year 0-30 loss trajectories.

    The model takes eps_F_PF from REGIONAL_ECON_PARAMS when a region_key is
    supplied, so the halved-elasticity arm patches the regional table rather
    than the scenario object. The calibrated price shock is held fixed across
    both arms: the disruption is the same physical event, and only the
    behavioural response to it is varied.
    """
    saved = {k: REGIONAL_ECON_PARAMS[k]['eps_F_PF'] for k in RO}
    if halve:
        for k in RO:
            REGIONAL_ECON_PARAMS[k]['eps_F_PF'] = saved[k] * 0.5
    per, yb = {}, []
    for rk in RO:
        e = copy.deepcopy(econ)
        df = CoupledMonthlyModel(region=regions[rk], econ=e, region_key=rk,
                                 t_max=30.0,
                                 yield_max_override=get_calibrated_ym(rk, mp)).run()
        per[rk] = {str(y): float((1 - df[df['year'] == y]['yield_fraction'].iloc[0]) * 100)
                   for y in range(31)}
        yb.append(float(df[df['year'] == 0]['yield_tha'].iloc[0]))
    area = np.array([regions[k].cropland_mha for k in RO])
    W = area * np.array(yb)
    W /= W.sum()
    g = [float(sum(per[k][str(y)] * W[j] for j, k in enumerate(RO))) for y in range(31)]
    for k in RO:                      # restore the regional table
        REGIONAL_ECON_PARAMS[k]['eps_F_PF'] = saved[k]
    return per, g


def main():
    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(0.20)
    base, gbase = trajectories(s3, regions, mp, halve=False)
    half, ghalf = trajectories(s3, regions, mp, halve=True)
    out = dict(RO=RO, base=base, half=half, gbase=gbase, ghalf=ghalf)
    json.dump(out, open(os.path.join(DATA, 'figS8_curves.json'), 'w'), indent=1)
    rho = np.corrcoef(
        np.argsort(np.argsort([base[k]['10'] for k in RO])),
        np.argsort(np.argsort([half[k]['10'] for k in RO])))[0, 1]
    print("global yr10 baseline / halved = %.2f / %.2f %%" % (gbase[10], ghalf[10]))
    print("regional Spearman rho (baseline vs halved, yr10) = %.2f" % rho)


if __name__ == '__main__':
    main()
