#!/usr/bin/env python3
"""
Canonical model run for Wallenstein & Manning, ERFS-100341.
Runs the coupled biophysical-economic model under the ERA5 data-based climate
out to year 30 (scenario S3, 100% fertilizer-price spike) and writes the single
frozen output used for central regional trajectories and correlations.

Outputs (written to ../../data/ and ../../outputs/):
  canonical_ERA5_y30.csv / .json  - per-region descriptors + yr1/10/30 losses
  global_S3_losses.txt            - production-weighted global loss, yr 1/10/30

Audited SOL result (production-weighted): 2.31 / 3.18 / 3.29 % for years
1 / 10 / 30.

v1.3: these values supersede the v1.2 figures of 4.33 / 5.58 / 5.95 %. Two
internal-consistency corrections were applied to the model (a stationary
Century spin-up that applies the same baseline water-stress multiplier the
simulation uses, and a re-solved rather than clipped equilibrium when the
physical fertilizer ceiling binds). See CHANGELOG.md.
"""
import os, sys, json, csv, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'model')
ERA5  = os.path.join(HERE, '..', 'era5')
DATA  = os.path.join(HERE, '..', '..', 'data')
OUT   = os.path.join(HERE, '..', '..', 'outputs')
sys.path.insert(0, MODEL)
import numpy as np
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file, get_regional_bnf
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params, calibrate_price_shock, REGIONAL_ECON_PARAMS
from soil_n_model import get_default_regions
from seams import outcome_weights

REGIONS = ['north_america','europe','east_asia','south_asia',
           'southeast_asia','latin_america','sub_saharan_africa','fsu_central_asia']
ABBR = {'north_america':'NA','europe':'EU','east_asia':'EA','south_asia':'SA',
        'southeast_asia':'SEA','latin_america':'LATAM','sub_saharan_africa':'SSA','fsu_central_asia':'FSU'}

def patch_era5_climate():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))

def main():
    patch_era5_climate()
    regions = get_default_regions(); mp = MonthlyNParams()
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(0.20)   # 100% price spike calibration
    rows = []
    for rk in REGIONS:
        r = regions[rk]; ym = get_calibrated_ym(rk, mp)
        df = CoupledMonthlyModel(region=r, econ=s3, region_key=rk,
                                 t_max=30.0, yield_max_override=ym).run()
        y0 = df[df['year'] == 0].iloc[0]
        loss = lambda yr: float((1 - df[df['year'] == yr]['yield_fraction'].iloc[0]) * 100)
        nmin0 = float(y0['n_mineralized'])
        # Derived share of gross N supply supplied by SOM mineralization.
        # All external sources use the same basis as the live monthly engine.
        bnf = get_regional_bnf(rk)
        total_n = nmin0 + r.synth_n_current + r.atm_n_deposition + bnf
        ep = REGIONAL_ECON_PARAMS[rk]
        rows.append(dict(
            region=rk, abbr=ABBR[rk],
            y_max=float(ym), y_base=float(y0['yield_tha']),
            loss_yr1=loss(1), loss_yr10=loss(10), loss_yr30=loss(30),
            buffer_ratio_pct=round(nmin0 / total_n * 100, 1),
            bnf=float(bnf), soc=float(r.soc_initial),
            water_deficit=float(r.baseline_water_deficit),
            synth_n=float(r.synth_n_current),
            eps_F_PF=float(ep['eps_F_PF']), eta=float(ep['eta']),
            cropland_mha=float(r.cropland_mha)))
    # freeze
    os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(DATA, 'canonical_ERA5_y30.csv'), 'w', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        ); w.writeheader()
        [w.writerow(x) for x in rows]
    # Global aggregation basis is declared, not improvised here. Until
    # 2026-07-25 (F-005) this line normalised its own production-tonnage
    # vector inline while two other files used two other bases; the vector is
    # now supplied and validated by seams.outcome_weights, which produces the
    # identical weights and additionally refuses a dropped or zero-weight
    # region and carries a provenance string into the frozen output.
    yb = [x['y_base'] for x in rows]
    W_out = outcome_weights([x['region'] for x in rows], yb, regions)
    W = W_out.as_array()
    gl = {y: round(float((np.array([x[f'loss_yr{y}'] for x in rows]) * W).sum()), 2) for y in (1, 10, 30)}
    json.dump({'regions': rows, 'global_prodweighted': gl,
               'aggregation_basis': W_out.basis,
               'aggregation_provenance': W_out.provenance},
              open(os.path.join(DATA, 'canonical_ERA5_y30.json'), 'w'), indent=1)
    with open(os.path.join(OUT, 'global_S3_losses.txt'), 'w') as f:
        f.write("Production-weighted global S3 yield loss (%)\n")
        f.write("year 1 : %.2f\nyear 10: %.2f\nyear 30: %.2f\n" % (gl[1], gl[10], gl[30]))
    print("Global production-weighted S3 loss  yr1/yr10/yr30 = %.2f / %.2f / %.2f %%" % (gl[1], gl[10], gl[30]))
    print("(SOL manuscript: 2.3 / 3.2 / 3.3)")
    print("buffer ratios:", [x['buffer_ratio_pct'] for x in rows])

if __name__ == '__main__':
    main()
