#!/usr/bin/env python3
"""F-002 evidence: the gap between the published calibration path and the one
that was actually run, and the recalibrated `yield_max` that closes it.

Writes results/calibration_production_path.csv with, per region:
  ym_legacy      yield_max from monthly_model_v3.calibrate_ym (roots run_model)
  ym_production  yield_max from coupled_monthly.calibrate_ym_production
                 (roots century_dynamic_spinup, the published path)
  y_prod_legacy  production-path baseline yield the legacy ym delivers
  y_prod_new     production-path baseline yield the new ym delivers
  faostat_target the calibration target
  gap_legacy_pct how far the published path missed FAOSTAT (F-002 headline)
  gap_new_pct    residual after recalibration (acceptance: <= 8e-3 percent)
"""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'model')
DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')
sys.path.insert(0, MODEL)
from monthly_model_v3 import (MonthlyNParams, apply_era5_climate_file,
                              FAOSTAT_TARGETS, century_dynamic_spinup, calibrate_ym)
from coupled_monthly import (calibrate_ym_production, CALIBRATION_SCHEME,
                             YM_REGION_FIELDS)
from soil_n_model import get_default_regions


def y_production(region_key, region, ym, mp):
    return century_dynamic_spinup(region_key, p=mp,
                                  synth_n=region.synth_n_current,
                                  yield_max_override=ym,
                                  region_override=region)['yield_eq']


def main():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))
    mp = MonthlyNParams()
    regions = get_default_regions()
    rows, worst_new, worst_legacy = [], 0.0, 0.0
    for rk, target in FAOSTAT_TARGETS.items():
        r = regions[rk]
        ym_legacy = calibrate_ym(rk, target, mp)
        ym_new = calibrate_ym_production(rk, target, mp, region=r)
        y_legacy = y_production(rk, r, ym_legacy, mp)
        y_new = y_production(rk, r, ym_new, mp)
        gap_legacy = 100.0 * (y_legacy - target) / target
        gap_new = 100.0 * (y_new - target) / target
        worst_legacy = max(worst_legacy, abs(gap_legacy))
        worst_new = max(worst_new, abs(gap_new))
        rows.append(dict(region=rk, scheme=CALIBRATION_SCHEME,
                         ym_legacy=round(ym_legacy, 6),
                         ym_production=round(ym_new, 6),
                         ym_change_pct=round(100 * (ym_new - ym_legacy) / ym_legacy, 4),
                         y_prod_legacy=round(y_legacy, 6),
                         y_prod_new=round(y_new, 6),
                         faostat_target=target,
                         gap_legacy_pct=round(gap_legacy, 4),
                         gap_new_pct=float('%.3e' % gap_new)))
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, 'calibration_production_path.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        [w.writerow(x) for x in rows]
    print("wrote %s" % out)
    print("legacy path missed FAOSTAT by up to %.3f percent" % worst_legacy)
    print("production path residual worst case %.3e percent" % worst_new)
    print("YM_REGION_FIELDS: %d" % len(YM_REGION_FIELDS))
    ok = worst_new <= 8e-3
    print("F-002 ACCEPTANCE (<= 8e-3 percent): %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
