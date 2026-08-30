#!/usr/bin/env python3
"""Calibration-path assertions (finding F-002).

The manuscript says yields are calibrated to FAOSTAT. Until 2026-07-25 that
was true of `monthly_model_v3.calibrate_ym`, which roots `run_model` — a path
that uses the global `CropParams.mitscherlich_c`, applies no baseline
water-stress multiplier, and was never run to produce a published number.
Every published run goes through `century_dynamic_spinup` plus
`MonthlyBiophysicalEngine`. No test caught the difference because every test
compared the model to itself.

This file asserts four things:

C1  The production path reproduces the FAOSTAT targets to 1e-3 relative.
C2  The legacy objective's gap is still there, and is still large. If this
    ever falls below 1% the two paths have converged and the second-path
    hazard this file exists to guard has gone away — at which point delete
    the guard deliberately rather than letting it quietly stop meaning
    anything.
C3  Every RegionParams field that moves `yield_max` is hashed into
    `calibration_fingerprint`. A field that reaches the calibration and is
    not in `YM_REGION_FIELDS` silently poisons the cache: two different
    regions parameterisations map to one cache key.
C4  The fingerprint leads with `CALIBRATION_SCHEME`, so a `yield_max` fitted
    under the legacy scheme can never be served for a production-path run.

OWED. F-002 also specifies an AST scan of the calibration path for region
fields the fingerprint does not hash. That scan is not implemented here; C3
is the empirical version of the same check and will catch any field that
actually moves the answer, but it cannot catch a field that is read and then
has no effect at the current parameter point. Tracked as owed work.

Writes results/calibration_fingerprint_checks.yaml.
"""
import os, sys, copy, warnings
from dataclasses import fields as dataclass_fields
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import numpy as np
from monthly_model_v3 import (MonthlyNParams, apply_era5_climate_file,
                              FAOSTAT_TARGETS, calibrate_ym,
                              century_dynamic_spinup)
from coupled_monthly import (calibrate_ym_production, calibration_fingerprint,
                             CALIBRATION_SCHEME, YM_REGION_FIELDS)
from soil_n_model import get_default_regions, RegionParams

DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')

#: C1 tolerance, relative to the FAOSTAT target.
FAOSTAT_RTOL = 1e-3
#: C2 floor. The legacy path must still miss by at least this much.
LEGACY_GAP_FLOOR_PCT = 1.0
#: C3 sensitivity. A perturbed field that moves `yield_max` by more than this
#: relative amount counts as reaching the calibration.
YM_MOVE_RTOL = 1e-6
#: Regions probed in C3. The perturbation sweep is the expensive check; two
#: contrasting regions (temperate high-input, tropical low-input) exercise
#: both the Mitscherlich and the stoichiometric-cap regimes.
C3_REGIONS = ('north_america', 'sub_saharan_africa')

checks = []


def record(section, name, ok, detail=''):
    checks.append(dict(section=section, check=name, pass_=bool(ok), detail=str(detail)))
    print('  [%s] %-56s %s' % ('PASS' if ok else 'FAIL', name, detail))
    return ok


def y_production(region_key, region, ym, mp):
    return century_dynamic_spinup(region_key, p=mp,
                                  synth_n=region.synth_n_current,
                                  yield_max_override=ym,
                                  region_override=region)['yield_eq']


# ============================================================

def c1_c2(regions, mp):
    print('\nC1  production path reproduces FAOSTAT     '
          'C2  legacy gap still present')
    ok = True
    worst_new, worst_legacy, worst_legacy_r = 0.0, 0.0, None
    for rk, target in FAOSTAT_TARGETS.items():
        r = regions[rk]
        ym = calibrate_ym_production(rk, target, mp, region=r)
        rel = abs(y_production(rk, r, ym, mp) - target) / target
        worst_new = max(worst_new, rel)

        ym_legacy = calibrate_ym(rk, target, mp)
        gap = abs(100.0 * (y_production(rk, r, ym_legacy, mp) - target) / target)
        if gap > worst_legacy:
            worst_legacy, worst_legacy_r = gap, rk

    ok &= record('C1', 'FAOSTAT reproduced to %g relative' % FAOSTAT_RTOL,
                 worst_new <= FAOSTAT_RTOL,
                 'worst %.3e relative (%.3e percent)' % (worst_new, worst_new * 100))
    # The guard is on the WORST-CASE legacy gap, not the smallest. Individual
    # regions can agree closely by coincidence — Southeast Asia's legacy gap
    # is 0.074% — without the two paths being the same path. What would make
    # this guard moot is the worst case collapsing.
    ok &= record('C2', 'legacy objective still misses by >= %.1f%% somewhere'
                 % LEGACY_GAP_FLOOR_PCT,
                 worst_legacy >= LEGACY_GAP_FLOOR_PCT,
                 'worst legacy gap %.3f percent (%s)' % (worst_legacy, worst_legacy_r))
    return ok


def c3(regions, mp):
    print('\nC3  every field that moves yield_max is fingerprinted')
    ok = True
    numeric = [f.name for f in dataclass_fields(RegionParams)
               if f.name != 'name']
    reached, missed = [], []
    for rk in C3_REGIONS:
        base_region = regions[rk]
        target = FAOSTAT_TARGETS[rk]
        ym0 = calibrate_ym_production(rk, target, mp, region=base_region)
        for fname in numeric:
            r = copy.copy(base_region)
            v = getattr(r, fname)
            if isinstance(v, bool):
                continue
            new = v * 1.10 if abs(v) > 0 else 0.10
            if fname == 'texture_class':
                new = int(v) + 1
            setattr(r, fname, type(v)(new) if not isinstance(v, float) else float(new))
            try:
                ym1 = calibrate_ym_production(rk, target, mp, region=r)
            except Exception as e:                      # a field that breaks the
                reached.append((rk, fname, 'raised %s' % type(e).__name__))
                continue
            moved = abs(ym1 - ym0) / max(abs(ym0), 1e-12) > YM_MOVE_RTOL
            if moved:
                reached.append((rk, fname, '%.6f -> %.6f' % (ym0, ym1)))
                if fname not in YM_REGION_FIELDS:
                    missed.append((rk, fname))

    ok &= record('C3', 'no unregistered field moves yield_max',
                 not missed,
                 'unregistered movers: %s' % (sorted({m[1] for m in missed}) or 'none'))
    declared_inert = sorted(set(YM_REGION_FIELDS) - {r[1] for r in reached})
    record('C3', 'declared-but-inert fields (informational, not a failure)',
           True, ', '.join(declared_inert) or 'none')
    record('C3', 'fields observed to reach the calibration', True,
           ', '.join(sorted({r[1] for r in reached})))
    return ok


def c4(regions, mp):
    print('\nC4  the fingerprint leads with the scheme')
    fp = calibration_fingerprint('north_america', mp, regions['north_america'])
    ok = record('C4', 'fingerprint[0] is CALIBRATION_SCHEME',
                fp[0] == CALIBRATION_SCHEME, repr(fp[0]))
    ok &= record('C4', 'YM_REGION_FIELDS has 13 entries',
                 len(YM_REGION_FIELDS) == 13, str(len(YM_REGION_FIELDS)))
    # a changed region field changes the fingerprint
    r = copy.copy(regions['north_america'])
    r.mitscherlich_c_regional = r.mitscherlich_c_regional * 1.10 or 0.1
    ok &= record('C4', 'perturbing a registered field changes the fingerprint',
                 calibration_fingerprint('north_america', mp, r) != fp)
    return ok


def main():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))
    mp = MonthlyNParams()
    regions = get_default_regions()
    ok = True
    ok &= c1_c2(regions, mp)
    ok &= c3(regions, mp)
    ok &= c4(regions, mp)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'calibration_fingerprint_checks.yaml'), 'w') as f:
        f.write('# Calibration-path checks (F-002)\n')
        f.write('# Generated by code/tests/test_calibration_fingerprint.py\n')
        f.write('calibration_scheme: %s\n' % CALIBRATION_SCHEME)
        f.write('ym_region_fields: %d\n' % len(YM_REGION_FIELDS))
        f.write('owed:\n')
        f.write('  - "AST scan of the calibration path for unhashed region fields"\n')
        f.write('checks:\n')
        for c in checks:
            f.write('  - section: %s\n' % c['section'])
            f.write('    check: "%s"\n' % c['check'].replace('"', "'"))
            f.write('    pass: %s\n' % ('true' if c['pass_'] else 'false'))
            f.write('    detail: "%s"\n' % c['detail'].replace('"', "'"))
        f.write('all_pass: %s\n' % ('true' if ok else 'false'))

    print('\nCALIBRATION FINGERPRINT: %s  (%d checks)'
          % ('PASS' if ok else 'FAIL', len(checks)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
