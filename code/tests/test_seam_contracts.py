#!/usr/bin/env python3
"""Seam D — aggregation-basis contract tests (finding F-005).

D1  Construction contract. Every negative control fires. A weight vector
    cannot be obtained without passing validation.
D2  External-validity probe. An independent hand implementation of the two
    bases, on a case constructed to separate them, with a guard that fails if
    the case ever stops separating them. A probe whose case has gone
    degenerate agrees with anything.
D3  Deposit consistency. The frozen headline is recomputed from its own
    per-region rows through the seam and must agree at the deposit's 2 dp.

Also asserts the behaviour-invariance claim the refactor rests on:
``calibrate_price_shock(0.20)`` returns 1.0389792148114703, unchanged.

Writes results/seam_contract_checks.yaml.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import numpy as np
from seams import (SeamD_AggregationWeights, SeamContractError,
                   outcome_weights, intensity_weights, nitrogen_weights,
                   assert_same_basis, basis_for_column,
                   BASIS_PRODUCTION_TONNAGE, BASIS_CROPLAND_AREA)
from soil_n_model import get_default_regions
from coupled_econ_biophysical import calibrate_price_shock

DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')

#: The published calibrated shock. Verified unchanged by the F-005 refactor.
PUBLISHED_PRICE_SHOCK = 1.0389792148114703

checks = []


def record(section, name, ok, detail=''):
    checks.append(dict(section=section, check=name, pass_=bool(ok), detail=str(detail)))
    print('  [%s] %-52s %s' % ('PASS' if ok else 'FAIL', name, detail))
    return ok


def fires(section, name, fn, detail=''):
    """A negative control passes when it raises SeamContractError."""
    try:
        fn()
    except SeamContractError as e:
        return record(section, name, True, str(e)[:70])
    return record(section, name, False, 'DID NOT RAISE ' + detail)


# ============================================================
# D1 — CONSTRUCTION CONTRACT
# ============================================================

def d1(regions):
    print('\nD1  construction contract — negative controls')
    keys = tuple(regions.keys())
    ok = True

    W_area = intensity_weights(keys, regions)
    W_n = nitrogen_weights(keys, regions)

    # 1. two different bases meeting in assert_same_basis
    ok &= fires('D1', 'different bases refused',
                lambda: assert_same_basis(W_area, W_n))
    # 2. no provenance
    ok &= fires('D1', 'weight vector with blank provenance refused',
                lambda: SeamD_AggregationWeights(
                    regions=keys, raw=tuple(1.0 for _ in keys),
                    weights=tuple(1.0 / len(keys) for _ in keys),
                    basis=BASIS_CROPLAND_AREA, provenance='   '))
    # 3. weights summing to 0.9
    ok &= fires('D1', 'weights summing to 0.9 refused',
                lambda: SeamD_AggregationWeights(
                    regions=keys, raw=tuple(1.0 for _ in keys),
                    weights=tuple(0.9 / len(keys) for _ in keys),
                    basis=BASIS_CROPLAND_AREA, provenance='synthetic'))
    # 4. a region silently dropped
    ok &= fires('D1', 'silently dropped region refused',
                lambda: intensity_weights(keys[:-1], regions))
    # 5. a region with zero weight
    zero = dict(regions)
    import copy
    z = copy.copy(regions[keys[0]])
    z.cropland_mha = 0.0
    zero[keys[0]] = z
    ok &= fires('D1', 'zero-weight region refused',
                lambda: intensity_weights(keys, zero))
    # 6. the vacuous assertion that was written and deleted during F-005
    ok &= fires('D1', 'same object passed N times refused (vacuous assertion)',
                lambda: assert_same_basis(*[W_area for _ in range(3)]))
    # 7. an undeclared column has no basis
    ok &= fires('D1', 'undeclared column has no basis',
                lambda: basis_for_column('n_leached'))

    # positive control: same basis, two independently constructed vectors
    W_area_2 = intensity_weights(keys, regions)
    ok &= record('D1', 'two independent same-basis vectors accepted',
                 assert_same_basis(W_area, W_area_2) == BASIS_CROPLAND_AREA)
    return ok


# ============================================================
# D2 — EXTERNAL VALIDITY PROBE
# ============================================================

def d2(regions):
    """Independent hand implementation on a case built to separate the bases."""
    print('\nD2  external-validity probe')
    keys = tuple(regions.keys())
    # A per-region quantity that differs between the two bases: give the
    # low-yield regions the large losses.
    y_base = [float(regions[k].soc_initial) / 20.0 for k in keys]
    x = [10.0 if y < np.median(y_base) else 1.0 for y in y_base]

    W_out = outcome_weights(keys, y_base, regions)
    W_int = intensity_weights(keys, regions)

    # Hand implementation, written from the definitions and not from seams.py
    num_p = sum(regions[k].cropland_mha * y * xi
                for k, y, xi in zip(keys, y_base, x))
    den_p = sum(regions[k].cropland_mha * y for k, y in zip(keys, y_base))
    hand_prod = num_p / den_p
    num_a = sum(regions[k].cropland_mha * xi for k, xi in zip(keys, x))
    den_a = sum(regions[k].cropland_mha for k in keys)
    hand_area = num_a / den_a

    seam_prod = float(np.dot(W_out.as_array(), x))
    seam_area = float(np.dot(W_int.as_array(), x))

    ok = record('D2', 'production basis matches hand implementation',
                abs(seam_prod - hand_prod) < 1e-12, '%.12f' % seam_prod)
    ok &= record('D2', 'area basis matches hand implementation',
                 abs(seam_area - hand_area) < 1e-12, '%.12f' % seam_area)
    # Guard: if the constructed case ever stops separating the two bases, this
    # probe is agreeing with everything and testing nothing.
    ok &= record('D2', 'GUARD: the case still separates the two bases',
                 abs(seam_prod - seam_area) > 1e-3,
                 'separation %.4f' % abs(seam_prod - seam_area))
    return ok


# ============================================================
# D3 — DEPOSIT CONSISTENCY
# ============================================================

def d3(regions):
    print('\nD3  deposited headline recomputed from its own rows')
    path = os.path.join(DATA, 'canonical_ERA5_y30.json')
    if not os.path.exists(path):
        return record('D3', 'canonical_ERA5_y30.json present', False, path)
    doc = json.load(open(path))
    rows = doc['regions']
    keys = [r['region'] for r in rows]
    W = outcome_weights(keys, [r['y_base'] for r in rows], regions,
                        universe=keys).as_array()
    ok = True
    for y in (1, 10, 30):
        recomputed = float((np.array([r['loss_yr%d' % y] for r in rows]) * W).sum())
        deposited = float(doc['global_prodweighted'][str(y)])
        gap = abs(recomputed - deposited)
        ok &= record('D3', 'year %-2d headline agrees at 2 dp' % y,
                     round(recomputed, 2) == round(deposited, 2),
                     'recomputed %.4f vs deposited %.2f (gap %.4f pp)'
                     % (recomputed, deposited, gap))
    return ok


# ============================================================

def invariance():
    print('\nINVARIANCE  the refactor changed no published number')
    v = calibrate_price_shock(0.20)
    return record('INV', 'calibrate_price_shock(0.20) unchanged',
                  v == PUBLISHED_PRICE_SHOCK, repr(v))


def main():
    regions = get_default_regions()
    ok = True
    ok &= d1(regions)
    ok &= d2(regions)
    ok &= d3(regions)
    ok &= invariance()

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'seam_contract_checks.yaml'), 'w') as f:
        f.write('# Seam D aggregation-basis contract checks (F-005)\n')
        f.write('# Generated by code/tests/test_seam_contracts.py\n')
        f.write('published_price_shock: %r\n' % PUBLISHED_PRICE_SHOCK)
        f.write('checks:\n')
        for c in checks:
            f.write('  - section: %s\n' % c['section'])
            f.write('    check: "%s"\n' % c['check'].replace('"', "'"))
            f.write('    pass: %s\n' % ('true' if c['pass_'] else 'false'))
            f.write('    detail: "%s"\n' % c['detail'].replace('"', "'").replace('\n', ' '))
        f.write('all_pass: %s\n' % ('true' if ok else 'false'))

    print('\nSEAM CONTRACTS: %s  (%d checks)'
          % ('PASS' if ok else 'FAIL', len(checks)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
