"""The SOC deposit is the same run as the canonical deposit, and it is monotone.

`make_soc_trajectories.py` runs the S3 scenario a second time, in a second
file, to deposit the carbon trajectory that `run_canonical.py` never wrote. A
quantity computed in two places is a quantity that can drift apart, and this
repository has an entry in FINDINGS.md for every time it has. So the
duplication is not trusted, it is checked.

Four properties:

1. **The two runs are the same run.** Both scripts deposit each region's
   year-1, year-10 and year-30 yield loss. If the configurations ever diverge,
   in the scenario, the calibration target, the climate, or the yield maxima,
   the loss columns diverge first and this fails before any SOC number is
   quoted from a run that is no longer canonical.

2. **The series matches its own summary.** The per-region summary rows are
   derived from the series, and a summary that has been regenerated while the
   series has not is a table nobody can trace.

3. **Carbon falls, and it falls everywhere.** The whole argument is that a
   fertilizer disruption draws down soil organic matter. A region whose SOC
   rose under sustained disruption would be a sign change, not a small error,
   and it should stop the build rather than appear in a figure.

4. **The comparison can fail.** A tampered loss value must register as a
   difference, or property 1 proves nothing.

Run: python3 code/tests/test_soc_trajectories.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

SOC = os.path.join(ROOT, 'data', 'soc_trajectories.json')
CANON = os.path.join(ROOT, 'data', 'canonical_ERA5_y30.json')

#: Yield losses are deposited to 4 and 2 decimals respectively, so agreement is
#: asserted at the coarser of the two. A difference above this is a
#: configuration difference, not a rounding one.
LOSS_TOL = 0.005


def _soc():
    with open(SOC) as fh:
        return json.load(fh)


def _canon():
    with open(CANON) as fh:
        return json.load(fh)


def test_the_soc_run_is_the_canonical_run():
    soc = {r['region']: r for r in _soc()['regions']}
    can = {r['region']: r for r in _canon()['regions']}
    assert set(soc) == set(can), (
        f'the two deposits cover different regions: '
        f'only-soc={sorted(set(soc) - set(can))}, '
        f'only-canonical={sorted(set(can) - set(soc))}')
    diffs = []
    for rk in sorted(soc):
        for col in ('loss_yr1', 'loss_yr10', 'loss_yr30'):
            a, b = float(soc[rk][col]), float(can[rk][col])
            if abs(a - b) > LOSS_TOL:
                diffs.append(f'{rk}.{col}: soc {a} vs canonical {b}')
    assert not diffs, (
        'make_soc_trajectories.py and run_canonical.py are no longer running '
        'the same scenario, so the SOC series does not belong to the yield '
        'losses the paper reports beside it:\n  ' + '\n  '.join(diffs))
    print(f'  [1] {len(soc)} regions: both deposits agree on every yield loss')


def test_the_summary_follows_from_the_series():
    doc = _soc()
    h = doc['horizon_years']
    bad = []
    for r in doc['regions']:
        s = doc['series'][r['region']]
        assert len(s) == h + 1, (
            f"{r['region']}: series has {len(s)} points, horizon is {h} years")
        for col, want in (('soc_yr0_tha', s[0]),
                          ('soc_yr30_tha', s[h]),
                          ('soc_decline_pct_yr30',
                           100.0 * (s[0] - s[h]) / s[0]),
                          ('soc_decline_pct_yr10',
                           100.0 * (s[0] - s[10]) / s[0])):
            if abs(float(r[col]) - want) > 5e-4:
                bad.append(f"{r['region']}.{col}: {r[col]} vs {want:.4f}")
    assert not bad, (
        'the summary does not follow from the deposited series; rerun '
        'code/repro/make_soc_trajectories.py:\n  ' + '\n  '.join(bad))
    print(f'  [2] every summary field recomputes from the series')


def test_carbon_declines_in_every_region():
    doc = _soc()
    rose = [r['region'] for r in doc['regions']
            if float(r['soc_decline_pct_yr30']) <= 0]
    assert not rose, (
        f'SOC does not decline under 30 years of sustained fertilizer '
        f'disruption in {rose}. That is a sign change in the mechanism the '
        f'paper is about, not a tolerance question.')
    nonmono = []
    for r in doc['regions']:
        s = doc['series'][r['region']]
        if any(s[i + 1] > s[i] + 1e-9 for i in range(len(s) - 1)):
            nonmono.append(r['region'])
    assert not nonmono, (
        f'SOC recovers at some point during sustained disruption in '
        f'{nonmono}. The shock never lifts in S3, so a rising year is a '
        f'model artifact and the trajectory should not be published until it '
        f'is explained.')
    worst = max(doc['regions'], key=lambda r: float(r['soc_decline_pct_yr30']))
    print(f"  [3] SOC falls monotonically in all "
          f"{len(doc['regions'])} regions; largest 30-yr decline "
          f"{worst['region']} {worst['soc_decline_pct_yr30']}%")


def test_the_comparison_can_fail():
    """A check that cannot fail ratifies rather than tests."""
    soc = {r['region']: dict(r) for r in _soc()['regions']}
    can = {r['region']: r for r in _canon()['regions']}
    rk = sorted(soc)[0]
    soc[rk]['loss_yr30'] = float(soc[rk]['loss_yr30']) + 10 * LOSS_TOL
    assert abs(soc[rk]['loss_yr30'] - float(can[rk]['loss_yr30'])) > LOSS_TOL, (
        'a tampered yield loss did not exceed the tolerance, so '
        'test_the_soc_run_is_the_canonical_run proves nothing')
    doc = _soc()
    s = list(doc['series'][rk])
    s[5] = s[4] + 1.0
    assert any(s[i + 1] > s[i] + 1e-9 for i in range(len(s) - 1)), (
        'a tampered series did not register as non-monotone')
    print('  [4] a tampered loss and a tampered series are both detected')


if __name__ == '__main__':
    print('SOC TRAJECTORIES')
    fails = 0
    for name in sorted(n for n in dir() if n.startswith('test_')):
        try:
            globals()[name]()
        except AssertionError as exc:
            fails += 1
            print(f'  FAIL {name}: {exc}')
    print('SOC TRAJECTORIES: ' + ('PASS' if not fails else f'FAIL ({fails})'))
    sys.exit(1 if fails else 0)
