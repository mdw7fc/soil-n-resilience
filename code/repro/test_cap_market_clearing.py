#!/usr/bin/env python3
"""Market-clearing test for the realized-yield equilibrium (v15, third form).

LINEAGE, BECAUSE THIS FILE HAS BEEN WRONG BEFORE
------------------------------------------------
Form 1 asserted the model's ``clearing_residual`` column stayed below 1e-6.
F-010 found that column was zero by algebra under the capped linear solver,
for any parameter values, correct or not. It tested that subtraction works.

Form 2 re-solved the four LINEAR structural equations externally with brentq
and required the reported price to be that root. That was a real test of the
linear clearing, and the linear clearing is gone: F-024 replaced it with a
price root-found against the realized biogeochemical yield, after F-022/F-023
measured the linearization biasing reported prices by about 1 pp.

WHAT IT DOES NOW
----------------
For EVERY step of every run (not only cap-binding ones), the test takes the
reported columns of the DataFrame and evaluates the structural equations of
the realized clearing, written out here independently of the solver:

    (1) land        L_hat = lambda_L * PY_hat
    (2) fertilizer  F_hat = eps_F_PF*PF_hat + eps_F_PY*PY_hat + eps_F_N*N_hat
                    unless the cap binds, in which case the QUANTITY is set by
                    physical availability: F_level = F0*L0*ceiling / L_level
    (3) clearing    eta*PY_hat = ln(yield_fraction) + alpha*L_hat

Equation (3) uses the REPORTED yield_fraction, which the solver obtained by
running the biophysics, so this is not the solver's own residual read back:
if brentq ever returns a bad root, or the clearing reverts to the linear
supply relation, (3) breaks at the size of the F-022 gap, about 1e-2 log
points, against a 1e-8 tolerance.

WHAT MAKES IT A TEST
--------------------
The old linear supply relation alpha*L_hat + beta*N_hat + gamma*F_hat is also
evaluated at every step, from the recorded diagnostic elasticities, and the
test REQUIRES it to disagree with eta*PY_hat by more than 1e-3 somewhere: a
check that cannot tell the new clearing from the old one would pass either,
and a comparison that cannot fail is not a comparison. The cap must also bind
at least once, or a model with no constrained branch would pass.

Outputs results/cap_market_clearing.txt.
"""
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
    get_supply_constrained_scenarios, get_scenario_params, calibrate_price_shock,
    supply_state,
)
from soil_n_model import get_default_regions  # noqa: E402

DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']

#: Structural tolerance. (1) and (2) are algebraic in reported quantities;
#: (3) is a root found to xtol=1e-12, so 1e-8 leaves headroom over rounding
#: while sitting six orders below the failure size.
STRUCT_TOL = 1e-8
#: The linear supply relation must disagree with the clearing by more than
#: this somewhere, or the test cannot tell the two clearings apart.
DISCRIMINATION_MIN = 1e-3
MIN_BINDING_STEPS = 1


def lambda_L_of(m):
    if abs(m.eps_LS_PL - m.eps_LD_PL) > 1e-10:
        return m.eps_LS_PL * m.eps_LD_PY / (m.eps_LS_PL - m.eps_LD_PL)
    return 0.0


def check_run(m, df):
    """Evaluate the structural equations over one run from reported columns."""
    lam = lambda_L_of(m)
    worst = 0.0
    lin_gap_max = 0.0
    binding = 0
    for i in range(1, len(df)):
        r = df.iloc[i]
        PY, F_hat, L_hat, N_hat = (float(r['PY_hat']), float(r['F_hat']),
                                   float(r['L_hat']), float(r['N_hat']))
        yf = float(r['yield_fraction'])
        sup = supply_state(m.econ, float(r['year']))
        PF_hat = m.PF_hat_base * sup.price_frac

        r1 = L_hat - lam * PY
        if int(r['cap_binding']):
            binding += 1
            F_expect = (m.F_baseline * m.L_baseline * sup.ceiling
                        / max(float(r['land_mha']), 1e-6))
            r2 = float(r['fert_applied_kgha']) - F_expect
        else:
            r2 = F_hat - (m.eps_F_PF * PF_hat + m.eps_F_PY * PY
                          + m.eps_F_N * N_hat)
        r3 = m.eta * PY - (np.log(max(yf, 1e-9)) + m.alpha * L_hat)
        worst = max(worst, abs(r1), abs(r2), abs(r3))

        lin = (m.alpha * L_hat + float(r['beta']) * N_hat
               + float(r['gamma']) * F_hat) - m.eta * PY
        lin_gap_max = max(lin_gap_max, abs(lin))
    return worst, lin_gap_max, binding


def main():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))
    reg = get_default_regions()
    mp = MonthlyNParams()
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(0.20)
    scen = [('S3', s3)] + sorted(get_supply_constrained_scenarios().items())

    lines = []
    worst_all = 0.0
    lin_max = 0.0
    binding_total = 0
    for sname, econ in scen:
        for key in RO:
            m = CoupledMonthlyModel(region=reg[key], econ=econ, region_key=key,
                                    t_max=30.0,
                                    yield_max_override=get_calibrated_ym(key, mp))
            df = m.run()
            w, lg, b = check_run(m, df)
            worst_all = max(worst_all, w)
            lin_max = max(lin_max, lg)
            binding_total += b
            lines.append('%-20s %-18s worst residual %.2e  linear-gap max %.2e  binding %d'
                         % (sname, key, w, lg, b))

    ok = True
    verdicts = []
    if worst_all > STRUCT_TOL:
        ok = False
        verdicts.append('FAIL structural residual %.2e > %.0e' % (worst_all, STRUCT_TOL))
    else:
        verdicts.append('worst structural residual %.2e (tol %.0e)' % (worst_all, STRUCT_TOL))
    if lin_max < DISCRIMINATION_MIN:
        ok = False
        verdicts.append('FAIL the linear supply relation also satisfies the clearing '
                        '(max gap %.2e); this test cannot distinguish the clearings '
                        'and therefore proves nothing' % lin_max)
    else:
        verdicts.append('linear-relation gap reaches %.2e: the check can fail' % lin_max)
    if binding_total < MIN_BINDING_STEPS:
        ok = False
        verdicts.append('FAIL the cap never bound; the constrained branch is unevidenced')
    else:
        verdicts.append('cap-binding steps: %d' % binding_total)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'cap_market_clearing.txt'), 'w') as f:
        f.write('\n'.join(lines + [''] + verdicts) + '\n')
    for v in verdicts:
        print(v)
    print('REALIZED MARKET CLEARING: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
