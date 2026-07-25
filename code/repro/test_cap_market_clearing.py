#!/usr/bin/env python3
"""Constrained-equilibrium market-clearing test — external re-solve (v15).

WHAT THIS REPLACED, AND WHY (finding F-010, 2026-07-25)
-------------------------------------------------------
This file was the evidence offered for the v1.3 constrained-cap fix. It
asserted that the model's ``clearing_residual`` column stayed below 1e-6.
That column is

    gamma * (F_hat - [ln(c) - lambda_L * PY_hat])

and the capped solver sets ``F_hat = ln(c) - L_hat`` with
``L_hat = lambda_L * PY_hat``. The two expressions are the same expression.
The residual is zero by algebra for every possible value of alpha, beta,
gamma, eta and lambda_L, correct or not, so the test could not fail and never
tested the equilibrium. It tested that subtraction works.

WHAT IT DOES NOW
----------------
For each cap-binding step the test takes the model's *reported* PY_hat,
F_hat, L_hat, N_hat and ln_cap, plus the *lagged* elasticities the solver
actually used (beta and gamma from step i-1, which is what the solver reads),
and evaluates the four structural equations independently of the solver:

    (1) land         L_hat  = lambda_L * PY_hat
    (2) fertilizer   F_hat  = ln(c) - L_hat          (quantity rationed)
    (3) supply       Y_hat  = alpha*L_hat + beta*N_hat + gamma*F_hat
    (4) clearing     Y_hat  = eta * PY_hat

It then root-finds the food price with ``scipy.optimize.brentq`` on the
excess-supply function built from those same four equations, bracketing
outward until the sign changes rather than assuming a bracket, and requires
the model's reported price to be that root to 1e-10.

WHAT MAKES IT A TEST
--------------------
Dropping the gamma term from the capped denominator, so that

    PY_hat = (beta*N_hat + gamma*ln(c)) / (eta - alpha*lambda_L)

instead of ``eta - (alpha - gamma)*lambda_L``, drives the structural residual
to 3.0e-03 and the root gap to 6.7e-03, and this test returns 1. The old
residual stays at zero under exactly that mutation, because the mutation
changes PY_hat and the identity is defined in terms of whatever PY_hat comes
out. Run the mutation with ``--mutate drop-gamma``.

The test also fails if the cap never binds: a run in which the constrained
branch is never entered would otherwise pass on a model that has no
constrained branch at all.

Outputs
-------
results/cap_market_clearing.txt — both scenarios, per-region losses, and the
worst structural residual and root gap.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import numpy as np
from scipy.optimize import brentq
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_supply_constrained_scenarios
from soil_n_model import get_default_regions
from seams import outcome_weights

DATA = os.path.join(HERE, '..', '..', 'data')
RESULTS = os.path.join(HERE, '..', '..', 'results')
RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia']

#: Structural residual tolerance. The four equations are evaluated in double
#: precision from reported quantities, so anything above rounding noise is a
#: real disagreement between the solver and the system it claims to solve.
STRUCT_TOL = 1e-12
#: Agreement required between the reported price and the independently
#: root-found price.
ROOT_TOL = 1e-10
#: A run with no binding cap cannot evidence a constrained solver.
MIN_BINDING_STEPS = 1


def patch_era5():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))


# ============================================================
# THE FOUR STRUCTURAL EQUATIONS, WRITTEN OUT INDEPENDENTLY
# ============================================================

def lambda_L_of(eps_LS_PL, eps_LD_PL, eps_LD_PY):
    """Land-market reduction coefficient. Recomputed here, not imported."""
    if abs(eps_LS_PL - eps_LD_PL) > 1e-10:
        return eps_LS_PL * eps_LD_PY / (eps_LS_PL - eps_LD_PL)
    return 0.0


def structural_residuals(PY_hat, F_hat, L_hat, N_hat, ln_c, p):
    """Residuals of the four capped-equilibrium equations at a reported point.

    ``p`` carries alpha, beta, gamma, eta and lambda_L for the step. Returns
    (land, fertilizer, market) residuals; the supply and clearing equations
    are combined into the single market residual Y_supply - eta*PY_hat,
    because the model never reports Y_hat itself.
    """
    lam = p['lambda_L']
    r_land = L_hat - lam * PY_hat
    r_fert = F_hat - (ln_c - L_hat)
    Y_supply = p['alpha'] * L_hat + p['beta'] * N_hat + p['gamma'] * F_hat
    r_market = Y_supply - p['eta'] * PY_hat
    return r_land, r_fert, r_market


def excess_supply(PY, N_hat, ln_c, p):
    """Excess supply as a function of the food price alone.

    Substitutes (1) and (2) into (3) and subtracts (4). The equilibrium price
    is the root. Written from the four equations above rather than from the
    solver's closed form, so a wrong closed form does not cancel out of this
    check.
    """
    lam = p['lambda_L']
    L_hat = lam * PY
    F_hat = ln_c - L_hat
    Y_supply = p['alpha'] * L_hat + p['beta'] * N_hat + p['gamma'] * F_hat
    return Y_supply - p['eta'] * PY


def solve_price(N_hat, ln_c, p):
    """Root-find the equilibrium food price, bracketing outward."""
    f = lambda PY: excess_supply(PY, N_hat, ln_c, p)
    lo, hi = -1.0, 1.0
    f_lo, f_hi = f(lo), f(hi)
    tries = 0
    while f_lo * f_hi > 0.0 and tries < 60:
        lo *= 2.0
        hi *= 2.0
        f_lo, f_hi = f(lo), f(hi)
        tries += 1
    if f_lo * f_hi > 0.0:
        raise AssertionError(
            'excess supply does not change sign on [%g, %g] after %d '
            'expansions; the capped system has no equilibrium price at '
            'N_hat=%g ln_c=%g' % (lo, hi, tries, N_hat, ln_c))
    return brentq(f, lo, hi, xtol=1e-15, rtol=8.9e-16, maxiter=200)


# ============================================================
# MUTATION
# ============================================================

def apply_mutation(name):
    """Install a deliberately wrong capped solver, to prove the test bites.

    'drop-gamma': denominator ``eta - alpha*lambda_L`` instead of
    ``eta - (alpha - gamma)*lambda_L``. This is the algebra error the rewrite
    exists to catch; the old identity-based assertion does not see it.
    """
    if name is None:
        return
    if name != 'drop-gamma':
        raise SystemExit('unknown mutation %r' % name)

    def _capped_dropped_gamma(self, beta, gamma, ln_c):
        lam = self._lambda_L()
        num = beta * self.N_hat + gamma * ln_c
        den = self.eta - self.alpha * lam          # MUTATION: gamma dropped
        PY_hat = num / den if abs(den) > 1e-10 else 0.0
        L_hat = lam * PY_hat
        F_hat = ln_c - L_hat
        return PY_hat, F_hat, L_hat

    CoupledMonthlyModel._solve_equilibrium_capped = _capped_dropped_gamma


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mutate', default=None,
                    help="install a wrong solver ('drop-gamma') and expect failure")
    args = ap.parse_args(argv)
    apply_mutation(args.mutate)

    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()
    scen = get_supply_constrained_scenarios()

    worst_struct = 0.0
    worst_root = 0.0
    worst_where = None
    binding_steps = 0
    lines = []

    for name, econ in [('SC1', scen['SC1_20pct']), ('SC2', scen['SC2_20pct_recovery'])]:
        per, yb = {}, []
        for rk in RO:
            m = CoupledMonthlyModel(region=regions[rk], econ=econ, region_key=rk,
                                    t_max=30.0, yield_max_override=get_calibrated_ym(rk, mp))
            df = m.run()
            per[rk] = {y: float((1 - df[df['year'] == y]['yield_fraction'].iloc[0]) * 100)
                       for y in (1, 10, 30)}
            yb.append(float(df[df['year'] == 0]['yield_tha'].iloc[0]))

            lam = lambda_L_of(m.eps_LS_PL, m.eps_LD_PL, m.eps_LD_PY)
            bind = list(df.index[df['cap_binding'].astype(bool)])
            for i in bind:
                # The solver reads the PREVIOUS step's elasticities. Use the
                # same lag here; step i's would be a different system from the
                # one that was actually solved.
                p = {'alpha': m.alpha, 'eta': m.eta, 'lambda_L': lam,
                     'beta': float(df['beta'].iloc[i - 1]),
                     'gamma': float(df['gamma'].iloc[i - 1])}
                PY_hat = float(df['PY_hat'].iloc[i])
                F_hat = float(df['F_hat'].iloc[i])
                L_hat = float(df['L_hat'].iloc[i])
                N_hat = float(df['N_hat'].iloc[i])
                ln_c = float(df['ln_cap'].iloc[i])
                if not np.isfinite(ln_c):
                    raise AssertionError(
                        'step %d of %s/%s is flagged cap_binding but reports no '
                        'ln_cap' % (i, name, rk))

                res = structural_residuals(PY_hat, F_hat, L_hat, N_hat, ln_c, p)
                s = max(abs(r) for r in res)
                root = solve_price(N_hat, ln_c, p)
                g = abs(PY_hat - root)
                binding_steps += 1
                if max(s, g) > max(worst_struct, worst_root):
                    worst_where = '%s/%s year %d' % (name, rk, int(df['year'].iloc[i]))
                worst_struct = max(worst_struct, s)
                worst_root = max(worst_root, g)

        W = outcome_weights(RO, yb, regions).as_array()
        g = {y: float(sum(per[k][y] * W[j] for j, k in enumerate(RO))) for y in (1, 10, 30)}
        lines.append("%s production-weighted global loss  yr1/yr10/yr30 = %.2f / %.2f / %.2f %%"
                     % (name, g[1], g[10], g[30]))
        for j, k in enumerate(RO):
            lines.append("    %-22s %6.2f %6.2f %6.2f" % (k, per[k][1], per[k][10], per[k][30]))

    lines.append("")
    lines.append("cap-binding steps checked            : %d" % binding_steps)
    lines.append("worst structural residual (log-pts)  : %.2e" % worst_struct)
    lines.append("worst root gap on the food price     : %.2e" % worst_root)
    lines.append("worst step                           : %s" % (worst_where or 'n/a'))
    if args.mutate:
        lines.append("MUTATION APPLIED                     : %s" % args.mutate)

    ok = (binding_steps >= MIN_BINDING_STEPS
          and worst_struct < STRUCT_TOL
          and worst_root < ROOT_TOL)
    if binding_steps < MIN_BINDING_STEPS:
        lines.append("FAIL: the cap never bound; this run evidences nothing about "
                     "the constrained solver.")
    lines.append("CONSTRAINED MARKET CLEARING: %s" % ("PASS" if ok else "FAIL"))

    text = "\n".join(lines) + "\n"
    print(text, end='')
    if not args.mutate:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, 'cap_market_clearing.txt'), 'w') as f:
            f.write(text)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
