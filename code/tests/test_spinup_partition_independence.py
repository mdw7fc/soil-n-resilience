#!/usr/bin/env python3
"""Is the reported result independent of the assumed initial SOM partition?

Rebuilt 2026-07-25 (WP3). This file was written during the v15 pass, lost with
the working tree, and rebuilt from the two places that still describe it: the
`mc_exempt_reason` on `som_pool_fractions` in `code/model/params.yaml`, which
cites it as the test that falsified an earlier claim, and finding F-011, which
records it as the test that catches eight of the twelve COVERED registry
leaves. No work package owned it, so `params.yaml` was citing a test that did
not exist and pointing the SI at a characterisation file nothing wrote.

WHAT IT ESTABLISHES

`som_pool_fractions` is exempt from the Monte Carlo ensemble. The reason
recorded before v15 was that the dynamic spin-up overwrites the initial
partition entirely, so the assumption cannot matter. That reason is false, and
this file is what falsifies it:

  A  The FAST POOLS are partition-independent. Active and slow reach
     equilibrium well inside the shipped spin-up and agree across every
     starting partition to within FAST_RTOL, a relative bound derived from
     the spin-up's own stopping rule.

  B  The PASSIVE POOL IS NOT. With k_passive at 0.000728 per year its turnover
     time is ~1374 years, while the shipped convergence criterion (fractional
     SOC drift below 0.002 over a 50-year window) is met far sooner. The
     passive pool therefore still carries the assumed fraction when the
     spin-up stops, and so does total SOC. This is asserted as a POSITIVE
     result -- the spread must EXCEED a threshold. If a future change ever
     makes the passive pool converge, this assertion fails loudly rather than
     silently restoring a claim the registry records as false.

  C  The EQUILIBRIUM ITSELF IS partition-independent. Run to a true fixed
     point (n_spinup 20000, tol 1e-6) every starting partition lands on the
     same SOC. The shipped spin-up simply does not run that far, and should
     not: the model's own equilibrium would replace a measured stock with an
     inferred one.

  D  What actually licenses the exemption is neither of the above but the
     measured sensitivity of the PUBLISHED quantities. Across the partition
     sweep the S3 year-1 and year-10 yield losses must move by less than
     PUBLISHED_TOL in every region, against a reporting precision of 0.1 pp,
     while absolute SOC moves by far more. Absolute SOC is initialization,
     not prediction.

The measured characterisation is written to
`results/spinup_partition_characterisation.yaml` on every run, and the SI
cites it from there rather than restating point values.

WHY IT ALSO PINS THE EQUILIBRIUM

Checks A-D are structural: they would hold for a range of parameter values.
On their own they catch almost nothing, which is how eight registry leaves
came to depend on this one file. Check E therefore pins the spin-up
equilibrium itself against a committed characterisation. Any parameter that
moves the equilibrium -- residue retention, root:shoot, the carbon retention
efficiency, the humification and decay rates, the initial stock, the yield
target -- moves these numbers and fails this test. That is the coverage.

Exit 0 means every check holds.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "model"))

import numpy as np  # noqa: E402

from monthly_model_v3 import (  # noqa: E402
    MonthlyNParams,
    apply_era5_climate_file,
    century_dynamic_spinup,
)
from soil_n_model import SOMPoolParams, get_default_regions  # noqa: E402
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym  # noqa: E402
from coupled_econ_biophysical import get_scenario_params  # noqa: E402
import registry as R  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
CHARACTERISATION = os.path.join(RESULTS, "spinup_partition_characterisation.yaml")
EXPECTED = os.path.join(ROOT, "baseline", "spinup_characterisation_expected.json")

#: f_passive values swept, with f_active held at its registered value. The
#: range brackets the registered 0.58 and is the range the SI limitation
#: quotes.
F_PASSIVE_SWEEP = (0.45, 0.58, 0.73)

#: Active and slow equilibrate inside the shipped spin-up, so they agree
#: across partitions. The bound is RELATIVE to pool size and derived from the
#: spin-up's own stopping rule rather than picked: the shipped criterion is a
#: fractional SOC drift below 0.002 over a 50-year window, and a pool that has
#: equilibrated under that rule can still sit a few multiples of it away from
#: where an infinitely long run would put it. Five times is the allowance. An
#: absolute bound in t C/ha would be meaningless here anyway, since the slow
#: pool ranges from about 4 t C/ha in sub-Saharan Africa to 14 in temperate
#: regions and the same absolute spread means different things in each.
FAST_RTOL = 5 * 0.002
#: The passive pool must NOT equilibrate. t C/ha, across the sweep. This is a
#: floor, not a ceiling: see check B.
PASSIVE_MIN_SPREAD = 0.20
#: A true fixed point run stops on a fractional drift below 1e-6 over a
#: 50-year window. That bounds the drift RATE, not the distance to the fixed
#: point: with a 1374-year passive turnover a run can be arbitrarily slow and
#: still not arrived. Two orders of magnitude above the stopping tolerance.
FIXEDPT_RTOL = 1e-4
#: Published yield losses, percentage points, against 0.1 pp reporting.
PUBLISHED_TOL = 0.05
#: Equilibrium characterisation, relative.
PIN_RTOL = 1e-6

TEMPERATE = ("north_america", "europe", "fsu_central_asia")

failures = []


def check(ok, msg):
    if not ok:
        failures.append(msg)
        print("FAIL  " + msg)
    return ok


def partition(f_passive):
    f_active = float(R.leaf("som_pool_fractions", "f_active"))
    return dict(f_active=f_active, f_slow=1.0 - f_active - f_passive,
                f_passive=f_passive)


def s3_losses(region_key, region, som, mp, ym, pools, t_max=10):
    """S3 yield loss against its own no-shock control, years 1 and 10."""
    out = []
    for shock in (None, 0.0):
        econ = copy.deepcopy(get_scenario_params()["S3"])
        if shock is not None:
            econ.fert_price_shock = shock
        m = CoupledMonthlyModel(copy.deepcopy(region), econ, region_key=region_key,
                                t_max=t_max, yield_max_override=ym,
                                initial_pools=pools, monthly_params=mp,
                                som_params=som)
        f = m.run()
        out.append(f)
    shocked, control = out
    res = {}
    for yr in (1, 10):
        s = float(shocked.loc[shocked.year == yr, "yield_tha"].iloc[0])
        c = float(control.loc[control.year == yr, "yield_tha"].iloc[0])
        res[yr] = 100.0 * (1.0 - s / c)
    return res


def main():
    apply_era5_climate_file(os.path.join(ROOT, "data", "era5_regional_climates.json"))
    mp = MonthlyNParams()
    regions = get_default_regions()
    base_som = SOMPoolParams()

    record = {}
    print("=" * 72)
    print("Spin-up partition sweep: f_passive %s" % (F_PASSIVE_SWEEP,))

    for rk in sorted(regions):
        shipped, fixedpt, losses = {}, {}, {}
        for fp in F_PASSIVE_SWEEP:
            som = dataclasses.replace(base_som, **partition(fp))
            s = century_dynamic_spinup(rk, p=mp, som_params=som)
            t = century_dynamic_spinup(rk, n_spinup=20000, tol=1e-6, p=mp,
                                       som_params=som)
            shipped[fp], fixedpt[fp] = s, t
            ym = get_calibrated_ym(rk, mp)
            # The spin-up result is the initial-pool contract in full: the
            # three pools, the equilibrium SOC the water-stress term measures
            # change against, and the equilibrium mineral N.
            losses[fp] = s3_losses(rk, regions[rk], som, mp, ym, s)

        def spread(d, key):
            v = [float(d[fp][key]) for fp in F_PASSIVE_SWEEP]
            return float(max(v) - min(v))

        def rel(d, key):
            v = [float(d[fp][key]) for fp in F_PASSIVE_SWEEP]
            m = sum(v) / len(v)
            return (max(v) - min(v)) / abs(m) if m else 0.0

        act, act_r = spread(shipped, "c_active"), rel(shipped, "c_active")
        slw, slw_r = spread(shipped, "c_slow"), rel(shipped, "c_slow")
        pas = spread(shipped, "c_passive")
        soc = spread(shipped, "soc")
        fpt, fpt_r = spread(fixedpt, "soc"), rel(fixedpt, "soc")
        y1 = float(max(losses[f][1] for f in F_PASSIVE_SWEEP) - min(losses[f][1] for f in F_PASSIVE_SWEEP))
        y10 = float(max(losses[f][10] for f in F_PASSIVE_SWEEP) - min(losses[f][10] for f in F_PASSIVE_SWEEP))

        print(f"  {rk:20s} active {act_r:.2e}  slow {slw_r:.2e}  passive {pas:6.3f}"
              f"  SOC {soc:6.3f}  fixedpt {fpt_r:.2e}  yr1 {y1:.4f}  yr10 {y10:.4f}")

        # A -- fast pools are partition-independent
        check(act_r <= FAST_RTOL,
              f"{rk}: active pool spread {act_r:.4g} relative > {FAST_RTOL}")
        check(slw_r <= FAST_RTOL,
              f"{rk}: slow pool spread {slw_r:.4g} relative > {FAST_RTOL}")
        # B -- the passive pool is not, and must not be
        check(pas >= PASSIVE_MIN_SPREAD,
              f"{rk}: passive pool spread {pas:.4g} < {PASSIVE_MIN_SPREAD}. The "
              f"shipped spin-up now appears to overwrite the initial partition. "
              f"params.yaml records the opposite as measured fact and exempts "
              f"som_pool_fractions from the ensemble on that basis; if this is "
              f"a real change, that reason has to be rewritten, not this bound.")
        # C -- the equilibrium itself is partition-independent
        check(fpt_r <= FIXEDPT_RTOL,
              f"{rk}: true fixed point SOC spread {fpt_r:.4g} relative > "
              f"{FIXEDPT_RTOL}")
        # D -- published quantities are insensitive
        check(y1 <= PUBLISHED_TOL,
              f"{rk}: S3 year-1 loss moves {y1:.4g} pp across the partition "
              f"sweep, above the {PUBLISHED_TOL} pp that licenses the exemption")
        check(y10 <= PUBLISHED_TOL,
              f"{rk}: S3 year-10 loss moves {y10:.4g} pp across the partition "
              f"sweep, above the {PUBLISHED_TOL} pp that licenses the exemption")

        record[rk] = {
            "shipped": {str(fp): {k: float(shipped[fp][k]) for k in
                                  ("c_active", "c_slow", "c_passive", "soc",
                                   "c_input_eq", "n_min_eq", "yield_eq")}
                        for fp in F_PASSIVE_SWEEP},
            "years_to_converge": {str(fp): int(shipped[fp]["years_to_converge"])
                                  for fp in F_PASSIVE_SWEEP},
            "fixed_point_soc": {str(fp): float(fixedpt[fp]["soc"])
                                for fp in F_PASSIVE_SWEEP},
            "s3_loss_pp": {str(fp): {str(y): float(losses[fp][y]) for y in (1, 10)}
                           for fp in F_PASSIVE_SWEEP},
            "spreads": {"c_active": act, "c_slow": slw, "c_passive": pas,
                        "c_active_rel": act_r, "c_slow_rel": slw_r,
                        "soc": soc, "fixed_point_soc": fpt,
                        "fixed_point_soc_rel": fpt_r,
                        "s3_yr1_pp": y1, "s3_yr10_pp": y10},
        }

    # ------------------------------------------------------------------
    # The SI cites the absolute-SOC sensitivity from this file. Measure it
    # rather than restating it.
    # ------------------------------------------------------------------
    temperate_soc = {rk: record[rk]["spreads"]["soc"] for rk in TEMPERATE
                     if rk in record}
    worst_pub = max(max(record[rk]["spreads"]["s3_yr1_pp"],
                        record[rk]["spreads"]["s3_yr10_pp"]) for rk in record)
    print("=" * 72)
    print("  absolute SOC spread, temperate regions (t C/ha): "
          + ", ".join(f"{k} {v:.2f}" for k, v in temperate_soc.items()))
    print(f"  worst published-quantity spread: {worst_pub:.4f} pp "
          f"(reporting precision 0.1)")
    print("  NOTE: the mc_exempt_reason on som_pool_fractions states that "
          "absolute SOC\n        moves by 'more than 8 t C/ha in every "
          "temperate region'. Measured\n        here it is "
          + ", ".join(f"{v:.2f}" for v in temperate_soc.values())
          + ". That sentence is cited by the SI\n        limitation and needs "
            "restating; the licensing argument itself is\n        unaffected, "
            "since it rests on the published quantities being flat.")

    # ------------------------------------------------------------------
    # E -- pin the equilibrium so parameter mutations are caught
    # ------------------------------------------------------------------
    os.makedirs(RESULTS, exist_ok=True)
    try:
        import yaml
        with open(CHARACTERISATION, "w") as fh:
            yaml.safe_dump(
                {"_comment": "Written by test_spinup_partition_independence.py "
                             "on every run. Cited by the SI limitation on "
                             "som_pool_fractions. Do not edit by hand.",
                 "f_passive_sweep": list(F_PASSIVE_SWEEP),
                 "regions": record}, fh, sort_keys=True, default_flow_style=False)
    except ImportError:
        with open(CHARACTERISATION.replace(".yaml", ".json"), "w") as fh:
            json.dump(record, fh, indent=1, sort_keys=True)

    if not os.path.exists(EXPECTED):
        print(f"\n  no pinned characterisation at {EXPECTED}; writing it")
        os.makedirs(os.path.dirname(EXPECTED), exist_ok=True)
        json.dump({"_comment": "Pinned spin-up equilibrium. Regenerate only "
                               "when a parameter change is intended.",
                   "regions": record}, open(EXPECTED, "w"), indent=1,
                  sort_keys=True)
    else:
        exp = json.load(open(EXPECTED))["regions"]
        n = 0
        for rk in sorted(record):
            if rk not in exp:
                check(False, f"{rk}: absent from the pinned characterisation")
                continue
            for fp in F_PASSIVE_SWEEP:
                for k, got in record[rk]["shipped"][str(fp)].items():
                    want = exp[rk]["shipped"][str(fp)][k]
                    n += 1
                    if not np.isclose(got, want, rtol=PIN_RTOL, atol=0.0):
                        check(False, f"{rk} f_passive={fp} {k}: {got!r} != "
                                     f"pinned {want!r}")
        print(f"  {n} pinned equilibrium fields compared")

    print("=" * 72)
    if failures:
        print(f"SPIN-UP PARTITION INDEPENDENCE FAILED ({len(failures)})")
        for f in failures[:10]:
            print("  " + f)
        return 1
    print("SPIN-UP PARTITION INDEPENDENCE PASSED")
    print("  fast pools partition-independent; passive pool is not and must "
          "not be;\n  equilibrium partition-independent; published quantities "
          "flat to "
          f"{worst_pub:.3f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
