#!/usr/bin/env python3
"""The disruption timeline is defined once and both models read that one place.

Four properties, one of them the standing requirement that the comparison be
able to fail.

Why this file exists. Until v15 the ceiling ramp and the price relaxation were
written out four times: once each in the annual and the monthly coupled model,
and within each of those, once for the ceiling and once for the price. The four
copies agreed. That is why nobody removed them, and it is not a reason to keep
them, because the monthly and annual models are coupled at different
resolutions and the interface between them is exactly where a drift would land
and be hard to see. supply_state() is now the only definition and this file is
what holds it to that.

The pulse shape is the substantive property. A one-year disruption expressed as
a one-year recovery ramp decays linearly through the year it is supposed to be
at full strength, so it would understate year 1 by roughly half while looking
entirely reasonable in a plot. The test pins the square shape directly.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))

from coupled_econ_biophysical import (  # noqa: E402
    EconParams, SupplyState, supply_state, get_pulse_scenario,
    get_supply_constrained_scenarios,
)

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def log(msg):
    sys.stdout.write(msg + "\n")


def test_pulse_is_square():
    """Full strength until the duration, nothing after. Not a ramp."""
    e = get_pulse_scenario()
    for t in (0.0, 0.25, 0.5, 0.999, 1.0):
        s = supply_state(e, t)
        check(s.price_frac == 1.0,
              "pulse price_frac at t=%r is %r, expected 1.0; a value between 0 "
              "and 1 inside the disruption year is the recovery-ramp shape the "
              "pulse is defined not to have" % (t, s.price_frac))
    for t in (1.0001, 1.5, 5.0, 30.0):
        s = supply_state(e, t)
        check(s.price_frac == 0.0,
              "pulse price_frac at t=%r is %r, expected 0.0" % (t, s.price_frac))
        check(s.ceiling == 1.0,
              "pulse ceiling at t=%r is %r, expected 1.0" % (t, s.ceiling))
    log("  [1] PULSE1 holds full strength through t = 1 inclusive and is gone past it")


def test_recovery_ramp_is_unchanged():
    """SC2's ramp is the pre-refactor arithmetic, evaluated by hand here.

    Deliberately recomputed from the definition rather than read back from
    supply_state, so that the test is a statement about what the ramp should be
    and not a restatement of what it is.
    """
    sc2 = get_supply_constrained_scenarios()['SC2_20pct_recovery']
    for t in (0.0, 1.0, 5.0, 10.0, 19.0, 20.0, 25.0):
        want_frac = min(1.0, t / 20.0) if t > 0 else 0.0
        want_ceiling = 0.80 + 0.20 * want_frac
        want_price = 1.0 - want_frac if t > 0 else 1.0
        s = supply_state(sc2, t)
        check(abs(s.ceiling - want_ceiling) < 1e-12,
              "SC2 ceiling at t=%r is %r, expected %r" % (t, s.ceiling, want_ceiling))
        check(abs(s.price_frac - want_price) < 1e-12,
              "SC2 price_frac at t=%r is %r, expected %r" % (t, s.price_frac, want_price))

    sc1 = get_supply_constrained_scenarios()['SC1_20pct']
    for t in (0.0, 1.0, 30.0):
        s = supply_state(sc1, t)
        check(s.ceiling == 0.80,
              "SC1 has no recovery; ceiling at t=%r is %r, expected 0.80" % (t, s.ceiling))
        check(s.price_frac == 1.0,
              "SC1 has no recovery; price_frac at t=%r is %r, expected 1.0" % (t, s.price_frac))
    log("  [2] SC1 and SC2 reproduce the pre-refactor ceiling and price paths")


def test_contradictory_disruption_raises():
    """A pulse that also recovers gradually is two scenarios; refuse both."""
    bad = EconParams(fert_supply_ceiling=0.80,
                     fert_capacity_recovery_years=20.0,
                     fert_disruption_years=1.0)
    try:
        supply_state(bad, 0.5)
    except ValueError:
        pass
    else:
        FAILURES.append("supply_state accepted fert_disruption_years=1 together "
                        "with fert_capacity_recovery_years=20 instead of raising")

    for bad_state in ((1.5, 1.0), (-0.1, 1.0), (0.8, 1.4), (0.8, -0.2)):
        try:
            SupplyState(ceiling=bad_state[0], price_frac=bad_state[1])
        except ValueError:
            continue
        FAILURES.append("SupplyState accepted out-of-range values %r; a contract "
                        "that can hold an impossible value is a comment"
                        % (bad_state,))
    log("  [3] a contradictory scenario and an out-of-range state both raise")


def test_the_comparison_can_fail():
    """The standing requirement: watch each assertion shape fail once."""
    seen = len(FAILURES)

    # The square-pulse assertion, applied to a ramp. If a one-year pulse were
    # implemented as a one-year recovery ramp this is the value it would give
    # at mid-year, and the check above must reject it.
    ramp = EconParams(fert_supply_ceiling=0.80, fert_capacity_recovery_years=1.0)
    mid = supply_state(ramp, 0.5)
    check(mid.price_frac == 1.0,
          "deliberate: a one-year recovery ramp gives price_frac %r at mid-year, "
          "not 1.0, which is the understatement the square pulse avoids"
          % (mid.price_frac,))

    # The range contract, deliberately violated.
    try:
        SupplyState(ceiling=2.0, price_frac=1.0)
        check(False, "deliberate: SupplyState should have rejected ceiling=2.0")
    except ValueError:
        FAILURES.append("deliberate: SupplyState rejected ceiling=2.0 as required")

    got = len(FAILURES) - seen
    if got != 2:
        log("  [4] FAIL: expected 2 deliberate failures, saw %d" % got)
        del FAILURES[seen:]
        FAILURES.append("the deliberate-failure probe did not fire; the "
                        "assertions in this file may not be able to fail")
        return
    del FAILURES[seen:]
    log("  [4] both assertion shapes were watched failing before being trusted")


def main():
    log("SUPPLY STATE (the disruption timeline)")
    test_pulse_is_square()
    test_recovery_ramp_is_unchanged()
    test_contradictory_disruption_raises()
    test_the_comparison_can_fail()
    if FAILURES:
        log("SUPPLY STATE: FAIL")
        for f in FAILURES:
            log("  " + f)
        return 1
    log("SUPPLY STATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
