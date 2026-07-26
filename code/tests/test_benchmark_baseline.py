#!/usr/bin/env python3
"""test_benchmark_baseline.py -- the benchmark suite gates (F-008, F-009).

WHY THIS RE-RUNS THE SUITE

It would be cheaper to read `outputs/benchmarks.csv` and compare that to the
baseline. It would also check nothing: a committed CSV agreeing with a
committed JSON only establishes that neither file has been edited. This test
re-runs `run_benchmarks.run()` against the live model, so what it compares is
what the model says about the field record today.

WHAT IT FAILS ON

F-008 records a failure -- `B3-europe-YR30`, model ratio 0.406 against an
observed 0.75-0.90 -- and that failure is frozen in
`data/benchmarks/baseline_verdicts.json` rather than fixed. A frozen failure is
only honest if every route away from it is closed, so this gate fails on five
distinct movements:

  1. REGRESSION      a row's verdict gets worse than the baseline.
  2. IMPROVEMENT     a row's verdict gets better. This one surprises people.
                     A number that gets better silently is a number nobody
                     read; the baseline must be regenerated deliberately, with
                     an entry in FINDINGS.md saying what moved and why.
  3. DISAPPEARANCE   a baselined row is no longer produced by the suite. The
                     simplest way to make a failure go away is to stop
                     computing it.
  4. NEW FAILURE     a row not in the baseline arrives already failing.
  5. DOWNGRADE       a row's informativeness drops (STRONG -> WEAK -> NONE).
                     This is the other way to make a failure go away: keep
                     computing the row, stop claiming it proves anything.

`test_gate_can_fail` runs the comparison against a synthetic version of each of
the five and asserts each is reported. A gate nobody has watched fail is a gate
nobody knows the polarity of.

Run directly (`python code/tests/test_benchmark_baseline.py`) or under pytest.
Exit 0 clean, 1 on any of the five.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "repro"))
sys.path.insert(0, os.path.join(ROOT, "code", "model"))

BASELINE_PATH = os.path.join(ROOT, "data", "benchmarks",
                             "baseline_verdicts.json")

# Worse to better. A verdict moving down this list is a regression; up is an
# improvement. OWED and NOT_APPLICABLE are not on the scale: they are states
# the suite is in, not judgements about the model, so any movement into or out
# of them is reported as a change rather than ranked.
VERDICT_RANK = {"FAIL": 0, "MARGINAL": 1, "PASS": 2, "INFORMATIVE": 3}
UNRANKED = {"OWED", "NOT_APPLICABLE"}

# More to less. A drop is a downgrade.
INFORMATIVENESS_RANK = {"NONE": 0, "WEAK": 1, "STRONG": 2}


def load_baseline() -> dict:
    with open(BASELINE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def current_verdicts() -> Dict[str, Dict[str, str]]:
    import run_benchmarks

    rows = run_benchmarks.run(os.path.join(ROOT, "outputs"))
    return {r.row_id: {"verdict": r.verdict,
                       "informativeness": r.informativeness} for r in rows}


def compare(baseline: Dict[str, Dict[str, str]],
            current: Dict[str, Dict[str, str]]) -> List[str]:
    """Return one message per defect. Empty list means the gate passes."""
    failures: List[str] = []

    for row_id, base in sorted(baseline.items()):
        if row_id not in current:
            failures.append(
                "DISAPPEARED  %s was baselined %s and the suite no longer "
                "produces it" % (row_id, base["verdict"]))
            continue
        now = current[row_id]
        bv, nv = base["verdict"], now["verdict"]
        if bv != nv:
            if bv in UNRANKED or nv in UNRANKED:
                failures.append(
                    "CHANGED      %s %s -> %s (a state change, not a ranked "
                    "movement; regenerate the baseline deliberately)"
                    % (row_id, bv, nv))
            elif VERDICT_RANK[nv] < VERDICT_RANK[bv]:
                failures.append("REGRESSION   %s %s -> %s" % (row_id, bv, nv))
            else:
                failures.append(
                    "IMPROVEMENT  %s %s -> %s (regenerate the baseline with "
                    "run_benchmarks.py --write-baseline and record it in "
                    "FINDINGS.md)" % (row_id, bv, nv))
        bi, ni = base["informativeness"], now["informativeness"]
        if (INFORMATIVENESS_RANK.get(ni, 0)
                < INFORMATIVENESS_RANK.get(bi, 0)):
            failures.append(
                "DOWNGRADE    %s informativeness %s -> %s; a row that stops "
                "claiming to prove anything is a failure removed by "
                "redefinition" % (row_id, bi, ni))

    for row_id, now in sorted(current.items()):
        if row_id in baseline:
            continue
        if now["verdict"] in ("FAIL", "MARGINAL"):
            failures.append(
                "NEW FAILURE  %s arrived at %s and is not in the baseline"
                % (row_id, now["verdict"]))
        else:
            failures.append(
                "NEW ROW      %s arrived at %s; add it to the baseline "
                "deliberately" % (row_id, now["verdict"]))
    return failures


def test_benchmarks_match_baseline() -> None:
    baseline = load_baseline()["verdicts"]
    failures = compare(baseline, current_verdicts())
    assert not failures, "\n".join(failures)


def test_frozen_failure_is_still_frozen() -> None:
    """The recorded failure must still be recorded, and still be a failure.

    F-008's finding is that the model's temperate nil-N yield ratio runs about
    twice as hard as the Broadbalk record shows. If that row ever silently
    becomes a PASS the paper's temperate loss figures change meaning, so the
    row is named here rather than left to the general comparison.
    """
    baseline = load_baseline()
    assert baseline["verdicts"]["B3-europe-YR30"]["verdict"] == "FAIL", (
        "B3-europe-YR30 is F-008's recorded failure; if it is no longer FAIL "
        "in the baseline, the baseline was regenerated without an entry in "
        "FINDINGS.md")
    assert baseline["tally"]["FAIL"] == 1


def test_gate_can_fail() -> None:
    """Watch the comparison fail on each of the five movements."""
    base = {
        "R-PASS": {"verdict": "PASS", "informativeness": "STRONG"},
        "R-FAIL": {"verdict": "FAIL", "informativeness": "STRONG"},
    }
    cases: List[Tuple[str, Dict[str, Dict[str, str]], str]] = [
        ("regression",
         {"R-PASS": {"verdict": "FAIL", "informativeness": "STRONG"},
          "R-FAIL": base["R-FAIL"]}, "REGRESSION"),
        ("improvement",
         {"R-PASS": base["R-PASS"],
          "R-FAIL": {"verdict": "PASS", "informativeness": "STRONG"}},
         "IMPROVEMENT"),
        ("disappearance", {"R-PASS": base["R-PASS"]}, "DISAPPEARED"),
        ("new failure",
         {**base, "R-NEW": {"verdict": "FAIL", "informativeness": "WEAK"}},
         "NEW FAILURE"),
        ("downgrade",
         {"R-PASS": {"verdict": "PASS", "informativeness": "NONE"},
          "R-FAIL": base["R-FAIL"]}, "DOWNGRADE"),
    ]
    for label, current, expect in cases:
        msgs = compare(base, current)
        assert any(m.startswith(expect) for m in msgs), (
            "the gate did not report %s: %r" % (label, msgs))
    assert not compare(base, dict(base)), (
        "the gate reports a defect on an unchanged suite")


def main() -> int:
    print("test_gate_can_fail ...", end=" ")
    test_gate_can_fail()
    print("ok")
    print("test_frozen_failure_is_still_frozen ...", end=" ")
    test_frozen_failure_is_still_frozen()
    print("ok")
    print("re-running the benchmark suite against the live model ...")
    baseline = load_baseline()
    failures = compare(baseline["verdicts"], current_verdicts())
    if failures:
        for f in failures:
            print("  " + f)
        print("%d movement(s) away from the frozen verdicts" % len(failures))
        return 1
    print("  %d rows, every verdict and informativeness matches the baseline"
          % baseline["n_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
