#!/usr/bin/env python3
"""The claim gate — every published number, scored against a live artifact.

Spec: FINDINGS.md F-012, F-013, F-015, F-016. Rebuilt by WP5 on 2026-07-26
after the v15 working tree was lost.

WHAT THIS TEST IS FOR
---------------------
Nothing else in this repository compares the model to the paper. The unit tests
compare the model to itself; the benchmark suite compares it to the field
record. This compares it to what has been written down, sentence by sentence,
and fails the build when a published number and the model disagree beyond the
precision the sentence itself states.

SIX GATES
---------
G1  Structure. Ids are unique, the header's declared counts are true, and every
    claim carries a location, a status and at least one check or an explicit
    reason for having none.
G2  Resolution. Every check resolves to a number. A check that cannot find its
    artifact is not an agreeing check; F-012 recorded "zero unresolved paths"
    as a result in its own right.
G3  Verdicts against `docs/claims_baseline.json`. The drifted set may only
    shrink. A check that comes into line without the baseline being regenerated
    ALSO fails, on the same principle as the benchmark and unstamped baselines:
    a number that gets better silently is a number nobody read.
G4  Owed generators. `owed_count` may only shrink. A claim recorded as having
    no script behind it is a debt, and the debt may not grow.
G5  The two-way index. `depends_on_params` here is the reverse of
    `affects_claims` in params.yaml, and BOTH directions are checked. On F-012's
    first run this failed on C-061, which named eps_F_PF and som_decay_rates
    while neither parameter named the claim back. A register that only pointed
    one way would let a parameter change without anyone knowing which published
    sentences moved.
G6  Tolerance provenance. No check may carry a tolerance wider than the
    precision of its own stated value. This is the mechanical form of the
    standing rule: do not widen a tolerance to make a claim agree.

WHAT THIS TEST WILL NOT DO
--------------------------
It will not repair a disagreement by editing `stated`. `stated` is what has been
published. When a check disagrees the default repair is to the document, and the
entry carries `document_edit_owed` saying so.

Writes results/claims_report.md and outputs/claims_status.csv through
`code/repro/make_claim_report.py`, which is a generator rather than this test,
so the SI can print the table without importing a test file.
"""

from __future__ import annotations

import json
import math
import os
import sys
from decimal import Decimal
from typing import Any, Dict, List, Mapping

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "code", "repro"))
sys.path.insert(0, os.path.join(REPO, "code", "model"))

import yaml  # noqa: E402

import claim_resolvers as cr  # noqa: E402

BASELINE_PATH = os.path.join(REPO, "docs", "claims_baseline.json")
INDEX_BASELINE_PATH = os.path.join(REPO, "docs", "claims_index_baseline.json")

failures: List[str] = []
notes: List[str] = []


def check(ok: bool, message: str) -> bool:
    if not ok:
        failures.append(message)
    return ok


# --- G1 structure ----------------------------------------------------------

def g1_structure(doc: Mapping[str, Any]) -> None:
    claims = doc["claims"]
    ids = [c["id"] for c in claims]
    check(len(ids) == len(set(ids)), f"G1 duplicate claim ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    check(
        len(claims) == int(doc["claim_count"]),
        f"G1 header says claim_count {doc['claim_count']}, file has {len(claims)}",
    )
    n_checks = sum(len(c.get("checks", [])) for c in claims)
    check(
        n_checks == int(doc["check_count"]),
        f"G1 header says check_count {doc['check_count']}, file has {n_checks}",
    )
    for c in claims:
        for field in ("text", "location", "status", "artifact"):
            check(bool(c.get(field)), f"G1 {c['id']} has no {field}")
        check(bool(c.get("checks")), f"G1 {c['id']} carries no checks")
        cids = [k["id"] for k in c.get("checks", [])]
        check(
            len(cids) == len(set(cids)),
            f"G1 {c['id']} has duplicate check ids",
        )
        if c.get("status") == "current" and "owed" in c:
            failures.append(
                f"G1 {c['id']} carries an `owed:` note while status says `current`. "
                "That is the F-016 shape: a debt on a claim nothing counts."
            )


# --- G2/G3/G4 verdicts -----------------------------------------------------

def _load_baseline(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        failures.append(f"G3 baseline missing: {os.path.relpath(path, REPO)}")
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def g2_g3_g4(scoring: Mapping[str, Any]) -> None:
    base = _load_baseline(BASELINE_PATH)
    if not base:
        return

    now = {f"{r['claim']}/{r['check']}" for r in scoring["checks"] if r["verdict"] == "DRIFTED"}
    was = set(base.get("drifted_checks", []))

    grew = sorted(now - was)
    check(not grew, f"G3 the drifted set grew: {grew}")

    improved = sorted(was - now)
    check(
        not improved,
        "G3 these checks came into line without the baseline being regenerated: "
        f"{improved}. Regenerate docs/claims_baseline.json and say why in FINDINGS.md.",
    )

    check(
        sorted(scoring["drifted_claims"]) == sorted(base.get("drifted_claims", [])),
        f"G3 drifted claim set is {scoring['drifted_claims']}, baseline says {base.get('drifted_claims')}",
    )

    check(
        scoring["owed_count"] <= int(base.get("owed_count", 0)),
        f"G4 owed_count rose from {base.get('owed_count')} to {scoring['owed_count']}",
    )
    if scoring["owed_count"] < int(base.get("owed_count", 0)):
        failures.append(
            f"G4 owed_count fell from {base.get('owed_count')} to {scoring['owed_count']} "
            "without the baseline being regenerated."
        )


# --- G5 the two-way index --------------------------------------------------

def g5_two_way_index(doc: Mapping[str, Any]) -> None:
    import registry as reg

    forward: Dict[str, set] = {}
    for name in reg.names():
        for cid in reg.affects_claims(name):
            forward.setdefault(cid, set()).add(name)

    reverse: Dict[str, set] = {
        c["id"]: set(c.get("depends_on_params") or []) for c in doc["claims"]
    }

    unknown = sorted(set(forward) - set(reverse))
    check(
        not unknown,
        f"G5 params.yaml names claims that do not exist in claims.yaml: {unknown}",
    )

    for cid, params in sorted(reverse.items()):
        fwd = forward.get(cid, set())
        missing_forward = sorted(params - fwd)
        missing_reverse = sorted(fwd - params)
        check(
            not missing_forward,
            f"G5 {cid} lists {missing_forward} but those parameters do not declare "
            f"affects_claims: [{cid}] in params.yaml",
        )
        check(
            not missing_reverse,
            f"G5 {missing_reverse} declare affects_claims: [{cid}] but {cid} does not "
            "list them in depends_on_params",
        )

    frozen = _load_baseline(INDEX_BASELINE_PATH)
    if frozen:
        live = {k: sorted(v) for k, v in reverse.items() if v}
        check(
            live == {k: sorted(v) for k, v in frozen.get("index", {}).items()},
            "G5 the claim/parameter index changed without docs/claims_index_baseline.json "
            "being regenerated. A documentation edit that moves the index moves which "
            "published sentences a parameter change is known to touch.",
        )


# --- G6 tolerance provenance ----------------------------------------------

def _stated_precision(stated: float) -> float:
    """The half-width implied by how the number was written.

    2.5 was written to one decimal, so it asserts nothing finer than 0.1; 1230
    was written to the unit, so 1.0. This is the precision the sentence itself
    claims, which is where every tolerance in the register came from.
    """
    d = Decimal(repr(float(stated))).normalize()
    exp = d.as_tuple().exponent
    if exp >= 0:
        return 1.0
    return float(10 ** exp)


def g6_tolerance_provenance(doc: Mapping[str, Any]) -> None:
    for c in doc["claims"]:
        for k in c.get("checks", []):
            tol = float(k.get("tol", 0.0))
            if tol == 0.0:
                continue
            allowed = _stated_precision(k["stated"])
            if tol <= allowed + 1e-12:
                continue
            # A tolerance may exceed the written precision only where the entry says
            # in writing why, and never by more than a factor of two. F-016's C-010
            # SOC check is the one such case in the register.
            check(
                bool(k.get("tol_note")) and tol <= 2.0 * allowed + 1e-12,
                f"G6 {c['id']}/{k['id']} carries tol {tol} against a value written to "
                f"{allowed}, with {'no' if not k.get('tol_note') else 'an insufficient'} "
                "`tol_note`. A tolerance may not be widened past a sentence's own "
                "precision without saying why, and never past twice it.",
            )


# --- report ----------------------------------------------------------------

def main() -> int:
    doc = cr.load_claims()
    g1_structure(doc)

    try:
        scoring = cr.evaluate(doc)
    except cr.ClaimResolutionError as exc:
        failures.append(f"G2 unresolved check: {exc}")
        print("\n".join(f"  FAIL  {f}" for f in failures))
        print("\nCLAIM GATE: FAIL (resolution)")
        return 1

    g2_g3_g4(scoring)
    g5_two_way_index(doc)
    g6_tolerance_provenance(doc)

    smallest = min(
        (r for r in scoring["checks"] if r["verdict"] == "DRIFTED"),
        key=lambda r: abs(r["delta"]),
        default=None,
    )
    largest = max(
        (r for r in scoring["checks"] if r["verdict"] == "DRIFTED"),
        key=lambda r: abs(r["delta"]),
        default=None,
    )

    print(
        "CLAIM REGISTER  %d claims, %d checks, %d AGREES, %d DRIFTED, %d unresolved, "
        "owed generators %d"
        % (
            scoring["n_claims"],
            scoring["n_checks"],
            scoring["n_agrees"],
            scoring["n_drifted"],
            0,
            scoring["owed_count"],
        )
    )
    print("DRIFTED CLAIMS  %s" % ", ".join(scoring["drifted_claims"]))
    if smallest:
        print(
            "SMALLEST DRIFT  %s/%s  %.4g against tol %g"
            % (smallest["claim"], smallest["check"], abs(smallest["delta"]), smallest["tol"])
        )
    if largest:
        print(
            "LARGEST DRIFT   %s/%s  %.4g"
            % (largest["claim"], largest["check"], abs(largest["delta"]))
        )

    for f in failures:
        print("  FAIL  %s" % f)
    print("\nCLAIM GATE: %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
