#!/usr/bin/env python3
"""Mutation coverage over the parameter registry (finding F-011).

WHAT THIS ANSWERS

A registry that documents the model is worth nothing; a registry that drives it
is worth exactly as much as the tests that notice when it changes. This harness
perturbs each mutable leaf of ``params.yaml`` in a sandbox copy of the
repository and asks two questions:

    REACH   did a published number move?
    CATCH   did any test object?

and scores the leaf on the cross of the two.

    GUARDED_AT_LOAD     the registry refused the mutated value outright. A
                        load-time refusal is stronger than a test: a value that
                        would make the model incoherent never reaches it.
    DECLARED_NOT_WIRED  the mutation changed no model state at all. The
                        registry declares the parameter; the model does not
                        take it. This is the defect F-006 recorded.
    INERT               model state moved, no published number did. The
                        parameter is wired but drives nothing the probe reads.
    UNTESTED            a published number moved and every test stayed green.
    COVERED             a published number moved and a test went red.

HOW REACH IS MEASURED

``_flatten_canonical`` reduces ``data/canonical_ERA5_y30.json`` to its numeric
fields -- eight regions of per-region descriptors plus ``global_prodweighted``.
That artifact is the published-number set and nothing else. It carries no gross
margins, no prices and no cost shares, so a price parameter that moves the
margin figures the abstract quotes still scores INERT here. F-011 records this
as a limitation of the probe rather than a fact about the parameters, and this
file prints the six affected rows under a `not_probed` flag rather than letting
INERT be read as "irrelevant".

The baseline is a fresh run of the current tree, never the committed artifact.
As of WP2 the committed ``canonical_ERA5_y30.json`` predates the F-002
production-path recalibration and disagrees with a live run in 50 of 107
fields; fingerprinting it would score every leaf as reaching.

HOW CATCH IS MEASURED

Only tests that are green on the unmutated tree can catch anything. The harness
runs the suite once at baseline, keeps the green set, and reports the excluded
red tests by name -- a permanently-red test that "fails" under mutation is not
coverage, and silently counting it would inflate COVERED.

Usage:  python3 run_mutation_coverage.py [--jobs N] [--leaf NAME ...]
Writes: results/mutation_coverage.csv, results/mutation_coverage_summary.txt
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))    # code/tests -> v15
sys.path.insert(0, os.path.join(REPO, "code", "model"))

RESULTS = os.path.join(REPO, "results")
CANONICAL = os.path.join("data", "canonical_ERA5_y30.json")
MARGIN = os.path.join("data", "figure1_farm_gradient.json")
PRICE_GEN = "run_price_shock_analysis.py"

#: Relative perturbation applied to a leaf. Large enough to clear solver
#: tolerance, small enough to stay inside every declared parameter bound so a
#: bounds test does not fire on the perturbation itself rather than its effect.
DELTA = 0.10

#: A field counts as moved above this. The canonical artifact rounds some
#: fields to 2 dp, so this is a floor on what the artifact can express, not on
#: what the model computes.
REACH_TOL = 1e-9

#: Every test that could object. Paths are relative to the repo root and each
#: is run from its own directory, which is how the suite expects to be invoked.
TEST_FILES = [
    "code/tests/test_calibration_fingerprint.py",
    "code/tests/test_seam_contracts.py",
    "code/tests/test_spinup_partition_independence.py",
    "code/tests/test_wp1_registry_wiring.py",
    "code/repro/test_cap_market_clearing.py",
    "code/repro/test_cross_document_consistency_sol.py",
    "code/repro/test_dimensional_consistency_sol.py",
    "code/repro/test_full_zero_shock_sol.py",
    "code/repro/test_mc_robustness_sol.py",
    "code/repro/test_parameter_boundaries_sol.py",
    "code/repro/test_parameter_consistency_sol.py",
    "code/repro/test_parameter_extremes_sol.py",
    "code/repro/test_zero_shock_invariance.py",
]

#: Registry entries whose effect lands outside the probe's fingerprint. F-011
#: recorded six price parameters here and read their INERT verdicts as "not
#: probed" rather than as "does not matter", which was the right reading of a
#: wrong probe: the fingerprint was `canonical_ERA5_y30.json` alone, which
#: carries yields and losses and no money at all, so those six could not have
#: come out any other way. A verdict that can only come out one way is not a
#: verdict. F-019 widens the fingerprint to the published Figure 1 margin
#: curves and the derived per-region nitrogen cost shares, which every one of
#: the six moves, so the set is now empty. It is kept rather than deleted so
#: that adding a name to it stays a deliberate act with a reason attached.
NOT_PROBED = frozenset()

TEST_TIMEOUT = 600
RUN_TIMEOUT = 900


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _flatten_canonical(doc) -> dict:
    """The published-number set: every numeric field of the canonical artifact."""
    out = {}

    def walk(o, p):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            out[p] = float(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}.{k}" if p else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{p}[{i}]")

    walk(doc, "")
    return out


def _flatten_margins(doc: dict) -> dict:
    """The farm-margin half of the published-number set.

    Point by point along each curve rather than at the three SOC levels the
    abstract quotes, because the whole curve is the published object: a
    mutation that bends the gradient without moving its endpoints has still
    moved a published number.
    """
    out = {}
    for rk, rec in (doc.get("regions") or {}).items():
        pcts = rec.get("soc_pct") or []
        for series in ("yield_pen", "fert_red", "margin_chg"):
            for pct, v in zip(pcts, rec.get(series) or []):
                out["margin.%s.%s@%s" % (rk, series, pct)] = float(v)
    for rk, v in (doc.get("derived_n_cost_share") or {}).items():
        out["price.%s.n_cost_share" % rk] = float(v)
    for rk, d in (doc.get("regional_prices") or {}).items():
        for k in ("nitrogen_usd_per_kg_n", "crop_usd_per_t"):
            if isinstance(d.get(k), (int, float)):
                out["price.%s.%s" % (rk, k)] = float(d[k])
    return out


def _diff(a: dict, b: dict) -> list:
    """Fields present in both that moved, worst first."""
    moved = []
    for k in a:
        if k not in b:
            continue
        d = abs(a[k] - b[k])
        if d > REACH_TOL:
            moved.append((k, a[k], b[k], d))
    moved.sort(key=lambda r: -r[3])
    return moved


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

def _sandbox(dst: str) -> str:
    """A copy of the repository sufficient to run the model and its tests."""
    os.makedirs(dst, exist_ok=True)
    for sub in ("code", "data", "outputs", "results", "baseline"):
        src = os.path.join(REPO, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, sub),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                            dirs_exist_ok=True)
    return dst


def _perturb(v):
    """Scale by DELTA, additively when scaling is a no-op.

    A categorical or zero-valued leaf cannot be scaled -- ``texture_class`` is
    0 in sub-Saharan Africa and 1 elsewhere -- so zero moves by DELTA outright.
    The point is to state a different value, not a physically meaningful one.
    """
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v * (1.0 + DELTA) if v != 0 else DELTA
    if isinstance(v, dict):
        return {k: _perturb(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_perturb(x) for x in v]
    return v


def _write_mutant(sandbox: str, leaf: str) -> None:
    import yaml
    path = os.path.join(sandbox, "code", "model", "params.yaml")
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    name, _, sub = leaf.partition(".")
    entry = doc["parameters"][name]
    if sub:
        entry["value"][sub] = _perturb(entry["value"][sub])
    else:
        entry["value"] = _perturb(entry["value"])
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Sandbox actions
# ---------------------------------------------------------------------------

def _run(cmd, cwd, timeout):
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode, p.stdout.decode("utf8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def _state(sandbox: str):
    """(guarded, state_dict). Guarded means the registry refused the value."""
    out = os.path.join(sandbox, "_state.json")
    probe = os.path.join(sandbox, "code", "tests", "_mutation_state_probe.py")
    rc, log = _run([sys.executable, probe, out],
                   os.path.join(sandbox, "code", "tests"), 300)
    if not os.path.exists(out):
        return None, {}, log
    d = json.load(open(out))
    return bool(d.get("guarded_at_load")), d.get("state", {}), log


def _canonical(sandbox: str):
    rc, log = _run([sys.executable, "run_canonical.py"],
                   os.path.join(sandbox, "code", "repro"), RUN_TIMEOUT)
    path = os.path.join(sandbox, CANONICAL)
    if rc != 0 or not os.path.exists(path):
        return None, log
    fp = _flatten_canonical(json.load(open(path)))

    # The margin half. Run unconditionally, for every leaf, rather than only
    # for the leaves judged likely to reach money: a probe that decides in
    # advance which parameters could matter is the assumption-as-constant
    # pattern this whole sweep exists to find.
    rc2, log2 = _run([sys.executable, PRICE_GEN],
                     os.path.join(sandbox, "code", "repro"), RUN_TIMEOUT)
    mpath = os.path.join(sandbox, MARGIN)
    if rc2 != 0 or not os.path.exists(mpath):
        return None, "price analysis failed: " + log2
    fp.update(_flatten_margins(json.load(open(mpath))))
    return fp, log


def _suite(sandbox: str, tests):
    verdicts = {}
    for rel in tests:
        d = os.path.join(sandbox, os.path.dirname(rel))
        rc, _ = _run([sys.executable, os.path.basename(rel)], d, TEST_TIMEOUT)
        verdicts[rel] = rc
    return verdicts


# ---------------------------------------------------------------------------
# One leaf
# ---------------------------------------------------------------------------

def score_leaf(leaf: str, base_state: dict, base_pub: dict, green: list) -> dict:
    row = {"leaf": leaf, "verdict": "", "state_fields_moved": 0,
           "published_fields_moved": 0, "worst_field": "", "worst_delta": 0.0,
           "caught_by": "", "not_probed": leaf.split(".")[0] in NOT_PROBED,
           "note": ""}
    tmp = tempfile.mkdtemp(prefix="mut_")
    try:
        sb = _sandbox(os.path.join(tmp, "s"))
        _write_mutant(sb, leaf)

        guarded, state, log = _state(sb)
        if guarded is None:
            row["verdict"] = "ERROR"
            row["note"] = log.strip().splitlines()[-1][:200] if log.strip() else "state probe produced nothing"
            return row
        if guarded:
            row["verdict"] = "GUARDED_AT_LOAD"
            row["note"] = "registry refused the value at load"
            return row

        row["state_fields_moved"] = len(_diff(base_state, state))

        # The canonical run happens for every leaf, never conditionally on the
        # state snapshot. An earlier revision short-circuited to
        # DECLARED_NOT_WIRED whenever the state fingerprint did not move, which
        # silently mis-scored every parameter consumed inside the run rather
        # than stored on a parameter object: residue_c_to_active_fraction and
        # both laub_tropical_ratios leaves move published numbers and were
        # being reported as reaching nothing. REACH is the stronger signal and
        # is now always measured; the state snapshot only ever breaks the tie
        # between INERT and DECLARED_NOT_WIRED.
        pub, log = _canonical(sb)
        if pub is None:
            row["verdict"] = "ERROR"
            row["note"] = "canonical run failed: " + log.strip().splitlines()[-1][:160]
            return row

        pmoved = _diff(base_pub, pub)
        row["published_fields_moved"] = len(pmoved)
        if pmoved:
            row["worst_field"], _, _, row["worst_delta"] = pmoved[0]

        if not pmoved:
            if row["state_fields_moved"]:
                row["verdict"] = "INERT"
                row["note"] = ("wired, but its effect is outside the canonical "
                               "artifact" if row["not_probed"] else
                               "wired, but no published number depends on it")
            else:
                row["verdict"] = "DECLARED_NOT_WIRED"
                row["note"] = "registry declares it; nothing the probe reads depends on it"
            return row

        verdicts = _suite(sb, green)
        caught = [t for t, rc in verdicts.items() if rc != 0]
        row["caught_by"] = ";".join(os.path.basename(t) for t in caught)
        row["verdict"] = "COVERED" if caught else "UNTESTED"
        return row
    except Exception as exc:                                # noqa: BLE001
        row["verdict"] = "ERROR"
        row["note"] = f"{type(exc).__name__}: {exc}"[:200]
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------

ORDER = ["COVERED", "UNTESTED", "DECLARED_NOT_WIRED", "GUARDED_AT_LOAD",
         "INERT", "ERROR"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2)))
    ap.add_argument("--leaf", action="append", default=None)
    args = ap.parse_args()

    import registry as R
    leaves = args.leaf or R.leaves()
    t0 = time.time()

    print(f"[baseline] building reference sandbox ({len(leaves)} leaves, "
          f"{args.jobs} jobs)", flush=True)
    base_dir = tempfile.mkdtemp(prefix="mut_base_")
    sb = _sandbox(os.path.join(base_dir, "s"))

    guarded, base_state, log = _state(sb)
    if guarded is not False:
        print("[baseline] FATAL: unmutated registry does not load\n" + log)
        return 2
    base_pub, log = _canonical(sb)
    if base_pub is None:
        print("[baseline] FATAL: unmutated canonical run failed\n" + log)
        return 2
    print(f"[baseline] {len(base_state)} model-state fields, "
          f"{len(base_pub)} published fields", flush=True)

    base_tests = _suite(sb, TEST_FILES)
    green = [t for t, rc in base_tests.items() if rc == 0]
    red = [t for t, rc in base_tests.items() if rc != 0]
    print(f"[baseline] {len(green)} tests green, {len(red)} excluded as "
          f"already red:", flush=True)
    for t in red:
        print(f"           {os.path.basename(t)}  (rc={base_tests[t]})", flush=True)
    shutil.rmtree(base_dir, ignore_errors=True)

    rows = []
    with futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        fut = {ex.submit(score_leaf, lf, base_state, base_pub, green): lf
               for lf in leaves}
        for i, f in enumerate(futures.as_completed(fut), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i:2d}/{len(leaves)}] {r['leaf']:44s} {r['verdict']}",
                  flush=True)

    rows.sort(key=lambda r: (ORDER.index(r["verdict"]) if r["verdict"] in ORDER
                             else 9, r["leaf"]))

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, "mutation_coverage.csv")
    cols = ["leaf", "verdict", "state_fields_moved", "published_fields_moved",
            "worst_field", "worst_delta", "caught_by", "not_probed", "note"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    tally = {v: sum(1 for r in rows if r["verdict"] == v) for v in ORDER}
    el = time.time() - t0
    lines = [
        "MUTATION COVERAGE OVER THE PARAMETER REGISTRY (F-011)",
        "",
        f"leaves          {len(rows)}",
        f"delta           {DELTA:+.0%} relative, additive on a zero-valued leaf",
        f"published set   {len(base_pub)} numeric fields of {CANONICAL}",
        f"model state     {len(base_state)} numeric fields",
        f"elapsed         {el/60:.1f} min",
        "",
    ]
    for v in ORDER:
        if tally[v] or v != "ERROR":
            lines.append(f"    {v:<20s}{tally[v]:>4d}")
    lines += ["    " + "-" * 24, f"    {'':<20s}{len(rows):>4d}", ""]
    lines.append("TESTS EXCLUDED FROM CATCH (already red on the unmutated tree)")
    if red:
        for t in red:
            lines.append(f"    {t}  rc={base_tests[t]}")
    else:
        lines.append("    none")
    lines += ["",
              "NOT PROBED. These are wired and move gross margins, prices or",
              "cost shares, none of which the canonical artifact carries. Read",
              "their INERT verdict as 'not probed', not as 'irrelevant'.",
              ]
    for r in rows:
        if r["not_probed"]:
            lines.append(f"    {r['leaf']}  ({r['verdict']})")
    lines += ["", "UNTESTED -- a published number moves and nothing objects."]
    untested = [r for r in rows if r["verdict"] == "UNTESTED"]
    if untested:
        for r in untested:
            lines.append(f"    {r['leaf']}  ({r['published_fields_moved']} fields, "
                         f"worst {r['worst_field']} {r['worst_delta']:.4g})")
    else:
        lines.append("    none")

    # ------------------------------------------------------------------
    # COVERED counts a leaf once for any test that objects, which flatters a
    # suite containing a whole-artifact fingerprint: such a test trips on
    # every reaching mutation by construction, so it drives UNTESTED to zero
    # without saying anything about the parameter. Depth is the honest metric.
    # F-011 made this criticism of the pre-v15 suite -- "a suite whose
    # catching power sits in one behavioural test and two mirror tests is
    # thin" -- and it has to be applied to the current one too.
    # ------------------------------------------------------------------
    covered = [r for r in rows if r["verdict"] == "COVERED"]
    per_test, sole = {}, {}
    for r in covered:
        t = [x for x in r["caught_by"].split(";") if x]
        for x in t:
            per_test[x] = per_test.get(x, 0) + 1
        if len(t) == 1:
            sole.setdefault(t[0], []).append(r["leaf"])
    lines += ["", "COVERAGE DEPTH -- how many tests object, not whether any does.",
              f"    {sum(1 for r in covered if r['caught_by'].count(';') == 0)} of "
              f"{len(covered)} COVERED leaves rest on a single test.", ""]
    for t, c in sorted(per_test.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {c:3d}  {t}")
    for t, leaves_ in sorted(sole.items(), key=lambda kv: -len(kv[1])):
        lines += ["", f"    Caught ONLY by {t} ({len(leaves_)}). If that test is "
                      f"ever retired or", "    rebaselined away, these return to "
                      "UNTESTED:"]
        for lf in leaves_:
            lines.append(f"        {lf}")

    txt = "\n".join(lines) + "\n"
    open(os.path.join(RESULTS, "mutation_coverage_summary.txt"), "w").write(txt)
    print("\n" + txt)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
