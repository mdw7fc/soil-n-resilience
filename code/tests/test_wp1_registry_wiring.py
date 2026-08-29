#!/usr/bin/env python3
"""WP1 acceptance: reversing the direction of authority changed no number.

A refactor that changes no number has to be SHOWN to change no number (F-011).
This script runs the three checks that showing takes:

  1. A 123-field canonical diff against a FRESHLY REGENERATED artifact. The
     canonical artifact is eight regions of fifteen fields plus three global
     production-weighted losses.

     Revised 2026-07-25 (WP3). This check originally required every field to
     be bit-identical, and read the artifact straight off the tree. Both parts
     had to change. It read a deposit file that is only rewritten when someone
     runs the model, so on an un-regenerated checkout it compared the 20defb2
     baseline against the 20defb2 artifact and reported "no number moved" --
     passing because the file was stale rather than because the code agreed.
     And WP2's F-002 production-path recalibration then moved 50 of the 123
     fields on purpose, so bit-identity can never hold again.

     The check now re-runs the model into a throwaway copy and asserts the
     DELTA: exactly these 50 fields moved, to these values, and nothing else
     did. That still fails on an unintended change in either direction, and it
     keeps the evidence of what WP2 moved instead of rebaselining it away.
     The pinned set is baseline/canonical_expected_delta.json.
  2. A field-by-field equality log over every regional field, plus the three
     parameter dataclasses, plus FAOSTAT_TARGETS and REGIONAL_ECON_PARAMS.
  3. The global S3 production-weighted yield loss, reported so the run's
     headline number is on the record next to the diff.

  4. Plus the registry's own load-time contract: 53 entries, 55 mutable leaves,
     and exactly six leaves refused at load by the two sum-to-one blocks and
     the profile-depth unit check.

The 'before' snapshots are committed under .baseline/ and were taken from the
reconstruction base, git 20defb2, with every literal still in place.

Exit 0 means WP1 holds and the canonical artifact moves only where WP2 is
recorded as having moved it. Any nonzero exit names the first field that
disagrees with the pinned delta.
"""

import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "code", "model"))

# The snapshots are committed as v15/baseline/ on Matthew's disk (a dotted
# directory is easy to lose in a file browser) and as .baseline/ in a working
# tree. Accept either; the contents are the same three JSON files.
BASELINE = os.path.join(ROOT, ".baseline")
if not os.path.isdir(BASELINE):
    BASELINE = os.path.join(ROOT, "baseline")
RESULTS = os.path.join(ROOT, "results")

failures = []
lines = []


def log(msg):
    lines.append(msg)
    print(msg)


def check(ok, msg):
    if not ok:
        failures.append(msg)
        log("FAIL  " + msg)
    return ok


# ---------------------------------------------------------------------------
# 1. 123-field canonical diff
# ---------------------------------------------------------------------------

def flatten_canonical(doc):
    """Every numeric field of the canonical artifact, as a flat dict."""
    flat = {}
    for r in doc["regions"]:
        for k, v in r.items():
            # All fifteen fields, strings included: the region key and its
            # abbreviation are part of the artifact's identity, and a rewiring
            # that silently reordered or renamed a region would otherwise pass.
            flat["%s.%s" % (r["region"], k)] = v
    for y, v in doc["global_prodweighted"].items():
        flat["global_prodweighted.%s" % y] = v
    return flat


def regenerate_canonical():
    """Run the canonical model in a throwaway copy and return its artifact.

    CHECK 1 used to read ``data/canonical_ERA5_y30.json`` straight off the
    tree. That file is a deposit artifact, not a run: on a checkout where it
    has not been regenerated it still holds the 20defb2 figures, so the check
    compared the baseline against itself and reported "no number moved". It
    passed by being stale. Re-running is the only way the comparison says
    anything about the code as it stands.
    """
    import shutil, subprocess, tempfile
    tmp = tempfile.mkdtemp(prefix="wp1_canon_")
    try:
        for sub in ("code", "data", "outputs", "results"):
            src = os.path.join(ROOT, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(tmp, sub),
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                                dirs_exist_ok=True)
        p = subprocess.run([sys.executable, "run_canonical.py"],
                           cwd=os.path.join(tmp, "code", "repro"),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=900)
        path = os.path.join(tmp, "data", "canonical_ERA5_y30.json")
        if p.returncode != 0 or not os.path.exists(path):
            return None
        return json.load(open(path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


log("=" * 72)
log("CHECK 1 -- canonical diff against a freshly regenerated artifact")
before = flatten_canonical(json.load(open(os.path.join(BASELINE, "canonical_before.json"))))
fresh = regenerate_canonical()

if not check(fresh is not None, "canonical model failed to run"):
    after = {}
else:
    after = flatten_canonical(fresh)

check(len(before) == 123, "canonical fingerprint is %d fields, expected 123" % len(before))
check(set(before) == set(after), "canonical field set changed")

# WP1 changed no number. WP2's F-002 production-path recalibration then moved
# 50 of the 123 deliberately, by solving y_max against the FAOSTAT target
# rather than reading a static value. Asserting "nothing moved" can therefore
# never pass again, and rebaselining to the post-WP2 figures would erase the
# evidence that anything moved at all. The gate instead pins the delta: these
# fields moved, by these amounts, and nothing else did. It still catches an
# unintended change -- in either direction -- and it keeps the record.
EXPECTED_DELTA = os.path.join(BASELINE, "canonical_expected_delta.json")
expected = {}
if os.path.exists(EXPECTED_DELTA):
    expected = json.load(open(EXPECTED_DELTA))["moved"]

DELTA_TOL = 1e-9
moved, unexpected, mismatched = {}, [], []
for k in sorted(before):
    if k in after and before[k] != after[k]:
        moved[k] = after[k]
        if k not in expected:
            unexpected.append(k)
            log("  UNEXPECTED MOVE  %-40s %r -> %r" % (k, before[k], after[k]))
        elif isinstance(after[k], (int, float)) and not isinstance(after[k], bool):
            if abs(float(expected[k]) - float(after[k])) > DELTA_TOL:
                mismatched.append(k)
                log("  WRONG VALUE      %-40s expected %r, got %r"
                    % (k, expected[k], after[k]))
        elif expected[k] != after[k]:
            mismatched.append(k)
            log("  WRONG VALUE      %-40s expected %r, got %r"
                % (k, expected[k], after[k]))

healed = [k for k in expected if k not in moved]
for k in healed:
    log("  NO LONGER MOVES  %-40s expected it to move to %r" % (k, expected[k]))

log("  %d fields compared, %d moved, %d pinned as expected"
    % (len(before), len(moved), len(expected)))
check(not unexpected, "%d field(s) moved that the pinned delta does not "
                      "account for: %s" % (len(unexpected), unexpected[:6]))
check(not mismatched, "%d field(s) moved to a value the pinned delta does not "
                      "expect: %s" % (len(mismatched), mismatched[:6]))
check(not healed, "%d field(s) the pinned delta expects to move no longer "
                  "move: %s" % (len(healed), healed[:6]))


# ---------------------------------------------------------------------------
# 2. field-by-field equality log
# ---------------------------------------------------------------------------

log("=" * 72)
log("CHECK 2 -- field-by-field equality over every regional field")

from soil_n_model import (  # noqa: E402
    FeedbackParams,
    SOMPoolParams,
    get_default_regions,
)
import coupled_econ_biophysical as econ  # noqa: E402
import monthly_model_v3 as mm  # noqa: E402

# Fields that existed when `baseline/regional_fields_before.json` was frozen and
# have since been deliberately deleted. A deletion is not a drift, but it is not
# nothing either: it must be named here, with the finding that authorised it,
# or the comparison below raises. Silently skipping a missing attribute would
# turn "the field is gone" and "the field never mattered" into the same result.
DELETED_FIELDS = {
    ("FeedbackParams", "cre_base"):
        "F-011 scored it INERT and v15 deleted the fallback it guarded; "
        "soil_n_model.region_cre() now raises instead of substituting 0.11. "
        "Verified to move no number: logs/run_203_canon.log diffs to zero "
        "against the pre-deletion canonical over all 125 fields.",
}

snap = json.load(open(os.path.join(BASELINE, "regional_fields_before.json")))
econ_snap = json.load(open(os.path.join(BASELINE, "econ_targets_before.json")))

regions = get_default_regions()
n_fields = n_moved = 0
for rk in sorted(snap["regions"]):
    r = regions[rk]
    for f in sorted(snap["regions"][rk]):
        n_fields += 1
        b = snap["regions"][rk][f]
        a = getattr(r, f)
        same = (a == b) and (type(a) is type(b))
        if not same:
            n_moved += 1
            log("  moved  %-24s %-24s %r -> %r" % (rk, f, b, a))
log("  RegionParams: %d fields over %d regions, %d moved"
    % (n_fields, len(snap["regions"]), n_moved))
check(n_moved == 0, "%d regional fields moved" % n_moved)

for label, obj in (
    ("SOMPoolParams", SOMPoolParams()),
    ("SOMPoolParams_tropical", SOMPoolParams.tropical()),
    ("FeedbackParams", FeedbackParams()),
):
    moved = 0
    for f in sorted(snap[label]):
        if not hasattr(obj, f):
            if (label, f) in DELETED_FIELDS:
                log("  deleted %-23s %-24s %s" % (label, f, DELETED_FIELDS[(label, f)]))
                continue
            moved += 1
            log("  MISSING %-23s %-24s gone from the model and not declared "
                "in DELETED_FIELDS" % (label, f))
            continue
        b, a = snap[label][f], getattr(obj, f)
        if a != b:
            moved += 1
            log("  moved  %-24s %-24s %r -> %r" % (label, f, b, a))
    log("  %-24s %d fields, %d moved" % (label, len(snap[label]), moved))
    check(moved == 0, "%s: %d fields moved" % (label, moved))

moved = 0
for k, v in econ_snap["FAOSTAT_TARGETS"].items():
    if mm.FAOSTAT_TARGETS[k] != v:
        moved += 1
        log("  moved  FAOSTAT_TARGETS %s %r -> %r" % (k, v, mm.FAOSTAT_TARGETS[k]))
log("  FAOSTAT_TARGETS: %d fields, %d moved" % (len(econ_snap["FAOSTAT_TARGETS"]), moved))
check(moved == 0, "FAOSTAT_TARGETS: %d moved" % moved)

moved = n = 0
for rk, d in econ_snap["REGIONAL_ECON_PARAMS"].items():
    got = econ.REGIONAL_ECON_PARAMS[rk]
    if set(got) != set(d):
        moved += 1
        log("  moved  REGIONAL_ECON_PARAMS %s field set %r -> %r"
            % (rk, sorted(d), sorted(got)))
    for f, v in d.items():
        n += 1
        if got[f] != v:
            moved += 1
            log("  moved  REGIONAL_ECON_PARAMS %s %s %r -> %r" % (rk, f, v, got[f]))
log("  REGIONAL_ECON_PARAMS: %d fields, %d moved" % (n, moved))
check(moved == 0, "REGIONAL_ECON_PARAMS: %d moved" % moved)


# ---------------------------------------------------------------------------
# 3. headline result
# ---------------------------------------------------------------------------

log("=" * 72)
log("CHECK 3 -- global S3 production-weighted yield loss")
# Read from the same fresh run CHECK 1 used, not from data/. Reading the
# deposit artifact made this check pass on a stale file, which is how it came
# to assert the 20defb2 figures long after WP2 had moved them.
gl = (fresh or {}).get("global_prodweighted", {})
if not gl:
    check(False, "no fresh canonical artifact to read the headline from")
else:
    log("  year 1  %.2f %%" % gl["1"])
    log("  year 10 %.2f %%" % gl["10"])
    log("  year 30 %.2f %%" % gl["30"])
    log("  NOTE: the reconstruction base (git 20defb2, pre-F-002) produces")
    log("        2.31 / 3.18 / 3.29. WP2's F-002 recalibration moved this to")
    log("        2.32 / 3.20 / 3.31, and F-025's realized-yield clearing")
    log("        (2026-08-29) to 2.32 / 3.18 / 3.30: clearing the market on the")
    log("        biophysical response instead of its linearization shifts the")
    log("        equilibrium fertilizer path slightly. The 3.03 year-10 figure")
    log("        in HANDOFF section 5 matches none of these and is owed a")
    log("        restatement -- see results/mutation_coverage_reconciliation.md.")
    check(gl["1"] == 2.32 and gl["10"] == 3.18 and gl["30"] == 3.30,
          "canonical global losses are %.2f/%.2f/%.2f, expected the post-F-025 "
          "2.32/3.18/3.30" % (gl["1"], gl["10"], gl["30"]))


# ---------------------------------------------------------------------------
# 4. registry load-time contract
# ---------------------------------------------------------------------------

log("=" * 72)
log("CHECK 4 -- registry load-time contract")
import tempfile  # noqa: E402

import yaml  # noqa: E402

import registry as reg  # noqa: E402

log("  entries %d, mutable leaves %d" % (len(reg.names()), len(reg.leaves())))
# 54 -> 53 and 56 -> 55 when v15 deleted cre_base (F-011 INERT, F-018).
check(len(reg.names()) == 53, "registry has %d entries, expected 53" % len(reg.names()))
check(len(reg.leaves()) == 55, "registry has %d leaves, expected 55" % len(reg.leaves()))

mc = {}
for n in reg.names():
    mc[reg.mc_status(n)] = mc.get(reg.mc_status(n), 0) + 1
log("  mc status: %s" % sorted(mc.items()))
check(mc.get("drawn") == 8, "expected 8 drawn priors, got %s" % mc.get("drawn"))
# 17 -> 16: cre_base was declared_fixed and is deleted, not reclassified.
check(mc.get("declared_fixed") == 16,
      "expected 16 declared-but-fixed uncertainties (F-007), got %s"
      % mc.get("declared_fixed"))


def refuses(mutate, label):
    doc = yaml.safe_load(open(reg.PARAMS_PATH))
    mutate(doc)
    path = os.path.join(tempfile.mkdtemp(), "params.yaml")
    yaml.safe_dump(doc, open(path, "w"))
    try:
        reg.reload(path)
    except reg.RegistryError:
        return True
    finally:
        reg.reload()
    return False


guarded = []
for lf in ("f_active", "f_slow", "f_passive"):
    guarded.append(("som_pool_fractions." + lf, refuses(
        lambda d, lf=lf: d["parameters"]["som_pool_fractions"]["value"].__setitem__(
            lf, d["parameters"]["som_pool_fractions"]["value"][lf] * 1.1),
        lf)))
for lf in ("cre_to_active", "cre_to_slow"):
    guarded.append(("cre_allocation." + lf, refuses(
        lambda d, lf=lf: d["parameters"]["cre_allocation"]["value"].__setitem__(
            lf, d["parameters"]["cre_allocation"]["value"][lf] * 1.1),
        lf)))
guarded.append(("soc_profile_depth_cm", refuses(
    lambda d: d["parameters"]["soc_profile_depth_cm"].__setitem__("value", 33),
    "depth")))

for name, ok in guarded:
    log("  %-34s %s" % (name, "refused at load" if ok else "NOT REFUSED"))
n_guarded = sum(1 for _, ok in guarded if ok)
check(n_guarded == 6, "expected 6 leaves refused at load, got %d" % n_guarded)


# ---------------------------------------------------------------------------

log("=" * 72)
if failures:
    log("WP1 FAILED (%d)" % len(failures))
    for f in failures:
        log("  " + f)
else:
    log("WP1 PASSED -- the registry drives the model; the canonical artifact moves\n              only the 50 fields WP2 pinned, by the pinned amounts.")

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "wp1_equality_log.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

sys.exit(1 if failures else 0)
