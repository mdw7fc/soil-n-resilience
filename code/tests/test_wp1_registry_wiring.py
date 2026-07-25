#!/usr/bin/env python3
"""WP1 acceptance: reversing the direction of authority changed no number.

A refactor that changes no number has to be SHOWN to change no number (F-011).
This script runs the three checks that showing takes:

  1. A 123-field canonical diff. The canonical artifact is eight regions of
     fifteen fields plus three global production-weighted losses. Every numeric
     field must be bit-identical before and after the rewiring.
  2. A field-by-field equality log over every regional field, plus the three
     parameter dataclasses, plus FAOSTAT_TARGETS and REGIONAL_ECON_PARAMS.
  3. The global S3 production-weighted yield loss, reported so the run's
     headline number is on the record next to the diff.

  4. Plus the registry's own load-time contract: 54 entries, 56 mutable leaves,
     and exactly six leaves refused at load by the two sum-to-one blocks and
     the profile-depth unit check.

The 'before' snapshots are committed under .baseline/ and were taken from the
reconstruction base, git 20defb2, with every literal still in place.

Exit 0 means WP1 holds. Any nonzero exit names the first field that moved.
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


log("=" * 72)
log("CHECK 1 -- canonical diff")
before = flatten_canonical(json.load(open(os.path.join(BASELINE, "canonical_before.json"))))
after = flatten_canonical(json.load(open(os.path.join(ROOT, "data", "canonical_ERA5_y30.json"))))

check(len(before) == 123, "canonical fingerprint is %d fields, expected 123" % len(before))
check(set(before) == set(after), "canonical field set changed")
n_diff = 0
for k in sorted(before):
    if k in after and before[k] != after[k]:
        n_diff += 1
        log("  moved  %-44s %r -> %r" % (k, before[k], after[k]))
log("  %d fields compared, %d numeric differences" % (len(before), n_diff))
check(n_diff == 0, "canonical diff returned %d numeric differences" % n_diff)


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
gl = json.load(open(os.path.join(ROOT, "data", "canonical_ERA5_y30.json")))["global_prodweighted"]
log("  year 1  %.2f %%" % gl["1"])
log("  year 10 %.2f %%" % gl["10"])
log("  year 30 %.2f %%" % gl["30"])
log("  NOTE: the reconstruction base (git 20defb2, pre-F-002) produces")
log("        2.31 / 3.18 / 3.29. The 2.32 / 3.03 recorded in HANDOFF section 5")
log("        is post-recalibration and is WP2's acceptance target, not WP1's.")
log("        WP1's requirement is that the rewiring moves nothing, which")
log("        CHECK 1 is the test of.")
check(gl["1"] == 2.31 and gl["10"] == 3.18 and gl["30"] == 3.29,
      "canonical global losses moved from the 20defb2 base of 2.31/3.18/3.29")


# ---------------------------------------------------------------------------
# 4. registry load-time contract
# ---------------------------------------------------------------------------

log("=" * 72)
log("CHECK 4 -- registry load-time contract")
import tempfile  # noqa: E402

import yaml  # noqa: E402

import registry as reg  # noqa: E402

log("  entries %d, mutable leaves %d" % (len(reg.names()), len(reg.leaves())))
check(len(reg.names()) == 54, "registry has %d entries, expected 54" % len(reg.names()))
check(len(reg.leaves()) == 56, "registry has %d leaves, expected 56" % len(reg.leaves()))

mc = {}
for n in reg.names():
    mc[reg.mc_status(n)] = mc.get(reg.mc_status(n), 0) + 1
log("  mc status: %s" % sorted(mc.items()))
check(mc.get("drawn") == 8, "expected 8 drawn priors, got %s" % mc.get("drawn"))
check(mc.get("declared_fixed") == 17,
      "expected 17 declared-but-fixed uncertainties (F-007), got %s"
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
    log("WP1 PASSED -- the registry drives the model and no number moved.")

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "wp1_equality_log.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

sys.exit(1 if failures else 0)
