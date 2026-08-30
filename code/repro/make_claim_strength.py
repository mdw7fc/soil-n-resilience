#!/usr/bin/env python3
"""How often does the ensemble actually produce the ordering the paper states?

Spec: FINDINGS.md F-013 (assurance plan 3.8). Rebuilt by WP5 on 2026-07-26
after the v15 working tree was lost.

THE QUESTION NO UNIT TEST ASKS
------------------------------
Every number behind the v14 regional rankings was computed correctly. What was
missing was the step between a computed ordering and a stated one. A ranking
read off the central run is a fact about one draw; a ranking written into a
paper is a claim about the system. This script scores the second against the
posterior.

    p >= 0.90   state the ranking outright
    p >= 0.60   state it hedged, with the probability given
    otherwise   report the group; the regions are not separable

The thresholds are declared HERE, once, above the data, and not chosen per
claim. A threshold picked after the probability is visible is not a threshold.

EVERY FAMILY ACCOUNTS FOR ITS FULL MASS
---------------------------------------
`p` is the probability of leading, not the probability of leading among the
regions the scoring happened to see. Each family reports the mass it covers and
refuses to score if that mass is not 1.

THE GATE CHECKS THE REGION, NOT ONLY THE BAND
---------------------------------------------
F-013's C-064 is why. "Most exposed" is a licensed form of statement about a
family whose leader clears 0.90, so a threshold check alone would have passed a
sentence that named the wrong region. Each scored claim therefore carries the
region it names, and `--gate` fails a claim naming a region the ensemble does
not put first, at any band.

WHAT THIS SCRIPT MAY NOT DO
---------------------------
It may not edit the ensemble to agree with `params.yaml`. The ensemble is what
ran. Editing it would make the published interval unreproducible from the code
that produced it.

Writes results/claim_strength.csv, results/claim_strength.md,
docs/claim_strength_baseline.json.

    python3 code/repro/make_claim_strength.py           # score and write
    python3 code/repro/make_claim_strength.py --freeze  # also rewrite the baseline
    python3 code/repro/make_claim_strength.py --gate    # score, then exit 1 on any
                                                        # overstatement not carried
                                                        # in the baseline
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "code", "model"))

import prices  # noqa: E402
import registry as reg  # noqa: E402

POSTERIOR = os.path.join(REPO, "data", "mc_ensemble", "mc_posterior.csv.gz")
RESULTS = os.path.join(REPO, "results")
DOCS = os.path.join(REPO, "docs")

#: Declared above the data, not chosen per claim.
STATE_THRESHOLD = 0.90
HEDGE_THRESHOLD = 0.60

#: The SOC level at which the paper states its cross-regional rankings.
RANKING_SOC_PCT = 100.0


# --- the posterior ---------------------------------------------------------

def load_posterior(path: str = POSTERIOR) -> Dict[int, Dict[str, Dict[str, float]]]:
    """draw -> region -> {yield_pen, profit_chg, F_shocked, y_shock}, at mean SOC."""
    out: Dict[int, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if abs(float(row["soc_pct"]) - RANKING_SOC_PCT) > 1e-9:
                continue
            rec: Dict[str, float] = {}
            for key in ("yield_pen", "profit_chg", "F_shocked", "y_shock"):
                raw = row.get(key, "")
                if raw not in ("", "nan", "None"):
                    rec[key] = float(raw)
            out[int(row["draw"])][row["region"]] = rec
    if not out:
        raise RuntimeError(f"{path} contained no draws at SOC {RANKING_SOC_PCT}%")
    return dict(out)


# --- the three families ----------------------------------------------------

def _leader_counts(
    draws: Mapping[int, Mapping[str, Mapping[str, float]]],
    score,
    better,
) -> Tuple[collections.Counter, set]:
    """Leader tally, and the set of regions that were ELIGIBLE to lead.

    Eligibility is reported separately from leadership on purpose. A family
    scored over four of eight regions is a conditional statement, and a reader
    who is only shown which regions actually led cannot tell the difference.
    """
    counts: collections.Counter = collections.Counter()
    eligible: set = set()
    for _, regions in draws.items():
        vals = {}
        for rk, rec in regions.items():
            v = score(rk, rec)
            if v is not None:
                vals[rk] = v
        if not vals:
            continue
        eligible |= set(vals)
        counts[better(vals, key=vals.get)] += 1
    return counts, eligible


def _p3(draws):
    return _leader_counts(
        draws,
        lambda rk, rec: rec.get("yield_pen"),
        max,
    )


def _p4(draws):
    return _leader_counts(
        draws,
        lambda rk, rec: rec.get("profit_chg"),
        min,
    )


def _p4b(draws):
    """Derived nitrogen cost share, from prices and the drawn outcome.

    F-013: the superseded 83.7% figure measured a hardcoded FERT_COST_FRAC
    dictionary that assigned SSA 0.25 and North America 0.08 and varied neither,
    so the probability measured the assumption. The share is derived here.
    """

    def score(rk: str, rec: Mapping[str, float]) -> Optional[float]:
        if rk not in prices.PRICED_REGIONS:
            return None
        if "F_shocked" not in rec or "y_shock" not in rec or rec["y_shock"] <= 0:
            return None
        return prices.nitrogen_cost_share(rk, rec["F_shocked"], rec["y_shock"])

    return _leader_counts(draws, score, max)


FAMILIES: List[Dict[str, Any]] = [
    {
        "family": "P3",
        "claim_id": "C-062",
        "question": "highest year-1 yield loss",
        "location": "SI [163]",
        "stated_region": "fsu_central_asia",
        "stated_text": (
            "the Former Soviet Union/Central Asia region carries the largest year-1 "
            "yield loss in 99.8% of draws"
        ),
        "scorer": _p3,
    },
    {
        "family": "P4",
        "claim_id": "C-063",
        "question": "worst year-1 net-revenue change",
        "location": "SI [163]",
        "stated_region": "sub_saharan_africa",
        "stated_text": (
            "Sub-Saharan Africa is the worst region for year-1 gross margin in 83.7% "
            "of draws"
        ),
        "scorer": _p4,
    },
    {
        "family": "P4b",
        "claim_id": "C-064",
        "question": "highest derived nitrogen cost share",
        "location": "SI [163]",
        "stated_region": "sub_saharan_africa",
        "stated_text": (
            "... reflecting its high fertilizer-cost share and inelastic food demand"
        ),
        "scorer": _p4b,
    },
]


def band(p: float) -> str:
    if p >= STATE_THRESHOLD:
        return "state"
    if p >= HEDGE_THRESHOLD:
        return "hedge"
    return "not_separable"


def licensed_sentence(p: float, leader: str, runner: Optional[str], runner_p: float) -> str:
    if p >= STATE_THRESHOLD:
        return "state the ranking outright"
    if p >= HEDGE_THRESHOLD:
        return f"state the ranking hedged, with p = {p:.3f} given"
    if runner:
        return f"state that the regions are not separable; report the group {leader} + {runner}"
    return "state that the regions are not separable"


def score_all() -> List[Dict[str, Any]]:
    draws = load_posterior()
    n_draws = len(draws)
    rows: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        counts, eligible = fam["scorer"](draws)
        total = sum(counts.values())
        if total == 0:
            raise RuntimeError(f"{fam['family']}: no draw could be scored")
        ordered = counts.most_common()
        leader, n_lead = ordered[0]
        runner, n_run = ordered[1] if len(ordered) > 1 else (None, 0)
        p = n_lead / total
        p_run = n_run / total
        mass = total / n_draws
        if abs(mass - 1.0) > 1e-9:
            raise RuntimeError(
                f"{fam['family']}: scored {total} of {n_draws} draws. A leader "
                "probability that does not account for its full mass is a "
                "probability of leading among the draws the scoring happened to see."
            )
        in_ensemble = sorted({r for d in draws.values() for r in d})
        rows.append(
            {
                "family": fam["family"],
                "claim_id": fam["claim_id"],
                "location": fam["location"],
                "question": fam["question"],
                "n": n_draws,
                "leader": leader,
                "p": p,
                "runner_up": runner or "",
                "p_runner_up": p_run,
                "pair_p": p + p_run,
                "band": band(p),
                "stated_region": fam["stated_region"],
                "region_agrees": leader == fam["stated_region"],
                "stated_text": fam["stated_text"],
                "n_regions_scored": len(eligible),
                "n_regions_in_ensemble": len(in_ensemble),
                "regions_scored": sorted(eligible),
                "licensed": licensed_sentence(p, leader, runner, p_run),
                "distribution": {k: v / total for k, v in ordered},
            }
        )
    return rows


# --- outputs ---------------------------------------------------------------

def write_csv(rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    cols = [
        "family", "claim_id", "location", "question", "n", "leader", "p",
        "runner_up", "p_runner_up", "pair_p", "band", "stated_region",
        "region_agrees", "n_regions_scored", "n_regions_in_ensemble", "licensed",
    ]
    with open(os.path.join(RESULTS, "claim_strength.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def write_md(rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out: List[str] = []
    out.append("# Claim strength against the Monte Carlo posterior\n")
    out.append(
        "Generated by `code/repro/make_claim_strength.py`. Assurance plan 3.8. "
        "Thresholds are declared in that script, not chosen per claim: p >= 0.9 "
        "states the ranking, p >= 0.6 states it hedged with the probability given, "
        "below that the regions are reported as not separable.\n"
    )
    for r in rows:
        out.append(f"## {r['family']}: {r['question']}\n")
        out.append(
            f"Leader **{r['leader']}**, p = {r['p']:.3f} (n = {r['n']})."
            + (
                f" Runner-up {r['runner_up']}, p = {r['p_runner_up']:.3f}.\n"
                if r["runner_up"]
                else "\n"
            )
        )
        out.append(f"Licensed: {r['licensed']}.\n")
        if r["band"] == "not_separable" and r["runner_up"]:
            out.append(
                f"Report the group: {r['leader']} + {r['runner_up']} "
                f"(combined p = {r['pair_p']:.3f}).\n"
            )
        if not r["region_agrees"]:
            out.append(
                f"**Overstated.** The sentence names {r['stated_region']}, which the "
                f"ensemble puts first in {r['distribution'].get(r['stated_region'], 0.0):.3f} "
                "of draws. A threshold check alone would have passed this; the claim is "
                "wrong about *which* region, not about how strongly to say it.\n"
            )
        if r["n_regions_scored"] < r["n_regions_in_ensemble"]:
            out.append(
                f"**Scored over {r['n_regions_scored']} of "
                f"{r['n_regions_in_ensemble']} regions.** The registry prices "
                f"{', '.join(prices.PRICED_REGIONS)} and raises for the rest, so this "
                "probability is conditional on the four audited price pairs. See "
                "`v15/RECONSTRUCTION_GAPS.md` G-4.\n"
            )
    with open(os.path.join(RESULTS, "claim_strength.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


def write_baseline(rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(DOCS, exist_ok=True)
    overstated = [
        {
            "claim_id": r["claim_id"],
            "family": r["family"],
            "location": r["location"],
            "reason": (
                "names a region the ensemble does not put first"
                if not r["region_agrees"]
                else "stated outright at a probability below the state threshold"
            ),
            "stated_region": r["stated_region"],
            "ensemble_leader": r["leader"],
            "p_leader": round(r["p"], 4),
            "p_stated_region": round(r["distribution"].get(r["stated_region"], 0.0), 4),
            "band": r["band"],
        }
        for r in rows
        if (not r["region_agrees"]) or r["band"] != "state"
    ]
    payload = {
        "generated_by": "code/repro/make_claim_strength.py --freeze",
        "spec": ["F-013"],
        "rule": (
            "This list may only shrink, and a claim that comes into line without this "
            "file being regenerated also fails. Carried pending the v15 document edits."
        ),
        "thresholds": {"state": STATE_THRESHOLD, "hedge": HEDGE_THRESHOLD},
        "n_draws": rows[0]["n"] if rows else 0,
        "overstated": overstated,
        "overstated_count": len(overstated),
        "probabilities": {r["family"]: round(r["p"], 4) for r in rows},
        "leaders": {r["family"]: r["leader"] for r in rows},
    }
    with open(os.path.join(DOCS, "claim_strength_baseline.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def gate(rows: Sequence[Mapping[str, Any]]) -> int:
    path = os.path.join(DOCS, "claim_strength_baseline.json")
    if not os.path.exists(path):
        print("  FAIL  docs/claim_strength_baseline.json is missing")
        return 1
    with open(path, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    carried = {o["claim_id"] for o in base.get("overstated", [])}
    now = {r["claim_id"] for r in rows if (not r["region_agrees"]) or r["band"] != "state"}
    failures: List[str] = []
    for cid in sorted(now - carried):
        failures.append(f"{cid} is overstated and is not carried in the baseline")
    for cid in sorted(carried - now):
        failures.append(
            f"{cid} came into line without docs/claim_strength_baseline.json being regenerated"
        )
    for f in failures:
        print("  FAIL  %s" % f)
    print("\nCLAIM STRENGTH GATE: %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="rewrite docs/claim_strength_baseline.json")
    ap.add_argument("--gate", action="store_true", help="exit 1 on an uncarried overstatement")
    args = ap.parse_args()

    rows = score_all()
    write_csv(rows)
    write_md(rows)
    if args.freeze:
        write_baseline(rows)

    for r in rows:
        flag = "" if r["region_agrees"] else "   <- names %s" % r["stated_region"]
        scope = (
            ""
            if r["n_regions_scored"] == r["n_regions_in_ensemble"]
            else "  [%d of %d regions]" % (r["n_regions_scored"], r["n_regions_in_ensemble"])
        )
        print(
            "%-4s %-34s %-20s p = %.3f  %s%s%s"
            % (r["family"], r["question"], r["leader"], r["p"], r["band"], scope, flag)
        )

    return gate(rows) if args.gate else 0


if __name__ == "__main__":
    sys.exit(main())
