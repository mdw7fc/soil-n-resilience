#!/usr/bin/env python3
"""claim_resolvers.py -- resolve one check in docs/claims.yaml to one number.

Spec: FINDINGS.md F-012, F-015, F-016. Rebuilt by WP5 on 2026-07-26 after the
v15 working tree was lost.

WHY A SEPARATE MODULE
---------------------
Both `code/tests/test_claims.py` (the gate) and `code/repro/make_claim_report.py`
(the reporter, which writes `docs/claims_baseline.json` and
`results/claims_report.md`) have to resolve the same check to the same number. A
generator that imported a test would rebuild the one-concept-two-places failure
this whole rebuild is about, so the resolution lives here and both import it.

FOUR SELECTOR KINDS, AND NO FIFTH
---------------------------------
    csv_cell   pick a row by column values, read one column
    json       a dotted path, with `list[i]` and `list[key=value]` steps
    registry   a params.yaml entry, optionally one region of it
    derived    a named function in DERIVED below

`derived` is the escape hatch and it is deliberately a *named* one. A check that
needs arithmetic over an artifact -- a gap between two SOC levels, a
production-weighted aggregate -- names a function here rather than carrying a
formula in the YAML. That keeps the arithmetic reviewable in one place and stops
`claims.yaml` from becoming a small programming language.

RESOLUTION MUST NOT SILENTLY FAIL
---------------------------------
Every failure to resolve raises. F-012's first run recorded "zero unresolved
paths" as a result in its own right: a check that cannot find its number is not
an agreeing check, and a register that quietly skipped one would be worse than
no register at all.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Mapping, Sequence

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
MODEL = os.path.join(REPO, "code", "model")
if MODEL not in sys.path:
    sys.path.insert(0, MODEL)

CLAIMS_PATH = os.path.join(REPO, "docs", "claims.yaml")

REGIONS: List[str] = [
    "north_america",
    "europe",
    "east_asia",
    "south_asia",
    "southeast_asia",
    "latin_america",
    "sub_saharan_africa",
    "fsu_central_asia",
]


class ClaimResolutionError(RuntimeError):
    """A check could not be resolved. Never swallowed, never defaulted."""


# --- loading ---------------------------------------------------------------

def load_claims(path: str = CLAIMS_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if "claims" not in doc:
        raise ClaimResolutionError(f"{path} has no `claims:` list")
    return doc


def _abs(rel: str) -> str:
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        raise ClaimResolutionError(f"artifact not found: {rel}")
    return p


_CACHE: Dict[str, Any] = {}


def _read_csv(rel: str) -> List[Dict[str, str]]:
    if rel not in _CACHE:
        with open(_abs(rel), "r", encoding="utf-8") as fh:
            _CACHE[rel] = list(csv.DictReader(fh))
    return _CACHE[rel]


def _read_json(rel: str) -> Any:
    if rel not in _CACHE:
        with open(_abs(rel), "r", encoding="utf-8") as fh:
            _CACHE[rel] = json.load(fh)
    return _CACHE[rel]


def _num(x: Any, where: str) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except (TypeError, ValueError) as exc:
        raise ClaimResolutionError(f"{where}: {x!r} is not numeric") from exc


# --- selector kinds --------------------------------------------------------

def _sel_csv_cell(sel: Mapping[str, Any], artifact: str) -> float:
    rel = sel.get("artifact", artifact)
    rows = _read_csv(rel)
    where = sel.get("where") or {}
    hits = []
    for row in rows:
        ok = True
        for col, want in where.items():
            if col not in row:
                raise ClaimResolutionError(f"{rel}: no column {col!r}")
            got = row[col]
            if isinstance(want, (int, float)) and not isinstance(want, bool):
                try:
                    ok = ok and abs(float(got) - float(want)) < 1e-9
                except ValueError:
                    ok = False
            else:
                ok = ok and str(got).strip() == str(want).strip()
            if not ok:
                break
        if ok:
            hits.append(row)
    if len(hits) != 1:
        raise ClaimResolutionError(
            f"{rel}: `where` {where} matched {len(hits)} rows, expected exactly 1"
        )
    col = sel["column"]
    if col not in hits[0]:
        raise ClaimResolutionError(f"{rel}: no column {col!r}")
    return _num(hits[0][col], f"{rel}:{col}")


_STEP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]+)\])?$")


def _sel_json(sel: Mapping[str, Any], artifact: str) -> float:
    rel = sel.get("artifact", artifact)
    node: Any = _read_json(rel)
    path = str(sel["path"])
    for raw in path.split("."):
        m = _STEP.match(raw)
        if not m:
            raise ClaimResolutionError(f"{rel}: cannot parse path step {raw!r}")
        name, idx = m.group(1), m.group(2)
        if not isinstance(node, Mapping) or name not in node:
            raise ClaimResolutionError(f"{rel}: no key {name!r} on the way through {path!r}")
        node = node[name]
        if idx is None:
            continue
        if "=" in idx:
            key, want = idx.split("=", 1)
            if not isinstance(node, list):
                raise ClaimResolutionError(f"{rel}: {name!r} is not a list, cannot select {idx!r}")
            picks = [e for e in node if str(e.get(key)) == want]
            if len(picks) != 1:
                raise ClaimResolutionError(f"{rel}: {name}[{idx}] matched {len(picks)} entries")
            node = picks[0]
        else:
            node = node[int(idx)]
    return _num(node, f"{rel}:{path}")


def _sel_registry(sel: Mapping[str, Any], artifact: str) -> float:
    import registry as reg  # noqa: WPS433 -- deliberate late import, see module docstring

    name = sel["name"]
    if "key" in sel:
        return _num(reg.region_value(name, sel["key"]), f"registry:{name}.{sel['key']}")
    return _num(reg.value(name), f"registry:{name}")


# --- derived ---------------------------------------------------------------

def _figure2_gradient_monotone(region: str) -> float:
    """1 if year-10 loss falls monotonically as farm SOC rises, else 0.

    `yield_pct` is a signed change, so a shrinking loss is an increasing series.
    """
    g = _read_json("data/figure2_panels.json")["gradient"][region]
    soc, ys = g["soc_pct"], g["yield_pct"]
    if sorted(soc) != list(soc):
        raise ClaimResolutionError(f"figure2 gradient for {region} is not sorted by SOC")
    return 1.0 if all(ys[i] <= ys[i + 1] + 1e-12 for i in range(len(ys) - 1)) else 0.0


def _mc_min_buffer_probability() -> float:
    rows = _read_csv("data/mc_ensemble/mc_probabilities.csv")
    ps = [
        float(r["value"])
        for r in rows
        if r["statement"].startswith("P(low-SOC yield loss > high-SOC yield loss")
    ]
    if len(ps) != len(REGIONS):
        raise ClaimResolutionError(
            f"expected {len(REGIONS)} per-region buffer probabilities, found {len(ps)}"
        )
    return min(ps)


def _figure1_at(region: str, soc_pct: float, field: str) -> float:
    d = _read_json("data/figure1_farm_gradient.json")["regions"][region]
    soc = d["soc_pct"]
    hits = [i for i, s in enumerate(soc) if abs(float(s) - float(soc_pct)) < 1e-9]
    if len(hits) != 1:
        raise ClaimResolutionError(
            f"figure1 {region}: SOC {soc_pct} matched {len(hits)} grid points"
        )
    return float(d[field][hits[0]])


def _figure1_margin_gap(agg: str) -> float:
    """Margin percentage points gained moving from 50% to 100% of regional mean SOC."""
    gaps = [
        _figure1_at(r, 100, "margin_chg") - _figure1_at(r, 50, "margin_chg")
        for r in _read_json("data/figure1_farm_gradient.json")["regions"]
    ]
    return {"min": min, "max": max}[agg](gaps)


def _figure1_margin_at(region: str, soc_pct: float) -> float:
    return _figure1_at(region, soc_pct, "margin_chg")


def _registry_n_price(region: str) -> float:
    import prices  # noqa: WPS433

    return float(prices.n_price_usd_kg(region))


def _priced_region_count() -> float:
    import prices  # noqa: WPS433

    return float(len(prices.PRICED_REGIONS))


def _priced_regions_match(expected: Sequence[str]) -> float:
    import prices  # noqa: WPS433

    return 1.0 if set(prices.PRICED_REGIONS) == set(expected) else 0.0


def _figS11_spread(shocks: Sequence[float], agg: str) -> float:
    """Spread in percentage points between SOC 25% and SOC 100% at given shocks."""
    d = _read_json("data/figS11_severity_sweep.json")
    grid = [float(s) for s in d["shocks"]]
    vals: List[float] = []
    for region, levels in d["regions"].items():
        for sh in shocks:
            hits = [i for i, g in enumerate(grid) if abs(g - float(sh)) < 1e-9]
            if len(hits) != 1:
                raise ClaimResolutionError(f"figS11: shock {sh} matched {len(hits)} grid points")
            i = hits[0]
            vals.append(float(levels["25"][i]) - float(levels["100"][i]))
    return {"min": min, "max": max}[agg](vals)


def _figS8_reduction_pct() -> float:
    d = _read_json("data/figS8_curves.json")
    return 100.0 * (1.0 - float(d["ghalf"][10]) / float(d["base"] and d["gbase"][10]))


def _food_price_regional_extreme(column: str, agg: str) -> float:
    rows = [r for r in _read_csv("data/food_price_response.csv") if r["abbr"] != "GLOBAL"]
    vals = [float(r[column]) for r in rows]
    if len(vals) != len(REGIONS):
        raise ClaimResolutionError(f"food_price_response.csv: {len(vals)} regions, expected 8")
    return {"min": min, "max": max}[agg](vals)


def _nitrogen_weights() -> Dict[str, float]:
    """Nitrogen-tonnage weights. F-005: the reduction target is a mass, so it is
    averaged on nitrogen tonnage, not on production."""
    import registry as reg  # noqa: WPS433

    tonnes = {
        r: float(reg.region_value("synth_n_current", r)) * float(reg.region_value("cropland_mha", r))
        for r in REGIONS
    }
    total = sum(tonnes.values())
    return {r: v / total for r, v in tonnes.items()}


def _s3_global_reduction(column: str) -> float:
    rows = {r["region"]: r for r in _read_csv("results/s3_shock_calibration.csv")}
    w = _nitrogen_weights()
    missing = set(w) - set(rows)
    if missing:
        raise ClaimResolutionError(f"s3_shock_calibration.csv missing regions: {sorted(missing)}")
    return 100.0 * sum(w[r] * float(rows[r][column]) for r in w)


def _s3_solved_shock_pct() -> float:
    rows = _read_csv("results/s3_shock_calibration.csv")
    vals = {round(float(r["solved_shock_pct"]), 6) for r in rows}
    if len(vals) != 1:
        raise ClaimResolutionError(f"solved_shock_pct is not uniform across regions: {vals}")
    return vals.pop()


def _total_cropland_mha() -> float:
    import registry as reg  # noqa: WPS433

    return sum(float(reg.region_value("cropland_mha", r)) for r in REGIONS)


def _worst_faostat_gap_pct() -> float:
    rows = _read_csv("results/calibration_production_path.csv")
    return max(abs(float(r["gap_new_pct"])) for r in rows)


def _bnf_ramp_not_wired() -> float:
    """1 if the registry marks `bnf_ramp_years` as declared-but-not-applied."""
    import registry as reg  # noqa: WPS433

    e = reg.entry("bnf_ramp_years")
    return 1.0 if ("superseded_by" in e or "superseded_note" in e) else 0.0


def _water_stress_min_factor() -> float:
    import parameter_registry as pr  # noqa: WPS433

    return float(pr.WATER_STRESS_MIN_FACTOR)


def _yield_min_regional_all_positive() -> float:
    import registry as reg  # noqa: WPS433

    return 1.0 if all(float(reg.region_value("yield_min_regional", r)) > 0 for r in REGIONS) else 0.0


DERIVED: Dict[str, Callable[..., float]] = {
    "figure2_gradient_monotone": _figure2_gradient_monotone,
    "mc_min_buffer_probability": _mc_min_buffer_probability,
    "figure1_margin_gap": _figure1_margin_gap,
    "figure1_margin_at": _figure1_margin_at,
    "registry_n_price": _registry_n_price,
    "priced_region_count": _priced_region_count,
    "priced_regions_match": _priced_regions_match,
    "figS11_spread": _figS11_spread,
    "figS8_reduction_pct": _figS8_reduction_pct,
    "food_price_regional_extreme": _food_price_regional_extreme,
    "s3_global_reduction": _s3_global_reduction,
    "s3_solved_shock_pct": _s3_solved_shock_pct,
    "total_cropland_mha": _total_cropland_mha,
    "worst_faostat_gap_pct": _worst_faostat_gap_pct,
    "bnf_ramp_not_wired": _bnf_ramp_not_wired,
    "water_stress_min_factor": _water_stress_min_factor,
    "yield_min_regional_all_positive": _yield_min_regional_all_positive,
}


def _sel_derived(sel: Mapping[str, Any], artifact: str) -> float:
    name = sel["name"]
    if name not in DERIVED:
        raise ClaimResolutionError(f"no derived resolver named {name!r}")
    return _num(DERIVED[name](**(sel.get("args") or {})), f"derived:{name}")


_KINDS: Dict[str, Callable[[Mapping[str, Any], str], float]] = {
    "csv_cell": _sel_csv_cell,
    "json": _sel_json,
    "registry": _sel_registry,
    "derived": _sel_derived,
}


# --- the public call -------------------------------------------------------

def resolve(check: Mapping[str, Any], claim: Mapping[str, Any]) -> float:
    sel = check["select"]
    kind = sel.get("kind")
    if kind not in _KINDS:
        raise ClaimResolutionError(f"{claim['id']}/{check['id']}: unknown selector kind {kind!r}")
    try:
        return _KINDS[kind](sel, claim.get("artifact", ""))
    except ClaimResolutionError:
        raise
    except Exception as exc:  # noqa: BLE001 -- re-raised with the check named
        raise ClaimResolutionError(f"{claim['id']}/{check['id']}: {exc}") from exc


def evaluate(doc: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Resolve every check. Returns the whole scoring, with nothing rounded away."""
    doc = doc or load_claims()
    results: List[Dict[str, Any]] = []
    for claim in doc["claims"]:
        for check in claim.get("checks", []):
            model = resolve(check, claim)
            stated = float(check["stated"])
            tol = float(check.get("tol", 0.0))
            delta = model - stated
            results.append(
                {
                    "claim": claim["id"],
                    "check": check["id"],
                    "stated": stated,
                    "model": model,
                    "delta": delta,
                    "tol": tol,
                    "units": check.get("units", ""),
                    "verdict": "AGREES" if abs(delta) <= tol + 1e-12 else "DRIFTED",
                }
            )
    drifted_claims = sorted({r["claim"] for r in results if r["verdict"] == "DRIFTED"})
    owed = sorted(
        c["id"] for c in doc["claims"] if c.get("status") in {"owed_generator", "pending_regeneration"}
    )
    return {
        "checks": results,
        "n_checks": len(results),
        "n_agrees": sum(1 for r in results if r["verdict"] == "AGREES"),
        "n_drifted": sum(1 for r in results if r["verdict"] == "DRIFTED"),
        "n_claims": len(doc["claims"]),
        "drifted_claims": drifted_claims,
        "owed_generators": owed,
        "owed_count": len(owed),
    }
