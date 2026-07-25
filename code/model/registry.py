"""registry.py -- the loader that makes params.yaml drive the model.

v15. Reconstructed 2026-07-25 (WP1) after the v15 working tree was lost.

WHAT CHANGED IN v15 (F-006 -> F-011). Before v15 this module was imported by
four files and supplied exactly two things: ``soc_tha_per_pct()`` and
``value('residue_c_to_active_fraction')``. Every other registered value was
restated as a literal in the model, and a mirror test compared the two. The
registry documented the model; it did not drive it. Perturbing a registry entry
therefore changed no published number and broke only the mirror test, which is
why mutation coverage scored 45 of 56 leaves DECLARED_NOT_WIRED.

The direction of authority is now reversed. ``soil_n_model.py``,
``coupled_econ_biophysical.py``, ``prices.py`` and ``monthly_model_v3.py`` read
their constants from here at import. Editing ``params.yaml`` changes the model.

THREE CONSTRAINTS ARE ENFORCED AT LOAD, NOT ASSERTED BY A TEST.

  1. ``som_pool_fractions``  f_active + f_slow + f_passive == 1
  2. ``cre_allocation``      cre_to_active + cre_to_slow == 1
  3. ``whc_sensitivity``     its units string must name the registered
                             ``soc_profile_depth_cm``

Six mutable leaves are refused by those three checks -- the three pool
fractions, the two allocation shares, and ``soc_profile_depth_cm`` itself.
A load-time refusal is stronger than a test: a perturbation that would make the
model incoherent cannot reach the model at all. The mutation harness scores
these six GUARDED_AT_LOAD.

RECONSTRUCTION NOTE. Two registry values could not be recovered in full from
the surviving artifacts and are recorded as gaps rather than invented; see
``v15/RECONSTRUCTION_GAPS.md`` and the ``documented_as`` keys in params.yaml.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

__all__ = [
    "PARAMS_PATH",
    "REGIONS",
    "ALLOWED_KEYS",
    "DOCUMENTARY_KEYS",
    "RegistryError",
    "entry",
    "value",
    "leaf",
    "region_value",
    "region_fields",
    "regional_map",
    "units",
    "category",
    "mc_status",
    "uncertainty",
    "affects_claims",
    "benchmark_ids",
    "names",
    "leaves",
    "soc_tha_per_pct",
    "global_aggregation_weights",
    "reload",
]


PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.yaml")


class RegistryError(ValueError):
    """Raised at load time. The registry refuses to hand the model a value it
    knows to be incoherent."""


# Every key an entry may carry. An unknown key is a typo or an undeclared
# convention, and either way the registry should not silently accept it.
# ``superseded_by`` and ``superseded_note`` were added by F-007 so that
# bnf_potential and bnf_ramp_years could be kept rather than deleted: deleting
# them would erase the fact that the manuscript still describes the superseded
# mechanism.
ALLOWED_KEYS = frozenset({
    "value",
    "units",
    "category",
    "source",
    "used_by",
    "affects_claims",
    "benchmark",
    "mc",
    "mc_exempt_reason",
    "uncertainty",
    "derived_from",
    "superseded_by",
    "superseded_note",
    "documented_as",
    "guarded_at_load",
})

# Keys that carry prose or provenance rather than a number the model reads.
# ``build.params_fingerprint()`` (WP6) hashes the registry with these removed,
# so editing a source string or a limitation sentence does not mark every
# downstream artifact stale.
DOCUMENTARY_KEYS = frozenset({
    "source",
    "used_by",
    "affects_claims",
    "benchmark",
    "mc_exempt_reason",
    "superseded_by",
    "superseded_note",
    "documented_as",
})

ALLOWED_CATEGORIES = frozenset({"measured", "calibrated", "judgment", "derived"})
ALLOWED_MC = frozenset({"drawn", "declared_fixed", "exempt", "not_applicable"})

# Entries whose value is a mapping of leaf name -> number rather than a mapping
# of region -> number. Everything else with a mapping value is regional.
_NAMED_LEAF_ENTRIES = frozenset({
    "som_decay_rates",
    "som_humification",
    "som_pool_cn",
    "som_pool_fractions",
    "laub_tropical_ratios",
    "cre_allocation",
})

_SUM_TO_ONE = {
    "som_pool_fractions": ("f_active", "f_slow", "f_passive"),
    "cre_allocation": ("cre_to_active", "cre_to_slow"),
}
_SUM_TO_ONE_TOL = 1e-12


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

_DOC: Dict[str, Any] = {}
_P: Dict[str, Dict[str, Any]] = {}
REGIONS: List[str] = []


def _check_schema(params: Mapping[str, Any]) -> None:
    for name, e in params.items():
        if not isinstance(e, dict):
            raise RegistryError(f"{name}: entry must be a mapping, got {type(e).__name__}")
        unknown = set(e) - ALLOWED_KEYS
        if unknown:
            raise RegistryError(f"{name}: unknown key(s) {sorted(unknown)}")
        if "value" not in e:
            raise RegistryError(f"{name}: no value")
        cat = e.get("category")
        if cat not in ALLOWED_CATEGORIES:
            raise RegistryError(f"{name}: category {cat!r} not in {sorted(ALLOWED_CATEGORIES)}")
        mc = e.get("mc")
        if mc not in ALLOWED_MC:
            raise RegistryError(f"{name}: mc {mc!r} not in {sorted(ALLOWED_MC)}")
        # F-007: an unpropagated uncertainty is a sentence the limitations owe
        # the reader, so a non-drawn entry must say why it is not drawn.
        if mc != "drawn" and not e.get("mc_exempt_reason"):
            raise RegistryError(f"{name}: mc={mc} with no mc_exempt_reason")
        if mc == "drawn" and not e.get("uncertainty"):
            raise RegistryError(f"{name}: mc=drawn with no uncertainty block")
        if cat == "derived" and not e.get("derived_from"):
            raise RegistryError(f"{name}: category=derived with no derived_from")


def _check_sum_to_one(params: Mapping[str, Any]) -> None:
    """Constraints 1 and 2. Five of the six GUARDED_AT_LOAD leaves."""
    for name, leaf_names in _SUM_TO_ONE.items():
        v = params[name]["value"]
        missing = [k for k in leaf_names if k not in v]
        if missing:
            raise RegistryError(f"{name}: missing leaf(s) {missing}")
        total = sum(float(v[k]) for k in leaf_names)
        if abs(total - 1.0) > _SUM_TO_ONE_TOL:
            raise RegistryError(
                f"{name}: {' + '.join(leaf_names)} = {total!r}, must sum to 1 "
                f"(within {_SUM_TO_ONE_TOL}). A partition that does not sum to "
                f"one silently creates or destroys the quantity it partitions, "
                f"so the registry refuses to hand it to the model."
            )


def _check_profile_depth_units(params: Mapping[str, Any]) -> None:
    """Constraint 3. The sixth GUARDED_AT_LOAD leaf.

    ``whc_sensitivity`` is millimetres of plant-available water per percentage
    point of SOC *over a stated profile depth*. The depth is registered once,
    as ``soc_profile_depth_cm``. If the two disagree the sensitivity means
    something other than what the model does with it, which is F-005's error
    class: a number recorded without its basis.
    """
    depth = params["soc_profile_depth_cm"]["value"]
    if float(depth) <= 0:
        raise RegistryError(f"soc_profile_depth_cm: {depth!r} must be positive")
    token = "%gcm_profile" % float(depth)
    u = str(params["whc_sensitivity"].get("units", ""))
    if token not in u:
        raise RegistryError(
            f"whc_sensitivity: units {u!r} does not name the registered "
            f"soc_profile_depth_cm of {depth} (expected the token {token!r}). "
            f"The profile depth is the basis of this number; a sensitivity "
            f"quoted over an unstated or a different depth is not the same "
            f"quantity."
        )


def _load(path: str = PARAMS_PATH) -> None:
    global _DOC, _P, REGIONS
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or "parameters" not in doc:
        raise RegistryError(f"{path}: no 'parameters' mapping")
    params = doc["parameters"]
    regions = list(doc.get("regions") or [])
    if len(regions) != 8:
        raise RegistryError(f"{path}: expected 8 regions, got {len(regions)}")

    _check_schema(params)
    _check_sum_to_one(params)
    _check_profile_depth_units(params)

    _DOC, _P, REGIONS = doc, params, regions


def reload(path: str = PARAMS_PATH) -> None:
    """Re-read the registry. Used by the mutation harness, which perturbs one
    leaf in a sandbox copy of params.yaml and re-imports the model."""
    _load(path)


_load()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def names() -> List[str]:
    return list(_P)


def entry(name: str) -> Dict[str, Any]:
    try:
        return _P[name]
    except KeyError:
        raise RegistryError(f"{name!r} is not a registered parameter") from None


def value(name: str) -> Any:
    return entry(name)["value"]


def leaf(name: str, leaf_name: str) -> Any:
    """A named leaf of a multi-part entry, e.g. leaf('som_decay_rates', 'k_slow')."""
    v = entry(name)["value"]
    if not isinstance(v, dict) or leaf_name not in v:
        raise RegistryError(f"{name}: no leaf {leaf_name!r}")
    return v[leaf_name]


def region_value(name: str, region_key: str) -> Any:
    """The value of ``name`` for ``region_key``.

    A scalar entry returns that scalar for every region -- whc_sensitivity is
    registered once and applies to all eight -- so a caller does not have to
    know which registry entries happen to be regionalised.
    """
    e = entry(name)
    v = e["value"]
    if isinstance(v, dict) and name not in _NAMED_LEAF_ENTRIES:
        if region_key not in v:
            raise RegistryError(
                f"{name}: no value for region {region_key!r}. See the "
                f"'documented_as' key on this entry if it records a "
                f"reconstruction gap."
            )
        return v[region_key]
    return v


def regional_map(name: str) -> Dict[str, Any]:
    """The whole region -> value mapping, broadcast if the entry is scalar."""
    return {rk: region_value(name, rk) for rk in REGIONS}


def region_fields(region_key: str, field_names: Sequence[str]) -> Dict[str, Any]:
    """Keyword arguments for one region, in registry order.

    This is what ``soil_n_model.get_default_regions`` calls. The eight regions'
    seventeen quantitative fields were literals in that function until v15.
    """
    if region_key not in REGIONS:
        raise RegistryError(f"{region_key!r} is not a registered region")
    return {f: region_value(f, region_key) for f in field_names}


def units(name: str) -> str:
    return str(entry(name).get("units", ""))


def category(name: str) -> str:
    return str(entry(name).get("category", ""))


def mc_status(name: str) -> str:
    return str(entry(name).get("mc", ""))


def uncertainty(name: str) -> Optional[Dict[str, Any]]:
    return entry(name).get("uncertainty")


def affects_claims(name: str) -> List[str]:
    return list(entry(name).get("affects_claims") or [])


def benchmark_ids(name: str) -> List[str]:
    return list(entry(name).get("benchmark") or [])


def _is_bounds(v: Any) -> bool:
    """True for an interval, or a region -> interval table."""
    if isinstance(v, (list, tuple)):
        return True
    if isinstance(v, dict):
        return bool(v) and all(isinstance(x, (list, tuple)) for x in v.values())
    return False


def leaves() -> List[str]:
    """Every mutable leaf, as 'name' or 'name.leaf'.

    The mutation harness (WP3) enumerates the registry through this function,
    so what counts as a leaf is defined once, here, rather than in the harness.

    Two kinds of entry are not leaves. Derived entries carry no independent
    value. Bounds declarations -- an interval, or a region -> interval table --
    are the prior's own support rather than a quantity the model reads, so
    perturbing one states a different prior rather than a different model.
    That excludes cost_share_band, crop_price_bounds and n_price_wedge_bounds
    and leaves 56 mutable leaves.
    """
    out: List[str] = []
    for name, e in _P.items():
        if e.get("category") == "derived":
            continue
        v = e["value"]
        if _is_bounds(v):
            continue
        if isinstance(v, dict) and name in _NAMED_LEAF_ENTRIES:
            out.extend(f"{name}.{k}" for k in v)
        else:
            out.append(name)
    return sorted(out)


# ---------------------------------------------------------------------------
# Derived quantities. These live here rather than in params.yaml because a
# derived value that can be edited independently of its factors is a second
# statement of the same fact, which is the condition F-007 was written about.
# ---------------------------------------------------------------------------

def soc_tha_per_pct() -> float:
    """t C/ha per percentage point SOC over the registered profile depth."""
    return (
        float(value("soc_bulk_density"))
        * float(value("soc_profile_depth_cm"))
        * float(value("cm2_per_ha"))
        / float(value("g_per_t"))
        * float(value("pct_to_fraction"))
    )


def global_aggregation_weights() -> Dict[str, float]:
    """AREA-basis weights.

    F-005: the paper reports global numbers on three different bases -- area,
    nitrogen and production -- which differ by up to 1.87 percentage points on
    a reported quantity. This function is the area basis and nothing else.
    ``seams.outcome_weights`` is the single place a basis is chosen; do not
    call this directly to weight an outcome.
    """
    area = regional_map("cropland_mha")
    total = float(sum(area.values()))
    return {rk: float(a) / total for rk, a in area.items()}
