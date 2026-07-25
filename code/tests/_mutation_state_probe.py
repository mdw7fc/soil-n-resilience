#!/usr/bin/env python3
"""Dump the model's registry-fed state as a flat numeric fingerprint.

Run inside a sandbox by ``run_mutation_coverage.py``. Importing the model is
enough: the four rewired modules read the registry at import (F-011), so the
objects below are exactly what the registry supplies. Nothing here runs a
scenario -- this is the *input* face of the model, and the difference between
this fingerprint and the published one is what separates a parameter the
registry fails to deliver from one it delivers to no effect.

Writes JSON to argv[1]. Exit 2 means the registry refused the mutated file at
load, which the caller scores GUARDED_AT_LOAD.
"""
import os, sys, json, warnings, dataclasses

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))

OUT = sys.argv[1]


def _flat(obj, prefix, sink):
    """Every float reachable from obj, keyed by path. Strings and bools are
    ignored: a mutation harness scores numbers."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        sink[prefix] = float(obj)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            _flat(getattr(obj, f.name, None), f"{prefix}.{f.name}", sink)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flat(v, f"{prefix}.{k}", sink)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flat(v, f"{prefix}[{i}]", sink)


def main():
    state = {}
    try:
        import registry as R
    except Exception as exc:                      # registry refused the value
        json.dump({"guarded_at_load": True, "error": f"{type(exc).__name__}: {exc}"},
                  open(OUT, "w"))
        return 2

    import soil_n_model as S
    import coupled_econ_biophysical as C
    import monthly_model_v3 as M

    _flat(S.get_default_regions(), "regions", state)
    _flat(C.REGIONAL_ECON_PARAMS, "econ", state)
    _flat(getattr(M, "FAOSTAT_TARGETS", {}), "faostat", state)

    for mod, attr in ((S, "FeedbackParams"), (S, "SOMPoolParams")):
        cls = getattr(mod, attr, None)
        if cls is None:
            continue
        try:                                       # default construction only
            _flat(cls(), f"cls.{attr}", state)
        except Exception:
            pass

    # Derived quantities that live in registry.py rather than params.yaml.
    for fn in ("soc_tha_per_pct", "global_aggregation_weights"):
        f = getattr(R, fn, None)
        if callable(f):
            try:
                _flat(f(), f"derived.{fn}", state)
            except Exception:
                pass

    # The price seam. These move gross margins, which the canonical artifact
    # does not carry -- see the INERT caveat in F-011.
    try:
        import prices as P
        for fn in ("nitrogen_price_in_yield_units", "n_price_usd_kg"):
            f = getattr(P, fn, None)
            if not callable(f):
                continue
            for rk in R.REGIONS:
                try:
                    _flat(f(rk), f"prices.{fn}.{rk}", state)
                except Exception:
                    pass
    except Exception:
        pass

    json.dump({"guarded_at_load": False, "state": state}, open(OUT, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
