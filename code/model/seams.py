"""Seam contracts: declared aggregation bases for global means.

Finding F-005 (2026-07-25). This repository computed "the global number"
three different ways in three different files and described it in words a
fourth way:

  * ``coupled_econ_biophysical.calibrate_price_shock`` weighted the regional
    fertilizer-demand responses by **nitrogen tonnage**
    (``cropland_mha * synth_n_current``) while its own docstring, and the
    docstring of ``get_scenario_params`` below it, said "area-weighted".
  * ``coupled_econ_biophysical.aggregate_global`` weighted every outcome
    column by **cropland area**.
  * ``code/repro/run_canonical.py`` normalised a third vector inline,
    **production tonnage** (``cropland_mha * y_base``), and that third one is
    what produced the published 2.30 / 3.41 / 3.64 % headline.

Nothing in the code compared them and no output labelled which one it used.
Measured spread on the reported quantities reached 1.87 pp, against a
reporting precision of 0.1 pp.

THE RESOLUTION IS CONCEPTUAL, NOT NUMERICAL
-------------------------------------------
There are two quantity classes, not two competing bases for one quantity.

``outcome_weights()`` returns the single declared basis for outcome shares
(``yield_fraction``, ``soc_fraction``, ``food_price_index``): **production
tonnage**, because a fifth of the world's grain lost in one region is not
offset by a fifth of a region that grows little.

``intensity_weights()`` returns the basis for per-hectare rates
(``fert_applied_kgha``, ``n_mineralized``, ``water_stress``): **cropland
area**, because the global mean of a per-hectare rate is its area-weighted
mean by definition.

``nitrogen_weights()`` is retained for ``calibrate_price_shock`` alone,
because a scenario defined as a reduction in nitrogen *mass* is correctly
weighted by nitrogen mass; weighting a mass-defined scenario by anything
else would mean the scenario does not deliver its own definition.

THE CHECK IS ON CONSTRUCTION, NOT ON INSPECTION
-----------------------------------------------
``SeamD_AggregationWeights`` is a frozen dataclass that validates in
``__post_init__``: weights sum to one, no region silently dropped, no
zero-weight region, and a non-empty provenance string. There is no way to
obtain a weight vector and skip the check.

``assert_same_basis()`` refuses two vectors built on different bases. It
also refuses to be called with the same object N times: the first version of
that assertion in this repository was
``assert_same_basis(*[W_prod for _ in OUTCOME_COLS])``, which passes the same
object repeatedly and therefore cannot fail. An assertion that cannot fail
ratifies rather than tests.

Author: Matthew Wallenstein & Dale Manning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


# ============================================================
# DECLARED BASES
# ============================================================

BASIS_PRODUCTION_TONNAGE = 'production_tonnage'
BASIS_CROPLAND_AREA = 'cropland_area'
BASIS_NITROGEN_TONNAGE = 'nitrogen_tonnage'

KNOWN_BASES = (
    BASIS_PRODUCTION_TONNAGE,
    BASIS_CROPLAND_AREA,
    BASIS_NITROGEN_TONNAGE,
)

#: Columns that are shares of an outcome. Aggregated on production tonnage.
OUTCOME_COLS = ('yield_fraction', 'soc_fraction', 'food_price_index')

#: Columns that are per-hectare rates. Aggregated on cropland area.
INTENSITY_COLS = ('fert_applied_kgha', 'n_mineralized', 'water_stress')

#: Tolerance on the weight-normalisation check.
WEIGHT_SUM_TOL = 1e-12


class SeamContractError(ValueError):
    """A declared aggregation-basis contract was violated."""


# ============================================================
# SEAM D — AGGREGATION WEIGHTS
# ============================================================

@dataclass(frozen=True)
class SeamD_AggregationWeights:
    """A weight vector that carries its own basis and provenance.

    Parameters
    ----------
    regions : tuple of str
        Region keys, in the order the weights are given.
    raw : tuple of float
        Unnormalised weight quantities in the basis's own units
        (t grain, Mha, t N). Retained so that call sites which historically
        accumulated-then-divided can reproduce their arithmetic bit for bit.
    weights : tuple of float
        Normalised weights. Must sum to one.
    basis : str
        One of ``KNOWN_BASES``.
    provenance : str
        Where the raw quantities came from. Must be non-empty: a weight
        vector whose origin is not stated is not auditable.

    Raises
    ------
    SeamContractError
        If the vector is empty, lengths disagree, the basis is undeclared,
        provenance is blank, any region carries zero or negative weight, or
        the normalised weights do not sum to one.
    """

    regions: Tuple[str, ...]
    raw: Tuple[float, ...]
    weights: Tuple[float, ...]
    basis: str
    provenance: str

    def __post_init__(self):
        object.__setattr__(self, 'regions', tuple(self.regions))
        object.__setattr__(self, 'raw', tuple(float(x) for x in self.raw))
        object.__setattr__(self, 'weights', tuple(float(x) for x in self.weights))

        if not self.regions:
            raise SeamContractError('empty weight vector: no regions')
        if len(self.regions) != len(self.weights) or len(self.regions) != len(self.raw):
            raise SeamContractError(
                'length mismatch: %d regions, %d raw, %d weights'
                % (len(self.regions), len(self.raw), len(self.weights)))
        if len(set(self.regions)) != len(self.regions):
            raise SeamContractError('duplicate region in weight vector: %r'
                                    % (self.regions,))
        if self.basis not in KNOWN_BASES:
            raise SeamContractError(
                'undeclared aggregation basis %r; known bases are %r'
                % (self.basis, KNOWN_BASES))
        if not str(self.provenance).strip():
            raise SeamContractError(
                'weight vector on basis %r has no provenance string' % (self.basis,))

        for k, w, r in zip(self.regions, self.weights, self.raw):
            if not np.isfinite(w) or not np.isfinite(r):
                raise SeamContractError('non-finite weight for region %r' % (k,))
            if r <= 0.0:
                raise SeamContractError(
                    'region %r carries zero or negative raw weight (%r) on basis %r; '
                    'a region with no weight is a region silently removed from the mean'
                    % (k, r, self.basis))
            if w <= 0.0:
                raise SeamContractError(
                    'region %r carries zero or negative normalised weight (%r)'
                    % (k, w))

        total = float(np.sum(self.weights))
        if abs(total - 1.0) > WEIGHT_SUM_TOL:
            raise SeamContractError(
                'weights on basis %r sum to %.17g, not 1 (tolerance %g)'
                % (self.basis, total, WEIGHT_SUM_TOL))

    # -- accessors ------------------------------------------------------
    def as_array(self) -> np.ndarray:
        """Normalised weights as a float array, in ``self.regions`` order."""
        return np.asarray(self.weights, dtype=float)

    def raw_array(self) -> np.ndarray:
        """Unnormalised quantities as a float array."""
        return np.asarray(self.raw, dtype=float)

    def for_region(self, region_key: str) -> float:
        """Normalised weight for one region."""
        try:
            return self.weights[self.regions.index(region_key)]
        except ValueError:
            raise SeamContractError(
                'region %r is not in this %s weight vector' % (region_key, self.basis))

    def label(self) -> str:
        """One-line basis label for writing into an output file."""
        return '%s (%s)' % (self.basis, self.provenance)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return ('SeamD_AggregationWeights(basis=%r, n=%d, provenance=%r)'
                % (self.basis, len(self.regions), self.provenance))


# ============================================================
# FACTORIES — THE ONLY SANCTIONED WAY TO OBTAIN A WEIGHT VECTOR
# ============================================================

def _normalise(raw: Sequence[float]) -> Tuple[float, ...]:
    arr = np.asarray(raw, dtype=float)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise SeamContractError('raw weights do not sum to a positive finite total')
    return tuple(arr / total)


def _check_complete(region_keys: Sequence[str],
                    universe: Iterable[str],
                    basis: str) -> None:
    """Fail if a region present in the model is missing from the weights."""
    missing = [k for k in universe if k not in set(region_keys)]
    if missing:
        raise SeamContractError(
            'basis %r drops %d region(s) present in the model: %s'
            % (basis, len(missing), ', '.join(sorted(missing))))
    unknown = [k for k in region_keys if k not in set(universe)]
    if unknown:
        raise SeamContractError(
            'basis %r weights region(s) the model does not have: %s'
            % (basis, ', '.join(sorted(unknown))))


def outcome_weights(region_keys: Sequence[str],
                    y_base: Sequence[float],
                    regions: Mapping[str, object] = None,
                    universe: Iterable[str] = None,
                    provenance: str = None) -> SeamD_AggregationWeights:
    """Weights for outcome shares: production tonnage.

    ``w_i ∝ cropland_mha_i * y_base_i`` — the tonnes of grain a region
    actually grows at its own baseline yield. This is the declared basis for
    ``yield_fraction``, ``soc_fraction`` and ``food_price_index``, and it is
    the basis on which the published headline losses were computed.

    Parameters
    ----------
    region_keys : sequence of str
        Region keys, in the order of ``y_base``.
    y_base : sequence of float
        Year-0 baseline yield (t/ha) for each region, from the model run.
        Passed in rather than recomputed: the production basis depends on
        model output, and silently recalibrating inside a weight factory
        would make the weights a function of whatever the caller had not
        yet run.
    regions : mapping, optional
        ``{region_key: RegionParams}``. Defaults to ``get_default_regions()``.
    universe : iterable of str, optional
        Region keys that must all be present. Defaults to ``regions`` keys.
    """
    if regions is None:
        from soil_n_model import get_default_regions
        regions = get_default_regions()
    region_keys = tuple(region_keys)
    y_base = tuple(float(y) for y in y_base)
    if len(region_keys) != len(y_base):
        raise SeamContractError(
            'outcome_weights: %d region keys but %d baseline yields'
            % (len(region_keys), len(y_base)))
    _check_complete(region_keys, universe if universe is not None else regions.keys(),
                    BASIS_PRODUCTION_TONNAGE)

    raw = tuple(float(regions[k].cropland_mha) * y for k, y in zip(region_keys, y_base))
    return SeamD_AggregationWeights(
        regions=region_keys,
        raw=raw,
        weights=_normalise(raw),
        basis=BASIS_PRODUCTION_TONNAGE,
        provenance=provenance or
        'cropland_mha (soil_n_model.get_default_regions) x year-0 yield_tha '
        '(CoupledMonthlyModel baseline)',
    )


def intensity_weights(region_keys: Sequence[str],
                      regions: Mapping[str, object] = None,
                      universe: Iterable[str] = None,
                      provenance: str = None) -> SeamD_AggregationWeights:
    """Weights for per-hectare rates: cropland area.

    ``w_i ∝ cropland_mha_i``. The global mean of a per-hectare rate is its
    area-weighted mean by definition, so ``fert_applied_kgha``,
    ``n_mineralized`` and ``water_stress`` take this basis and no other.
    """
    if regions is None:
        from soil_n_model import get_default_regions
        regions = get_default_regions()
    region_keys = tuple(region_keys)
    _check_complete(region_keys, universe if universe is not None else regions.keys(),
                    BASIS_CROPLAND_AREA)

    raw = tuple(float(regions[k].cropland_mha) for k in region_keys)
    return SeamD_AggregationWeights(
        regions=region_keys,
        raw=raw,
        weights=_normalise(raw),
        basis=BASIS_CROPLAND_AREA,
        provenance=provenance or
        'cropland_mha (soil_n_model.get_default_regions)',
    )


def nitrogen_weights(region_keys: Sequence[str],
                     regions: Mapping[str, object] = None,
                     universe: Iterable[str] = None,
                     provenance: str = None) -> SeamD_AggregationWeights:
    """Weights for a scenario defined as a reduction in nitrogen mass.

    ``w_i ∝ cropland_mha_i * synth_n_current_i``. Used by
    ``calibrate_price_shock`` alone. A 20% reduction in applied nitrogen is
    exactly 20% only on this basis; on the production basis the same shock
    delivers 19.56%. Both numbers are correct, and a paper that prints them
    in the same sentence owes the reader the basis of each.
    """
    if regions is None:
        from soil_n_model import get_default_regions
        regions = get_default_regions()
    region_keys = tuple(region_keys)
    _check_complete(region_keys, universe if universe is not None else regions.keys(),
                    BASIS_NITROGEN_TONNAGE)

    raw = tuple(float(regions[k].cropland_mha) * float(regions[k].synth_n_current)
                for k in region_keys)
    return SeamD_AggregationWeights(
        regions=region_keys,
        raw=raw,
        weights=_normalise(raw),
        basis=BASIS_NITROGEN_TONNAGE,
        provenance=provenance or
        'cropland_mha x synth_n_current (soil_n_model.get_default_regions)',
    )


# ============================================================
# CROSS-VECTOR CONTRACT
# ============================================================

def assert_same_basis(*vectors: SeamD_AggregationWeights,
                      context: str = '') -> str:
    """Refuse two weight vectors that are not on the same declared basis.

    Returns the shared basis string so that a caller can label its output
    with it.

    Guards, in order:

    1. At least two vectors, and at least two *distinct objects*. Passing the
       same object N times cannot fail and therefore is not a check. This is
       the exact vacuous assertion that was written and deleted during F-005.
    2. Every argument is a ``SeamD_AggregationWeights`` (so it has already
       passed construction validation).
    3. All bases equal.
    4. All region orderings equal — two vectors on the same basis in
       different orders will silently mismatch a column-wise dot product.
    """
    where = (' [%s]' % context) if context else ''
    if len(vectors) < 2:
        raise SeamContractError(
            'assert_same_basis needs at least two vectors, got %d%s'
            % (len(vectors), where))
    if len({id(v) for v in vectors}) < 2:
        raise SeamContractError(
            'assert_same_basis was given the same object %d times%s; an '
            'assertion that compares an object with itself cannot fail'
            % (len(vectors), where))
    for v in vectors:
        if not isinstance(v, SeamD_AggregationWeights):
            raise SeamContractError(
                'assert_same_basis got %r, which is not a checked weight vector%s'
                % (type(v).__name__, where))

    bases = {v.basis for v in vectors}
    if len(bases) != 1:
        raise SeamContractError(
            'aggregation bases disagree%s: %s' % (where, ', '.join(sorted(bases))))

    orders = {v.regions for v in vectors}
    if len(orders) != 1:
        raise SeamContractError(
            'weight vectors share basis %r but not region order%s'
            % (vectors[0].basis, where))

    return vectors[0].basis


def basis_for_column(column: str) -> str:
    """Declared basis for a named output column.

    Raises rather than guessing: a column with no declared class has no
    global mean until someone declares one.
    """
    if column in OUTCOME_COLS:
        return BASIS_PRODUCTION_TONNAGE
    if column in INTENSITY_COLS:
        return BASIS_CROPLAND_AREA
    raise SeamContractError(
        'column %r has no declared aggregation basis; add it to OUTCOME_COLS '
        'or INTENSITY_COLS in code/model/seams.py and say why' % (column,))


__all__ = [
    'BASIS_PRODUCTION_TONNAGE', 'BASIS_CROPLAND_AREA', 'BASIS_NITROGEN_TONNAGE',
    'KNOWN_BASES', 'OUTCOME_COLS', 'INTENSITY_COLS',
    'SeamContractError', 'SeamD_AggregationWeights',
    'outcome_weights', 'intensity_weights', 'nitrogen_weights',
    'assert_same_basis', 'basis_for_column',
]
