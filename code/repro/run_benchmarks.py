#!/usr/bin/env python3
"""Score the model against external evidence it was never fitted to.

Spec: FINDINGS.md F-008. Rebuilt by WP4 on 2026-07-26 after the v15
working tree was lost.

WHAT THIS IS FOR
----------------
Every other test in this repository compares the model to itself: to a
calibration fingerprint, a registry entry, a seam contract, a previous
canonical run. Those catch drift. None of them can catch a model that is
internally consistent and externally wrong, which is what F-002 and F-004
turned out to be. This is the only check of the other kind.

Nothing in code/model/params.yaml was fitted to anything in
data/benchmarks/observed_values.yaml. The model's only calibration target
is `faostat_yield_target`, which appears in no benchmark source. Every row
is therefore held out by construction, and the planned "calibrate on
Broadbalk, then predict Morrow and Sanborn" split was unnecessary.

    THE MODEL MUST NOT BE TUNED TO ANY ROW IN THIS SUITE.

Fitting to these rows would convert the only external check this project
has into calibration data and would reproduce, at a larger scale, the
error F-002 records. If a row fails, the failure is reported. See
`data/benchmarks/baseline_verdicts.json` and
`code/tests/test_benchmark_baseline.py`, which freeze the verdicts so a
failure is tracked without having to be fixed first.

TWO COLUMNS, NOT ONE
--------------------
Every row carries `informativeness` beside `verdict`, because a benchmark
can be passed for reasons that have nothing to do with the model being
right.

    STRONG  the model quantity and the observed quantity are the same
            quantity, measured at a matched elapsed horizon, and where it
            matters at a matched nitrogen rate.
    WEAK    the row compares real things but something is mismatched — the
            horizon, the nitrogen rate, the quantity kind (a derivative
            against an average), or the verdict follows from the model's
            initialisation rather than from its dynamics.
    NONE    the row carries no verdict at all: an inversion, a diagnostic,
            or an observation the compilation owes.

B2-EUROPE-DRIFT-FERT is the worked example of why this column exists. It
reports a 96-year soil carbon drift of -0.21 percent under unchanged
management against an observed envelope of -8.3 to +4.5 percent. That
looks like a pass and is worth nothing: the engine is initialised at a
spun-up equilibrium, so near-zero drift is a property of the
initialisation, and any model with a spin-up would reproduce it. It is
marked WEAK and carries no verdict.

COMPARATORS EXAMINED AND REJECTED
---------------------------------
Recorded here rather than dropped silently, so a reader can see the suite
was not assembled by keeping whatever agreed. Full reasons are in
`observed_values.yaml` under `rejected_comparators`.

  Broadbalk plot 3 time series  Nil since 1843 and drifting around its own
      low equilibrium. Not a fertilizer-withdrawal transient, which is the
      quantity the model produces.
  Morrow unfertilized  A prairie-conversion legacy the model does not
      represent; the unfertilized trajectory is dominated by the loss of
      native prairie carbon.
  Morrow 1964  79.87 against 58.0 Mg C/ha, nine years after fertilization
      began on that plot. A 37.7 percent carbon difference cannot be a
      fertilization response over nine years.
  Bad Lauchstadt reversal rates  Both arms involve farmyard manure and the
      model has no organic amendment pathway.
  Absolute soil carbon stocks  An initialisation in this model, not a
      prediction (F-001). Comparing them measures the choice of initial
      condition.

RECONSTRUCTION NOTE
-------------------
F-008 records 41 rows scoring 11 PASS / 3 MARGINAL / 1 FAIL /
18 INFORMATIVE / 7 OWED / 1 NOT_APPLICABLE, and its prose names and
describes 16 of them. The other 25 are not individually recoverable from
any surviving artifact. This suite carries every row F-008 names plus the
per-region extensions its own naming convention implies, and reports what
that actually scores. The shortfall is reconciled row by row in
`results/benchmark_reconciliation.md` rather than closed by inventing
rows until the count reaches 41.

Usage:
    python3 code/repro/run_benchmarks.py [--out-dir OUTPUTS]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'model'))

import yaml  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from coupled_monthly import MonthlyBiophysicalEngine, get_calibrated_ym  # noqa: E402
from soil_n_model import get_default_regions  # noqa: E402
import registry as _reg  # noqa: E402

OBSERVED_PATH = os.path.join(ROOT, 'data', 'benchmarks', 'observed_values.yaml')

REGIONS = ['north_america', 'europe', 'east_asia', 'south_asia',
           'southeast_asia', 'latin_america', 'sub_saharan_africa',
           'fsu_central_asia']

# Horizons in elapsed years from a common spun-up equilibrium.
#   1   the first year of the transient, which is what F-004's crop
#       calendar moves and what the manuscript's year-1 loss reports
#   30  the manuscript's own horizon, and the matched horizon for the
#       Prague-Ruzyne 1983-2020 window
#   96  the long tail, for comparison with century-scale experiments
HORIZONS = (1, 30, 96)

PASS, MARGINAL, FAIL = 'PASS', 'MARGINAL', 'FAIL'
INFORMATIVE, OWED, NOT_APPLICABLE = 'INFORMATIVE', 'OWED', 'NOT_APPLICABLE'
STRONG, WEAK, NONE = 'STRONG', 'WEAK', 'NONE'

# A model value outside the observed band but within this factor of the
# nearer band edge is MARGINAL rather than FAIL. The band edge is itself an
# estimate carrying its own error, so a small excursion is not evidence of
# disagreement; a large one is.
MARGINAL_FACTOR = 1.5


# =====================================================================
# Row container
# =====================================================================

@dataclass
class Row:
    row_id: str
    family: str
    quantity: str
    model_value: Optional[float]
    observed_low: Optional[float]
    observed_high: Optional[float]
    units: str
    horizon_years: Optional[int]
    n_rate_kg_ha: Optional[float]
    verdict: str
    informativeness: str
    note: str
    source: str = ''
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('extra')
        for k, v in self.extra.items():
            d[k] = v
        return d


def band_verdict(model: float, low: float, high: float,
                 tol: float = 0.0) -> str:
    """PASS inside the band, MARGINAL just outside it, FAIL beyond.

    `tol` widens the band by the observation's own reporting precision.
    Widening it for any other reason is forbidden: F-008's standing rule is
    that tolerances come from each observation's stated precision, never
    from what would make a row agree.
    """
    lo, hi = min(low, high) - tol, max(low, high) + tol
    if lo <= model <= hi:
        return PASS
    edge = lo if model < lo else hi
    if edge == 0:
        return FAIL
    # Multiplicative closeness, and it must be two-sided: a model value at
    # 0.6 times the lower edge is as far outside as one at 1.6 times the
    # upper edge. An earlier one-sided form scored B3-europe-YR30 MARGINAL,
    # which would have hidden the suite's only failure.
    ratio = model / edge
    if (1.0 / MARGINAL_FACTOR) <= ratio <= MARGINAL_FACTOR:
        return MARGINAL
    # Also treat an excursion small in absolute terms relative to the band
    # width as MARGINAL, so a band straddling zero behaves sensibly.
    width = abs(hi - lo)
    if width > 0 and abs(model - edge) <= (MARGINAL_FACTOR - 1.0) * width:
        return MARGINAL
    return FAIL


# =====================================================================
# Model side: one spun-up equilibrium per region, then paired trajectories
# =====================================================================

class RegionRuns:
    """Nil-N and current-rate trajectories from one common equilibrium.

    Both arms start from the same spun-up pools, which is what makes this a
    paired-plot design rather than two independent runs. Spinning up each
    arm separately would compare two different equilibria and would not be
    the quantity a long-term experiment measures.
    """

    def __init__(self, region_key: str, regions=None, n_years: int = max(HORIZONS)):
        regions = regions or get_default_regions()
        self.region_key = region_key
        self.region = regions[region_key]
        self.ym = get_calibrated_ym(region_key)
        base = MonthlyBiophysicalEngine(self.region, region_key=region_key,
                                        yield_max_override=self.ym)
        self.pools = {
            'c_active': base.C_active, 'c_slow': base.C_slow,
            'c_passive': base.C_passive, 'soc': base.soc_initial,
            'mineral_n': base.mineral_n, 'yield_eq': base.yield_baseline,
            'n_min_eq': base.n_min_baseline,
        }
        self.current_rate = float(self.region.synth_n_current)
        self.n_years = n_years
        self._cache: Dict[float, List[Dict]] = {}
        self.fert = self.trajectory(self.current_rate)
        self.nil = self.trajectory(0.0)

    def trajectory(self, rate: float, n_years: Optional[int] = None) -> List[Dict]:
        n_years = n_years or self.n_years
        key = (round(rate, 9), n_years)
        if key in self._cache:
            return self._cache[key]
        eng = MonthlyBiophysicalEngine(self.region, region_key=self.region_key,
                                       yield_max_override=self.ym,
                                       initial_pools=dict(self.pools))
        out = [eng.step(fert_applied=rate) for _ in range(n_years)]
        self._cache[key] = out
        return out

    # -- B3 -----------------------------------------------------------
    def nil_ratio(self, year: int) -> float:
        return self.nil[year - 1]['yield_tha'] / self.fert[year - 1]['yield_tha']

    def nil_ratio_at_rate(self, rate: float, year: int) -> float:
        return (self.trajectory(0.0, year)[year - 1]['yield_tha']
                / self.trajectory(rate, year)[year - 1]['yield_tha'])

    def implied_rate_for_ratio(self, target: float, year: int = 1) -> Optional[float]:
        try:
            return brentq(lambda x: self.nil_ratio_at_rate(x, year) - target,
                          0.5, 500.0, xtol=1e-4)
        except (ValueError, RuntimeError):
            return None

    # -- B2 -----------------------------------------------------------
    def soc_drift_pct(self, year: int, nil: bool = False) -> float:
        arm = self.nil if nil else self.fert
        return 100.0 * (arm[year - 1]['soc_total'] / self.pools['soc'] - 1.0)

    def fert_minus_nil_excess_pct(self, year: int) -> float:
        """Soil carbon under fertilization as an excess over soil carbon
        without it, at a matched elapsed year. A difference, not a level:
        absolute stocks in this model are an initialisation (F-001)."""
        return 100.0 * (self.fert[year - 1]['soc_total']
                        / self.nil[year - 1]['soc_total'] - 1.0)

    # -- B1 -----------------------------------------------------------
    def yield_at_rate(self, rate: float, year: int = 1) -> float:
        return self.trajectory(max(0.0, rate), year)[year - 1]['yield_tha']

    def mpp(self, rate: float, year: int = 1, h: float = 0.01) -> float:
        """Marginal physical product of applied N, kg grain per kg N.

        A central difference on the first year's yield from the common
        equilibrium. This is dY/dN_applied, a derivative — not
        (Y_fert - Y_control)/N, which is an average. For a concave response
        the average exceeds the derivative at any positive N, so the two
        must not be compared to each other. Rows that do are marked WEAK.

        The step is small enough that the result is insensitive to it:
        h = 0.5 and h = 0.01 agree to five significant figures.
        """
        lo_rate = max(0.0, rate - h)
        hi_rate = rate + h
        lo = self.yield_at_rate(lo_rate, year)
        hi = self.yield_at_rate(hi_rate, year)
        return 1000.0 * (hi - lo) / (hi_rate - lo_rate)

    def implied_rate_for_mpp(self, target: float, year: int = 1) -> Optional[float]:
        try:
            return brentq(lambda x: self.mpp(x, year) - target, 0.05, 500.0,
                          xtol=1e-4)
        except (ValueError, RuntimeError):
            return None


# =====================================================================
# Families
# =====================================================================

def build_b1(runs: Dict[str, RegionRuns], obs: Dict) -> List[Row]:
    rows: List[Row] = []
    b1 = obs['B1']
    ssa = runs['sub_saharan_africa']
    env = b1['ssa_on_farm_mpp_envelope']
    rate = ssa.current_rate

    model_mpp = ssa.mpp(rate)
    rows.append(Row(
        row_id='B1-SSA-MPP', family='B1',
        quantity='marginal physical product of applied N at the regional mean rate',
        model_value=round(model_mpp, 4),
        observed_low=env['low'], observed_high=env['high'],
        units='kg_grain_per_kg_N', horizon_years=1, n_rate_kg_ha=rate,
        verdict=band_verdict(model_mpp, env['low'], env['high']),
        informativeness=WEAK,
        note=(
            'WEAK for three reasons, any one of which is disqualifying for a '
            'STRONG verdict. (1) The rate is not matched: a marginal product '
            'is a derivative at a point, 7 kg N/ha is the steepest part of the '
            'response curve, and the surveys measured wherever those farmers '
            'actually were. (2) The envelope mixes true marginal products '
            '(Sheahan, Liverpool-Tasie) with agronomic efficiencies '
            '(Vanlauwe, Ichami, Chivenge), which are averages; for a concave '
            'response the average exceeds the derivative, biasing the envelope '
            'upward. (3) observed_values.yaml records the envelope itself as '
            'UNSOURCED — no publication states it. See B1-SSA-IMPLIED-RATE-'
            'RANGE, which settles the comparison without inventing an '
            'observation.'),
        source=env['source'],
        extra={'quantity_kind': 'derivative',
               'observed_status': env['verification']['status']}))

    r20 = ssa.implied_rate_for_mpp(env['high'])
    r_low = ssa.implied_rate_for_mpp(env['low'])
    rows.append(Row(
        row_id='B1-SSA-IMPLIED-RATE-RANGE', family='B1',
        quantity='N rate at which the model reproduces each end of the observed MPP envelope',
        model_value=None, observed_low=None, observed_high=None,
        units='kg_N_per_ha', horizon_years=1, n_rate_kg_ha=None,
        verdict=INFORMATIVE, informativeness=NONE,
        note=(
            'An inversion, not a test — it carries no verdict by construction. '
            'It settles B1-SSA-MPP without inventing an observation: the '
            'Kenyan estimates correspond to model rates in the mid-twenties '
            'kg N/ha, which is where Kenyan maize farmers who fertilize '
            'actually are (Sheahan et al. put the sample mean at 25.2). The '
            'Nigerian plot-level estimate would require a rate near the '
            'model\'s upper root, which those farmers are not at '
            '(Liverpool-Tasie et al. report 40-47 kg N/ha). So the model\'s '
            'nitrogen response in SSA agrees with the Kenyan on-farm evidence '
            'and disagrees with the single Nigerian estimate, which is the one '
            'whose gap most plausibly reflects constraints this model does not '
            'represent.'),
        source=env['source'],
        extra={'implied_rate_at_observed_high': None if r20 is None else round(r20, 4),
               'implied_rate_at_observed_low': None if r_low is None else round(r_low, 4),
               'kenya_survey_mean_rate': b1['kenya_mpp_overall']['n_rate_kg_ha'],
               'nigeria_survey_mean_rate_2010': b1['nigeria_mpp_plot_level']['n_rate_kg_ha_2010'],
               'nigeria_survey_mean_rate_2012': b1['nigeria_mpp_plot_level']['n_rate_kg_ha_2012']}))
    return rows


def build_b2(runs: Dict[str, RegionRuns], obs: Dict) -> List[Row]:
    rows: List[Row] = []
    b2 = obs['B2']
    eu = runs['europe']
    excess = b2['broadbalk_fert_minus_nil_excess']
    ov = excess['value']
    lo, hi = ov / 2.0, ov * 2.0   # half-to-twice a single observed value

    for horizon, informativeness in ((96, STRONG), (30, STRONG)):
        m = eu.fert_minus_nil_excess_pct(horizon)
        rows.append(Row(
            row_id=f'B2-BROADBALK-FERT-MINUS-NIL-YR{horizon}', family='B2',
            quantity='soil carbon under fertilization as an excess over soil carbon without it',
            model_value=round(m, 4), observed_low=round(lo, 4), observed_high=round(hi, 4),
            units='percent', horizon_years=horizon, n_rate_kg_ha=144.0,
            verdict=band_verdict(m, lo, hi), informativeness=informativeness,
            note=(
                'Scored against a half-to-twice band because the observation is '
                'a single value, not an interval. The comparison is a '
                'difference, never a level: absolute soil carbon in this model '
                'is an initialisation and not a prediction (F-001), so the '
                'model\'s 42 Mg C/ha temperate start and Broadbalk\'s 28.8 are '
                'not comparable quantities and their difference under a '
                'treatment contrast is.'
                + (' At the manuscript\'s own 30-year horizon the model is '
                   'inside the band; the discrepancy is in the extrapolation, '
                   'not at the published horizon.' if horizon == 30 else
                   ' This row and B3-europe-YR30 agree on direction and roughly '
                   'on magnitude, at two sites, in two different quantities, '
                   'one a yield and one a carbon stock. They are one mechanism '
                   'seen twice, not two independent failures.')),
            source=excess['source'],
            extra={'quantity_kind': 'ratio',
                   'observed_point_value': ov,
                   'band_rule': 'half_to_twice',
                   'observed_status': excess['verification']['status'],
                   'denominator_disputed': True}))

    drift = b2['broadbalk_mineral_plot_drift_envelope']
    m_fert = eu.soc_drift_pct(96, nil=False)
    rows.append(Row(
        row_id='B2-EUROPE-DRIFT-FERT', family='B2',
        quantity='96-year soil carbon drift under unchanged current management',
        model_value=round(m_fert, 4),
        observed_low=drift['low'], observed_high=drift['high'],
        units='percent', horizon_years=96, n_rate_kg_ha=eu.current_rate,
        verdict=INFORMATIVE, informativeness=WEAK,
        note=(
            'This is the worked example of why the informativeness column '
            'exists. The model lands inside the observed envelope, which looks '
            'like a pass and is worth nothing: the engine is initialised at a '
            'spun-up equilibrium, so near-zero drift under unchanged '
            'management is a property of the initialisation, and any model '
            'with a spin-up would reproduce it. It is marked WEAK and carries '
            'no verdict. Reporting it as a pass would be the same defect as a '
            'test that compares the model to itself.'),
        source=drift['source'],
        extra={'quantity_kind': 'rate',
               'would_have_been': band_verdict(m_fert, drift['low'], drift['high']),
               'observed_status': drift['verification']['status']}))

    m_nil = eu.soc_drift_pct(96, nil=True)
    rows.append(Row(
        row_id='B2-EUROPE-DRIFT-NIL', family='B2',
        quantity='96-year soil carbon drift with no synthetic nitrogen',
        model_value=round(m_nil, 4), observed_low=None, observed_high=None,
        units='percent', horizon_years=96, n_rate_kg_ha=0.0,
        verdict=INFORMATIVE, informativeness=NONE,
        note=(
            'No matched observation — Broadbalk plot 3 was rejected as a '
            'comparator because it drifts around its own low equilibrium '
            'rather than through a withdrawal transient. Reported because it '
            'is what makes B3-europe-YR30 and B2-BROADBALK-FERT-MINUS-NIL one '
            'finding rather than two: a nil-N soil carbon loss of this size is '
            'internally consistent with a yield ratio near 0.36 halving '
            'residue inputs, so the carbon side and the yield side are not '
            'independent failures.'),
        source='no comparator; see observed_values.yaml rejected_comparators',
        extra={'quantity_kind': 'rate'}))

    fym = b2['broadbalk_fym_2010']
    rows.append(Row(
        row_id='B2-BROADBALK-FYM', family='B2',
        quantity='soil carbon under farmyard manure',
        model_value=None, observed_low=fym['value'], observed_high=fym['value'],
        units='Mg_C_per_ha', horizon_years=167, n_rate_kg_ha=None,
        verdict=NOT_APPLICABLE, informativeness=NONE,
        note=(
            'The model has no organic amendment pathway at all: carbon enters '
            'only as crop residue and roots. The manure plot holds 73.3 Mg '
            'C/ha against 30.0 on the inorganic plot, so the omitted pathway '
            'is large — larger than the treatment effect the model does '
            'represent. The manuscript must not claim the model represents '
            'organic amendment. Recorded as NOT_APPLICABLE rather than omitted '
            'so that the omission is visible in the suite.'),
        source=fym['source'],
        extra={'quantity_kind': 'stock',
               'inorganic_plot_stock': b2['broadbalk_soc_inorganic_npkmg_2010']['value'],
               'observed_status': fym['verification']['status']}))
    return rows


def build_b3(runs: Dict[str, RegionRuns], obs: Dict) -> List[Row]:
    rows: List[Row] = []
    b3 = obs['B3']
    prague = b3['prague_ruzyne_nil_over_npk4']
    tol = float(prague.get('tolerance', 0.0))
    eu = runs['europe']

    for horizon in HORIZONS:
        m = eu.nil_ratio(horizon)
        matched = horizon == prague['horizon_years']
        if matched:
            verdict = band_verdict(m, prague['low'], prague['high'], tol=tol)
            informativeness = STRONG
            note = (
                'The suite\'s only failure, and the row the manuscript\'s S3 '
                'temperate yield losses bear on directly. Yield with no '
                'synthetic nitrogen as a fraction of yield at the region\'s '
                'current rate, read at the same elapsed year from a common '
                'spun-up equilibrium — the paired-plot design of a long-term '
                'experiment. The observed arm is Prague-Ruzyne, unfertilized '
                'since 1954, against its NPK4 arm at 95 kg N/ha, close to the '
                'model\'s European regional mean of 85. The model loses about '
                '59 percent of yield where the experiment lost 22 to 32 '
                'percent: the mechanism runs about twice as hard as the field '
                'record shows. '
                'The tolerance of 0.02 is the published table\'s own reporting '
                'precision (one decimal place on yields of 3.8 to 6.9 t/ha), '
                'not a widening chosen to change the verdict — the model is '
                'far outside the band either way. '
                'It is scored against one site. Broadbalk plot 3 has been '
                'unfertilized since 1843 and its soil carbon is already '
                'compiled, but its grain yields are not; see '
                'B3-OWED-BROADBALK-YIELD-RATIO. Until they are compiled this '
                'is a failure against one observation and should be reported '
                'as one. '
                'What the benchmark constrains is the size of the yield '
                'penalty when nitrogen is withdrawn, not whether soil organic '
                'matter cushions it. The buffering claim itself is not '
                'contradicted by any row in this suite.')
        else:
            verdict, informativeness = INFORMATIVE, NONE
            note = (
                'No matched observation at this horizon; the Prague-Ruzyne '
                'windows are 7-27 and 29-66 years. Reported because the shape '
                'of the trajectory is itself the diagnostic: the model '
                'front-loads the collapse, spending most of the loss in the '
                'first three decades and then flattening, while Prague '
                'declines gently and is still declining at 66 years. Whatever '
                'is too strong is therefore in the fast part of the coupling — '
                'the residue-to-active-pool and mineralization feedback on 3- '
                'and 27-year turnover — rather than in the slow carbon '
                'kinetics.')
        rows.append(Row(
            row_id=f'B3-europe-YR{horizon}', family='B3',
            quantity='nil-N yield as a fraction of yield at the current rate',
            model_value=round(m, 4),
            observed_low=prague['low'] if matched else None,
            observed_high=prague['high'] if matched else None,
            units='dimensionless', horizon_years=horizon,
            n_rate_kg_ha=eu.current_rate,
            verdict=verdict, informativeness=informativeness, note=note,
            source=prague['source'] if matched else 'no matched observation',
            extra={'quantity_kind': 'ratio',
                   'observed_n_rate_kg_ha': prague['n_rate_kg_ha'] if matched else None,
                   'tolerance': tol if matched else None,
                   'observed_status': prague['verification']['status'] if matched else None}))

    for rk in REGIONS:
        if rk == 'europe':
            continue
        m = runs[rk].nil_ratio(30)
        rows.append(Row(
            row_id=f'B3-{rk}-YR30', family='B3',
            quantity='nil-N yield as a fraction of yield at the current rate',
            model_value=round(m, 4), observed_low=None, observed_high=None,
            units='dimensionless', horizon_years=30,
            n_rate_kg_ha=runs[rk].current_rate,
            verdict=INFORMATIVE, informativeness=NONE,
            note=(
                'No long-term nitrogen-withdrawal experiment was located for '
                'this region at a matched horizon, so the row carries no '
                'verdict. Reported at the manuscript\'s own horizon so that '
                'the one region that does have a comparator can be seen in '
                'context: europe is the model\'s most severe temperate '
                'response, and it is the one that fails.'),
            source='no comparator located',
            extra={'quantity_kind': 'ratio'}))

    ssa_obs = b3['ssa_nil_over_fertilized_ratio']
    ssa = runs['sub_saharan_africa']
    implied = ssa.implied_rate_for_ratio(ssa_obs['value'], year=1)
    rows.append(Row(
        row_id='B3-sub_saharan_africa-IMPLIED-RATE', family='B3',
        quantity='N rate at which the model reproduces the observed nil-to-fertilized ratio',
        model_value=None if implied is None else round(implied, 4),
        observed_low=None, observed_high=None,
        units='kg_N_per_ha', horizon_years=1, n_rate_kg_ha=None,
        verdict=INFORMATIVE, informativeness=NONE,
        note=(
            'An inversion, and it carries no verdict for two separate reasons. '
            'First, the compilation does not record the nitrogen rate on the '
            'trials\' fertilized arm, which is what would turn this into a '
            'verdict — see B3-OWED-SSA-TRIAL-RATE, the single most valuable '
            'missing number in the whole compilation. Second, and newly, WP4 '
            'could not source the 0.572 ratio at all: no publication reports '
            'it. The implied rate below is therefore an inversion of an '
            'unsourced target and must not be read as agreement. '
            'The rate it returns is nonetheless inside the range of '
            'recommended maize rates that on-farm trials in the region would '
            'have used, which is the reading F-008 gave it.'),
        source=ssa_obs['source'],
        extra={'quantity_kind': 'ratio',
               'observed_ratio': ssa_obs['value'],
               'observed_status': ssa_obs['verification']['status'],
               'model_ratio_at_current_rate': round(ssa.nil_ratio(1), 4)}))
    return rows


def build_b4(obs: Dict) -> List[Row]:
    b4 = obs['B4']['whc_per_soc_percentage_point']
    registered = float(_reg.value('whc_sensitivity'))
    implied = b4['value'] * (b4['profile_mm'] / 100.0)
    return [Row(
        row_id='B4-WHC-SENSITIVITY', family='B4',
        quantity='water-holding-capacity gain per percentage point of SOC over the profile',
        model_value=round(registered, 4),
        observed_low=round(implied, 4), observed_high=round(implied, 4),
        units='mm_water_per_pct_SOC_per_300mm', horizon_years=None,
        n_rate_kg_ha=None,
        verdict=band_verdict(registered, implied, implied, tol=0.005),
        informativeness=STRONG,
        note=(
            'An arithmetic check, not a field comparison: does the registered '
            'parameter equal the conversion the cited meta-analysis implies? '
            '1.16 mm per 100 mm of soil per percentage point of SOC, over a '
            '300 mm profile, is 3.48. The registry carries 3.48. The tolerance '
            'is half of the last printed digit of the source value, nothing '
            'more. F-008 records the registry as carrying 3.5; it does not, '
            'and RECONSTRUCTION_GAPS.md G-1 records why registering 3.5 was '
            'rejected — it moves sixteen of the 123 canonical fields and would '
            'have failed WP1\'s own acceptance. The manuscript states 3.5, '
            'which is 3.48 at the two significant figures it uses.'),
        source=b4['source'],
        extra={'quantity_kind': 'arithmetic', 'registered_value': registered,
               'source_slope_per_100mm': b4['value'],
               'profile_mm': b4['profile_mm']})]


def build_b5(obs: Dict) -> List[Row]:
    rows: List[Row] = []
    b5 = obs['B5']
    wide = b5['own_price_elasticity_retrieved_range']
    tight = b5['own_price_elasticity_tight_range']
    no_est = set(b5['regional_coverage']['no_usable_estimate'])

    for rk in REGIONS:
        eps = float(_reg.region_value('eps_F_PF', rk))
        verdict = band_verdict(eps, wide['low'], wide['high'])
        note = (
            'The model\'s regional own-price elasticity of fertilizer demand '
            'against the range spanned by the retrieved literature.')
        if verdict == MARGINAL:
            note += (
                ' Outside the extreme retrieved estimate by 0.01, and reported '
                'MARGINAL rather than FAIL because that endpoint is itself an '
                'estimate carrying standard errors of 0.14 to 0.84; an '
                'excursion of 0.01 is inside the reporting precision of the '
                'band, not evidence of disagreement.')
        informativeness = STRONG
        if rk in no_est:
            informativeness = WEAK
            note += (
                ' WEAK: WP4 searched in English and Russian across six '
                'countries and found no published own-price elasticity of '
                'fertilizer demand for this region at all. The value is '
                'therefore scored against a range assembled entirely from '
                'other regions, which tests only that it is not implausible '
                'globally. See B5-OWED-FSU-ELASTICITY.')
        rows.append(Row(
            row_id=f'B5-{rk}-OWN-PRICE', family='B5',
            quantity='own-price elasticity of fertilizer demand',
            model_value=round(eps, 4),
            observed_low=wide['low'], observed_high=wide['high'],
            units='dimensionless', horizon_years=None, n_rate_kg_ha=None,
            verdict=verdict, informativeness=informativeness, note=note,
            source=wide['source'],
            extra={'quantity_kind': 'elasticity',
                   'inside_tight_range': bool(
                       min(tight['low'], tight['high']) <= eps
                       <= max(tight['low'], tight['high'])),
                   'observed_status': wide['verification']['status']}))

    inside_tight = [rk for rk in REGIONS
                    if min(tight['low'], tight['high'])
                    <= float(_reg.region_value('eps_F_PF', rk))
                    <= max(tight['low'], tight['high'])]
    rows.append(Row(
        row_id='B5-TIGHT-RANGE-ALL', family='B5',
        quantity='regional elasticities against the tighter multi-study range',
        model_value=float(len(inside_tight)),
        observed_low=tight['low'], observed_high=tight['high'],
        units='count_of_regions_inside', horizon_years=None, n_rate_kg_ha=None,
        verdict=INFORMATIVE, informativeness=WEAK,
        note=(
            'Carries no verdict because it is nested inside the eight '
            'B5-<region>-OWN-PRICE rows and would double-count them. Reported '
            'because the tighter range is where the central mass of national '
            'aggregate estimates actually sits, and because the provenance of '
            'the 14-study count could not be verified: no review reporting '
            'exactly fourteen studies spanning -0.7 to -0.2 was found. The '
            'best candidate is a count over Nicolella, Dragone & Bacha (2005), '
            'recorded in observed_values.yaml as a hypothesis and not as a '
            'citation.'),
        source=tight['source'],
        extra={'quantity_kind': 'elasticity', 'regions_inside': inside_tight,
               'observed_status': tight['verification']['status']}))
    return rows


def build_owed(obs: Dict) -> List[Row]:
    """Observations the compilation owes.

    An owed row is not a failure and not a pass. It is a specific,
    obtainable number whose absence weakens a row, kept in the suite so the
    absence stays visible instead of becoming a silent omission.
    """
    rows: List[Row] = []
    for entry in obs['owed']:
        rows.append(Row(
            row_id=entry['id'], family=entry['id'].split('-')[0],
            quantity='observation owed by the compilation',
            model_value=None, observed_low=None, observed_high=None,
            units='', horizon_years=None, n_rate_kg_ha=None,
            verdict=OWED, informativeness=NONE,
            note=' '.join(entry['what'].split()),
            source='',
            extra={'status_2026_07_26': ' '.join(
                entry.get('status_2026_07_26', '').split())}))
    return rows


# =====================================================================
# Provenance
# =====================================================================

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def run(out_dir: str) -> List[Row]:
    with open(OBSERVED_PATH) as fh:
        obs = yaml.safe_load(fh)

    regions = get_default_regions()
    runs = {rk: RegionRuns(rk, regions) for rk in REGIONS}

    rows: List[Row] = []
    rows += build_b1(runs, obs)
    rows += build_b2(runs, obs)
    rows += build_b3(runs, obs)
    rows += build_b4(obs)
    rows += build_b5(obs)
    rows += build_owed(obs)
    return rows


def summarise(rows: List[Row]) -> Dict[str, int]:
    tally: Dict[str, int] = {}
    for r in rows:
        tally[r.verdict] = tally.get(r.verdict, 0) + 1
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'outputs'))
    ap.add_argument(
        '--write-baseline', action='store_true',
        help=('Freeze the current verdicts into '
              'data/benchmarks/baseline_verdicts.json. Regenerating the '
              'baseline is how a change in what the model says about the '
              'field record gets written down at the moment it happens, so '
              'it must be a deliberate act and never a side effect of a '
              'normal run.'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = run(args.out_dir)

    import csv
    csv_path = os.path.join(args.out_dir, 'benchmarks.csv')
    fields = ['row_id', 'family', 'quantity', 'model_value', 'observed_low',
              'observed_high', 'units', 'horizon_years', 'n_rate_kg_ha',
              'verdict', 'informativeness', 'quantity_kind',
              'observed_status', 'note', 'source']
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r.as_record())

    tally = summarise(rows)
    strong = sum(1 for r in rows if r.informativeness == STRONG)
    payload = {
        'generated_by': 'code/repro/run_benchmarks.py',
        'spec': 'FINDINGS.md F-008',
        'git_commit': _git_commit(),
        'params_sha256': _sha256(os.path.join(ROOT, 'code', 'model', 'params.yaml')),
        'script_sha256': _sha256(os.path.abspath(__file__)),
        'observed_values_sha256': _sha256(OBSERVED_PATH),
        'n_rows': len(rows),
        'tally': tally,
        'n_strong': strong,
        'acceptance_f008': {'n_rows': 41, 'PASS': 11, 'MARGINAL': 3, 'FAIL': 1,
                            'INFORMATIVE': 18, 'OWED': 7, 'NOT_APPLICABLE': 1,
                            'n_strong': 15},
        'rows': [r.as_record() for r in rows],
    }
    json_path = os.path.join(args.out_dir, 'benchmarks.json')
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)

    if args.write_baseline:
        baseline_path = os.path.join(ROOT, 'data', 'benchmarks',
                                     'baseline_verdicts.json')
        baseline = {
            'note': (
                'Frozen verdicts for the F-008 benchmark suite. This file is '
                'what lets a failure be TRACKED without having to be FIXED '
                'first: B3-europe-YR30 is recorded here as FAIL, and the '
                'build stops on any movement away from this state. '
                'code/tests/test_benchmark_baseline.py re-runs the suite '
                'rather than reading outputs/benchmarks.csv, because reading '
                'the committed CSV would only check that a file had not been '
                'edited. It fails on a regression, on a baselined row '
                'disappearing, on a new row arriving already failing, and on '
                'a row\'s informativeness being downgraded — which is the '
                'other way to make a failure go away: stop claiming the row '
                'proves anything. It also fails on an IMPROVEMENT until this '
                'baseline is regenerated with --write-baseline.'),
            'generated_by': 'code/repro/run_benchmarks.py --write-baseline',
            'spec': 'FINDINGS.md F-008',
            'git_commit': payload['git_commit'],
            'params_sha256': payload['params_sha256'],
            'observed_values_sha256': payload['observed_values_sha256'],
            'n_rows': len(rows),
            'tally': tally,
            'verdicts': {r.row_id: {'verdict': r.verdict,
                                    'informativeness': r.informativeness}
                         for r in rows},
        }
        with open(baseline_path, 'w') as fh:
            json.dump(baseline, fh, indent=2, sort_keys=False)
        print(f'  -> {baseline_path}  (baseline rewritten)')

    order = [PASS, MARGINAL, FAIL, INFORMATIVE, OWED, NOT_APPLICABLE]
    print(f'{len(rows)} rows  ' +
          '  '.join(f'{k} {tally.get(k, 0)}' for k in order) +
          f'   STRONG {strong}')
    print(f'  -> {csv_path}')
    print(f'  -> {json_path}')
    for r in rows:
        if r.verdict in (FAIL, MARGINAL):
            print(f'  {r.verdict:9s} {r.row_id:42s} model {r.model_value} '
                  f'vs [{r.observed_low}, {r.observed_high}]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
