# WP4 — benchmark suite reconciliation

**Written 2026-07-26.** What the rebuilt suite scores, against what F-008
records, and exactly where the two differ.

Acceptance, from `v15_REBUILD_STATE.md` and F-008:

> 41 rows, 11 PASS / 3 MARGINAL / 1 FAIL / 18 INFORMATIVE / 7 OWED /
> 1 NOT_APPLICABLE, with B3-europe-YR30 failing at a model ratio of 0.406.
> Fifteen rows marked STRONG.

Result:

| | PASS | MARGINAL | FAIL | INFORMATIVE | OWED | N/A | rows | STRONG |
|---|---|---|---|---|---|---|---|---|
| F-008 | 11 | 3 | 1 | 18 | 7 | 1 | 41 | 15 |
| rebuilt | **9** | **3** | **1** | **14** | **7** | **1** | **35** | **11** |
| difference | −2 | 0 | 0 | −4 | 0 | 0 | −6 | −4 |

**MARGINAL, FAIL, OWED and NOT_APPLICABLE reproduce exactly, and so do the
row identities behind them.** The three MARGINAL rows are the three F-008
names — `B1-SSA-MPP`, `B2-BROADBALK-FERT-MINUS-NIL`, and North America's
own-price elasticity. The single failure is `B3-europe-YR30`. The one
NOT_APPLICABLE is `B2-BROADBALK-FYM`. The shortfall is entirely 2 PASS and
4 INFORMATIVE rows that could not be identified.

**Nothing was tuned.** No parameter was touched, and no band, tolerance or
verdict rule was chosen to move a row. The one verdict rule that was
changed during the rebuild was changed *against* the acceptance: an early
one-sided form of `band_verdict` scored `B3-europe-YR30` MARGINAL, which
would have hidden the suite's only failure, and it was corrected to a
two-sided test.

---

## 1. The model side reproduces F-008 to three or four significant figures

This is the part that matters most, because it says the rebuilt model —
after WP1's rewiring and WP2's recalibration — still produces the numbers
the finding was written about. Five independent quantities were checked
before any row was defined:

| quantity | F-008 | rebuilt | |
|---|---|---|---|
| `B3-europe-YR30` nil-N yield ratio | 0.406 | **0.4063** | exact |
| `B3-europe` ratio at 96 years | 0.364 | 0.3653 | +0.0013 |
| `B3-europe` ratio at 1 year | 0.763 | 0.7680 | +0.005 |
| `B2-EUROPE-DRIFT-FERT`, 96-year drift | −0.21 % | **−0.2098 %** | exact |
| `B2-EUROPE-DRIFT-NIL`, 96-year drift | −28.0 % | **−27.95 %** | exact |
| fert-minus-nil SOC excess at 30 years | 21.0 % | **21.0 %** | exact |
| fert-minus-nil SOC excess at 96 years | 38.6 % | 38.49 % | −0.11 |
| `B3-sub_saharan_africa` implied rate | 47.6 | **47.59** | exact |

The year-1 europe ratio is the only model number that moved by more than
rounding, and 0.005 on a ratio is consistent with WP2's recalibration of
`y_max`, which moves year-1 yields and leaves the 30-year ratio to three
decimal places unchanged.

### The one model quantity that did not reproduce: B1's MPP triple

F-008 records the sub-Saharan marginal physical product as 24.8 kg grain
per kg N at 7 kg N/ha, reproducing an MPP of 20.0 at 25.9 kg N/ha and of
7.7 at 109.2. The rebuilt suite gives 24.72, 26.10 and 112.18.

Six finite-difference conventions were tested and none reproduces the
triple:

| definition | MPP(7) | rate at MPP 20.0 | rate at MPP 7.7 |
|---|---|---|---|
| F-008 | 24.8 | 25.9 | 109.2 |
| central, h = 0.5 | 24.716 | 26.095 | 112.180 |
| central, h = 0.01 | 24.716 | 26.095 | 112.179 |
| forward, h = 1 | 24.580 | 25.595 | 111.680 |
| forward, h = 0.1 | 24.702 | 26.045 | 112.129 |
| backward, h = 1 | 24.854 | 26.595 | 112.680 |
| average product from 0 | 25.701 | 54.973 | 301.843 |

The central difference is step-independent to five significant figures, so
it is the model's true derivative and not an artifact of the step. The
deviation is not a constant scale factor (+0.34 %, −0.75 %, −2.66 %), so it
is not a units or rounding difference either. **The convention F-008 used
could not be recovered.** The suite uses the central difference on the
first year's yield from the common equilibrium, which is the defensible
definition and which reproduces the neighbouring `B3` inversion exactly.
Recorded as a gap rather than resolved by picking whichever convention
lands nearest 24.8 — the same discipline WP3 applied to `texture_class`.

---

## 2. What the six missing rows are, and why they are not invented

F-008's prose names and describes sixteen rows. The suite implements all
sixteen, plus the per-region extensions its own naming convention implies
(`B3-<region>-YR30` for the seven regions without a comparator, and
`B5-<region>-OWN-PRICE` for all eight), plus the three owed observations
WP4's own literature work opened. That reaches 35.

The remaining six are **2 PASS and 4 INFORMATIVE**. Their identities are
not recoverable. Every candidate partition that reaches 41 was tested
against the six-way tally and rejected:

- **`B3` at all three horizons for all eight regions** (24 rows) would put
  23 rows in INFORMATIVE on its own, against a budget of 18.
- **`B3` at YR1 and YR30 for all eight regions** (16 rows) gives 21
  INFORMATIVE. Also too many.
- **Per-region `B2` drift rows** in the style of `B2-EUROPE-DRIFT-*` would
  add 8 or 16 INFORMATIVE rows. Too many.
- **Splitting `B1-SSA-IMPLIED-RATE-RANGE` into its two endpoints** adds one
  INFORMATIVE, not four, and F-008 writes it as a single row.
- **The 2022 fertilizer-price hindcast** (`data/benchmarks/hindcast_benchmark_sol.csv`,
  12 rows over 3 elasticity scenarios × 4 regions) is the most plausible
  home for extra PASS rows and is thematically `B5`. But F-008's `B5`
  paragraph is entirely about elasticity magnitudes against the literature
  and never mentions a hindcast, and the hindcast's own numbers disagree
  with observation in sign for South Asia (predicted −18.4 %, observed
  +2.3 %), so folding it in would have produced failures F-008 does not
  record. Rejected.

Adding rows until the count reaches 41 would be the benchmark-suite
equivalent of tuning the model to the benchmarks. The count is reported
short and the gap is named.

**What would close it:** any surviving copy of `outputs/benchmarks.csv`
from the crashed v15 session. It is confirmed absent from this tree, from
`_transfer/`, and from the GitHub deposit at `7026193c` / tag `v1.2`. The
`logs/` sequence jumps 01 → 02 → 03 → 21 → 51 → 52 → 67, so
`run_32_benchmarks.log` through `run_36_benchmarks.log` never reached
disk either.

---

## 3. What the research half changed, and what it did not

The observed-value compilation was the expensive part, and rebuilding it
against the primary literature rather than copying F-008's prose changed
several things. **None of these changes moved a verdict**, but several
change what the SI may claim.

### Verified, and now better sourced than F-008 was

- **Prague-Ruzyně.** Both ratios reproduce exactly from Hlisnikovský et
  al. 2022, *Plants* 11:1825, Table 1 — control 3.8 / NPK4 4.9 for
  1961–1981, control 4.7 / NPK4 6.9 for 1983–2020. NPK4 is confirmed at 95
  kg N/ha for the wheat crop. Four qualifications are now on record and
  belong in the SI: the ratios are derived rather than published; the table
  is one decimal place, so they carry about ±0.02 and the third digit is
  not meaningful; they are ratios of period means, not means of annual
  ratios; and the yields are winter wheat *after potatoes* only, 9 and 14
  seasons, not every year. The start year is disputed between 1954 and 1955
  across five papers on the same site, and F-008's elapsed-year arithmetic
  only works with 1954.
- **Nigeria MPP** 7.75 (2010) and 7.71 (2012), verified in Liverpool-Tasie
  et al. 2017 Table 4, **with their nitrogen rates** — 40.19 and 46.56 kg
  N/ha.
- **Own-price elasticity endpoints.** −1.87 is Williamson 2011 (US
  nitrogen, IV, SE 0.41); −0.21 is Denbaly & Vroomen 1993 (US corn,
  short run).

### Corrections F-008 owes

- **The Kenyan MPP is 17.5, not 17**, and it is evaluated at a sample mean
  of **25.2 kg N/ha** (Sheahan, Black & Jayne 2012). F-008 records it as
  "17" with no rate.
- **The 7.7–20.0 SSA envelope is not a published envelope.** No source
  states it. Both endpoints come from a single comparison paragraph in
  Liverpool-Tasie et al. 2017, and the envelope silently mixes true
  marginal products with agronomic efficiencies, which are averages. For a
  concave response the average exceeds the derivative at any positive N, so
  the mixture biases the envelope upward. The published literature also
  runs well outside it in both directions — 2.6 kg/kg for Central Malawi
  (Burke, Snapp & Jayne 2020), means of 42 for Kenya and 18 for the rest of
  SSA (Ichami et al. 2019).
- **The low Nigerian value's explanation is misattributed.**
  Liverpool-Tasie et al. emphasise transport costs — "about 70% of the
  actual cost incurred by farmers using fertilizer is due to transportation
  costs" — not phosphorus, weeds and application timing. That framing
  belongs to Burke, Snapp & Jayne 2020 on Central Malawi, a different
  paper and a different country. The SI must not carry the spliced
  explanation.
- **"−1.87 to −0.21" is a reconstructed span, not a citable finding.** The
  endpoints come from different literatures. The honest statement is that
  the central mass of national aggregate estimates sits at −0.2 to −0.7 and
  that all eight model values lie in it.
- **The "14 studies spanning −0.7 to −0.2" could not be verified.** No such
  review was found. A count over Nicolella, Dragone & Bacha 2005 gives
  about 14 in that interval and is recorded as a hypothesis.

### The one finding that should worry a reader

**The sub-Saharan response ratio of 0.572 could not be sourced at all.** No
publication reports it. The nearest genuine syntheses give 0.50 (Ichami et
al. 2019, mean fertilizer response 2.0), 0.556 / 0.588 (their Kenya and
other-SSA medians), and 0.543 (Chivenge et al. 2011).

A lead, recorded as speculation: both of those meta-analyses work in the
**log** response ratio, ln(Xe/Xc). A log ratio of 0.572 corresponds to a
raw ratio of 1.77 — squarely inside the Kenya/SSA median band of 1.7 to 1.8
— and its reciprocal is 0.564. **A value of 0.572 presented as a raw
proportion has the exact signature of a log response ratio read as a linear
one.** If that is what happened, the number is an error rather than a
citation gap, and `B3-sub_saharan_africa-IMPLIED-RATE` is an inversion of a
target that does not exist.

This does not change any verdict: the row carries none, for exactly the
reason F-008 gave. It does mean the row must not be reported in the SI as
agreement between the model and the field record until it is resolved.

### A material gap the SI should state

**There is no published own-price elasticity of fertilizer demand for the
former Soviet Union or Central Asia, in any language.** Six countries were
searched in English and Russian. The nearest work analyses
fertilizer-to-output price ratios and optimal N rates, not demand
elasticities. This is a gap in the literature, not a retrieval failure, and
it means `B5-fsu_central_asia-OWN-PRICE` is scored against a range built
entirely from other regions. The row is marked WEAK for that reason and a
new owed observation, `B5-OWED-FSU-ELASTICITY`, records it.

Sub-Saharan coverage is weak in a different way: all three retrieved
sources are 2024–25, cover five countries, and their estimates span −0.01
to −3.40 depending almost entirely on how prices are imputed for
non-purchasers. No pre-2024 SSA elasticity exists. Latin American coverage
is Brazil-only; East Asian coverage is China-only and crop-specific.

---

## 4. Two observations that disagree with the surviving pre-v15 compilation

Both are recorded in `observed_values.yaml`, flagged, and left unresolved.

- **Broadbalk plot 3, 2010.** F-008 uses 25.3 Mg C/ha.
  `data/benchmarks/validation_data_extraction.csv`, compiled from the same
  e-RA dataset before v15, gives 24.5 for the same plot and year. F-008's
  value reproduces its own quoted 18.6 % excess exactly, so it is
  internally consistent; 24.5 would give 22.4 %, a different benchmark.
  Both fall inside the half-to-twice band, so `B2-BROADBALK-FERT-MINUS-NIL-YR30`
  passes either way and `-YR96` is MARGINAL either way. One read of e-RA
  02-BKSOC1843 v2 closes it. Recorded as `B2-OWED-BROADBALK-NIL-2010`.
- **The Broadbalk FYM plot.** F-008 uses 73.3 Mg C/ha; the surviving
  extraction gives 92.2 for plot 2.2 in 2015. Different plot or different
  year. The row is NOT_APPLICABLE either way, because the model has no
  organic amendment pathway, so no verdict depends on it.

---

## 5. Three new owed observations, opened by this pass

Added to the seven-row OWED block, which means three of F-008's seven owed
rows are *not* the same three. F-008 names four: `B1-OWED-SURVEY-RATE`,
`B3-OWED-SSA-TRIAL-RATE`, `B3-OWED-BROADBALK-YIELD-RATIO`,
`B5-EPS-F-N-NO-ANALOGUE`. The three added here are:

- `B5-OWED-FSU-ELASTICITY` — any estimate at all for that region.
- `B2-OWED-BROADBALK-NIL-2010` — the disputed plot-3 stock.
- `B1-OWED-MPP-QUANTITY-KIND` — separate the SSA estimates into true
  marginal products and agronomic efficiencies, so the model derivative is
  compared only to derivatives.

That the count lands on seven either way is a coincidence and is recorded
as one. F-008's other three owed rows are among the six that could not be
recovered.

**`B1-OWED-SURVEY-RATE` is now partly closed** — WP4 found the rates for two
of the four SSA estimates (25.2 kg N/ha for Sheahan, 40.19 / 46.56 for
Liverpool-Tasie). It is deliberately left OWED rather than re-scored,
because re-scoring `B1-SSA-MPP` at a matched rate changes what the model is
compared against, and that belongs in a pass that says so rather than in a
reconstruction that is supposed to reproduce a prior state.

---

## 6. What this suite does not do

- It does not gate. `code/tests/test_benchmark_baseline.py` is **not
  written** — it is F-009's artifact and belongs to WP6, which owns the
  build graph and `make verify`. `data/benchmarks/baseline_verdicts.json`
  is written and frozen, so the gate has something to compare against the
  moment it exists.
- It does not touch the manuscript. F-008's four consequences for the
  manuscript — report the suite including the failure, present the S3
  temperate losses with the doubling stated, stop describing the magnitude
  as validated, and note that the buffering claim itself is not
  contradicted — are D2's work, and the benchmark section D2 needs is this
  file plus `outputs/benchmarks.csv`.
- It does not resolve the 0.572 problem. That needs a decision, not more
  code.
