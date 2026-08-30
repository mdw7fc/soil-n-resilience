# FINDINGS

Dated record of defects found during the v15 hardening pass, what was measured,
and where the numbers live. Each entry names the test that will fail if the
finding is reversed, so that no finding rests on this file being read.

---

## F-001 — 2026-07-25 — The spin-up does not reach a steady state, and the registry's exemption said it did

`som_pool_fractions` was exempted from the Monte Carlo on the grounds that the
dynamic spin-up overwrites the initial 4/38/58 partition of SOC. That claim was
false. With `k_passive = 0.000728/yr` the passive pool has a 1374-year turnover,
while the spin-up's convergence criterion (fractional SOC drift below 0.002 over
a 50-year window) is met after 101 to 130 years. The active and slow pools do
equilibrate and are partition-independent to better than 1.5%; the passive pool
is essentially inherited.

Measured. Starting North America from `f_passive` 0.10, 0.58 and 0.90 gives
converged passive stocks of 6.98, 28.91 and 43.00 t C/ha and total SOC of 29.23,
51.01 and 64.98 t C/ha. Run to a true fixed point (`n_spinup=20000, tol=1e-6`)
all three land on 49.79 t C/ha to within 5.4e-5, so the equilibrium itself is
partition-free; the shipped spin-up does not run that far. Against the analytic
fixed point `c_p* = h_slow_to_passive * k_slow * c_slow* / k_passive` the passive
pool covers 7.5-8.5% of the distance in the five temperate regions and 90.9-91.6%
in the four regions on the Laub tropical parameterisation.

Consequence. Temperate SOC is anchored on the measured stock (North America 51.0
against a registered `soc_initial` of 50.0) and tropical SOC on the model's own
kinetics (Sub-Saharan Africa 5.99 against a registered 9.0). Absolute stocks are
therefore not comparable across regimes even though relative changes are.
Absolute SOC is initialization, not prediction, and the manuscript must not
present it as one. **Owed to the SI as a stated limitation.**

What licenses the exemption instead. Over `f_passive` 0.45 to 0.73 the S3 year-1
and year-10 yield losses move by at most 0.01 and 0.02 percentage points in any
region, against a shift of up to 12.3 t C/ha in absolute SOC. The water-stress
term responds to SOC change measured against each run's own equilibrium, and the
passive pool contributes about 2% of N mineralization.

Written to: `code/model/params.yaml` (`som_pool_fractions.mc_exempt_reason`),
`monthly_model_v3.century_dynamic_spinup` docstring.
Asserted by: `code/tests/test_spinup_partition_independence.py`.

---

## F-002 — 2026-07-25 — The calibration was fitted on a code path nobody published

`monthly_model_v3.calibrate_ym` roots on `run_model`, which uses the global
`CropParams.mitscherlich_c` and applies no baseline water-stress multiplier.
Every published run goes through `century_dynamic_spinup` plus
`MonthlyBiophysicalEngine`, which use `region.mitscherlich_c_regional` and do
apply water stress. The manuscript's statement that yields are calibrated to
FAOSTAT was true of a path that was never run. No test caught it because every
test compared the model to itself, which is the user's condition #3 (tests
checking internal reproducibility rather than external validity) intersecting
condition #2 (calibration masking errors).

Measured, under the published ERA5 forcing. Production baseline yields missed
their FAOSTAT targets by -3.87% (South Asia) to +4.19% (Latin America).
Recalibrating on the production path moves `yield_max` by -3.36% to +3.78% and
the reported S3 losses by at most 0.10 pp (Latin America year-10 2.32 to 2.42
pp), with no change in the regional ranking: FSU/Central Asia remains the
highest year-1 loss.

Fixed by `coupled_monthly.calibrate_ym_production`, which roots the production
path itself; `get_calibrated_ym` now calls it. `CALIBRATION_SCHEME =
'production_path_v2'` is the first element of `calibration_fingerprint`, so every
`yield_max` cached on disk under the old scheme is stale by construction and
cannot be reused. `YM_REGION_FIELDS` grew from 9 to 13 fields, adding
`mitscherlich_c_regional`, `baseline_water_deficit`, `water_stress_coeff` and
`whc_sensitivity`. The legacy `calibrate_ym` is left importable on purpose: the
test measures the gap rather than deleting the evidence.

Written to: `results/calibration_production_path.csv`,
`logs/run_01_calib_fingerprint.log`.
Asserted by: `code/tests/test_calibration_fingerprint.py`, which reproduces the
FAOSTAT targets to 1e-3 relative (achieved: 8e-3 percent worst case), AST-scans
the calibration path for region fields the fingerprint does not hash, perturbs
all 19 RegionParams fields and fails if an unregistered field moves `yield_max`,
and fails if the legacy objective's gap ever falls below 1%.

Consequence. **The 1000-draw Monte Carlo and every figure must be regenerated
under the new calibration.**

---

## F-003 — 2026-07-25 — `k_slow` uncertainty cannot reach the baseline, which is why one `yield_max` per region is sound

`run_mc_ensemble.py` computes `yield_max` once outside the draw loop (line ~481)
and then perturbs `k_slow` inside it by a truncated normal on [0.60, 1.40]. That
would be a stale-calibration bug if `k_slow` reached the baseline. It does not,
structurally: a first-order pool at steady state passes its input through
unchanged, `c_slow* = 0.46 c_in / k_slow`, so the mineralization flux
`k_slow * c_slow*` is invariant to `k_slow`.

Measured. Baseline yield spans 0.098% at worst across the prior's full range,
against a calibration tolerance of 0.20%. `k_slow` uncertainty therefore
propagates only into the transient response, never into the equilibrium. This is
what licenses omitting `som_params` from `calibration_fingerprint`.

Written to: `logs/run_01_calib_fingerprint.log` section [7].
Asserted by: `test_k_slow_does_not_enter_the_baseline`, which fails if the span
exceeds 0.5% and names the line in `run_mc_ensemble.py` that depends on it.

---

## F-004 — 2026-07-25 — The deposited ERA5 climate module did not reproduce the published runs

`code/era5/REGIONAL_CLIMATES_era5.py` is the file a reader of the deposit would
import to get the published forcing. It was hand-maintained, and it disagreed
with what ran. Its temperature, precipitation and PET arrays matched
`data/era5_regional_climates.json` exactly (max absolute difference 0.0 across
all eight regions and all three arrays), but its maturity months differed for
three regions: Latin America 4 against 3, Sub-Saharan Africa 10 against 9,
FSU/Central Asia 9 against 8. `run_canonical.patch_era5_climate` reads the
arrays from the JSON and keeps planting and maturity months from the built-in
`monthly_model_v3.REGIONAL_CLIMATES`, so the deposited module was never the
configuration that produced any figure.

Measured, under the production-path calibration. Re-running S3 for 30 years with
the deposited calendar moves the year-1 yield loss by -0.163 pp (Latin America),
+0.129 pp (Sub-Saharan Africa) and -0.263 pp (FSU/Central Asia), and `yield_max`
by -1.51%, -2.62% and +0.11%. North America, whose calendar agreed, reproduced
bit-identically, which confirms the deltas are the calendar and not run-to-run
noise. Five of the six affected loss values move by more than the paper's 0.1 pp
reporting precision, and all of them by more than the 0.05 pp tolerance the
spin-up test uses to license its own exemption.

Why nothing caught it, and this is the same mechanism as F-002. `yield_max` is
refitted to the FAOSTAT target on whatever calendar it is handed, so the
baseline yield lands within 4e-5 relative of the target under either calendar.
Every check that compares the model to its own calibration target passes. The
error survives only in the shock response, which is the quantity the paper
reports and no test compared. Calibration absorbed the defect in the checked
quantity and left it in the reported one.

Ruled out while sizing this. `calibration_fingerprint` does include
`planting_month` and `maturity_month` (`coupled_monthly.py:731`) and the
fingerprints did differ across the two calendars, so there is no stale-`ym`
cache hazard here.

Fixed by making the file generated rather than maintained.
`code/era5/generate_era5_module.py` emits it from the JSON extract plus the
built-in calendar, the same two sources `patch_era5_climate` reads. The published
calendar is kept; the deposited months were not adopted, because there is no
evidence they are the better crop calendar and adopting them would change
published numbers on no authority.

Registry gap this exposed, not yet closed. The crop calendar is a per-region
empirical assumption (dominant regional crop after Sacks et al. 2010) that lives
in code, not in `code/model/params.yaml`, and it is not varied in the Monte
Carlo. It moves the reported loss by up to 0.26 pp per month of growing season,
which is larger than several parameters that are sampled. **Owed: register the
twelve planting and maturity months with provenance, and state in the SI that
the growing season is fixed.**

Written to: `results/era5_calendar_discrepancy.csv`,
`results/era5_calendar_discrepancy_summary.csv`,
`code/repro/measure_era5_calendar.py`, `logs/run_13_era5_calendar.log`.
Asserted by: `code/tests/test_era5_deposit_matches_runtime.py`, which fails if
regenerating the module would change a byte and, independently, if the parsed
deposit disagrees with the patched runtime state on any field. Negative control
run: reverting one maturity month produced 3 failures
(`logs/run_17_era5_negctl.log`).

---

## F-005 — 2026-07-25 — Three aggregation bases in one paper, and two docstrings named the wrong one

A global figure is a weighted mean, and the weights are a scientific claim. This
repository made that claim three different ways, in three files, and described it
in words a fourth way.

What was where. `coupled_econ_biophysical.calibrate_price_shock` weighted the
regional fertilizer-demand responses by **nitrogen tonnage**
(`cropland_mha * synth_n_current`) while its own docstring, and the docstring of
`get_scenario_params` below it, said "area-weighted".
`coupled_econ_biophysical.aggregate_global` weighted every outcome column by
**cropland area**. `code/repro/run_canonical.py` normalised a third vector
inline, **production tonnage** (`cropland_mha * y_base`), and that third one is
what produced the published 2.30 / 3.41 / 3.64 % headline. Nothing in the code
compared them, and no output labelled which one it used.

Measured spread, from `results/aggregation_basis_comparison.csv` (per-region
losses reweighted from the frozen `data/scenario_trajectories.csv`; weights from
`data/canonical_ERA5_y30.csv`). Global S3 yield loss, area / nitrogen /
production: year 1 **2.652 / 2.156 / 2.305 %**, year 10 **4.006 / 3.310 /
3.412 %**, year 30 **4.289 / 3.508 / 3.636 %**. The delivered year-1 fertilizer
reduction from the calibrated shock (`fert_price_shock = 1.0389792148`) is
**21.42 / 20.00 / 19.56 %**. Largest spread on any reported quantity: **1.87 pp**,
on the fertilizer reduction, against a reporting precision of 0.1 pp.

The scenario headline is basis-dependent in a way the paper does not state. "A
20% reduction in fertilizer application" is exactly 20.00% only on the
nitrogen-tonnage basis the shock was calibrated on. On the production basis that
the yield headline uses, the same shock delivers 19.56%. Two numbers in the same
sentence, computed on different weights.

Why no published number moved when this was fixed. `aggregate_global`, the
function carrying the area basis, has exactly two call sites
(`coupled_econ_biophysical.py:1052` and `:1072`), both inside that module's
`__main__` demonstration printout. It fed no figure, no table and no manuscript
sentence. The published path was `run_canonical.py`'s inline normalisation, which
was already production-weighted and stayed so.

Resolution, and it is a resolution of the concept and not of the numbers. There
are two quantity classes, not two competing bases for one quantity.
`outcome_weights()` returns the single declared basis for outcome shares
(`yield_fraction`, `soc_fraction`, `food_price_index`): production tonnage,
because a fifth of the world's grain lost in one region is not offset by a fifth
of a region that grows little. `intensity_weights()` returns the basis for
per-hectare rates (`fert_applied_kgha`, `n_mineralized`, `water_stress`):
cropland area, because the global mean of a per-hectare rate is its area-weighted
mean by definition. Nitrogen tonnage is retained in `calibrate_price_shock` alone
and now says so, because a scenario defined as a reduction in nitrogen mass is
correctly weighted by nitrogen mass; weighting a mass-defined scenario by
anything else would mean the scenario does not deliver its own definition. Both
false docstrings were corrected and both name the date they were wrong until.
Every call site now goes through one of the two factories, and `run_canonical.py`
takes the published vector from `outcome_weights` rather than building its own.

The check is on construction, not on inspection. `SeamD_AggregationWeights`
(`code/model/seams.py`) is a frozen dataclass that validates in `__post_init__`:
weights sum to one, no region silently dropped, no zero-weight region, and a
non-empty provenance string. `assert_same_basis()` refuses two vectors built on
different bases. There is no way to obtain a weight vector and skip the check.

A vacuous assertion was written and removed during this work. The first version
called `assert_same_basis(*[W_prod for _ in OUTCOME_COLS])` inside
`aggregate_global`, which passes the same object N times and therefore cannot
fail. It was deleted; the real cross-quantity check lives in the test file, where
two independently constructed vectors meet. An assertion that cannot fail
ratifies rather than tests.

Caveat carried forward. The rerun used to fill the columns with no per-region
file on disk (SOC change, delivered fertilizer) does not reproduce the frozen
canonical, because current HEAD recalibrates `yield_max` on the production path
(F-002). Maximum residual 1.4502 pp at south_asia year 30; at year 1 only
0.0702 pp. Those rows are labelled `_rerun_currentcode` in the CSV and must not
be mixed with the frozen rows.

Owed to the manuscript. The MS and SI must state, for every global number, which
basis it is on, and must not present a nitrogen-weighted 20% and a
production-weighted 2.30% as if they were the same average.

Written to: `results/aggregation_basis_comparison.csv`,
`results/aggregation_basis_weights.csv`,
`results/aggregation_basis_comparison_README.txt`,
`results/seam_contract_checks.yaml`, `logs/run_21_seams.log`.
Asserted by: `code/tests/test_seam_contracts.py` sections D1-D3. D2 is an
external-validity probe with an independent hand implementation and a guard that
fails if the constructed case stops separating the two bases. D3 recomputes the
deposited headline from its own per-region rows and requires exact agreement at
the deposit's 2 dp (measured unrounded gap 0.0049 pp). Verified no behaviour
change: `calibrate_price_shock(0.20)` returns 1.0389792148114703 before and
after. Negative controls firing in this area: a production vector and an area
vector passed to `assert_same_basis`, a vector with no provenance, weights
summing to 0.9, a region silently dropped, a region with zero weight.

---

## F-006 — 2026-07-25 — The registry documents the model; it does not drive it

Mutation coverage was run over the whole registry for the first time
(`code/tests/run_mutation_coverage.py`, 56 mutable leaves, one canonical S3 run
plus six test files per leaf, ~38 s each). The plan expected the output to be a
list of parameters nothing tests. The actual output is one line down from that:
**perturbing 45 of the 56 registered leaves by 10% changes no published number
at all.**

Only five leaves reach the canonical run: `soc_bulk_density`, `cm2_per_ha`,
`g_per_t` and `pct_to_fraction`, all of which reach it through the single
derived quantity `soc_tha_per_pct`, and `residue_c_to_active_fraction`, which
was wired earlier today. Six more are refused at load by the registry's own
constraints (the two sum-to-one blocks and the profile-depth unit check), which
is the registry working. Everything else is declared and mirrored.

The mechanism. `code/model/registry.py` is imported by exactly four modules and
supplies exactly two things: `soc_tha_per_pct()` and, since today,
`value('residue_c_to_active_fraction')`. Every other registered value is
restated as a literal in code, and `code/tests/test_registry_consistency.py`
compares the two, entry by entry, with an explicit wiring table. So the registry
cannot drift from the code, which is what that test was built to guarantee and
it does guarantee it. But the direction of authority runs from the code to the
registry, not the other way. `params.yaml` is a documentation layer with a drift
alarm attached.

Why this matters more than it sounds. Mutation coverage is only informative
about parameters the model actually reads. Run against a registry the model does
not read, it returns "everything is caught" for the same reason a thermometer in
a drawer reports a stable temperature. The 45 leaves are not proven safe; they
are unmeasured, and the harness cannot measure them until they are wired. The
verdict `DECLARED_NOT_WIRED` exists in the harness so this result cannot be read
as coverage.

The sharper case is the Monte Carlo. **33 of the 45 unwired leaves carry an
`uncertainty:` block**, including `som_decay_rates.k_slow`, `whc_sensitivity`,
`eps_F_N`, `eps_F_PF`, `cre_base`, `residue_retention` and `soc_initial`.
`code/repro/run_mc_ensemble.py` imports nothing from `registry`. The prior
stated in `params.yaml`, which is what the SI parameter table is generated from,
and the prior the ensemble actually samples are two separate statements of the
same thing, and nothing compares them. This is systemic condition 1, "the same
concept specified in multiple places", in the one place where it would move a
published uncertainty interval rather than a central estimate.

What is owed, and it is a refactor, not a patch. Each unwired leaf needs its
literal replaced by a registry read, in dependency order, with the mutation
harness rerun after each batch so the verdict moves from DECLARED_NOT_WIRED to
COVERED or UNTESTED and the UNTESTED rows become the real worklist the plan
asked for. The highest-value batch is the one the ensemble samples: the SOM
kinetics, `whc_sensitivity`, the demand elasticities, and the region tables. A
narrower interim step, if the refactor cannot be completed before v15, is a test
asserting that every `uncertainty:` block in `params.yaml` matches the
distribution `run_mc_ensemble.py` draws, which closes the specific gap above
without rewiring anything.

One caveat on REACH. It is measured on the canonical S3 run only. A parameter
that matters solely in the Monte Carlo tails, the price-shock analysis or the
four-pool comparison would read as not reaching even once wired. The verdict
column carries that limit.

Written to: `results/mutation_coverage.csv` (one row per leaf, with the
mutation rule, the maximum relative move, the quantity that moved most, and the
tests that failed), `results/mutation_coverage_summary.txt`,
`logs/run_28_mutation_coverage.log`.
Asserted by: nothing yet, deliberately. The harness is a measuring instrument
run on demand, not a test, because a 36-minute sweep in the suite would stop
being run. Whether it becomes a gate is a decision for after the rewiring, when
its output is capable of failing for a reason other than the one above.


## F-007 — 2026-07-25 — Every prior the SI reports and the ensemble draws that could be compared numerically disagreed

`code/tests/test_uncertainty_completeness.py` was written as the narrower
interim step F-006 named, and run for the first time today
(`logs/run_30_uncertainty.log`, exit 1, 27 findings). It compares the
`uncertainty:` block of every registry entry against the prior
`code/repro/run_mc_ensemble.py` actually draws.

`params.yaml` declares 25 uncertainties. Eight are mapped to a drawn prior; two
of those (`n_price_wedge`, `crop_price_usd_t`) defer to a per-region bounds
table that the ensemble reads directly, so they cannot disagree. **Of the six
that could be compared number by number, all six differed**, in 13 of the 18
compared fields:

| parameter | params.yaml declared | run_mc_ensemble drew |
|---|---|---|
| `som_decay_rates` (k_slow) | sd 0.15, [0.6, 1.5] | sd 0.20, [0.60, 1.40] |
| `eps_F_PF` | sd 0.2, [0.5, 1.5] | sd 0.30, [0.50, 1.50] |
| `eta` | sd 0.2, [0.5, 1.5] | sd 0.25, [0.60, 1.40] |
| `residue_retention` | sd 0.15, [0.6, 1.3] | sd 0.10, [0.80, 1.15] |
| `cre_regional` | sd 0.2, [0.05, 0.6] absolute clip | sd 0.30, [0.40, 1.80] multiplier, then clipped to [0.01, 0.99] |
| `whc_sensitivity` | [2.3, 8.4] mm/pp | [2.2995, 8.4] mm/pp |

The direction of correction is not symmetric. The ensemble is what ran, so
`params.yaml` is what was wrong, and every block above has been rewritten to the
drawn prior with a dated note saying so. Editing the ensemble instead would have
made the published interval unreproducible from the code that produced it.

Two of the six are worth separating from the arithmetic. `residue_retention`'s
declared prior was *wider* than the drawn one in both directions, so the SI
table has been advertising an input-carbon uncertainty three times the width of
the one that was propagated. And `cre_regional` wrote two different quantities
into one pair of fields: the numbers `[0.05, 0.6]` are an absolute retention
range, while `low` and `high` in every other block are multipliers on the
central value. That is F-005's error class in a new place, a number recorded
without its basis, and it is why the block now carries `clip_absolute` as an
explicit `[0.01, 0.99]` pair rather than as a bare `true`.

`whc_sensitivity` is a transcription, not a disagreement: the ensemble writes
the lower bound as `0.657`, three significant figures of `2.3 / 3.5 =
0.657143`, so it draws from 2.2995 mm rather than 2.3 mm. Neither side was
edited. Changing the ensemble literal would move the support of a truncated
normal and therefore every draw, to correct 0.02% of one bound of a
meta-analytic range. The test declares the rounding explicitly
(`ensemble_sig_figs=3` with a required `rounding_reason`) and still fails on any
residual the declared rounding cannot produce.

**Seventeen declared uncertainties are never sampled.** Fourteen had no recorded
reason, which is the finding underneath the finding: a prior in `params.yaml` is
what the SI parameter table prints, so the paper has been stating priors on
fourteen parameters it holds fixed. Each now carries a reason in `MAPPING`, and
each reason is required to state both the mechanism and the consequence for the
SI limitations section, because an unpropagated uncertainty is a sentence the
limitations owe the reader. Three of those reasons are themselves findings:

- **`root_shoot_c_ratio` is an understatement, not a cancellation.** The model
  forms residue carbon as `c_in ∝ (residue_retention + root_shoot_c_ratio)`
  (`monthly_model_v3:505-507`), a sum, not a product. Only the shoot term is
  drawn, so input-side carbon uncertainty is propagated on roughly half the
  input and the reported SOC interval is narrower than an input-complete
  ensemble would give. This one should be drawn in the next ensemble.
- **`bnf_potential` and `bnf_ramp_years` document a mechanism the model does not
  have.** Fixation in the published run comes from
  `monthly_model_v3.get_regional_bnf`, a constant landscape average computed
  from legume rotation fraction and net nitrogen credit; it never reads either
  parameter, and there is no ramp anywhere. `bnf_potential` survives only
  because `run_canonical.py:71` copies it into the canonical CSV as a reported
  column; `bnf_ramp_years` has no consumer at all. Both `used_by` fields named
  `soil_n_model.get_regional_bnf`, a function that is neither in that module nor
  a reader of these values. Both entries now carry `superseded_by` and
  `superseded_note` (new registry keys, `code/model/registry.py`), because
  deleting them would erase the fact that the manuscript still describes the
  superseded mechanism. **The manuscript must stop saying fixation ramps in over
  8 to 15 years, and fixation carries no sampled uncertainty at all**, since
  `MANAGED_TRANSITION_PARAMS` is neither registered nor drawn.
- **`residue_c_to_active_fraction` is wired and unsampled.** It is the only
  registered non-conversion parameter the canonical run reads (F-006), the
  mutation sweep put its reach at 2.5e-2 pp on southeast Asia year 1, and it was
  a bare literal until today, so no ensemble that has run could have drawn it.

What this does not assert. That any of these priors is right. It asserts only
that the paper's two statements of each are now the same statement. Whether a
truncated normal on [0.60, 1.40] is the correct prior for `k_slow` is benchmark
B2's question.

Consequence for the manuscript. The SI parameter table is generated from
`params.yaml`, so every number in the six rows above was wrong in the submitted
version, and the fourteen rows carrying a prior the ensemble never drew were
misleading in a worse way, because a declared prior reads as a propagated one.
The corrected table must also mark which rows are declared-but-fixed. No
published result changes: the ensemble drew what it drew, and this finding
changes only what the paper says it drew.

Written to: `code/model/params.yaml` (six corrected `uncertainty:` blocks, each
with a dated note; two `superseded_*` blocks), `code/model/registry.py`
(`superseded_by`, `superseded_note` added to `ALLOWED_KEYS`),
`code/tests/test_uncertainty_completeness.py` (14 reasons, the declared-rounding
mechanism), `logs/run_30_uncertainty.log` (the failing first run),
`logs/run_31_uncertainty.log` (green).
Asserted by: `code/tests/test_uncertainty_completeness.py`, which fails if a new
`uncertainty:` block appears with no decision recorded, if a mapped pair
disagrees on family, sd or either bound, if a NOT_SAMPLED entry carries no
reason, if a declared rounding carries no cause, or if the ensemble draws a
prior no registry entry declares.

---

## F-008 — 2026-07-25 — The first comparison to data the model was never fitted to: temperate yield under nitrogen withdrawal falls about twice as far as a 66-year unfertilized control shows

`code/repro/run_benchmarks.py`, `outputs/benchmarks.csv`, run log
`logs/run_36_benchmarks.log`. 41 rows: 11 PASS, 3 MARGINAL, 1 FAIL, 18
INFORMATIVE, 7 OWED, 1 NOT_APPLICABLE. Fifteen rows are marked STRONG, meaning
the model quantity and the observed quantity are the same quantity measured at
a matched horizon and, where it matters, a matched nitrogen rate.

Every test written for this model before today compares the model to itself: to
a calibration fingerprint, a registry entry, a seam contract, a previous run.
Those catch drift. None of them can catch a model that is internally consistent
and externally wrong, which is exactly what F-002 and F-004 turned out to be.
This is the first check of the other kind. Nothing in `params.yaml` was fitted
to anything in `data/benchmarks/observed_values.yaml`; the only calibration
target in the model is `FAOSTAT_TARGETS`, which appears in none of the
benchmark sources. Every row is therefore held out by construction, and the
planned "calibrate on Broadbalk, then predict Morrow and Sanborn" split was
unnecessary.

### The failure

**B3-europe-YR30.** Yield with no synthetic nitrogen, as a fraction of yield at
the region's current rate, read at the same elapsed year from a common spun-up
equilibrium. This is the paired-plot design of a long-term experiment.

| | model | observed |
|---|---|---|
| 1 year | 0.763 | — |
| 30 years | **0.406** | **0.681 to 0.776** |
| 96 years | 0.364 | beyond the observation window |

The observed values are Prague-Ruzyne, unfertilized since 1954 against its NPK4
arm at 95 kg N/ha, which is close to the model's European regional mean of 85.
The two published ratios are 0.776 over harvest years 1961-1981 (7 to 27 years
unfertilized) and 0.681 over 1983-2020 (29 to 66 years), so the model's 30-year
row is the matched horizon, and it is also the manuscript's own horizon. The
model loses 59 percent of yield where the experiment lost 22 to 32 percent.

The decline between the two observed windows is confounded with the change from
long-strawed to short-strawed varieties, which raised the fertilized arm from
4.9 to 6.9 t/ha, so the observed trend in the ratio is not a clean measure of
soil depletion and no benchmark row is built on it.

A second quantity at a second site points the same way. **B2-BROADBALK-FERT-
MINUS-NIL**: soil carbon under fertilization as an excess over soil carbon
without it, model 38.6 percent at 96 years against 18.6 percent observed at
Broadbalk in 2010 (plot 8 inorganic NPKMg, 30.0 Mg C/ha, against plot 3, nil
since 1843, 25.3 Mg C/ha). MARGINAL against a band of half to twice the single
observed value. At the manuscript's 30-year horizon the model gives 21.0
percent, which is inside the band; the discrepancy is in the extrapolation, not
in the published horizon.

The two rows agree on direction and roughly on magnitude, at two sites, in two
different quantities, one a yield and one a carbon stock. The model's response
to complete nitrogen withdrawal in a temperate system is about twice what the
field record shows.

### What the shape of the model trajectory says

The model front-loads the collapse. Europe's nil ratio goes 0.763 → 0.406 →
0.364 at 1, 30 and 96 years: most of the loss is spent in the first three
decades and the curve then flattens. Prague declines gently and is still
declining at 66 years. Whatever is too strong is therefore in the fast part of
the coupling, the residue-to-active-pool and mineralization feedback that
operates on a 3-year and a 27-year turnover, rather than in the slow carbon
kinetics. `B2-EUROPE-DRIFT-NIL` puts the model's 96-year soil carbon loss under
nil at -28.0 percent, which is internally consistent with a yield ratio of 0.36
halving residue inputs, so the carbon side and the yield side are not
independent failures. They are one mechanism seen twice.

### What is not a failure

**Sub-Saharan Africa is consistent with the meta-analysis once the rate is
matched.** The raw comparison looked bad: the model's marginal physical product
of nitrogen at the regional mean of 7 kg N/ha is 24.8 kg grain per kg N against
an on-farm envelope of 7.7 to 20.0 (B1-SSA-MPP, MARGINAL). But a marginal
product is a derivative at a point, and 7 kg N/ha is the steepest part of the
response curve, while the surveys measured wherever those farmers actually
were. Two inversions settle it without inventing an observation:

* B1-SSA-IMPLIED-RATE-RANGE: the model reproduces an MPP of 20.0 at 25.9 kg
  N/ha and of 7.7 at 109.2 kg N/ha. The Kenyan estimates (17 overall, 17.6 in
  Vihiga, 11 to 20 in the western districts) correspond to model rates of
  roughly 26 to 30 kg N/ha, which is where Kenyan maize farmers who fertilize
  actually are. The Nigerian plot-level estimate of 7.7 would require 109 kg
  N/ha, which they are not.
* B3-sub_saharan_africa-IMPLIED-RATE: the rate at which the model reproduces
  the observed nil-to-fertilized ratio of 0.572 is 47.6 kg N/ha, which is
  inside the range of recommended maize rates the on-farm trials behind that
  meta-analysis would have used.

So the model's nitrogen response in SSA agrees with the Kenyan on-farm
evidence and with the response-ratio meta-analysis, and disagrees with the
single Nigerian plot-level estimate, which is the one estimate whose gap most
plausibly reflects constraints this model does not represent (phosphorus,
weeds, application timing). Neither B1 nor B3 for SSA carries a verdict,
because the compilation does not record the nitrogen rate on either the
surveyed plots or the trials' fertilized arm. Those two missing numbers are
recorded as B1-OWED-SURVEY-RATE and B3-OWED-SSA-TRIAL-RATE; the second is the
single most valuable missing number in the whole compilation.

**B5.** All eight regional own-price elasticities of fertilizer demand fall
inside the range spanned by the retrieved estimates (-1.87 to -0.21) and inside
the tighter 14-study range (-0.7 to -0.2). North America's -0.20 sits 0.01
outside the extreme retrieved estimate and is reported MARGINAL rather than
FAIL, because the endpoint is itself an estimate carrying standard errors of
0.14 to 0.84 and an excursion of 0.05 is inside the reporting precision of the
band. **B4** confirms the whc_sensitivity arithmetic: 1.16 mm per 100 mm soil
per percentage point SOC over a 300 mm profile is 3.48, registered as 3.5.

**`eps_F_N` has no published analogue and must stop being presented as though it
has one.** B5-EPS-F-N-NO-ANALOGUE is filed as OWED. Every study in the
compilation estimates a response to a *price*; `eps_F_N` is the response to a
soil nitrogen *stock*. It is the dial that turns the substitution of soil
nitrogen for fertilizer on and off, so it is load-bearing for the paper's
central claim, and it is the least externally constrained number in the model.
Its numerical coincidence with the sub-Saharan own-price elasticity, both
-0.50, is a coincidence and must not be read as corroboration.

### Rows that pass for the wrong reason, and are marked as such

`benchmarks.csv` carries `informativeness` beside `verdict` because a benchmark
can be passed for reasons that have nothing to do with the model being right.
B2-EUROPE-DRIFT-FERT reports a 96-year soil carbon drift of -0.21 percent under
unchanged management, against an observed Broadbalk mineral-plot envelope of
-8.3 to +4.5 percent. That looks like a pass and is worth nothing: the engine
is initialised at a spun-up equilibrium, so near-zero drift is a property of
the initialisation and any model with a spin-up would reproduce it. It is
marked WEAK and carries no verdict. B2-BROADBALK-FYM is NOT_APPLICABLE: the
model has no organic amendment pathway at all, carbon enters only as crop
residue and roots, and the manure plot holds 73.3 Mg C/ha against 30.0 on the
inorganic plot, so the omitted pathway is large and the manuscript must not
claim the model represents organic amendment.

Three B2 time series were examined and rejected as comparators, with the reason
recorded in the code rather than by silent omission: Broadbalk plot 3 (nil since
1843, drifting around its own low equilibrium, not a fertilizer-withdrawal
transient), Morrow unfertilized (a prairie-conversion legacy the model does not
represent), and Morrow 1964 (79.87 against 58.0 Mg C/ha, nine years after
fertilization began on that plot, so a 37.7 percent carbon difference cannot be
a fertilization response).

### Consequence for the manuscript

The direction is stated, not corrected. **The model must not be retuned to
these benchmarks.** Fitting to them would convert the only external check this
project has into calibration data and would reproduce, at a larger scale, the
error F-002 records.

1. The SI must report the benchmark suite, including the failure. A validation
   section that reports only the eleven passes would be the same defect as a
   test that compares the model to itself.
2. The S3 temperate yield losses are the quantity B3-europe-YR30 bears on
   directly, and the benchmark says the mechanism runs about twice as hard as
   one long-term control shows. The manuscript's temperate loss figures should
   be presented with that stated, and the abstract should not describe the
   magnitude as validated.
3. B3-europe-YR30 is scored against one site. Broadbalk plot 3 has been
   unfertilized since 1843 and its soil carbon is already compiled here, but
   its grain yields are not, and they are the natural second temperate
   comparator (B3-OWED-BROADBALK-YIELD-RATIO). Until they are compiled this is
   a failure against one observation and should be reported as one.
4. The buffering claim itself is not contradicted by any row. What the
   benchmark constrains is the size of the yield penalty when nitrogen is
   withdrawn, not whether soil organic matter cushions it.

Written to: `code/repro/run_benchmarks.py` (new, with the rejected-comparator
reasoning in the docstrings), `outputs/benchmarks.csv`, `outputs/benchmarks.json`
(git commit, params SHA, script SHA, observed-values SHA),
`logs/run_32_benchmarks.log` through `logs/run_36_benchmarks.log`.
Asserted by: `code/tests/test_benchmark_baseline.py`, via `make verify`. The
suite now gates. The verdict recorded here is frozen in
`data/benchmarks/baseline_verdicts.json`, so this failure is tracked without
having to be fixed first, and any movement away from it stops the build. See
F-009.

## F-009 — 2026-07-25 — Two artifacts were being cited as results with no live generator, and one of them held pre-recalibration numbers

The build graph in `code/build.py` declares every generated artifact together
with the script that makes it and the files it reads, then compares recorded
hashes against current ones. Standing it up over the existing deposit found
exactly the two orphans the assurance plan predicted, and one of them was worse
than an orphan.

`data/climate_swap_comparison.csv` was a duplicate of
`outputs/climate_swap_comparison.csv` holding numbers from before the production
path recalibration. The two files disagreed on every row: south Asia's year-10
climate-swap shift read 0.54 pp in the data copy against 0.58 pp in the current
output, sub-Saharan Africa's year-10 loss read 14.0% against 4.87%. Only
`code/repro/climate_comparison.py` regenerates the output copy; nothing had
regenerated the data copy since the recalibration, and nothing could, because no
script writes to that path. It has been deleted. `MANIFEST.md` still describes
this comparison as having a maximum year-10 shift of 0.74 pp, which matches
neither the deleted file (max 0.54) nor the current output (max 0.58); that
number is from a third generation of the analysis and has to be corrected in the
manifest pass.

`data/figS12_curves.json` is read by `make_figure_s12.py` and written by
nothing. The README says it has a generator. This is the second direction of the
same check and needed its own detection rule: an orphan is a file nobody
produces *and* nobody declares, whereas this file is declared as an input, so
only a graph that reasons about inputs as well as outputs sees it. `build.py`
now reports it as UNSOURCED. Figure S12 is therefore drawn from a file of
unknown provenance and stays that way until the generator is written or the
curves are recomputed.

`figures/Figure_S5_flux_decomposition.png` is the one remaining orphan and is a
known gap: `make_figure_s5.py` has not been written yet.

Twenty-three nodes are declared. Twenty-two carry no provenance sidecar because
they predate the graph, which makes that set the regeneration to-do list. It is
recorded in `.build/unstamped_baseline.json` and is pruned on every successful
`make verify`, so a node that has been stamped once loses its exemption
permanently and cannot silently regress.

The same baseline logic carries F-008's failure. `code/tests/test_benchmark_baseline.py`
re-runs the benchmark suite (about six seconds; it does not read the committed
CSV, which would only check that a file had not been edited) and compares every
verdict against `data/benchmarks/baseline_verdicts.json`. It fails on a
regression, on a baselined row disappearing from the suite, on a new row
arriving already failing, and on a row's informativeness being downgraded, which
is the other way to make a failure go away: stop claiming the row proves
anything. It also fails on an *improvement*, until the baseline is regenerated,
so that a change in what the model says about the field record gets written down
at the moment it happens.

Both gates were watched failing before being trusted. Appending a comment to
`params.yaml` turned the stamped `benchmarks` node STALE and returned exit 1;
`test_gate_can_fail` runs the verdict comparison against a synthetic regression,
improvement, disappearance, downgrade and new-failure and asserts each is
reported.

Written to: `code/build.py` (new), `Makefile` (new),
`code/tests/test_benchmark_baseline.py` (new),
`data/benchmarks/baseline_verdicts.json` (generated, 41 rows),
`.build/unstamped_baseline.json` (generated),
`code/repro/run_benchmarks.py` (split into `build_rows` and `main`, plus
`--write-baseline`), `logs/run_41_build_status.log` through
`logs/run_50_build_verify.log`.
Asserted by: `make verify`, which runs nine test files and the build graph and
currently exits 0.
Owed: MANIFEST.md's 0.74 pp climate-swap figure; a generator for
`data/figS12_curves.json`; `make_figure_s5.py`.

## F-010 — 2026-07-25 — The constrained market-clearing test asserted an identity, and a wrong denominator passed it

`code/repro/test_cap_market_clearing.py` was the evidence offered for the v1.3
constrained-cap fix. It asserted that the model's `clearing_residual` column
stayed below 1e-6. That column is

    gamma * (F_hat - [ln(c) - lambda_L * PY_hat])

and the capped solver sets `F_hat = ln(c) - L_hat` with
`L_hat = lambda_L * PY_hat`. The two expressions are the same expression. The
residual is zero by algebra for every possible value of alpha, beta, gamma, eta
and lambda_L, correct or not, so the test could not fail and never tested the
equilibrium. It tested that subtraction works.

The test now re-solves the market from outside the class. For each step it
takes the model's reported PY_hat, F_hat, L_hat, N_hat and the *lagged*
elasticities the solver actually used, and evaluates the four structural
equations (land, fertilizer, supply, clearing) independently; then it
root-finds the food price with `scipy.optimize.brentq` on the excess-supply
function built from those same four equations, bracketing outward until the
sign changes rather than assuming a bracket. The reported price must be that
root to 1e-10.

On the current model the worst structural residual across 337 cap-binding steps
and both scenarios is 1.4e-17 and the worst root gap is 2.8e-17, which is
floating-point noise. The point of the rewrite is what happens when the algebra
is wrong: dropping the gamma term from the capped denominator, so that

    PY_hat = (beta*N_hat + gamma*ln(c)) / (eta - alpha*lambda_L)

instead of `eta - (alpha - gamma)*lambda_L`, drives the structural residual to
3.0e-03 and the root gap to 6.7e-03 and the test returns 1. The old residual
stays at zero under exactly that mutation, because the mutation changes PY_hat
and the identity is defined in terms of whatever PY_hat comes out.

The rewrite also fails if the cap never binds, since a run in which the
constrained branch is never entered would otherwise pass on a model that has no
constrained branch at all.

Two diagnostic columns were added to `CoupledMonthlyModel.run`, `N_hat` (which
already existed) and `ln_cap`, so that the check can be made from the
DataFrame without reaching into solver internals.

Written to: `code/repro/test_cap_market_clearing.py` (rewritten),
`code/model/coupled_monthly.py` (`ln_cap` column),
`results/cap_market_clearing.txt` (new, both scenarios plus the residuals),
`logs/run_51_capclear.log`, `logs/run_52_capclear_mutation.log`,
`logs/run_53_capclear.log`.
Asserted by: `make verify`. The mutation run is recorded in
`logs/run_52_capclear_mutation.log` rather than automated; wiring it into the
mutation harness is owed.

## F-011 — 2026-07-25 — The registry now supplies the model rather than documenting it, and the mutation sweep says so: 45 declared-not-wired leaves fell to 3

F-006 recorded the defect: `params.yaml` held a value, the model held a literal
of the same value, and a consistency test compared them. Perturbing the registry
entry therefore changed no published number and broke only the mirror test. The
registry documented the model. It did not drive it.

The direction of authority has been reversed. `code/model/soil_n_model.py`,
`code/model/coupled_econ_biophysical.py`, `code/model/prices.py` and
`code/model/monthly_model_v3.py` now read their constants from the registry at
import. The eight regions' seventeen quantitative fields in `soil_n_model.py`
were literals until today.

A refactor that changes no number has to be shown to change no number, so three
things were checked rather than asserted. A field-by-field equality log compares
every regional field before and after the rewiring. The benchmark baseline gate
is unchanged, which means the model's agreement with the field record did not
move. A 123-field canonical diff returns zero numeric differences.

WHAT THE SWEEP SAYS

`code/tests/run_mutation_coverage.py` perturbs each registry leaf, re-runs the
canonical S3 model in a sandbox copy of the repository, and scores the leaf on
two axes: did a published number move (REACH), and did any test object (CATCH).

                        before    after
    COVERED                  5       12
    UNTESTED                 0       22
    DECLARED_NOT_WIRED      45        3
    GUARDED_AT_LOAD          6        6
    INERT                    0       13
                            --       --
                            56       56

The before column is `logs/mutation_coverage_prerefactor.csv`; the after column
is `results/mutation_coverage.csv`.

The 45-to-3 collapse is the finding. Forty-two parameters that the model
previously ignored now drive it. The three that remain are `eps_F_N`,
`fert_reduction_target` and `texture_class`, and each is a separate small piece
of unfinished wiring rather than a systemic condition.

UNTESTED IS NOT A REGRESSION, IT IS THE BILL COMING DUE

Nothing moved from COVERED to UNTESTED. The 22 UNTESTED leaves were all
DECLARED_NOT_WIRED before, which is to say the reason no test caught a
perturbation was that the perturbation did nothing. Now it does something and
still nothing catches it. That is the honest state of the test suite and it is
the worklist:

    alpha, atm_n_deposition, baseline_water_deficit, bnf_potential, cn_bulk,
    cropland_mha, eps_F_PF, eps_F_PY, eps_LD_PL, eps_LD_PY, eps_LS_PL, eta,
    laub_tropical_ratios.k_passive_ratio, laub_tropical_ratios.k_slow_ratio,
    physical_feedback_strength, som_decay_rates.k_active,
    som_decay_rates.k_slow, som_humification.h_slow_to_passive,
    synth_n_current, water_stress_coeff, whc_sensitivity, yield_min_regional

Eight of the twelve COVERED leaves are caught by a single test,
`test_spinup_partition_independence.py`. Four more (`cm2_per_ha`, `g_per_t`,
`pct_to_fraction`, `soc_bulk_density`) are caught by `registry_consistency` and
`soc_conversion_invariance`, both of which compare the registry against a
literal. Their coverage is real but it rests on the literal continuing to exist.
A suite whose catching power sits in one behavioural test and two mirror tests
is thin, and the claim register (`docs/claims.yaml`) is the next layer, because
it turns each published number into something a mutation can break.

TWO WAYS AN INERT VERDICT CAN BE WRONG

INERT went from 0 to 13, and two of the thirteen are artifacts of how the probe
is built rather than statements about the parameters.

First, the fingerprint is too narrow. `probe()` builds its comparison from
`data/canonical_ERA5_y30.json` alone, flattened by `_flatten_canonical()` into
the per-region numeric fields plus `global_prodweighted`. Gross margins, prices
and cost shares are not in that artifact. So `crop_price_usd_t`,
`n_price_wedge`, `n_price_usd_kg_farmer_paid`, `n_benchmark_usd_kg`,
`urea_n_fraction` and `price_benchmark_max_factor` score INERT by construction,
not because they are irrelevant: every one of them moves the gross-margin
figures the abstract quotes. The published-number set the probe fingerprints has
to include the margin outcomes before an INERT verdict on a price parameter
means anything. Until it does, those six rows should be read as "not probed".

Second, `cre_base` scores INERT because `cre_regional` overrides it in all eight
regions. The base value is dead code reached by nothing. That is a real INERT,
and the repair is to delete the fallback rather than to widen the probe.

THREE INERT VERDICTS THAT CORROBORATE SOMETHING ELSE

`yield_max_regional` is INERT, and SI [65] already says why: the static values
are legacy fallbacks, not the reported calibration, which solves for y_max
against the FAOSTAT target. Document and sweep agree.

`bnf_ramp_years` is INERT, and both MS [78] and SI [14] state that fixation is
held static during disruption scenarios. The string "ramp" does not appear
anywhere in the v14 documents. So a ramp parameter is registered, carries
per-region values of 8 to 15 years, drives nothing, and is denied by the text.
It must be marked declared-but-fixed in the SI parameter table or removed.

`pop_supported` and the three `som_pool_cn` entries are reporting quantities
that no published number in the canonical artifact depends on. They are
candidates for the same treatment once the probe is widened.

Written to: `results/mutation_coverage.csv`,
`results/mutation_coverage_summary.txt`, `logs/run_67_mutation.log`.
Compared against: `logs/mutation_coverage_prerefactor.csv`,
`logs/mutation_coverage_summary_prerefactor.txt`.
Still owed: widen `_flatten_canonical` to the margin and price outcomes; delete
the `cre_base` fallback; mark `bnf_ramp_years` and `yield_max_regional`
declared-but-fixed; write a test for each of the 22 UNTESTED leaves.

## F-012 — 2026-07-25 — The claim register found nineteen drifted numbers on its first run, and the largest block traces to one stale artifact

`docs/claims.yaml` records 19 quantitative claims made in the manuscript, the
SI or the author response, each with the sentence that states it, the artifact
that should produce it, and one or more arithmetic checks over that artifact.
`code/tests/test_claims.py` resolves every check and fails the build when a
published number and the model disagree beyond tolerance.

First full run: **60 checks, 41 agreeing, 19 drifted across five claims, zero
unresolved paths.** Every drift is a real edit owed to v15, not a tolerance
artefact; the smallest is 0.118 pp against a 0.1 pp tolerance and the largest
is 14.98 pp.

**The single largest block is C-060, and it has one cause.** Every year-10
regional figure in MS [56] — East Asia 1.3, South Asia 6.0, FSU 5.5, SSA 5.4,
global 3.4 — matches `data/figure2_panels.json`, which predates the
production-path recalibration. None matches `data/canonical_ERA5_y30.json`,
which is what the deposit ships (1.182, 4.812, 5.126, 4.749, 3.03). Two checks
in the same claim still agree: `global_yr1` (2.32) and total cropland
(1230 Mha). So this is not a paragraph that was written carelessly. It is a
paragraph whose numbers were correct when written and were never re-read
against a rerun. That is exactly the failure the register exists to catch, and
the register caught it on the first run rather than at proof stage.

**C-021 shows the same failure in the other direction.** The registry carries
`whc_sensitivity` = 3.5 mm per pp SOC (Minasny & McBratney 2018); MS [31],
MS [28] and AR [40] all still say 8.4. The code change landed and the prose
did not follow it. A register that reads the registry rather than a document
is the only thing that would have noticed.

**C-014 and C-030 confirm the fertilizer-cost-share correction propagated
into the artifacts and not into the text.** The v14 margin gaps of 2.5–4.2 pp
were computed with the hardcoded 25% cost share; with derived regional shares
they are 0.27–0.99 pp. SSA's nitrogen price moved from the implied $1.40/kg N
to the registry's $2.30. The abstract edit already on the worklist (~0.3–1.0 pp)
is the right one, and now a test will fail if it is not made.

**C-031 settles a conflict between two documents in the main text's favour.**
MS [53] says the SOC-related spread is 0.2–1.5 pp; SI [197] says 0.1–1.5. All
four checks agree with the model, whose minimum spread across regions at the
100–150% shock is 0.214. SI [197] is the sentence to correct. A claim stated
twice is a claim that can drift apart, and the register is what makes the
two statements comparable instead of merely coexisting.

**What the register refuses to do.** `text` and `location` are transcribed
verbatim from v14. They record what has been claimed; they are not a target
the model may be tuned toward. When a check disagrees, the default repair is
to the document. Four claims carry `status: owed_generator` — no script yet
produces the number, so the claim is recorded rather than silently published
unbacked (C-011 figure-2 regeneration, C-041 figure S8 curves, C-050 the S3
shock calibration, C-061 the one-year pulse). `test_owed_count_may_only_shrink`
means that debt cannot grow.

**The gate is two-way.** `depends_on_params` in the register is the reverse
index of `affects_claims` in `params.yaml`, and both directions are checked.
On the first run this failed on C-061, which listed `eps_F_PF` and
`som_decay_rates` as affecting it while neither parameter declared the claim.
Fixed in `params.yaml`. A register that only pointed one way would have let
a parameter change without anyone knowing which published sentences moved.

**An improvement also stops the build.** `docs/claims_baseline.json` records
the five drifted claims and the four owed generators. A check moving to AGREES
without the baseline being regenerated fails, on the same principle as the
benchmark and unstamped baselines: a number that gets better silently is a
number nobody read.

**Side effect on the build graph.** Adding `affects_claims: C-061` to two
parameters marked every params-dependent node STALE, because the node
fingerprint hashed the bytes of `params.yaml`. A documentation edit that
cannot change any output should not invalidate artifacts, or STALE stops
carrying information. `build.params_fingerprint()` now hashes the document
with `DOCUMENTARY_KEYS` removed (provenance, note, used_by, affects_claims,
benchmark, source, citation and their kin). The list is a denylist so a new
key is fingerprinted by default and must be exempted deliberately.
`results/climate_swap_stats.txt` also stopped being an orphan:
`climate_comparison.py` now writes it instead of printing the two numbers for
a human to transcribe, which is how 0.74 pp and rho=0.93 survived
recalibration in both the README and the response letter (F-009).

Written to: `docs/claims.yaml`, `docs/claims_baseline.json`,
`code/tests/test_claims.py`, `logs/run_76_report.log`.
Compared against: `data/canonical_ERA5_y30.json`, `data/figure2_panels.json`,
`data/figure1_farm_gradient.json`, `data/figS11_severity_sweep.json`,
`data/food_price_response.csv`, `code/model/registry.py`.
Still owed: the four `owed_generator` scripts; the document edits for C-010,
C-014, C-021, C-030, C-060 and SI [197]; regeneration of
`data/figure2_panels.json` (C-011).

## F-013 — 2026-07-25 — Two of the three regional rankings the paper asserts are not supported by the posterior, and one of them names the wrong region

Run IDs: `logs/run_86_cs.log` (scoring, post-recalibration ensemble),
`logs/run_88_cs.log` (gate), `logs/run_90_neg.log` (gate watched failing),
`logs/run_94_verify.log` (full gate, exit 0, 15 tests).
Artifacts: `results/claim_strength.csv`, `results/claim_strength.md`,
`docs/claim_strength_baseline.json`.

Assurance plan 3.8 asks a question no unit test asks: before a ranking is
written down, how often does the ensemble actually produce it. Every number
behind the v14 rankings was computed correctly. What was missing was the step
between a computed ordering and a stated one.

Three ordering families are scored against the 1000-draw post-recalibration
ensemble, under thresholds declared in `code/repro/make_claim_strength.py`
rather than chosen per claim (0.90 to state a ranking, 0.60 to hedge it):

    P3   highest year-1 yield loss          fsu_central_asia  p = 0.998  state
    P4   worst year-1 net-revenue change    south_asia        p = 0.542  not separable
    P4b  highest derived nitrogen cost share south_asia       p = 0.958  state

Each family accounts for its full mass, so the leader probabilities are
probabilities of leading and not of leading among the regions the scoring
happened to see.

SI [163] states all three in one sentence. Registered as C-062, C-063 and
C-064, that sentence fails twice.

**C-063, the P4 claim.** "Sub-Saharan Africa is the worst region for year-1
gross margin in 83.7% of draws." The 83.7% was computed against a hardcoded
`FERT_COST_FRAC` dictionary that assigned Sub-Saharan Africa 0.25 and North
America 0.08 and varied neither, so the probability measured the assumption.
With regional nitrogen and crop prices registered and drawn as per-region
multipliers, Sub-Saharan Africa is worst in 0.000 of draws, South Asia leads at
0.542, East Asia is at 0.447, and the two together account for 0.989. Below
0.60 the paper is licensed to report the group and nothing more. The v15 text
should read that South Asia and East Asia are not separable.

**C-064, the P4b claim.** The same sentence attributes the P4 result to Sub-
Saharan Africa's "high fertilizer-cost share." Under derived regional prices
the highest share is South Asia in 0.958 of draws and Sub-Saharan Africa leads
in none. The 0.25 that clause rested on implied a Sub-Saharan African baseline
price of $15.53 per kg N.

C-064 is the finding that changed the gate. A threshold check alone would have
passed it: "most exposed" is a licensed form of statement about a family whose
leader clears 0.90. The claim is wrong about *which* region, not about how
strongly to say it. So `claim_strength` carries a `region` and the test now
fails a claim that names a region the ensemble does not put first, at any band.
`logs/run_90_neg.log` records the gate rejecting a deliberately mislabelled
C-062 before the real claims were trusted to it.

**C-062, the P3 claim,** survives: FSU/Central Asia at 0.998 may be stated
outright. It is registered anyway so that it is scored rather than assumed. A
later ensemble that moves it below 0.90 stops the build instead of leaving a
sentence standing on the strength of having once been true.

The two overstatements are carried in `docs/claim_strength_baseline.json`
pending the v15 document edits, on the same terms as the other four baselines:
the list may only shrink, and a claim that comes into line without the baseline
being regenerated also fails.

**Rescoring against the post-recalibration ensemble moved the numbers but not
the verdicts.** P4b fell 0.984 to 0.958 and P4 fell 0.559 to 0.542. Any
claim-strength figure quoted from before `logs/run_83_mc.log` is stale.

Also this entry: SI Table S1 is now generated from the registry
(`code/repro/make_table_s1.py`, gated by `code/tests/test_table_s1.py`), and
its `varied_in_mc` column reports the ensemble rather than the declaration. Of
54 registered parameters the ensemble draws 8, holds 17 fixed that declare an
uncertainty, exempts 25 with a stated reason, and 4 are not applicable. The 17
are the honest disclosure: the credible intervals this paper reports do not
contain them. Their justifications moved out of a test file into
`code/model/mc_mapping.py`, because the SI has to print them and a generator
importing a test would have rebuilt the one-concept-two-places failure the
whole rebuild is about.

Still owed: the C-063 and C-064 sentence edits; the four `owed_generator`
scripts; the document edits for C-010, C-014, C-021, C-030, C-060 and SI [197];
regeneration of `data/figure2_panels.json` (C-011) and the remaining downstream
artifacts under the new ensemble.

## F-014 — 2026-07-25 — Regenerating the chain moved one artifact and one claim, and exposed a node that had been reporting STALE for a reason nobody had read

Run IDs: `logs/run_95_all.log` (26 nodes, all succeeded), `logs/run_98_prices.log`,
`logs/run_101_verify.log` (exit 0, 14 suites).
Report: `logs/regeneration/regeneration_report.csv`, `.md`.

The whole build graph was re-run against the recalibrated production path and
the post-recalibration ensemble. Thirty-two artifacts changed, forty-six were
byte-identical. The Monte Carlo reproduces exactly: `mc_summary.csv`,
`mc_probabilities.csv` and `mc_priors.json` are unchanged bit for bit, and only
the gzip container of `mc_posterior.csv.gz` differs.

**`data/canonical_ERA5_y30.json` did not move.** Production-weighted global
yield loss stands at 2.32% in year 1 and 3.03% in year 10, and every regional
year-10 loss is unchanged: FSU/Central Asia 5.126, South Asia 4.812, sub-Saharan
Africa 4.749, Southeast Asia 3.675, Europe 3.429, Latin America 2.418, North
America 1.726, East Asia 1.182.

**`data/figure2_panels.json` is where the recalibration had not landed**, which
is what F-012 predicted from the register. Global total 3.412% to 3.032%
(direct 3.094 to 2.870, SOM penalty 0.319 to 0.161). South Asia 5.979 to 4.812,
Southeast Asia 4.485 to 3.675, sub-Saharan Africa 5.415 to 4.749. Its regional
totals now agree with the canonical year-10 losses to three decimal places,
which they did not before. C-011 leaves `pending_regeneration`.

**One claim newly drifted: C-042**, the regional output-price indices. Model
now gives year-1 production-weighted 5.34% against a stated 5.5%, year-10 5.95%
against 5.0%, Latin America year-10 3.97% against 1.0%, FSU/Central Asia 10.16%
against 10.3%. The Latin American figure is the one that matters: 1.0% was the
low end of the stated year-10 range and is now nearly four times that, so the
sentence's claim about how wide the regional spread is has changed, not only
its numbers. Recorded as `document_edit_owed` on C-042 and added to
`docs/claims_baseline.json` deliberately, per the rule that a baseline grows
only with an entry here saying why.

**The `prices` node had been reporting STALE for a defect in its own
declaration.** It claimed three outputs and writes one.
`data/eurostat_price_observations.json` is a hand-transcribed observation file
`derive_prices.py` only reads, and `data/nitrogen_price_table.json` is written
by `run_price_shock_analysis.py`. So the node was stale immediately after a
clean run of itself, permanently, and the only way to read that signal was to
learn to ignore it. This is the same failure as the params fingerprint in
F-012, arriving by a different route: a staleness signal that fires when
nothing is wrong stops being a staleness signal. The declaration now names one
output, takes the observation file as an input, and lists it under
`EXTERNAL_INPUTS` with a note on where the numbers come from. `nitrogen_price_table
.json` moved to the `price_shock` node that writes it.

Still owed: the C-042, C-063 and C-064 sentence edits, on top of the list in
F-013.

---

## F-015. The +104% shock is right; "averages approximately 20%" is stated on a basis the paper never names, and on the paper's own basis it is 18.7%

**Date** 2026-07-25. **Run** `logs/run_104_s3cal.log`, `logs/run_105_rep.log`.
**Artifacts** `results/s3_shock_calibration.csv`, `results/s3_shock_calibration.json`.
**Generator** `code/repro/make_s3_shock_calibration.py`. **Claim** C-050,
`owed_generator` -> `current`.

Three numbers appear in three document locations (MS [56], MS [65], SI [78]):
a +104% fertilizer price shock, a 20% S1 reduction it is calibrated to
deliver, and an S3 reduction that "averages approximately 20% over the
sustained-disruption period." The 20% target was registered as
`fert_reduction_target` and therefore checkable. The other two were not
written anywhere. They could be confirmed only by rerunning the model and
reading a number off a console, which is the F-009 mechanism: a number a
person transcribes is a number that survives the recalibration which
invalidated it.

The generator now solves and deposits all three. The solve reproduces:
**+103.90%**, which rounds to the stated +104%. S1 realizes **0.2000** by
construction on the nitrogen-tonnage basis. The sustained mean under S3, with
`eps_F_N` active, is **0.1911**.

The defect the deposit exposes is not in either number. It is that the sentence
does not say on what basis the reduction is averaged, and the two available
bases are not close. The scenario removes a fifth of the world's applied
fertilizer nitrogen, which is a mass, so the calibration targets nitrogen
tonnage. The paper's outcome aggregates are production-weighted. On the
production basis the same two figures are **0.1964** and **0.1872**. This is
F-005 again, in a sentence F-005 did not reach.

Two edits are owed and are recorded in C-050's `document_edit_owed`. State the
basis. And report the S3 figure as 19%, not 20%: the gap between the calibrated
reduction and the realized one *is* the depletion feedback, and rounding it away
erases the mechanism the S3 scenario exists to demonstrate. Farmers buy back
part of the reduction as soil nitrogen falls, and they buy back more of it over
time, from 0.1911 averaged over years 1-10 down to a year-30 realized reduction
that is smaller again in every region.

The sustained-disruption period is declared in the generator as
`SUSTAINED_YEARS = (1..10)` rather than chosen once the numbers were visible.
"Averages approximately 20% over the period" is only a claim after the period
is fixed; before that it is a search over windows.

Per-region realized S3 means span 0.126 (North America) to 0.292 (sub-Saharan
Africa). That is the own-price elasticity acting on regions with very different
baseline application rates. The shock itself is uniform.

**Baseline growth.** `docs/claims_baseline.json` gained C-041 in `drifted`, and
`owed` lost C-041 and C-050 (`owed_count` 4 -> 2). C-041's three new drifts are
the Figure S8 caption numbers that only became checkable once
`compute_figS8_curves.py` deposited its `summary` block: document 3.4/1.9/46%
against model 3.032/1.519/49.91%. They are not new errors, they are newly
visible ones, and the caption edit is owed under C-041.

**A second reverse index that was decoration.** Adding checks to C-041 and
C-050 tripped `test_depends_on_params_mirrors_the_registry`: `eps_F_PF`
declared `affects_claims: [C-041]` while C-041 listed no parameters, and C-050
named `eps_F_N` in its own text without `eps_F_N` declaring the claim. Both
directions are now stated: `eps_F_N` gained `affects_claims: [C-040, C-050]`.
The forward index existed the whole time; nothing read it until a claim
acquired a check.

---

## F-016. The last three owed generators are closed; the one-year pulse recovers thirty times faster than the paper says, and SSA's 30-year SOC decline is 2.14% not 2.5%

Runs: `logs/run_120_traj.log` (pulse), `logs/run_121_soc.log` (SOC),
`logs/run_122_soc_test.log` (SOC test), `logs/run_126_verify.log` (gate).
Artifacts: `data/scenario_trajectories.csv` (new `PULSE1_global` column),
`data/soc_trajectories.csv` / `.json`.

**The register now has zero owed generators.** `docs/claims_baseline.json`
`owed_count` goes 4 to 0 across F-015 and this entry: C-050 and C-061 out of
`owed_generator`, C-011 out of `pending_regeneration`, and C-010 out of an
`owed:` note. Every sentence in the register is scored against a live artifact
on every `make verify`. Nothing is left that the register admits it cannot
check.

**C-010 carried an `owed:` note while its status said `current`, so nothing
counted it.** `write_baseline`'s owed counter reads `status`, and C-010's said
the claim was checked. Four of its five checks were. The fifth, the SOC
decline, had no artifact behind it and said so in a field the counter does not
read. An `owed:` note on a claim nothing counts is the same object as an
unwired contract or a registry the model does not read: a comment. No other
claim in the register has this shape; that was checked, not assumed.

**The SOC deposit.** `code/repro/make_soc_trajectories.py` runs S3 at the
calibrated shock for 30 years and writes the carbon stock per region per year,
which the canonical run has never deposited. For a paper arguing that soil
organic matter buffers a fertilizer disruption, the SOC trajectory was the
wrong thing to be missing. Sub-Saharan Africa declines **2.145%** over 30 years
against a stated 2.5% (`tol: 0.15`, taken from the sentence's own precision
rather than widened to fit). Per region, year-0 to year-30 stock in t/ha and
the 10-/30-year decline: NA 50.69 to 50.39 (0.30/0.61%), EU 42.60 to 42.12
(0.58/1.12%), EA 35.47 to 35.32 (0.22/0.44%), SA 17.37 to 17.01 (1.14/2.09%),
SEA 22.24 to 21.87 (0.88/1.65%), LATAM 31.26 to 30.91 (0.58/1.13%), SSA 6.18 to
6.04 (1.09/2.14%), FSU 35.31 to 34.72 (0.85/1.67%).

**The SOC ordering is not the yield-loss ordering, and the paper reads as if it
were.** South Asia loses 2.09% of its carbon from a stock less than a third of
North America's, so a small percentage there is a smaller absolute buffer than
the same percentage anywhere else. FSU declines 1.67%, mid-pack, while carrying
one of the two largest year-30 yield losses. Percentage SOC decline is not a
proxy for exposure and should not be presented beside the loss ranking without
saying so.

**The duplication is checked rather than trusted.** The SOC generator is a
second script running the same configuration as `run_canonical.py`, which is
exactly the condition that has produced entries in this file before. It is a
separate script because canonical is the root of the build graph and widening
it restales every downstream node for a change that alters no existing number.
So it also deposits each region's year-1/10/30 yield loss, and
`code/tests/test_soc_trajectories.py` fails if any differs from
`data/canonical_ERA5_y30.json` by more than 0.005 pp. If the two configurations
diverge, the loss columns diverge first and the test names the region and the
column. All eight regions agree today. The test also asserts SOC falls
monotonically in every region, because the shock never lifts in S3 and a rising
year would be a sign change in the mechanism the paper is about.

**The one-year pulse: year 1 reproduces, year 5 does not.** C-061 states that a
single-year disruption costs 2.3% in year 1 and still leaves about 0.3% in year
5. The model gives **2.316%** and **0.009%**. The full curve, by year:
1: 2.316, 2: 0.492, 3: 0.044, 4: 0.015, 5: 0.009, 6: 0.007, 7: 0.006,
8: 0.005, 9: 0.005, 10: 0.004. The residual crosses 0.3% between year 2 and
year 3, so if 0.3% was ever measured it was measured in year 2. The model
recovers roughly thirty times more completely by year 5 than the sentence
claims.

**And the sentence credits a mechanism that is off.** It attributes the
recovery to soils re-equilibrating *and* food prices normalizing. Under a
pulse the price returns to baseline by construction at the end of year 1, so
everything after year 1 is the soil alone. The second half of the attribution
is not a magnitude error; it names a process the scenario has already switched
off.

**A pulse is a new capability, not a reinterpreted old one.**
`EconParams.fert_price_shock_years` is a square pulse: full strength for
`t < years`, exactly zero after. It is deliberately distinct from the existing
`fert_capacity_recovery_years` ramp, and `get_pulse_scenario` refuses to build
a pulse of a recovering scenario, because two recovery mechanisms at once give
a trajectory attributable to neither. `get_pulse_scenario` takes an
already-configured scenario and `dataclasses.replace`s one field, which
introduces zero new numeric literals and makes "identical to S3 in every other
parameter" true by construction rather than by docstring.

**The `>=` that zeroed every year.** The pulse first read 0.000 at every
reported year. The condition was `t >= fert_price_shock_years`, and the model
samples year 1 at `t = 1.0`, so a one-year pulse ended before the harvest it
was supposed to damage. Corrected to `t >`, with the failure mode recorded in
the comment. Worth noting how this one presented: not as an error but as a
clean column of zeros, which is a readable result. A one-year disruption
costing nothing is a publishable-looking finding.

**The claim-register resolver could not select a CSV row by an integer key.**
Registering C-061's checks failed three times: `[year=1]` (the CSV loads as
`{'rows': [...]}`), `rows[year=1]` (`_maybe_float` turns the key into `1.0` and
`_apply` string-compares), `rows[year=1.0]` (`.` is a path separator, so the
segment does not parse). The resolver reported this as "matched 0 elements",
which reads as a claim about the artifact rather than about the resolver. Fixed
in the resolver with `_sel_eq`, not worked around in the claim.

**Baseline growth.** No id enters `docs/claims_baseline.json` `drifted` in this
entry that was not already there; C-010's fifth check adds a drift to a claim
already carried. The drifted set stands at C-010, C-011, C-014, C-021, C-030,
C-041, C-042, C-060, C-061. Register report: AGREES 42, DRIFTED 28. Build graph
28 nodes, all OK, one orphan (`figures/Figure_S5_flux_decomposition.png`) and
one unsourced input (`data/figS12_curves.json`), both tracked in F-009/F-010.

**Document edits owed from this entry:** SSA 30-year SOC decline 2.5% to 2.14%;
pulse year-5 residual 0.3% to 0.009%, and drop the food-price half of its
attribution; state that percentage SOC decline does not order regions the way
yield loss does.

---

## F-017 — 2026-07-26 — Supplementary Table S4 is transcribed from a file nothing wrote, the graph declared the wrong input for the figure that was blamed for it, and the figure the handoff owed a generator for is not in the paper

D3's three owed items, worked against the rebuilt tree rather than against
HANDOFF §7's list. All three had moved since the handoff was written.

### The calibration table was stale, and it is the SI's table

`data/crop_response_calibration_table.csv` was written by no script in the
deposit and had not been since at least the v14 deposit. `MANIFEST.md` credited
`make_table_s4_sol.py` with regenerating it, and that script did not. F-009
recorded the class of defect but named `data/figS12_curves.json`, which has had
a live generator (`make_table_s4_sol.py` line 63) since before the
reconstruction base; the wrong file got the work package.

The file matters more than "an unsourced input read by Figure S13" suggests,
and in a different way, because **it is not read by Figure S13 at all**.
`make_ofra_validation.py` line 15 opens `outputs/Table_S4_calibration_sol.csv`.
The build graph declared `crop_response_calibration_table.csv` as that node's
input, so a file nothing wrote was recorded as feeding a figure that never
opened it. That is the third instance of declared-versus-actual in this deposit,
after F-014's `prices` node and F-009's `table_s4` node, and the first in the
input direction.

What actually reads the file is **the Supplementary Information**.
Supplementary Table S4 in `v14_sol` is a row-for-row transcription of it, and
the transcription is of the pre-recalibration copy:

| | frozen file / SI Table S4 | regenerated |
|---|---|---|
| N America N_current | 223.9 | **142.7** |
| N America N no-synth | 147.9 | **66.7** |
| N America y_max | 6.277 | **6.198** |
| S Asia y_max | 3.636 | **3.773** |
| L America y_max | 5.602 | **5.414** |
| SSA y_max | 3.876 | **3.967** |
| SSA y(no-synth) sim | 1.26 | **1.29** |

The regenerated values reproduce `data/figS12_curves.json` and
`outputs/Table_S4_calibration_sol.csv` — the two artifacts of the same script
that did have generators — to the digit, both of which are byte-identical
before and after this pass. So the two sourced outputs were current and the
unsourced one was two recalibrations behind, in the same script's output set.
**Every numeric column of Supplementary Table S4 except `FAOSTAT y_obs`, `c` and
`Floor` is wrong in `v14_sol`.** The nitrogen columns are wrong by 36% and 55%.

`make_table_s4_sol.py` now writes all three outputs. Only `floor_source`
survives as a literal, and it is documentary.

Figure S13 is unaffected and its PNG is byte-identical after regeneration: it
reads the sourced table, and its caption claim — the SSA ceiling below the OFRA
median and inside the IQR — holds at the regenerated ceiling (3.967 against a
median of 4.47, IQR 3.07–6.73) as it did at 3.876.

### The climate-swap number was smaller than recorded, and had a console for a source

F-009's "0.74 pp" is not in this tree. `MANIFEST.md` and `README.md` both read
0.69 pp with a Spearman ρ of 0.95. The regenerated comparison gives **0.70 pp
and ρ = 0.98**; the shift was nearly right and the ρ had drifted. Both are now
in `results/climate_swap_stats.txt`, written by `climate_comparison.py`, which
until this pass printed them to a console — the mechanism F-009 named, still
operating on the finding that named it, three generations after 0.74 pp entered
the manifest by transcription.

### Figure S5 is not in the paper

`make_figure_s5.py` has been carried as a debt since F-009 and appears in §7 of
the handoff. It should not be written. `figures/Figure_S5_flux_decomposition.png`
is not in this tree; the file exists only under `excluded_legacy_sol/`. The
scheme it draws — CUE respiration against non-recycled necromass in the
microbially-explicit 4-pool comparison — was never deposited, so it cannot be
regenerated. And the SI's own paragraph 200 reads "The unsupported 4-pool
quantitative comparison and Figure S5 were removed", which is the **only**
surviving reference to S5 in either document. The debt is retired in
`MANIFEST.md` and `README.md` with the reason recorded, rather than discharged
by generating a figure the paper does not contain.

Written to: `code/repro/make_table_s4_sol.py` (third output plus `FLOOR_SOURCES`),
`code/repro/climate_comparison.py` (stats deposit), `code/build.py` (`table_s4`
third output, `climate_swap` stats output, `ofra_validation` input corrected),
`Makefile` (allowance removed, not exempted), `MANIFEST.md`, `README.md`,
`data/crop_response_calibration_table.csv` (regenerated),
`results/climate_swap_stats.txt` (new).
Asserted by: `make verify` — thirteen suites and the build graph, 30 nodes at
BLOCKED 2 / OK 28, one orphan and two unsourced inputs allowed by name, exit 0.
The unsourced list is down from three to two and the removed entry is this one.

**Document edits owed from this entry, and they belong to D2:** Supplementary
Table S4 must be reprinted from the regenerated
`data/crop_response_calibration_table.csv`. Supplementary paragraph 200's claim
that "the deposit now includes generators for every reported numeric table" was
false when written and is true as of this commit.

## F-018 — 2026-07-28 — The v15 tree was lost and rebuilt from your Mac; the SOC deposit it carried was computed at the superseded eps_F_N and every year-10 and year-30 number in it is low

The working tree at `/tmp/repo14` was destroyed between two consecutive
commands. Nothing in this container survived it. What did survive was the git
repository in the `v15` folder on the author's machine: 29 commits, clean
working tree, HEAD `d979a0c` of 2026-07-26. It was bundled on that machine,
staged, and cloned to `/tmp/repo16`, where `make verify` returns exit 0 with
thirteen suites passing and the build graph at 30 nodes, 28 OK and the two
documented BLOCKED. The rebuild is therefore intact. What was lost is the two
days of work standing on it, and the lesson is the obvious one: a tree that
exists in exactly one place is a tree that can stop existing.

`code/repro/make_soc_trajectories.py` and `code/tests/test_soc_trajectories.py`
were recovered verbatim from the session transcript and replayed onto this tree
(`logs/run_201_soc.log`, `logs/run_202_soc_test.log`). The test passes here,
including its property that both deposits agree on every yield loss.

That agreement is what exposed the error. `data/soc_trajectories.json` as
deposited by the lost tree does NOT match what the generator produces here, and
the pattern of the mismatch names its cause: year-1 losses agree to five
decimals and every year-10 and year-30 loss is higher in the regenerated file.
That is the eps_F_N signature described in the `scenario_trajectories` block
note. The deposit was made at eps_F_N = -0.5, the superseded family. So the SOC
numbers reported in F-016 belong to the family this repository has already
retired, and F-016 should be read with that correction in front of it.

Under the current family, 30-year SOC decline, largest first:

| region | yr0 t/ha | yr30 t/ha | yr10 decline | yr30 decline |
|---|---|---|---|---|
| south_asia | 17.37 | 16.98 | 1.1888% | 2.2411% |
| sub_saharan_africa | 6.18 | 6.04 | 1.1151% | 2.2404% |
| fsu_central_asia | 35.31 | 34.67 | 0.8963% | 1.8247% |
| southeast_asia | 22.24 | 21.85 | 0.9101% | 1.7295% |
| europe | 42.60 | 42.09 | 0.6030% | 1.2062% |
| latin_america | 31.26 | 30.90 | 0.5974% | 1.1713% |
| north_america | 50.69 | 50.37 | 0.3133% | 0.6325% |
| east_asia | 35.47 | 35.31 | 0.2232% | 0.4508% |

Three things follow for the manuscript, and the third is the one that matters.

First, C-010's number changes again. The paper says sub-Saharan Africa's SOC
declines 2.5% over thirty years; F-016 corrected that to 2.14%; the current
family gives 2.2404%. The correction owed to the text is 2.5% to 2.24%.

Second, the year-10 figures move with it, so any sentence quoting a decade
decline needs re-reading against this table rather than against F-016's.

Third, and this is not a number correction: South Asia at 2.2411% and
sub-Saharan Africa at 2.2404% are separated by 0.0007 percentage points. Any
sentence naming sub-Saharan Africa as the region whose carbon falls furthest is
asserting an ordering the model does not resolve. Seven ten-thousandths of a
percentage point is not a ranking, and reporting it as one repeats the error
F-013 found in the regional rankings: a difference too small to survive its own
uncertainty, stated as though it were a result. The honest sentence is that
South Asia and sub-Saharan Africa lose the largest share of their carbon and
are not separable from each other.

`document_edit_owed` for C-010 accordingly becomes: correct 2.5% to 2.24%, and
replace the sub-Saharan-Africa-highest phrasing with the two-region statement.

## F-019 — 2026-07-28 — The global carbon-retention fallback is deleted; it moved no number because the model could never reach it, and that was the problem

`FeedbackParams.cre_base` was a registered parameter: value 0.11, provenance
Lehtinen et al. (2014) *Soil Use and Management* 30:524-538,
doi:10.1111/sum.12134, `used_by: [soil_n_model.FeedbackParams]`,
`affects_claims: [C-010]`, `mc: declared_fixed` with a written exemption
reason. It appeared in `params.yaml`, in the registry's entry count, in the
parameter ledger, in Table S1, and in C-010's `depends_on_params`. By every
signal this repository emits, it was part of the model.

It was not. All three call sites read
`region.cre_regional if region.cre_regional > 0 else fb.cre_base`, and all
eight regions set `cre_regional` (0.28, 0.259, 0.226, 0.341, 0.307, 0.308,
0.20, 0.35). The branch was reached by nothing. F-011's mutation sweep scored
the leaf INERT, which was correct and was not a clean bill of health: it meant
the documentation described an assumption the model does not make.

v15 deletes it. `soil_n_model.region_cre()` replaces the guard and raises on an
unset regional value. Six files changed: `soil_n_model.py` (field, two call
sites, one comment, the new helper), `coupled_econ_biophysical.py` (import and
call site), `params.yaml` (the entry, saved to `/tmp/cre_base_block.txt`),
`make_parameter_ledger_sol.py` (two dictionary entries), `run_mc_ensemble.py`
(docstring), `docs/claims.yaml` (C-010's `depends_on_params`).

The verification is the point. `logs/run_203_canon.log` regenerates the
canonical artifact after the deletion and it diffs to **zero over all 125
fields** against the pre-deletion copy: no field moved, none appeared, none
went missing. That is what an unreachable parameter looks like when you remove
it, and it is why the deletion is safe. It is also why the deletion matters.
The old code silently substituted a pooled cross-site mean for a regional one
and left no trace of having done so; nothing in any output distinguished a
region running at its own measured efficiency from a region running at 0.11
because its entry was blank. A registered parameter the model can never read is
a documented assumption that is not in the model.

Three gates fired, and each fired correctly rather than needing to be
persuaded.

`test_wp1_registry_wiring.py` crashed with `AttributeError: 'FeedbackParams'
object has no attribute 'cre_base'`, because it compares the live model against
a frozen field snapshot. The repair is not to make a missing attribute pass.
The test now carries a `DELETED_FIELDS` map: a field that has left the model
must be named there with the finding that authorised it, and an undeclared
disappearance is counted as a move and fails. "The field is gone" and "the
field never mattered" must not produce the same result. Its three registry
counts drop with the deletion: 54 entries to 53, 56 leaves to 55, 17
declared-fixed uncertainties to 16.

`test_claims.py` failed G5: the claim/parameter reverse index changed without
`docs/claims_index_baseline.json` being regenerated. Refreezing it removes
exactly one line, `"cre_base"`, and `docs/claims_baseline.json` is byte
identical, so no claim drifted. The index baseline shrank, which is the
permitted direction.

The build graph restaled 21 nodes on the params fingerprint, none on a changed
number. They are being regenerated rather than stamped
(`logs/run_208_stale.log`): the fingerprint is the mechanism that notices a
parameter set has changed, and stamping around it on the strength of one node's
zero-diff would teach the graph to trust an argument instead of an artifact.

Also in this pass, `soc_trajectories` becomes a build node and its two
`--allow-orphan` / `--allow-unsourced` lines leave the Makefile. One debt line
remains there: `results/s3_shock_calibration.csv`, owed
`make_s3_shock_calibration.py` (F-015).

Profiling, which was asked for and is recorded here so it is written before it
is reported: a canonical run is 6.4 s wall against 6.6 s CPU, 103% of one core,
single-threaded, peak RSS 108 MB, no major faults. The work is CPU-bound and
this container has two cores, so the parallelism ceiling on any sweep is 2x
regardless of how many workers are scheduled.

## F-020 — 2026-07-28 — The widened fingerprint splits the six unprobed price parameters three and three: half reach money untested, half are inert against money itself

F-011 could not score six leaves. `crop_price_usd_t`, `n_price_wedge`,
`n_price_usd_kg_farmer_paid`, `n_benchmark_usd_kg`, `urea_n_fraction` and
`price_benchmark_max_factor` were listed in the harness's `NOT_PROBED`
frozenset and each came back INERT, and the harness said in as many words that
the verdict should be read as "not probed" rather than "irrelevant", because
the published-number set was the canonical artifact alone and that artifact
carries no gross margins, no prices and no cost shares. F-019 deleted the
frozenset and widened the fingerprint to the money the abstract actually
quotes: the Figure 1 margin curves point by point along each curve, plus the
derived per-region nitrogen cost shares and the regional price pair. The
published set goes from 107 numeric fields to 587, the added 480 being the
money half. This entry is the re-sweep, `logs/run_210_mutation.log`, run 210,
55 leaves, two workers on two cores, 64.2 minutes, results in
`results/mutation_coverage.csv` and `results/mutation_coverage_summary.txt`.

The six do not move together. `crop_price_usd_t`, `n_price_wedge` and
`n_benchmark_usd_kg` score UNTESTED. Each moves 164 published fields, all of
them in the money half, since the same three scored INERT against the
107-field canonical set in the F-011 re-sweep and the canonical half has not
changed. The worst movements are `price.sub_saharan_africa.crop_usd_per_t` at
30.0 for the crop price and `margin.south_asia.margin_chg@10` at 1.042 for
both nitrogen price terms, and no test in the green baseline set objected to
any of it. The other three, `n_price_usd_kg_farmer_paid`, `urea_n_fraction`
and `price_benchmark_max_factor`, stay INERT: each moves exactly one model
state field and no published field at all.

Those are the only verdict changes against the F-011 sweep at 4cf2b72. Every
other leaf holds its verdict, and the one further difference in the table is
`cre_base`, which is absent rather than changed because F-019 deleted it. The
totals are 32 COVERED, 3 UNTESTED, 6 GUARDED_AT_LOAD, 2 DECLARED_NOT_WIRED and
12 INERT over 55 leaves; INERT falls from 16 to 12 and UNTESTED rises from 0 to
3. The largest fingerprint movements belong to `faostat_yield_target` and
`synth_n_current` at 523 fields each and `residue_c_to_active_fraction` at 514,
all three COVERED.

The lesson is about what a verdict costs to earn. An INERT verdict is only as
strong as the set of numbers the probe looks at, and a probe that omits half
the published output turns "we did not look" into a scored result that reads
like a finding. Widening the fingerprint did not change any parameter; it
changed what the harness was entitled to say about them, and it converted three
silent passes into three named coverage gaps. The remaining nine INERT leaves,
and in particular the three price leaves that survived the widening, now carry
a much stronger claim than they did in F-011: they are wired, they move model
state, and they move nothing in either the canonical artifact or the farm
margins. That is no longer a limitation of the probe. It is a statement about
the parameters, and each of them is now a candidate for the same treatment
`cre_base` received.

## F-021 - 2026-07-28 - The pulse is rebuilt on a seam rather than restored as a copy, and regenerating it brought two manuscript numbers back into agreement

The one-year pulse scenario PULSE1, which C-061 reads, was written in the v15
tree that was lost and had no surviving generator (F-018). The obvious rebuild
is a one-year capacity recovery, since the model already carried a recovery
ramp. That would have been wrong in a way that is hard to see: a ramp decays
the shock linearly across the year it is supposed to hold at full strength, so
year 1 would have come out near half of S3's year 1 and the curve would still
have looked entirely reasonable. What the pulse needs is a square shape, and
the model had no way to express one.

Adding a field for it would have meant writing the disruption timeline a fifth
time. The ceiling ramp and the price relaxation each existed twice, once in
`coupled_econ_biophysical` and once in `coupled_monthly`, and those two models
are coupled at different resolutions, which is precisely the interface where a
drift would land and go unnoticed. The four copies agreed, which is why they had
survived, and agreement is not a reason to keep a claim written down four times.
So the timeline now exists once, as `coupled_econ_biophysical.supply_state()`,
returning a frozen self-validating `SupplyState` of a ceiling and a price
fraction; both models call it and neither computes a recovery fraction any more.
The refactor was required to move no number and did not: the canonical artifact
diffs to zero across all 125 fields.

Two gates fired. The generator asserts that PULSE1's year 1 equals S3's year 1,
which is true by construction because they are the same shock, and the first
implementation failed it at 2.4e-05 against 2.3162
(`logs/run_212_scenario.log`). The cause was a boundary convention: row `y` is
the state at `t = y` with the forcing applied across the step that reached it,
so an exclusive end at `t >= 1` removes the shock from the only year the pulse
contains. The boundary is inclusive and the docstring now says why. That
assertion is worth more than it looks, because it is the one check on the
rebuild that does not depend on remembering what the lost code did.

The rebuild reproduces the lost column exactly where it should. Years 1 and 2
come out at 2.316 and 0.492 against the lost artifact's 2.316 and 0.492, and
they diverge from year 3 onward. That pattern is the eps_F_N signature: year 1
is identical at -0.5 and at 0 to five decimals and the later years are not. So
the agreement is evidence that this is the same scenario, and the divergence is
evidence that it is now running in the current family, which is what the node
was blocked on. The year-5 residual moves from 0.009% to 0.037%, so the owed
correction to C-061 is 0.3% to 0.037% and not to 0.009%.

`data/scenario_trajectories.csv` was the last artifact in the deposit still
carrying the superseded family. Regenerating it moved S3 year-10 from 3.032 to
3.198, into agreement with the canonical, and the claim register no longer reads
two families at once.

That regeneration then stopped the build on the improvement gate, which is what
that gate is for. `C-060/east_asia_yr10` and `C-060/fsu_yr10` came into line
without the baseline being touched: East Asia's year-10 loss moved from 1.182 to
1.210 against a stated 1.3 at a 0.1 pp tolerance, and the FSU's from 5.126 to
5.553 against a stated 5.5. Both now agree. The reading to resist is that two
manuscript numbers have been vindicated. What actually happened is that these
two sentences were written under a model state closer to the current family than
to the one the lost deposit was computed in, so the deposit was the anomaly and
not the paper. Twenty-four checks remain DRIFTED and none of them moved.
`docs/claims_baseline.json` is regenerated on the strength of this entry
(`logs/run_217_freeze.log`).

## F-022 - 2026-08-07 - There is one yield and one fertilizer quantity, but the reported food price clears a production change the biogeochemistry did not deliver

Dale Manning, reading the 7_22 draft, asked two things: whether the economic
and biogeochemical models each carry a yield that could drift apart, and
whether the fertilizer cap feeds back into `F_hat` inside the equilibrium or is
applied after it is solved. The second has a clean answer. The first has an
answer that is clean about yield and not clean about price.

**The cap.** In the model that produces every published number it is inside the
solve. `coupled_monthly` detects a binding cap, discards the unconstrained
solution and calls `_solve_equilibrium_capped`, which sets `F_hat = ln(c) -
L_hat` and re-derives the food price and land allocation from the fertilizer
physically available, with the fertilizer-price term dropping out because price
no longer rations demand once quantity does. `test_cap_market_clearing.py`
checks this the only way that means anything, by re-solving the four structural
equations outside the model with `brentq` and requiring the reported price to be
that root: 237 cap-binding steps, worst structural residual 1.4e-17, worst root
gap 2.2e-16 (`logs/run_225_cap.log`). F-010 records that the previous version of
this test asserted a residual that was zero by algebra and so could not fail.

There is a second model, `CoupledEconBiophysicalModel` in
`coupled_econ_biophysical`, which is annual rather than monthly and which still
does exactly what Dale was worried about: it solves the unconstrained
equilibrium, stores `PY_hat`, `F_hat` and `L_hat` from that solve, and then
clips the fertilizer level with `F_level = min(F_level, F_max)`. Its reported
`F_hat` is the unconstrained one and does not correspond to the fertilizer it
went on to apply. No published generator uses it: all seventeen call
`CoupledMonthlyModel` and the annual class has no importer outside its own
module. So no published number is affected, and a class that carries the
superseded behaviour, is importable, and is named as though it were the coupled
model is a trap rather than a spare.

**The yield.** There is one reported yield. `yield_fraction` is written only by
the Mitscherlich response in the monthly biogeochemical engine, and the economic
block produces no yield at all: it produces a fertilizer rate, a land area and a
food price. `beta` and `gamma` are not free economic parameters either; they are
recomputed every step as the local elasticities of that same Mitscherlich curve,
partitioned between soil and fertilizer nitrogen by gross input share. Dale's
description of the intended design, an economic yield parameterised on the
biogeochemistry, is what the elasticities do, and the yield itself is taken
straight from the biogeochemistry rather than from the linearisation.

There is nonetheless a second yield, implicit and unreconciled. The equilibrium
closes on market clearing, `Y_hat = eta * PY_hat`, against the log-linear supply
relation `Y_hat = alpha*L_hat + beta*N_hat + gamma*F_hat`. So the food price the
model reports is the price that clears a market for the log-linear production
change, while the production the model reports is the nonlinear one the
biogeochemistry delivered. Nothing forces those to agree and until now nothing
measured them. `code/repro/diagnose_yield_consistency.py` does, over three
scenarios by eight regions by thirty years.

The equilibrium itself is exact: demand and log-linear supply agree to 1.1e-14
pp. Getting that number required using the elasticities the solver actually
used, which are the previous step's, because the current step's are not known
until the biogeochemistry has been advanced and that happens after the solve. A
first version of the diagnostic compared against the current step's stored
`beta` and `gamma` and reported a 0.948 pp residual, which was the size of the
one-step lag and not a defect. The lag is real and moves `beta + gamma` by up to
3.25 pp in a step.

The gap that matters is realized production against econ-implied production. It
is a year-1 transient on a persistent floor: mean 0.71 pp and max 1.54 pp in
year 1, falling to roughly 0.22 pp mean by year 3 and staying there through year
30. The sign is almost always the same. Realized production exceeds the
log-linear production, because the Mitscherlich curve is concave and a
first-order expansion taken at a large one-year move overstates the loss. The
worst cell is South Asia in year 1 of S3: the biogeochemistry gives -2.14% and
the price is clearing -3.68%.

Where that lands on a published number is the food price index, which C-060
quotes. Reported indices are biased high, by up to 4.35 pp in year 1 (Europe
under SC1: 1.1212 reported against 1.0777 clearing realized production) and by
0.60 pp on average. At year 10, which is what the manuscript reports, the errors
run from -0.22 pp (sub-Saharan Africa) to +1.00 pp (Europe), with the FSU, whose
10.3% index is a C-060 check, reading 1.1103 against 1.1028. So the direction is
that the paper's food price responses are modestly too pessimistic, and the
year-1 price numbers are the ones to treat with most caution.

This is a limitation to state rather than a bug to fix. Reconciling the two
would mean iterating the equilibrium against the nonlinear biogeochemical
response within each step instead of solving the linearisation once, which is a
different model and not a correction to this one. What is owed is the SI
sentence: the food price is the market-clearing price for a first-order
expansion of the production response, the expansion is re-anchored every month,
and the residual between it and realized production is under 0.25 pp after year
2 and up to 1.54 pp in year 1. Nothing about the yield trajectories changes,
because they never came from the linearisation.

Numbers in `baseline/f022_f025_evidence/econ_biophysical_yield_gap.csv`;
`logs/run_224_yieldgap.log`.

## F-023 - 2026-08-29 - The yield gap decomposes into two offsetting terms, and the sign of the price bias depends on which output concept the market is asked to clear

Dale Manning's follow-up on F-022 asked three quantitative questions: whether
the 0.71 pp year-1 gap is small relative to the yield changes themselves,
whether the gap is explained by an assumption of constant elasticity, and what
the per-region production and price errors are. Answering them properly
required decomposing the gap, and the decomposition changes the story F-022
told about sign.

The gap between realized production (yield_fraction times land) and the
econ-implied production (eta times PY_hat) is the sum of two terms
(`baseline/f022_f025_evidence/yield_gap_decomposition.csv`, S3, `logs/run_226_decomp.log`):

The land term. The realized index credits land at elasticity 1, while the
supply relation credits it at alpha, about 0.10. With land expanding up to 1.4
percent, this contributes a steady +0.70 pp. It is a definitional difference,
not an error: alpha embeds diminishing returns at the extensive margin, the
yield-times-land index assumes new land yields the average. No published
number reads `total_production_index`, so this term touches nothing published
directly.

The yield term. Realized ln(yield_fraction) against the linearized beta*N_hat
+ gamma*F_hat. The elasticities are re-anchored annually, so this is not a
constant-elasticity error over time; it is the within-step error of a
first-order expansion across a finite move. Signed, it averages +0.08 pp in
year 1 (mixed across regions, up to about +1.5 pp where the move is large) and
-0.44 pp in the chronic phase: the linearized yield response is too
OPTIMISTIC after year 2, understating the loss, because the accumulating
soil-N decline compounds nonlinearly. F-022's statement that the sign of the
gap comes from concavity overstating the loss was true of the worst year-1
cells and wrong as the general mechanism; the persistent positive gap is the
land term winning over a negative yield term.

Relative size, over all three scenarios: the gap is 13 percent of the realized
production change on average (median 10 percent), concentrated where one-year
moves are large; corr(|gap|, squared move) = 0.35, so the second-order story
is partial, consistent with the two-term decomposition.

The consequence for price is concept-dependent
(`baseline/f022_f025_evidence/price_error_econ_consistent.csv`, `logs/run_227_price_consistent.log`).
Against a market that clears yield-times-land, reported price indices are HIGH
by up to 4.35 pp in year 1 (F-022's numbers). Against the economically
consistent concept, realized yield with land at alpha, reported prices are LOW:
mean -0.97 pp, mean magnitude 1.09 pp, worst -3.0 pp, and at year 10 between
+0.37 pp (East Asia) and -2.11 pp (FSU). Since the chronic-phase yield
linearization is too optimistic, the defensible statement for the paper is
that reported food price responses are modestly conservative in the chronic
phase, with year-1 values the least certain in either direction. C-060 is the
claim affected; the yield-loss series are not, as they never pass through the
linearization.

Dale's preferred remedy, clearing the market on the biogeochemical response
itself, has its first iterate already computed: `price_econ_consistent` is the
price that clears realized yield with land at alpha. A full fix is a
fixed-point iteration per step with biophysical state rollback; the remaining
adjustment beyond the first iterate enters through fertilizer demand
responding to the revised price and is second-order. Adopting it would
regenerate every price-bearing artifact and is a model change to decide
deliberately, not to slip into a revision.

His question about beta and gamma has a direct answer: they are not estimated
by regression. The SI (7_22) already states they are local Mitscherlich
elasticities recomputed each year at the current operating point and
partitioned between mineralized soil N and applied fertilizer by gross input
share, collapsing to unit elasticity where the stoichiometric cap binds.

Owed edits recorded from Dale's 8/28-29 review, to be executed with the D1
batch: move the AI-use statement from page 12 to back matter; add the SI
paragraph explaining how the regional price change computed at mean SOC is
applied to farms across the SOC distribution (his manuscript comment on the
farm-distribution paragraph); acknowledge in the econ-component paragraph that
food production and prices are outputs and state the difference from reported
production (this entry supplies the numbers); fix the "By contrast" sentence
that repeats its predecessor; in the author response, paste the revised
manuscript text under each reply, replace "FAOSTAT ~2020" with "FAOSTAT
2019-2021 mean" and say why that vintage (matches the IFA fertilizer-rate and
cropland vintages and pre-dates the 2021-22 price spike the scenarios
simulate), repair the cross-reference Dale could not find (the optional
two-crop sensitivity note), and state explicitly where each clarification
landed in the paper. A clean standalone model description document is owed for
resubmission or on request.

## F-024 - 2026-08-29 - Dale's remedy prototyped: clearing on the realized yield moves prices 1-2 pp, yields by at most 0.05 pp, and removes the two-yields objection by construction

`code/repro/prototype_realized_clearing.py` implements the fix Dale proposed:
root-find the food price at which demand equals the production change the
biogeochemistry actually delivers, eta*PY = ln(yield_frac(F(PY))) + alpha*
lambda_L*PY, with each candidate price evaluated by running the monthly
biophysical step from a soil-state snapshot. beta and gamma drop out of the
clearing entirely and revert to being diagnostics.

The prototype carries its own gate: run in linear mode it must reproduce
CoupledMonthlyModel.run() before its realized mode is believed. It reproduces
it to zero, literally 0.0e+00 across PY_hat, yield_fraction and fertilizer
over 30 years in two regions (`logs/run_228_proto.log`). A compact reimplementation
that had NOT been held to that standard would have been a second copy of the
model, which is the disease this rebuild treats.

Results over S3, eight regions, 30 years, 240 clearings: brentq converges in 8
evaluations per step, every step, no bracket failures. Prices move +0.83 pp on
average (range -1.5 to +2.8); at year 10, +0.9 to +1.9 pp everywhere except
East Asia at -0.36. This confirms F-023's first-iterate prediction and bounds
the second-order fertilizer-demand feedback at about 0.2 pp. Yield fractions
move by at most 0.054 pp and fertilizer by at most 0.25 percent, so the
published yield-loss claims are untouched to the precision the manuscript
quotes; the change is confined to the price channel, where the current numbers
were the ones F-023 showed to be conservative.

Matthew's question, whether the 10-13 percent gap warrants remedy, resolves as
follows. That ratio measured the internal inconsistency against a production
concept nothing published uses; the published casualties were only ever the
price indices, low by about 1 pp mean. The remedy costs one wiring change,
raises price responses by 1-2 pp at year 10, and eliminates the objection
rather than caveating it: after it, there is no second yield anywhere in the
system, implicit or otherwise, because the market clears the biogeochemical
response itself.

Cost: roughly 8x the biophysical work per run (a 6-second canonical run
becomes under a minute; the Monte Carlo ensemble is year-1-scoped and scales
less than proportionally). Adopting it regenerates every price-bearing
artifact, reopens C-060 and C-042 deliberately, and needs a FINDINGS entry
plus refreeze when it lands. Not yet wired; awaiting Matthew and Dale's
go-ahead. Numbers in `baseline/f022_f025_evidence/realized_clearing_comparison.csv`.

## F-025 - 2026-08-29 - The market now clears on the realized biogeochemical yield; the linearization Dale challenged is out of the model

Matthew and Dale adopted the F-024 remedy. `CoupledMonthlyModel.run()` now
root-finds the food price at which demand equals the production change the
biophysical model actually delivers, eta*PY = ln(yield_frac(F(PY))) +
alpha*lambda_L*PY, evaluating each candidate price by running the monthly
nitrogen balance at that price's fertilizer rate from a soil-state snapshot.
beta and gamma no longer enter the clearing anywhere; they are recorded
diagnostics. The physical supply ceiling became a quantity constraint inside
the residual, so the separate capped solver, the constrained-cap fix of v1.3
and object of F-010, has nothing left to solve and no copy left to drift.

`run_price_shock_analysis.py` got the same treatment: Figure 1's output-price
recovery was its own closed-form linear clearing, and under a 100 percent
spike the one-year move is exactly where F-023 showed the first-order
expansion at its worst. Its closed form survives only as the bracket guess.

`test_cap_market_clearing.py` is in its third form. It evaluates the three
structural equations of the realized clearing from reported DataFrame columns
at every step of every scenario, and it additionally REQUIRES the old linear
supply relation to disagree with the clearing by more than 1e-3 somewhere:
a check that cannot tell the new clearing from the old proves nothing. Worst
structural residual 1.51e-13 over all steps, linear-relation gap reaches
2.1e-2, 237 cap-binding steps (`logs/run_229_cap.log`).

What moved (`logs/run_230_stale.log`, `run_231_stale2.log`, claim register):
the canonical global loss path goes 2.32/3.20/3.31 to 2.32/3.18/3.30, and
regional year-10 losses shift by up to 0.9 pp (South Asia 6.0 to 5.10), so a
handful of yield sentences move at the manuscript's own precision even though
the clearing change alone moved yields by under 0.06 pp; most of those
sentences were already DRIFTED from earlier findings and are finally being
restated. Food prices move the other way and further: year-10 indices are
5.97 (NA), 11.00 (EU), 2.70 (EA), 8.89 (SA), 7.16 (SEA), 4.93 (LATAM), 7.19
(SSA), 12.96 (FSU) percent, against a register that stated 5.0 global and
10.3 FSU. Figure 1 margin gaps between mean and half-mean SOC now run 0.27
to 0.99 pp across the four fine-sweep regions. The pulse year-5 residual is
0.038 percent.

Both gates that exist for exactly this moment fired and were answered with
authorised refreezes rather than stamps: the WP1 pinned canonical delta
(baseline/canonical_expected_delta.json, 49 moved fields re-frozen, hard-coded
globals updated with the lineage comment) and the claims baseline, which
records three checks newly drifted (C-033 farmer-paid net revenue, C-042
year-1 regional min and max) on top of the standing document debt. The
document edits themselves are executed in the tracked-changes batch that
accompanies this entry.

The lesson this entry adds to the ledger: a linearization can sit at the
center of a model for three review rounds when every test compares the model
to itself. What surfaced it was a coauthor asking which of two yields the
price was clearing. The clearing now has no answer other than "the model's".

## F-026 - 2026-08-29 - The eps_F_N central is restored to -0.50, closing the code-document fork the release audit found, and the register reads 70 of 70 for the first time

An external audit of the release (run with a second AI system, Codex, whose
disclosure status in the AI-use statement is Matthew's decision) made eight
substantive claims. Each was verified against the tree before anything was
changed. Five were confirmed, two refuted, one was already-known debt.

**Confirmed, and the blocker: the fork.** `SOIL_N_RESPONSE_ELASTICITY_CENTRAL`
sat at 0.0 while the SI stated "default -0.50; active in S3, SC1 and SC2", the
manuscript attributed the 20-to-19-percent realized-reduction gap to the
depletion channel, and params.yaml registered -0.5. Under a zero central that
attribution was false (the gap came from the food-price cross-elasticity), and
every published trajectory came from a family the documents did not describe.
The audit's predicted post-restore family, 2.32/3.02/3.07, was confirmed
against our own structural-sensitivity output before the change and reproduced
exactly by the regenerated canonical after it. The central is now -0.50, the
registry comment records the decision, and the registered parameter is wired
(used_by names the central constant).

**Also confirmed.** Table S4's y_max values in the SI were stale (6.277/3.876
against live 6.198/3.967); embedded Figures S12 and S13 differ from the live
renders; Figure S6's buffer-panel correlation disagreed three ways (live
+0.19/0.10, SI text +0.02, response letter +0.07/0.03); the response letter
still said halving elasticities compresses year-10 loss by ~25 percent (it is
~50) and claimed the abstract range was 2.6-4.6 pp (it is 0.3-1.0); and the
SI sentence that "every number has been regenerated" was false while the
ensemble and four-pool results stood un-rerun. Typos confirmed: 0.65reduces,
whilecompressing, S11).The, region.Larger. All are repaired in the document
pass accompanying this entry.

**Refuted.** `make verify` from a clean checkout exits 0 with 15 suites; the
audit's claim that the gate errors did not reproduce. The two excluded tests
are excluded with written reasons, one red-on-purpose pending a document edit
that has now landed; both are being repaired and re-included rather than left
out.

**The consequences of the restore.** The graph regenerated in full (23 nodes,
logs/run_242_stale_eps3.log). Yield losses shrink because farmers partially
compensate declining soil N: global 2.32/3.02/3.07, South Asia year-10 4.80,
SSA 4.74, FSU 5.09. Food prices ease similarly (FSU year-10 11.89, global
6.72). The one-year pulse residual falls to 0.010 percent. SOC declines
soften (SSA 2.14, South Asia 2.09: under this family they separate, SSA
higher, reversing F-018's not-separable finding, which was a property of the
zero-central family). And the depletion channel produces year-10 low-SOC
non-monotonicity in three regions (LATAM, SSA, FSU), the behaviour v16_sol
documented as conditional on the global elasticity: C-001's year-10 gradient
checks are restated to the observed pattern and the documents carry the
year-1-universal, year-10-conditional scope statement. The realized clearing
survived one repair: the severity gradient's 10-percent-SOC farms put the
clearing price outside the fixed bracket, which now expands geometrically.

**The Monte Carlo is rerun, not retained.** The realized clearing was wired
into the ensemble's per-draw price (the same first-order closed form survives
only as the bracket guess), and 1,000 joint draws regenerated in 20 minutes.
Buffering is universal (P = 1.0 in every region), the median cross-region
buffer is 0.88 pp, the global median year-1 loss is 2.51 percent, and
P(SSA worst net revenue among priced regions) is 0.001 against the withdrawn
claim's 83.7 percent, closing that retraction with a number. The pre-rerun
deposit is snapshotted outside the tree.

**The register is rebased.** document_basis v14 -> v17, register_version
v17-f026, all 34 drifted checks restated to the regenerated artifacts at
each sentence's own precision, two tolerance-hygiene gates (G6) answered by
tightening rather than noting, and the claims baseline refrozen: 19 claims,
70 checks, 70 AGREES, 0 DRIFTED. The remaining debt is now entirely in the
documents, which the accompanying tracked pass brings to the register.

## F-027 - 2026-08-29 - The second audit round: clean files that lost their mathematics, tables nobody re-read, and the last unsourced input finally gets a generator

The second external audit declared the v17 folder no-go on eight grounds. As
with the first round, each claim was verified before being acted on; this
round all eight were substantively right, and two exposed defects in the
release TOOLING rather than in the documents.

**The clean files were mathematically blank.** The LibreOffice accept path
used to produce clean copies drops OMML equation objects: the clean SI carried
zero of the tracked SI's 103 equations, and inline symbols vanished with them.
The replacement is an XML-level accepter (`/tmp` tooling, recorded here
because the lesson matters more than the path) that touches only revision
markup. Writing it surfaced three regex traps in sequence, each caught by
schema validation: a self-closed `<w:ins/>` matched by the unwrap pattern and
swallowing its neighbours; the same class of bug for `<w:del/>` paragraph
marks; and tracked table-row deletion leaving doubled `trPr` elements. The
final clean files carry all 103 equations byte-identical to the tracked
copies, zero revision markup, and zero comments; the blank equations in
locally rendered PDFs are a LibreOffice rendering limitation, and the one
check this container cannot run is opening the files in Word, which is
Matthew's.

**Tables nobody re-read.** Supplementary Table 1's BNF row still carried the
legacy static parameterisation (25/25/20/...) rather than the derived
per-cereal-hectare values (31.9/18.3/13.8/22.1/23.0/37.7/15.0/13.8) and its
buffer-ratio row the pre-audit percentages; Supplementary Table 3 and the
Note 3 prose carried pre-F-026 Spearman values (eight cells and four prose
rhos updated); Table S4 prose still said y_max = 3.88 where the table said
3.967; the response letter said 0.54 pp where the climate swap now gives
0.62, said "roughly twice" where the SC1/SC2 year-30 contrast is 3.8 percent
against 0.02 (stated as values now, since a ratio against a near-zero
denominator means nothing), and claimed an abstract range that was two
revisions old. Figure S9 was still the pre-audit ensemble figure claiming
SSA-worst in 83.7 percent of draws; the regenerated figure and caption say
0.1 percent. The MS's NUE claim said a capture-efficiency increase cuts SSA
year-10 loss by ~55 percent; the artifact says 8 percent, at both places the
sentence appeared.

**The four-pool contradiction is resolved by withdrawal.** Figure S5,
Supplementary table 2 and the 0.9x-2.5x ratios were tracked-deleted; Note 2
now states the withdrawal and keeps the mechanism qualitative; the MS
sentence says the quantitative comparison is withdrawn because the engine is
not in the deposit. This follows the recommendation Matthew forwarded.

**The last unsourced input has a generator.** `results/s3_shock_calibration.csv`
was a deposit from the lost v15 tree that nothing regenerated, and it was
stale in exactly the way an unsourced input goes stale: computed under the
zero central, it let C-050 read a 19.1 percent realized reduction while the
actual value under -0.50 is 18.7. `make_s3_shock_calibration.py` now
generates it as a build node, the Makefile's `--allow-unsourced` flag is
gone, and C-050 is restated to 19 (18.74 within its stated tolerance).

**Register and packaging honesty.** The register's document_basis edit from
the first round had been lost in a crashed heredoc and the generated report
still said v14; it is now genuinely v17-f026 and the report says so. The
release bundle was incremental (unclonable alone) and the zip omitted
directories its own README promised; both are rebuilt self-contained. The
submission checklist was the v10 relic and is rewritten. The scenario
docstring still describing a zero central is fixed. README's climate figure
0.70 goes to 0.62.

The lesson this round adds: a release gate that does not read the released
files is a gate on something else. The cross-document test now requires the
corrected values, forbids twenty stale fragments, and hash-checks ten
embedded figures against the deposited renders, so each of this round's
findings is a permanent regression check rather than a memory.

## F-028 - 2026-08-29 - The third audit round: one decision carried everywhere, canonical bytes, and repo docs join the gate

The third external audit found four residues, all real, all fixed in one
synchronization pass. Verdicts on the four:

**1. The four-pool withdrawal now follows one decision (CONFIRMED, fixed).**
The manuscript withdrew the quantitative engine comparison on p.20 and then
re-cited Supplementary Fig. S5, Supplementary table 2 and the uncertainty
range in the parameter-uncertainty discussion; the SI kept an orphan
"Supplementary table 2" heading, its caption and its CUE-decomposition note
(the earlier deletion loop matched the table rows but not those three
paragraphs); SI Note 8 said the ratios were "retained here as originally
reported"; the response letter told the reviewer "we retain this analysis."
All four sites now state the same thing: the quantitative comparison
(Fig. S5, table 2, the SOC-loss ratios) is withdrawn because the alternative
engine is not part of the deposit, and only the qualitative
direction-of-response statement survives. Tracked deletions, author
"Matthew Wallenstein". Two XML traps surfaced and are fixed in the tooling
sense too: a paragraph whose pPr already carried an empty `<w:rPr/>` got a
second rPr from the deletion mark (schema violation; the marks are now
merged), and deleting a paragraph that contained a previous round's tracked
deletion nested `<w:del>` inside `<w:del>`, which the accepter's non-greedy
removal turned into a stray close tag (nesting is now flattened before
accept).

**2. Gross-margin language is gone (CONFIRMED, fixed).** Manuscript:
"Gross-margin losses follow..." became a partial net-revenue sentence scoped
to the four priced regions, and "the fertilizer share in regional gross
margin" became "the land-response coefficient" - which is what Supplementary
Note 3 actually reports, since Note 3 explicitly EXCLUDES cost share from the
cross-regional diagnostic (prices exist for only four regions). SI Note 6 now
counts "24,000 region-x-SOC-x-draw yield evaluations plus 12,000 evaluations
of the year-1 change in crop revenue net of nitrogen-fertilizer expenditure
across the four regions with audited price pairs" (1,000 draws x 3 SOC x 8
regions; x 4 priced regions), and its universality sentence is scoped to
those four regions - which the regenerated ensemble supports at P = 1.000 in
each. The Figure S7 caption's two "gross-margin" mentions became
net-revenue. Fixing the caption exposed a value drift nobody had flagged:
it claimed the 50-vs-100% SOC gap is "comparable or larger under halved
elasticities (2.5-4.2 pp baseline against 3.0-5.9 pp halved)" while the
deposited figS7 data gives 0.3-1.0 pp baseline against 0.2-0.7 pp halved -
smaller under halved elasticities, sign preserved. The caption now says
that. All the retired phrases joined the cross-document forbid list.

**3. The staleness gate is byte-deterministic (CONFIRMED, fixed at the
write side).** The auditor's machine regenerated one node's output with
1e-14-scale float differences and the gate correctly-but-uselessly called it
stale. Every node's textual outputs (.json, .csv, .csv.gz) are now
canonicalized immediately after the generator runs and before the sidecar is
stamped: float literals carrying more than six significant digits are
re-rendered at six (`%.6g`), JSON is re-dumped in one canonical form, and
gzip containers are rewritten with mtime=0. Six significant digits is two
orders finer than anything the documents quote and coarse enough that
1e-13-relative noise cannot move a rounding boundary (~1e-7 flip probability
per value). Two suites double as node generators
(test_parameter_extremes_sol, test_zero_shock_invariance) and re-staled
their own nodes when run as tests; they now canonicalize their own output.
The whole graph was regenerated under the hook (32 nodes including the
Monte Carlo ensemble, rerun in full), and five nodes rerun afterwards
reproduced their outputs byte-identically - only sidecar timestamps moved.
Figure PNGs changed bytes when their canonicalized inputs shifted values at
the 1e-6 level (pixel-identical renders); the five doc-embedded ones
(Figures 1, 2, S6, S8, S12) were re-embedded and the embedded-figure hash
check passes.

**4. Repo docs carry the released numbers or say they are history
(CONFIRMED, fixed, and now gated).** README still led with 2.31/3.18/3.29,
SSA y_max 3.88, "unsupported eps_F_N = -0.50", a stale expected-results
table and a reproduce list that bypassed the build graph; MANIFEST described
CLAIM_REGISTER_sol.csv as the decision table; the legacy register itself
retained pre-rebuild claim values (C02: 2.31/3.18/3.29 "retain"). README and
MANIFEST are rewritten to the released family (every expected-results row
recomputed from the regenerated artifacts), CHANGELOG gained a dated v17
entry, CLAIM_REGISTER_sol.csv/.md moved to docs/archive/ (preserved, exempt
from the orphan scan by prefix), and the surviving working records
(D1_D2_HOLD, RECONSTRUCTION_GAPS, v15_REBUILD_STATE, HANDOFF_v15,
EVIDENTIARY_STANDARD_sol, the two results/ reconciliation notes) carry a
SUPERSEDED banner stating the released family. The hole the audit named -
"the automated 70/70 claim check is not catching stale quoted text" in repo
docs - is closed structurally by a new 18th suite,
`code/repro/test_repo_docs_consistency.py`: README/MANIFEST/HANDOFF must
contain the released headline family and must not contain the stale one;
FINDINGS and CHANGELOG are append-only ledgers and exempt; every other
markdown at the root and under results/ must either carry SUPERSEDED in its
head or contain no stale fragment.

State at close: `make verify` = 18 suites + 32-node graph, exit 0, zero
orphans, zero unsourced inputs; claim register 70/70 AGREES on document
basis v17; the three tracked documents and their equation-preserving cleans
all validate, with the required withdrawal/net-revenue fragments present and
twelve new stale fragments forbidden.

## F-029 - 2026-08-29 - The four-pool comparison comes back regenerable, and the published documents stop talking about their own revisions

Matthew's decision, after a fresh-look port showed the withdrawn story
survives the corrected core: reinstate. Three strands landed together.

**1. The engine is in the deposit.** The April four-pool line (recovered
from tropical-reparam-2026-04-14/backups and matched-mems-2026-04-15;
byte-identical copies) was adopted as `code/model/som_4pool_monthly.py`
(pool dynamics unchanged, plus a respiration split the figure needs) and
`code/model/coupled_4pool.py` (the coupled model on the corrected core:
ERA5, stationary spin-up with the baseline water-stress multiplier,
production-path y_max calibration, the shared two-sided water-stress
response, realized-yield clearing, scenario-supplied eps_F_N). Node
`fourpool_comparison` (33rd) regenerates results/fourpool_comparison.json,
outputs/Table_S2_fourpool_sol.csv and figures/Figure_S5_fourpool_flux.png
(pixel-matched to the old embed's 3618x1317 so the SI media slot could be
reused). Numbers, vs the withdrawn family: ratios 0.90-2.16 (was
0.86-2.51), median 1.65 (was 1.80), SSA still the only region below parity
(0.90 vs 0.86); CUE-downregulation share 6-59% where defined (was 1-29%;
only SSA is now excluded - FSU's fixed-CUE anomaly vanished on the
corrected core); CUE-step respiration 80-87% of total (was 79-87%). One new
disclosure: under ERA5 the four-pool equilibrium N supply cannot reach the
SSA FAOSTAT yield target at any ceiling (plateau 1.37 vs 1.50 t/ha);
disclosed in Note 2, not retuned (F-008 rule). Claim C-073 pins the family
(register v17-f029: 20 claims, 74 checks, 74 AGREES).

**2. The documents were reinstated by UN-DELETING, then editing.** Rather
than inserting new blocks, the tracked deletions from the withdrawal rounds
were reverted (Note 2's quantitative text, the table-2 heading + caption +
9 rows + note, the Figure S5 heading + image + caption, the SI overview
fragment), and the numbers were then updated as small tracked edits - so
the reviewer sees the original text lightly edited, not deleted-and-
replaced. The media file behind the restored Figure S5 drawing was swapped
for the regenerated render.

**3. The published documents are standalone (Matthew's directive: only one
version is ever published).** Every reference to "the submitted version",
"this revision" or "added in revision" was removed from the manuscript and
SI - the abstract's duplicated "and and" went with them - and the
cross-document gate now BANS version-referential language in those two
documents (the response letter, where revision history belongs, is
exempt). Note 8 became "Model verification and reproducibility" (what the
deposit guarantees, not what was corrected); the MC-ranking note lost its
pre-audit-price narrative; and the figS10 caption was carrying stale values
all three audits missed - "from 21.0% to 9.4%, a 55% reduction" against
regenerated data reading 5.3% to 4.9% (~8%) - now fixed and forbidden.
The crop calendar R2 asked about is documented in the Supplementary
Methods (per-region planting/maturity months) and cited to Sacks et al.
2010 (new ref [41] in both reference lists).

**Tracked-changes discipline (Matthew's directive).** The sentence-level
replace helpers had been marking whole sentences deleted+reinserted when a
few words changed. redline.py now diffs at word level
(replace_text_minimal / worddiff_xml), rebuild_para_worddiff re-expresses a
paragraph as a word diff against its original text, and a retrofit pass
collapsed the existing coarse pairs across all three documents (61 MS, 140
SI, 20 AR) - verified byte-equivalent in both the accepted text and the
original-view text. Two tooling traps found and fixed on the way: the
retrofit must skip NESTED revision markup (a tempered non-greedy match
pairs an outer open with an inner close and orphans the outer close), and
accept.py must remove w:del/w:ins blocks innermost-first for the same
reason.

**Response letter (external review of the letter, relayed).** Paraphrased
bullet summaries for Referee 2's four substantive concerns replaced with
verbatim excerpts of the revised Methods text (and the revised uncertainty
paragraph quoted under (vi)); the malformed elasticity notation fixed
(eps_F_N = -0.50; active in S3, SC1, and SC2); "independently validated"
moderated to "evaluated against independent temperate and tropical
benchmarks" (twice); the crop-calendar non-answer replaced with the SI
documentation + Sacks citation; the declined manuscript-structure
paragraph added to the Introduction and the response changed to "Added";
"We thank reviewer" -> "We thank the reviewer".

State at close: 18 suites + 33/33 nodes, exit 0; register 74/74 AGREES;
all six documents validate; cross-document gate requires the reinstated
family and bans the withdrawn one and all version-referential language in
the published documents.

## F-030 - 2026-08-30 - The fourth audit round: the four-pool comparison learns to describe itself honestly

The fourth external audit accepted the reproducibility (independent clone,
18 suites + 33 nodes, exit 0; reported values match the generated output)
and rejected the scientific interpretation on five grounds. All five were
verified against the code before acting, and all five were right.

**1. The pathway description was false (CONFIRMED, fixed).** The documents
said "all SOM formation" passes through microbial biomass via POM -> DOM ->
MBC -> MAOM; the code carries a direct DOM -> MAOM sorption pathway, and at
equilibrium ~49% of MAOM input takes it (the auditor measured ~52% by a
slightly different accounting; "roughly half" is the honest statement, now
in the manuscript Methods, the SI scheme description and Note 2, with
per-region shares in the comparison JSON).

**2. The baseline-state claim was unsupported (CONFIRMED, withdrawn).**
The 4-pool spin-up equilibrates away from the observed SOC (NA 50.0 ->
45.7, SA 25.0 -> 29.2, SSA 9.0 -> 13.5 t C/ha; SSA yield 1.37 vs 1.50
target). Note 2 no longer claims normalization makes the ratio
"unaffected"; it now reports the departures, frames the comparison as
exploratory structural sensitivity, and adds the demonstration the auditor
asked for: in ABSOLUTE terms the 4-pool scheme loses more
disruption-attributable carbon in every region including SSA (0.26 vs 0.12
t C/ha), so SSA's sub-parity ratio partly reflects its larger equilibrium
baseline.

**3. The parameters were off-ledger (CONFIRMED, fixed).** All 21
FourPoolParams fields plus the uniform clay+silt 0.55 assumption are now
in PARAMETER_LEDGER_sol (604 entries, was 577), the governing update
equations and every value are written into the Supplementary Methods, and
clay_silt is a threaded parameter. The texture sensitivity the auditor
probed is reproduced and deposited: clay+silt 0.35 -> median 1.77, SSA
1.00; 0.75 -> median 1.56, SSA 0.85 (their 1.77/1.56 and ~1.00/0.85
match), reported in Note 2 and pinned by claim checks.

**4. Table 2 and Figure S5 labeled the wrong quantities (CONFIRMED,
fixed).** Table 2's caption now says what the CUE decomposition actually
partitions - the 4-pool MINUS Century loss difference - and carries the
absolute year-30 losses of both engines next to the ratio (two inserted
columns, tracked with w:cellIns; accept.py strips the marks). Figure S5 is
regenerated as the disruption-attributable carbon budget against no-shock
controls: the disruption REDUCES gross respiration (the old figure's
quantity), and the SOC loss is the gap between the larger fall in
residue-C inputs and the smaller fall in respiration - shown for both
engines, which is where the mechanism is actually visible.

**5. The conclusion overreached (CONFIRMED, tempered).** "Amplifying its
magnitude" and "the Century engine is the conservative choice" are gone;
the documents now say: larger year-30 SOC loss in seven of eight regions,
smaller in relative terms only in SSA, with the 2.2x maximum flagged as
occurring where the Century loss is smallest (0.48%) and absolute losses
reported alongside ratios. The cross-document gate bans the overreaching
phrases in the published documents.

**Response letter.** The market-clearing correction is summarized
prominently in the opening (with the full account still under "Analysis
and figure tasks"); the orphaned "3." (an empty numbered paragraph whose
content had been deleted but whose paragraph mark survived) is
tracked-deleted; the four-pool item now presents the exploratory framing.

State at close: 18 suites + 33/33 nodes, exit 0; register v17-f030, 78/78
AGREES; all six documents validate.

## F-031 - 2026-08-30 - The fifth audit round: mechanism attribution bounded to what was demonstrated

The fifth external audit accepted the F-030 corrections and left one
scientific issue: the interpretation attributed the 4-pool-Century loss
difference to a single mechanism (Supplementary Note 2 called the microbial
respiratory cost "the dominant mechanism"; the Figure S5 caption said the
loss is larger "because" carbon routed through microbes pays the
respiratory cost). The engines also differ in sorption, desorption,
priming, stoichiometry, pool structure, and equilibrium baselines, and the
fixed-CUE analysis partitions only 6-59% of the positive difference; the
remainder was never mechanistically partitioned. Three cleanup items
accompanied it. All were verified before acting, and all were right.

**1. Over-attribution (CONFIRMED, fixed).** Note 2 now reads: the engines
"differ in several coupled respects", the larger losses "arise from the
combined effects of its alternative stabilization, respiration, desorption,
and baseline-state dynamics", holding CUE constant assigns 6-59% of the
positive 4-pool-Century loss difference to CUE downregulation, "and the
remaining difference cannot be uniquely attributed to a single mechanism."
The table 2 note drops "Non-CUE mechanisms are dominated by respiratory
losses" (the fixed-CUE run cannot support it) for "not further partitioned
by mechanism"; the Figure S5 caption's "because carbon routed through
microbial biomass pays the (1-CUE) respiratory cost" becomes "reflecting
the combined effects..."; the East Asia ratio note now points at "the
engine difference" rather than "the added microbial respiratory cost". The
manuscript Discussion replaces "microbially driven stabilization feedbacks
could accelerate soil nitrogen depletion" with "alternative representations
of SOM stabilization can produce larger estimates of long-run buffer
erosion... and shorter estimated windows over which the buffer remains
intact." The response letter's verbatim Discussion excerpt is synced to the
final manuscript text (it had also missed the F-030 exploratory-framing and
texture-range additions). All edits are word-level tracked
(Matthew Wallenstein).

**2. Code description (CONFIRMED, fixed).** `som_4pool_monthly.py`'s
opening docstring still said "All SOM formation routes through microbial
biomass" — the F-030 correction had reached the documents but not the
code's own description. It now states the two-route scheme (microbial
assimilation with necromass partitioning, and direct DOM -> MAOM sorption,
roughly half of equilibrium MAOM formation each).

**3. Ledger units and uncertainty treatment (CONFIRMED, fixed).** The 22
four-pool rows carried the generic fallback unit ("fraction or elasticity"
— wrong for the five yr^-1 rate constants, the four C:N ratios and the
t C/ha sorption ceiling) and an uncertainty-treatment column reading
"fixed; texture sensitivity via clay_silt 0.35-0.75", which implied these
parameters were varied. The texture sensitivity varies clay_silt_fraction
only. Every row now carries its real unit (FOURPOOL_UNITS in
`make_parameter_ledger_sol.py`) and the honest treatment "fixed; not varied
in submitted analyses (the texture sensitivity varies clay_silt_fraction
only)"; the clay_silt_fraction row alone says "varied 0.35-0.75".

**4. SI page S-5 formatting (CONFIRMED, fixed).** Every run of the
four-pool overview paragraph carried `w:vertAlign val="subscript"` — the
whole paragraph rendered subscript-small. The mark is stripped (text
byte-identical); a scan found no other wholly-subscript paragraph in any
document.

Gate: the cross-document test now requires "alternative representations of
SOM stabilization" (MS), "cannot be uniquely attributed to a single
mechanism" and "combined effects of its alternative stabilization" (SI),
and bans "The dominant mechanism is", "dominated by respiratory losses",
"because carbon routed through microbial biomass" and "microbially driven
stabilization feedbacks" from the published documents. Register
v17-f031 (claims unchanged in number: 20 claims, 78 checks).

State at close: 18 suites + 33/33 nodes, exit 0; register 78/78 AGREES;
all six documents validate; ledger regenerated (604 semantic entries,
per-parameter units on the four-pool rows).
