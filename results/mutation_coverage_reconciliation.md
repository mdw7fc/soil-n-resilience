# Mutation coverage: WP3 rebuild against F-011

Rebuilt sweep, 2026-07-25, on the WP1+WP2 tree (`2b76b16`).

|                     | F-011 | WP3 rebuild |
|---------------------|------:|------------:|
| COVERED             |    12 |           5 |
| UNTESTED            |    22 |          27 |
| DECLARED_NOT_WIRED  |     3 |           2 |
| GUARDED_AT_LOAD     |     6 |           6 |
| INERT               |    13 |          16 |
| **total**           |    56 |          56 |

**44 of 56 leaves agree.** The leaf list is identical — `registry.leaves()`
returns the same 56 names — and `GUARDED_AT_LOAD` reproduces exactly, both in
count and in membership. Every one of the twelve disagreements has an
identified cause. None of them was tuned away.

## 1. Eight COVERED leaves fell to UNTESTED: the catching test does not exist

`cre_regional`, `faostat_yield_target`, `residue_c_to_active_fraction`,
`residue_retention`, `root_shoot_c_ratio`, `soc_initial`,
`som_decay_rates.k_passive`, `som_humification.h_active_to_slow`.

F-011: *"Eight of the twelve COVERED leaves are caught by a single test,
`test_spinup_partition_independence.py`."* That file is not in the tree. It was
lost with the v15 working tree and no work package in `v15_REBUILD_STATE.md`
rebuilds it. The eight leaves are exactly the eight F-011 names — the match is
by construction, not by coincidence.

This is a live documentation defect, not only a coverage one.
`code/model/params.yaml` cites the test twice as an authority:

- under `som_pool_fractions`, as the test that *falsified* an earlier claim
  about the dynamic spin-up overwriting the initial partition;
- as the writer of `results/spinup_partition_characterisation.yaml`
  "on every run", from which the SI is told to cite measured values.

So the registry currently cites a test that does not exist for a file that is
never written. Rebuilding it would restore eight COVERED verdicts and close the
dangling citation. It is the single highest-value item on the test worklist.

## 2. One UNTESTED leaf rose to COVERED

`eps_F_PF`, caught by `test_seam_contracts.py` — written by WP2 under F-005,
after F-011 was recorded. A real gain, not a discrepancy.

## 3. Two UNTESTED leaves fell to INERT

`bnf_potential` and `yield_min_regional` move no field of the canonical
artifact. This follows from WP2's F-002 production-path recalibration, which
solves `y_max` against the FAOSTAT target rather than reading a static value.
F-011 already recorded `yield_max_regional` as INERT for precisely this reason
and cited SI [65]: *"the static values are legacy fallbacks, not the reported
calibration."* `yield_min_regional` has now joined it, and the same SI sentence
covers both.

## 4. `texture_class`: DECLARED_NOT_WIRED or INERT

F-011 scores it DECLARED_NOT_WIRED. This sweep scores it INERT.

The boundary F-011 drew between these two verdicts could not be recovered from
the finding text, and three candidate rules were tested and rejected before
this one was adopted:

- *never requested from the registry* — separates `eps_F_N` and
  `fert_reduction_target` correctly but not `texture_class`, which **is**
  requested, into `RegionParams`;
- *the entry's `used_by` key is empty* — `eps_F_N` (DNW) and `bnf_ramp_years`
  (INERT) both declare `[]`, so it separates nothing;
- *static AST check for a consuming read* — too coarse; values fetched inline
  as `registry.value(...)` are never bound to a name and so read as unwired.

The rule adopted is dynamic and is stated in the harness docstring:
**DECLARED_NOT_WIRED** means the mutation moved no model state at all;
**INERT** means model state moved but no published number did. Under it,
`texture_class` reaches `RegionParams.texture_class` and stops there, which is
model state, so it scores INERT. `cre_base` scores INERT on the same grounds,
which is what F-011 says it should.

This is recorded as a reconstruction gap in the sense of
`RECONSTRUCTION_GAPS.md` rather than resolved by choosing a rule that happens
to reproduce 3 and 13. The totals for these two verdicts are therefore off by
one against F-011 in compensating directions.

## Two defects found in this harness and fixed before the reported run

Both were caught by reconciling against F-011 rather than by a test, which is
itself worth noting.

1. **The canonical run was conditional.** The first revision short-circuited to
   DECLARED_NOT_WIRED whenever the model-state snapshot did not move, without
   ever running the model. Any parameter consumed *inside* the run rather than
   stored on a parameter object was mis-scored:
   `residue_c_to_active_fraction` and both `laub_tropical_ratios` leaves move
   43, 27 and 27 published fields respectively and were being reported as
   reaching nothing. REACH is now always measured; the state snapshot only ever
   breaks the INERT / DECLARED_NOT_WIRED tie.

2. **The state probe missed the price seam and the tropical SOM variant.**
   `SOMPoolParams()` built with defaults does not read `laub_tropical_ratios`,
   and three price constants are consumed only inside validators. Six leaves
   scored DECLARED_NOT_WIRED for want of a probe rather than for want of
   wiring. The probe now exercises the tropical constructor, `check_price_bounds`,
   and the module-level price constants.

The first defect is the same class of error F-011 recorded against its own
probe — *"the fingerprint is too narrow"* — one layer further in. A narrow
fingerprint does not report uncertainty; it reports a confident negative.

## Four tests excluded from CATCH because they are already red

A permanently-red test cannot catch a mutation: it fails either way. Counting
one would inflate COVERED. The harness runs the suite once at baseline, keeps
the green set, and names the excluded:

| test | why |
|---|---|
| `code/tests/test_wp1_registry_wiring.py` | **Real.** WP1's own acceptance gate: 50 numeric differences against the `20defb2` base. WP2's F-002 recalibration deliberately moved those numbers, so this gate is red and will stay red until it is rebaselined to post-WP2 values or retired. It needs a decision. |
| `code/repro/test_parameter_consistency_sol.py` | **Real.** SSA fertilizer cost share 0.03578 against a hardcoded 0.037. Post-WP2 drift; the literal is the thing to fix. |
| `code/repro/test_parameter_extremes_sol.py` | **Real.** Non-finite values in `structural_cases`. Not yet diagnosed. |
| `code/repro/test_cross_document_consistency_sol.py` | Environmental. Wants the v14 manuscript `.docx` at an absolute path outside the repo. |

## Also observed

The committed `data/canonical_ERA5_y30.json` no longer reproduces from this
tree: 50 of 107 numeric fields differ, and `y_base` is now pinned to the
FAOSTAT target. Global production-weighted S3 loss runs 2.32 / 3.20 / 3.31 %
for years 1 / 10 / 30, against the 3.03 % year-10 figure WP6's acceptance
quotes. That is WP6's regeneration debt and is flagged here only because this
harness had to fingerprint a freshly-run baseline rather than the committed
artifact to avoid scoring all 56 leaves as reaching.
