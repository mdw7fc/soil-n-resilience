# Mutation coverage: WP3 rebuild against F-011

Two sweeps, 2026-07-25/26. The first scored the suite as WP3 found it; the
second scored it after the repairs that sweep exposed.

|                     | F-011 | as found | after repairs |
|---------------------|------:|---------:|--------------:|
| COVERED             |    12 |        5 |        **32** |
| UNTESTED            |    22 |       27 |         **0** |
| DECLARED_NOT_WIRED  |     3 |        2 |             2 |
| GUARDED_AT_LOAD     |     6 |        6 |             6 |
| INERT               |    13 |       16 |            16 |
| **total**           |    56 |       56 |            56 |

The leaf list is identical throughout — `registry.leaves()` returns the same 56
names — and `GUARDED_AT_LOAD` reproduces F-011 exactly, in count and in
membership. Nothing was tuned toward any acceptance figure.

## Read the headline sceptically

**UNTESTED 0 is not as good as it looks, and the depth table in
`mutation_coverage_summary.txt` is the number to read instead.**

`test_wp1_registry_wiring.py` now regenerates the canonical artifact and
asserts a 123-field delta. That makes it a whole-artifact fingerprint, so it
objects to *every* mutation that moves a published number, by construction.
It catches all 32 COVERED leaves. Twelve of them — `alpha`, `cropland_mha`,
`eps_F_PY`, `eps_LD_PL`, `eps_LD_PY`, `eps_LS_PL`, `eta`,
`faostat_yield_target`, both `laub_tropical_ratios` leaves,
`physical_feedback_strength`, `whc_sensitivity` — are caught by nothing else.
Retire or rebaseline that one file and those twelve return to UNTESTED
immediately.

This is F-011's own criticism in a new form. It wrote that "a suite whose
catching power sits in one behavioural test and two mirror tests is thin". The
current suite's catching power sits in one fingerprint test and one
behavioural test. Genuine per-parameter coverage is what
`test_spinup_partition_independence.py` provides for 15 leaves and
`test_dimensional_consistency_sol.py` for 4. The remaining 12 have a tripwire,
not a test. F-011's standing worklist item — write a test for each leaf that
nothing specific catches — is not discharged; it has moved from 22 leaves to
12.

## What the repairs were

Four problems, all surfaced by reconciling the first sweep against F-011
rather than by any test.

### 1. `test_spinup_partition_independence.py` did not exist

F-011 records it as catching eight of twelve COVERED leaves. It was written
during the v15 pass, lost with the working tree, and owned by no work package.
`code/model/params.yaml` cited it twice — as the test that falsified the claim
that the spin-up overwrites the initial pool partition, and as the writer of
`results/spinup_partition_characterisation.yaml`, which the SI limitation
cites. So the registry documented a test that was not there and pointed the SI
at a file nothing wrote.

Rebuilt from those citations. Sweeping `f_passive` over 0.45 / 0.58 / 0.73 in
all eight regions, it establishes: the fast pools are partition-independent;
the passive pool is not, and must not be (asserted as a floor, so that if it
ever starts converging the test fails loudly instead of silently restoring a
claim the registry records as false); the true fixed point is
partition-independent; and the published quantities are flat, worst-case
0.0133 pp against 0.1 pp reporting precision, which is what actually licenses
the ensemble exemption. It also pins the spin-up equilibrium, which is what
gives the 15 leaves their coverage — checks A–D are structural and would hold
across a range of parameter values.

**One prose number did not survive.** The `mc_exempt_reason` states absolute
SOC moves "more than 8 t C/ha in every temperate region". Measured: Europe
10.53 and FSU 8.63 clear it, North America is 2.32. The SI limitation cites
that sentence and needs restating. The licensing argument is unaffected — it
rests on the published quantities being flat, not on the size of the SOC move.

### 2. `test_wp1_registry_wiring.py` was green by being stale

It read `data/canonical_ERA5_y30.json` off the tree. That is a deposit
artifact, only rewritten when someone runs the model, so on an un-regenerated
checkout it compared the 20defb2 baseline against the 20defb2 artifact and
reported "no number moved" — passing because the file was stale, not because
the code agreed. It went red only inside the mutation sandboxes, which run the
model first. That is why the first sweep excluded it.

Both halves had to change: it now re-runs the model into a throwaway copy, and
asserts the *delta* rather than zero — exactly these 50 fields moved, to these
values, and nothing else did, pinned in
`baseline/canonical_expected_delta.json`. Rebaselining to post-WP2 figures
would have erased the evidence that WP2 moved anything. CHECK 3 had the same
staleness defect and was still asserting the 20defb2 headline; it now reads
the fresh run and expects 2.32 / 3.20 / 3.31.

### 3. `test_parameter_extremes_sol.py` failed on a correct run

Its blanket finiteness assertion over every numeric column tripped on
`ln_cap`, the diagnostic column F-010 added, which is NaN whenever the
fertilizer ceiling does not bind. NaN is the correct encoding of "not
applicable". `ln_cap` is now asserted finite exactly where `cap_binding` is
true and NaN everywhere else — stronger than the assertion it replaces, not an
exemption.

### 4. `test_parameter_consistency_sol.py` is still red, deliberately

WP2 froze `EXPECTED_SHARES` at pre-F-002 figures on purpose and owes them to
WP5's claim register; rebaselining would lose the evidence that a published
number moved. That decision stands. But its note claimed only sub-Saharan
Africa had drifted, and the assertion stopped at the first region, so it never
evaluated the rest. Three of four have drifted:

| region | pinned | model | delta |
|---|---:|---:|---:|
| sub_saharan_africa | 0.037 | 0.035778 | −1.22e-3 |
| south_asia | 0.153 | 0.147321 | −5.68e-3 |
| latin_america | 0.047 | 0.049145 | +2.15e-3 |
| north_america | 0.060 | 0.060800 | +8.00e-4 (within tolerance) |

South Asia is the largest mover and the one that matters: SI [163] and claims
C-063 / C-064 turn on which region carries the highest derived nitrogen cost
share. WP5 should carry all three, not one. All four are now measured before
anything is asserted, and reported together.

## Leaf-level disagreements with F-011 that remain

Both are unchanged by the repairs and both are accounted for.

**`bnf_potential` and `yield_min_regional` fell UNTESTED → INERT.** They move
no field of the canonical artifact now that WP2 solves `y_max` against the
FAOSTAT target. F-011 already recorded `yield_max_regional` as INERT for
exactly this reason and cited SI [65]: "the static values are legacy
fallbacks, not the reported calibration." The same sentence now covers all
three.

**`texture_class` scores INERT, not DECLARED_NOT_WIRED.** F-011's boundary
between those two verdicts could not be recovered from the finding text. Three
candidate rules were tested and rejected: *never requested from the registry*
(separates `eps_F_N` and `fert_reduction_target` correctly but not
`texture_class`, which is requested, into `RegionParams`); *empty `used_by`*
(`eps_F_N` and `bnf_ramp_years` both declare `[]`, so it separates nothing);
and a *static AST check for a consuming read* (too coarse — values fetched
inline as `registry.value(...)` are never bound to a name and read as
unwired). The rule adopted is dynamic and stated in the harness docstring:
DECLARED_NOT_WIRED means the mutation moved no model state at all, INERT means
model state moved but no published number did. Under it `texture_class`
reaches `RegionParams.texture_class` and stops there, so it scores INERT — and
`cre_base` scores INERT on the same grounds, which is what F-011 says it
should. Recorded as a reconstruction gap rather than resolved by picking the
rule that reproduces 3 and 13.

## Two defects in this harness, found by reconciliation and fixed

1. **The canonical run was conditional.** The first revision short-circuited to
   DECLARED_NOT_WIRED whenever the model-state snapshot did not move, without
   running the model at all. Any parameter consumed *inside* the run rather
   than stored on a parameter object was mis-scored:
   `residue_c_to_active_fraction` and both `laub_tropical_ratios` leaves move
   43, 27 and 27 published fields and were reported as reaching nothing.
2. **The state probe missed the price seam and the tropical SOM variant.**
   `SOMPoolParams()` built with defaults does not read `laub_tropical_ratios`,
   and three price constants are consumed only inside validators. Six leaves
   scored DECLARED_NOT_WIRED for want of a probe rather than for want of
   wiring.

The first is the same class of error F-011 recorded against its own probe —
"the fingerprint is too narrow" — one layer further in. A narrow fingerprint
does not report uncertainty; it reports a confident negative.

## Still owed

- Real per-parameter tests for the 12 leaves that only the canonical
  fingerprint catches.
- WP5 to carry the three drifted cost shares.
- The `mc_exempt_reason` SOC sentence, and the SI limitation citing it.
- WP6 owns the stale deposit artifact: the committed
  `canonical_ERA5_y30.json` is the 20defb2 file. A live run gives global S3
  losses of 2.32 / 3.20 / 3.31 % for years 1 / 10 / 30, against the 3.03 %
  year-10 figure quoted in HANDOFF §5 and in WP6's acceptance. That acceptance
  number may itself need restating.
- `n_price_usd_kg_farmer_paid`, `price_benchmark_max_factor` and
  `urea_n_fraction` join the three F-011 already listed as `not_probed`: they
  move gross margins, prices or cost shares, none of which the canonical
  artifact carries. Widening the published set to the margin outcomes is
  F-011's own outstanding item.
