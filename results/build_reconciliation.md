# WP6 reconciliation — the build graph, the Makefile, and what regenerating the chain found

**Date** 2026-07-26. **Spec** F-009 and F-014 in `FINDINGS.md`.
**Commits** `fa83999` (graph, Makefile, benchmark gate), `4de0d19` (regenerated chain).
**Runs** `logs/build/_run_all.log` plus one log per node under `logs/build/`.

---

## 1. What was built

| Artifact | What it is |
|---|---|
| `code/build.py` | 30 declared nodes, each a generator with its inputs and outputs; status, verify, run, stamp, graph, fingerprint |
| `Makefile` | `make verify` (13 suites plus the graph), `make all`, `make stale`, `make figures`, `make status`, `make mutation` |
| `code/tests/test_benchmark_baseline.py` | F-009's missing gate: re-runs the benchmark suite and compares every verdict against `data/benchmarks/baseline_verdicts.json` |
| `.build/*.json` | one provenance sidecar per stamped node |
| `.build/unstamped_baseline.json` | the pre-graph exemption list, pruned on every successful verify |
| `baseline/surviving_v15/` | frozen copies of the four artifacts whose generators did not survive the crash |

`params_fingerprint()` hashes `params.yaml` with `DOCUMENTARY_KEYS` removed, per
F-012. The list is a denylist — `provenance`, `note`, `source`, `citation`,
`used_by`, `affects_claims`, `benchmark`, `units`, `label` and their kin — so a
key nobody has argued about is fingerprinted by default. Adding a comment to
`params.yaml` no longer restales twenty-eight nodes; changing a value still does.
That was watched both ways before it was trusted.

Three declaration classes exist besides node outputs, and each exists because a
file was otherwise going to be reported as a defect it is not: `EXTERNAL_INPUTS`
(observations and retrievals, each with a note on where the numbers come from —
F-014's `prices` node failed for want of exactly this), `TEST_ARTIFACTS` (files a
test deposits, evidence about the code rather than results the manuscript cites)
and `CACHES`. `NARRATIVE` lists the hand-written reconciliation notes by name
rather than by pattern, so that a generated `.md` cannot hide among them.

---

## 2. The acceptance, item by item

> **Acceptance:** 28 nodes OK, one orphan (`figures/Figure_S5_flux_decomposition.png`),
> one unsourced input (`data/figS12_curves.json`); `canonical_ERA5_y30.json`
> unchanged at global year-1 2.32% and year-10 3.03%.

| Acceptance | Result | Verdict |
|---|---|---|
| 28 nodes OK | **28 OK**, 2 BLOCKED, of 30 declared | count matches, composition differs |
| one orphan | **one orphan**, `data/soc_trajectories.csv` | count matches, identity differs |
| one unsourced input | **three**, none of them `figS12_curves.json` | does not reproduce |
| canonical year-1 2.32% | **2.32%** | reproduces |
| canonical year-10 3.03% | **3.20%** | does not reproduce |

Nothing was tuned to close any of these. Each deviation is accounted for below,
and the fifth one turned out to be the finding of this package.

---

## 3. The canonical run does not reproduce after year two, and the cause is one parameter

This is the important part of this document.

`data/scenario_trajectories.csv` survived the crashed v15 session and carries
F-014's and F-016's numbers exactly. Running the present tree's generators
against it gives a clean signature:

| year | v15 global S3 loss | rebuilt | difference |
|---|---|---|---|
| 1 | 2.316 | 2.316 | 0.000 |
| 2 | 2.895 | 2.895 | 0.000 |
| 3 | 2.954 | 2.984 | +0.030 |
| 5 | 2.992 | 3.087 | +0.095 |
| 10 | 3.032 | 3.198 | +0.166 |
| 30 | 3.081 | 3.309 | +0.228 |

Years 1 and 2 agree to three decimals in all eight regions. From year 3 the two
diverge monotonically. Soil carbon shows the same shape: year-0 stocks agree to
four decimals in every region, and the 30-year decline is uniformly steeper in
the rebuild (sub-Saharan Africa 2.240% against F-016's 2.145%, FSU 1.825%
against 1.673%).

The realized fertilizer reduction under S3 localizes it exactly:

| region | v15 yr1 | rebuilt yr1 | v15 yr10 | rebuilt yr10 |
|---|---|---|---|---|
| north_america | 0.12897 | 0.12897 | 0.12345 | 0.12850 |
| europe | 0.15611 | 0.15611 | 0.14553 | 0.15513 |
| south_asia | 0.24326 | 0.24326 | 0.22948 | 0.24222 |
| sub_saharan_africa | 0.29859 | 0.29859 | 0.28712 | 0.29851 |
| fsu_central_asia | 0.18533 | 0.18533 | 0.16995 | 0.18394 |

Year 1 is identical to five decimals. Over the following decade the v15 model
buys back roughly a tenth of the reduction as soil nitrogen falls; the rebuilt
model buys back almost none — an order of magnitude less in every region, and
essentially nothing in sub-Saharan Africa. That buy-back is F-015's mechanism:
*"Farmers buy back part of the reduction as soil nitrogen falls, and they buy
back more of it over time."* It runs through one parameter, `eps_F_N`, the
elasticity of fertilizer demand with respect to the soil nitrogen stock.

**Setting `eps_F_N = -0.5` reproduces the deposited v15 canonical exactly.**

| `eps_F_N` | mean absolute error against the deposited v15 year-10 regional losses |
|---|---|
| 0.0 (what this tree runs) | 0.1889 pp |
| **-0.5** | **0.0002 pp** |
| -0.25 | 0.0927 pp |
| -1.0 | 0.1751 pp |

At -0.5 all eight regions land on the deposited value to the precision the
artifact carries: NA 1.726, EU 3.429, EA 1.182, SA 4.812, SEA 3.675, LATAM
2.418, SSA 4.749, FSU 5.126. This is not a fit — one value was set and every
one of eight numbers fell into place.

**So the deposited v15 results require `eps_F_N = -0.5` in S3, and this tree
runs S3 at `eps_F_N = 0`.** The tree is explicit and internally consistent
about running it at zero:

- `parameter_registry.py`: `SOIL_N_RESPONSE_ELASTICITY_CENTRAL = 0.0`.
- `coupled_econ_biophysical.py`, the S3 definition: *"eps_F_N is zero
  centrally"*, and the module docstring: *"The soil-N response elasticity is
  zero in the central case because no empirical regional estimate is available.
  Negative values are evaluated separately as structural sensitivities."*
- The comment above `REGIONAL_ECON_PARAMS`: *"The registered value (-0.5) is the
  S4 structural-sensitivity setting; the reported S1–S3 runs hold it at
  SOIL_N_RESPONSE_ELASTICITY_CENTRAL = 0.0 … F-011 scores it
  DECLARED_NOT_WIRED for this reason; the verdict is correct and is not a gap
  to close."*

And the evidence on the other side is just as explicit:

- F-015: *"The sustained mean under S3, **with `eps_F_N` active**, is 0.1911."*
- `params.yaml` registers `eps_F_N: -0.5` with `affects_claims: [C-040, C-050]`.
  C-050 **is** the S3 shock-calibration claim. A parameter held at zero in S3
  cannot affect it.
- `results/s3_shock_calibration.csv`, the surviving v15 artifact, shows the
  buy-back over time that only a nonzero `eps_F_N` produces.

Both statements cannot be true of the same model. Either the v15 session
changed the central value and the change died with the tree, or the deposited
numbers were produced with a setting the deposited code never declared. **This
is a decision for Matthew, not more code**, and it is a larger one than it
looks:

1. **Every multi-year number in the paper moves.** Global year-10 loss 3.03% or
   3.20%; year-30 3.08% or 3.31%; every regional trajectory; the SOC declines;
   the realized S3 reduction; Figure 2's decomposition; the food-price indices.
   Year 1 is unaffected either way, so the abstract's year-1 figures stand.
2. **The direction is counter-intuitive.** Switching the feedback *off* makes
   the losses *larger*, because the buy-back is a cushion. The published numbers
   are the more conservative pair.
3. **The mechanism attribution changes with it.** At `eps_F_N = -0.5` part of
   the multi-year damping is farmers responding to their own declining soil
   nitrogen — a behavioural channel. At zero, everything after year 1 is soil
   and price alone. The paper argues about soil buffering, so which channel is
   carrying the multi-year result is not a detail.
4. **`eps_F_N` is the registry's weakest parameter.** Its own `source` field
   reads *"No clean regional estimates exist; chosen for the S4 feedback
   channel"*, and F-008 records that it *"has no published analogue and must
   stop being presented as though it has one."* If the central results depend
   on it at -0.5, that sentence has to be in the SI, and the number needs a
   sensitivity band rather than a point value.

The experiment is one line and takes ten seconds: set `s3.eps_F_N = -0.5` after
`get_scenario_params()['S3']` and re-run `run_canonical.py`. Nothing was changed
in the tree to make it come out either way; this package regenerated the chain
with the model exactly as WP1 and WP2 left it.

**Consequence for the regenerated artifacts.** Everything regenerated in
`4de0d19` carries the `eps_F_N = 0` numbers. The chain is internally consistent
and the graph is green, but **the regenerated artifacts should not be treated as
the paper's numbers until this is settled.** The pre-regeneration state is
recoverable in full from `fa83999` (`git show fa83999:data/figure2_panels.json`
and so on), and the four artifacts with no generator are additionally frozen in
`baseline/surviving_v15/`.

---

## 4. F-009's two named defects, re-examined

**The stale duplicate is confirmed and deleted.** `data/climate_swap_comparison.csv`
disagreed with `outputs/climate_swap_comparison.csv` on every row, exactly as
F-009 records: sub-Saharan Africa's year-10 loss 14.0% against 4.92%, South
Asia's year-10 shift 0.54 pp against 0.69 pp. Nothing writes the data-directory
copy and nothing could. Removed (the file is preserved under `_transfer/`, which
is now gitignored, because the bridge cannot delete on this volume).

**`data/figS12_curves.json` is not unsourced.** `code/repro/make_table_s4_sol.py`
writes it, at line 63, and has since commit `ca87332` — before the reconstruction
base. `README.md` line 60 says so too: *"make_table_s4_sol.py → outputs/Table_S4_calibration_sol.csv
and data/figS12_curves.json"*. The v15 graph must have declared only the first of
that script's two outputs, which is precisely the defect F-014 found later in the
`prices` node — *"It claimed three outputs and writes one"* — arriving from the
other direction. One script, two writes, one declaration. The node here declares
both, and Figure S12's input has a live generator.

**The file that really is unsourced is `data/crop_response_calibration_table.csv`.**
`make_ofra_validation.py` reads it to draw Figure S13. `MANIFEST.md` credits
`make_table_s4_sol.py` with regenerating it. That script does not write it, and
no script in the deposit does. So Figure S13 is the figure drawn from a file of
unknown provenance, and it has been since at least the v14 deposit. F-009's
verdict was right about the class and wrong about the file, and the wrong file
is the one that got a work package.

**`figures/Figure_S5_flux_decomposition.png` is not an orphan in this tree
because it is not in this tree.** It exists only as
`excluded_legacy_sol/Figure_S5_flux_decomposition_legacy_sol.png`, which
`MANIFEST.md` places outside the evidentiary chain, and the orphan scan does not
walk that directory. `make_figure_s5.py` is still unwritten; that debt is real
and belongs to D3, but it does not show up as an orphan here and no orphan was
invented to make the count match.

---

## 5. The orphan and the three unsourced inputs

| file | verdict | why |
|---|---|---|
| `data/soc_trajectories.csv` | ORPHAN | `make_soc_trajectories.py` (F-016) did not survive the crash |
| `data/soc_trajectories.json` | UNSOURCED | same generator; read by the claim register for C-010 |
| `results/s3_shock_calibration.csv` | UNSOURCED | `make_s3_shock_calibration.py` (F-015) did not survive the crash |
| `data/crop_response_calibration_table.csv` | UNSOURCED | no script has ever written it (§4) |

Three of the four are the same story: **the v15 pass closed the last owed
generators in F-015 and F-016, and those two scripts were lost with the tree
while their outputs survived.** F-016's closing sentence — *"Nothing is left
that the register admits it cannot check"* — is no longer true of this tree, and
no work package in the rebuild plan writes either script. That is a hole in the
plan, the third of its kind after `test_spinup_partition_independence.py` (since
rebuilt) and the eight-region price table.

Each of the four is allowed by name on the `make verify` command line, with the
reason written into the Makefile beside it. Writing any one of the generators
removes its line. Adding a fifth line requires an entry in `FINDINGS.md` saying
why.

---

## 6. Two nodes refused to run

`build.py` has a `BLOCKED` state: a node whose generator is behind its deposited
artifact, so that running it would destroy evidence the tree cannot rebuild.
Blocked nodes are skipped by `run --all`, are reported with their reason on
every `status` and every `verify`, and can be overridden with `--force`.

**`scenario_trajectories`.** The deposited CSV carries a `PULSE1_global` column.
F-016 added the one-year pulse to the model and to this generator; that work
died with the tree, and the surviving generator is the pre-pulse one. Running it
drops the column, and C-061's two checks read it. The blocked node is the
difference between a build graph and a script that overwrites things.

**`mc_ensemble`.** The deposited ensemble is the only surviving v15 one and is
what F-013's claim strength reproduces against (P3 = 0.998 on the nose).
Rerunning it costs about ninety minutes and would overwrite that evidence with
draws from the configuration §3 puts in question. It should be rerun once
`eps_F_N` is settled, and not before.

---

## 7. What the regeneration moved

Forty-nine artifacts changed; `outputs/benchmarks.csv`, `outputs/benchmarks.json`,
`data/benchmarks/broadbalk_yield_benchmark_sol.csv` and the two blocked nodes'
outputs did not.

- **The benchmark suite reproduced byte for byte** — 35 rows, same verdicts,
  `B3-europe-YR30` still failing at 0.406. `test_benchmark_baseline.py` re-runs
  the suite against the live model rather than reading the committed CSV, and
  passes.
- **The claim gate did not move**: 19 claims, 70 checks, 42 AGREES, 28 DRIFTED,
  0 unresolved, same drifted set, smallest drift still `C-060/east_asia_yr10` at
  0.118 pp. WP5 predicted `docs/claims_baseline.json` would need regenerating
  after WP6; it does not, because the baseline records claim identities rather
  than values, and no claim crossed its tolerance. The three claims WP5 offered
  as a free acceptance test for WP6 (C-010, C-060, C-061) read the two artifacts
  this package refused to regenerate, so that test was not run rather than
  passed — and §3 is the reason it would have failed.
- `data/figure2_panels.json` now agrees with the canonical year-10 losses
  (global total 3.1975 against the canonical 3.198), which was C-011's
  `pending_regeneration`. It agrees at the `eps_F_N = 0` values.
- **Every figure changed by more bytes than its data did.** The figures were
  rendered in the reconstruction container, not in the environment that produced
  the deposited PNGs, and most of the byte difference is the renderer. A deposit
  that pins `requirements.txt` but not a matplotlib version cannot promise
  byte-identical figures; the numbers behind them are what the graph fingerprints.

---

## 8. Three suites are not in the gate, by name

`make verify` runs thirteen. Two are excluded with the reason written into the
Makefile: `test_parameter_consistency_sol.py` is red on purpose (three derived
nitrogen cost shares moved when F-002 recalibrated the production path; the
repair is a document edit owed to the register, not a code change), and
`test_cross_document_consistency_sol.py` reads a manuscript `.docx` that is not
in the deposit and belongs to D1/D2. `run_mutation_coverage.py` is a
twenty-minute sweep rather than a gate and has its own target.

`test_wp1_registry_wiring.py`, which the state file records as red and expected
to stay red, **passes** — it was rebaselined in `8573179`. The status table was
not updated; that is now the fourth package in a row where the git log and the
table disagree.

---

## 9. For whoever runs the next package

1. **Settle `eps_F_N` before anything else.** D1 and D2 write numbers into the
   manuscript, and which numbers they are depends on this. So does whether the
   ensemble is rerun.
2. **`make_soc_trajectories.py` and `make_s3_shock_calibration.py` need
   rebuilding**, and no package owns them. They are small — each runs S3 for
   thirty years and deposits per-region series — and both should be written
   against whatever `eps_F_N` decision is made, not before it.
3. **The pulse capability is missing from `make_scenario_trajectories.py`**
   (F-016's `get_pulse_scenario`, `EconParams.fert_price_shock_years`, and the
   `t >` rather than `t >=` condition). Until it is restored, that node stays
   blocked and the deposited CSV is the only copy of the pulse column.
4. `MANIFEST.md` owes two corrections, and one it was said to owe it does not.
   The claim that `make_table_s4_sol.py` regenerates
   `crop_response_calibration_table.csv` is wrong (§4), and the Figure S5 entry
   needs revisiting. But **F-009's "0.74 pp" is not in this tree**: `MANIFEST.md`
   line 88 and `README.md` line 108 both read 0.69 pp, and the regenerated
   output's maximum year-10 shift is 0.70 pp with a Spearman ρ of 0.98 against
   the README's 0.95. So the manifest edit that was owed is a smaller one than
   recorded, and the ρ is the number that has drifted.

5. **`climate_comparison.py` still prints its two headline numbers to a console
   instead of depositing them.** F-012 records that it was changed to write
   `results/climate_swap_stats.txt` precisely so that 0.74 pp and ρ = 0.93 could
   not survive a recalibration by being transcribed; that change was lost with
   the tree. The numbers above were read out of `logs/build/climate_swap.log`,
   which is the F-009 mechanism operating on the very finding that named it.
   The one-line repair belongs to D3.
