# v15 REBUILD — STATE

> **SUPERSEDED working record (marked 2026-08-29, F-028).** This file is a
> preserved process document from the v15 rebuild. Numbers quoted in it predate
> F-025 (realized-yield market clearing) and F-026 (central eps_F_N = -0.50)
> and are NOT the released results. The released headline family is: global S3
> yield loss 2.32 / 3.02 / 3.07 % at years 1/10/30; SSA y_max 3.97. Current
> truth lives in README.md, docs/claims.yaml (v17 basis) and FINDINGS.md.


**Read this file first. It is deliberately short.**
Full specification: `FINDINGS.md` (73 KB — read only the section named for your step, never the whole file).
Full context: `HANDOFF_v15_model_assurance.md`.

**Reconstruction base:** `../resumbission/v14_sol/reproducibility/ERFS-100341-soil-resilience_sol/`, git commit `20defb2`, tree clean.
**Rebuild target:** `paper2-soil-resilience/v15/` on Matthew's disk.

---

## The rule that matters

The v15 pass did sixteen findings in one context and died holding eight hours of work that existed nowhere but a container. The crash was survivable. Not having committed was not.

**Commit every artifact to Matthew's disk the moment it exists.** Not at the end of the step. Not when it is tested. When it exists. A broken file on disk beats a perfect file in a container.

---

## Work packages

Each is one Cowork task. Do not combine them. Mark status here and commit this file before the task ends.

| # | Package | Spec | Status |
|---|---|---|---|
| WP1 | Parameter registry (`params.yaml`, `registry.py`) + wire it into the four model modules | F-001, F-006, F-007, F-011 | **done** — `9b2af15`; acceptance gate now red, see Log |
| WP2 | Production-path calibration + seam contracts + `test_cap_market_clearing` rewrite | F-002, F-005, F-010 | **done** — `9b2af15`, `2b76b16` |
| WP3 | Mutation coverage harness | F-011 | **done** — `3307523`; 44/56 leaves agree with F-011, see Log |
| WP4 | Benchmark suite + `observed_values.yaml` + baseline verdicts | F-008 | **done** — `97ccc58`, `4733bc3`; 35 rows against the acceptance 41, see Log |
| WP5 | Claim register (`claims.yaml`) + claim strength | F-012, F-013, F-015, F-016 | **done** — `86bf0b1`, `c143f2f`, `ae3dac9`; acceptance met exactly, see Log |
| WP6 | Build graph + Makefile + full regeneration | F-009, F-014 | **done** — `fa83999`, `4de0d19`, `+reconciliation`; 30 nodes, 28 OK, 2 blocked; **the canonical does not reproduce after year 2 and one parameter explains it — see Log** |
| D1 | Main-text edits | HANDOFF §7 | **held by Matthew, 2026-07-26** — will not run until `eps_F_N` is settled by him rather than by WP6; §7's list is anchored on the v14 submission and v14_sol has already made most of it |
| D2 | SI edits + regenerated Table S1 + new limitations + benchmark section | HANDOFF §7 | **held**, same reason. Now also owes a reprint of Supplementary Table S4 (F-017) |
| D3 | Deposit docs: MANIFEST.md, Figure S5, the genuinely unsourced input | F-009, F-017 | **done** — `1037e39`, `d6272b0`; two of the three items were the wrong items, see Log |

**Dependencies.** WP1 blocks WP2–WP6. WP2 blocks WP6. WP4 and WP5 are independent of each other. D1, D2 and D3 depend on nothing and can run first or in parallel with any WP.

---

## Discipline inside a task

1. **Read narrowly.** `Grep` FINDINGS.md for the finding ID. Do not `Read` the whole file. It is 28,000 tokens.
2. **Delegate exploration.** Reading five model files to find where a constant lives is a subagent's job. It returns twenty lines; reading them yourself costs five thousand tokens.
3. **Never paste full test output into the thread.** Redirect to a log file, commit the log, report the exit code and the failing line.
4. **Commit at every artifact.** If fifteen tool calls have passed with nothing written to Matthew's disk, stop and write something.
5. **Update this file and commit it before the task ends,** including a one-line note on anything surprising. If the task dies mid-turn, this file is what the next one inherits.
6. **One package per task.** When the package is done, stop. Do not start the next one because there is room.

---

## Kickoff prompts

Start a new Cowork task with the project folder connected and paste the matching block.

### WP1
> Read `paper2-soil-resilience/v15/v15_REBUILD_STATE.md` first, then WP1 only. Reconstruct `code/model/params.yaml` and `code/model/registry.py` from `v15/outputs/table_S1_parameters.md` (which mirrors all 54 entries) plus findings F-001, F-006, F-007 and F-011 in `v15/FINDINGS.md` — grep for those IDs, do not read the whole file. Then wire the registry into `soil_n_model.py`, `coupled_econ_biophysical.py`, `prices.py` and `monthly_model_v3.py` so the model reads it at import. Acceptance: a 123-field canonical diff returns zero numeric differences. Commit each file to my disk as you finish it. Update the state file and stop when WP1 is done.

### WP2
> Read `paper2-soil-resilience/v15/v15_REBUILD_STATE.md` first, then WP2 only. Grep F-002, F-005 and F-010 in `v15/FINDINGS.md`. Build `calibrate_ym_production` (acceptance: FAOSTAT targets to 8e-3 percent worst case), `code/model/seams.py` with `outcome_weights` / `intensity_weights` / `assert_same_basis` (acceptance: `calibrate_price_shock(0.20)` returns 1.0389792148114703 unchanged), and rewrite `test_cap_market_clearing.py` to re-solve with `brentq` (acceptance: worst residual 1.4e-17; the dropped-gamma mutation drives it to 3.0e-03). Commit as you go. Update the state file and stop.

### WP3
> Read `paper2-soil-resilience/v15/v15_REBUILD_STATE.md` first, then WP3 only. Grep F-011 in `v15/FINDINGS.md`. Rebuild `code/tests/run_mutation_coverage.py`. Acceptance: 56 leaves scoring COVERED 12, UNTESTED 22, DECLARED_NOT_WIRED 3, GUARDED_AT_LOAD 6, INERT 13. Commit the harness and `results/mutation_coverage.csv`. Update the state file and stop.

### WP4
> Read `paper2-soil-resilience/v15/v15_REBUILD_STATE.md` first, then WP4 only. Grep F-008 in `v15/FINDINGS.md` — it names every observed value and its source. Rebuild `data/benchmarks/observed_values.yaml`, `code/repro/run_benchmarks.py` and `data/benchmarks/baseline_verdicts.json`. Acceptance: 41 rows, 11 PASS / 3 MARGINAL / 1 FAIL / 18 INFORMATIVE / 7 OWED / 1 N/A, with B3-europe-YR30 failing at a model ratio of 0.406. **Do not tune the model to any benchmark.** This package is research as much as code; the observed-value compilation is the expensive part. Commit each source as you verify it. Update the state file and stop.

### WP5
> Read `paper2-soil-resilience/v15/v15_REBUILD_STATE.md` first, then WP5 only. Grep F-012, F-013, F-015 and F-016 in `v15/FINDINGS.md`. Rebuild `docs/claims.yaml` (19 claims, 70 checks, two-way index against `affects_claims`), `code/tests/test_claims.py`, `code/repro/make_claim_strength.py` and the three baseline JSONs. Acceptance: AGREES 42, DRIFTED 28, owed generators 0; drifted set C-010, C-011, C-014, C-021, C-030, C-041, C-042, C-060, C-061; the three probabilities in `v15/results/claim_strength.md`. Commit as you go. Update the state file and stop.

### WP6
> Read `paper2-soil-resilience/v15/v15_REBUILD_STATE.md` first, then WP6 only. Grep F-009 and F-014 in `v15/FINDINGS.md`. Build `code/build.py` and the `Makefile`, with `params_fingerprint()` hashing the registry with `DOCUMENTARY_KEYS` removed. Then regenerate the full chain. Acceptance: 28 nodes OK, one orphan (`figures/Figure_S5_flux_decomposition.png`), one unsourced input (`data/figS12_curves.json`); `canonical_ERA5_y30.json` unchanged at global year-1 2.32% and year-10 3.03%. Commit as you go. Update the state file and stop.

### D1 — SUPERSEDED, do not paste
> The original prompt said "every replacement number is in §5; do not rerun the model." Both halves are now wrong. §5's canonical family was produced at `eps_F_N = -0.5` (WP6), and §7's list is anchored on the v14 submission, not v14_sol. Pasting it would overwrite correct v14_sol figures with the -0.5 family. Rewrite it after Matthew settles `eps_F_N`; it should start by diffing `resumbission/v14/…_v14-clean.docx` against `resumbission/v14_sol/…_v14_sol.docx` and land only what v14_sol has not already landed.

### D2 — SUPERSEDED, do not paste
> Same defect as D1, plus two of its own. The SSA 30-year SOC decline is 2.145% at `eps_F_N = -0.5` and **2.24% at zero**, so the single number the prompt is most specific about depends on the open decision. And F-017 added an item the prompt does not contain: Supplementary Table S4 must be reprinted from the regenerated `data/crop_response_calibration_table.csv`. The three new limitations and the benchmark-suite section (F-001, F-004, F-008) are independent of `eps_F_N` and could be split into a package that runs now.

### D3 — done, `1037e39` and `d6272b0`
> Two of its three items were the wrong items. `figS12_curves.json` has had a generator since before the reconstruction base; the unsourced file was `data/crop_response_calibration_table.csv`, and it is the source of Supplementary Table S4. `make_figure_s5.py` should never be written: Figure S5 is not in the paper. See F-017.

---

## Log

- **2026-07-26 — D3 done, and two of its three items were the wrong items.**
  `1037e39` (generators and stats deposit) and `d6272b0` (docs, F-017, stamps).
  `make verify` runs thirteen suites and the graph and exits 0; 30 nodes at
  BLOCKED 2 / OK 28; the unsourced list is down from three to two. Full account
  in FINDINGS.md F-017. Four things the next task should know:

  1. **`data/crop_response_calibration_table.csv` is Supplementary Table S4.**
     Not "an input to Figure S13" — `make_ofra_validation.py` line 15 reads
     `outputs/Table_S4_calibration_sol.csv` and always has, so the graph's input
     declaration was wrong too: the third declared-versus-actual in this deposit,
     and the first in the input direction. The file's real consumer is the SI,
     which transcribes it row for row, and the frozen copy was two recalibrations
     behind. **Every numeric column of SI Table S4 except `FAOSTAT y_obs`, `c` and
     `Floor` is wrong in v14_sol**; the nitrogen columns by 36% and 55%. That is a
     D2 edit and it is not in D2's prompt.
  2. **`make_figure_s5.py` must not be written.** Figure S5 was withdrawn from the
     paper — SI paragraph 200 says so and is the only surviving reference — the
     PNG is not in `figures/`, and the 4-pool code it draws was never deposited.
     Retired in MANIFEST.md and README.md with the reason.
  3. **F-009's 0.74 pp is in no file in this tree.** MANIFEST and README both read
     0.69 pp / rho 0.95; the regenerated values are **0.70 pp / rho 0.98**. Both now
     come from `results/climate_swap_stats.txt`, because `climate_comparison.py`
     printed them to a console until this pass — F-009's own mechanism, still
     running on the finding that named it.
  4. **The two sourced outputs of `make_table_s4_sol.py` were byte-identical
     before and after regeneration** (`figS12_curves.json`,
     `Table_S4_calibration_sol.csv`), and so is Figure S13's PNG. Only the
     unsourced output had drifted. That is the cleanest available demonstration of
     what the graph is for, and it is worth one sentence in the response letter.

  **D1 and D2 were not started.** Matthew held both pending his own reading of
  WP6's `eps_F_N` resolution. The kickoff prompts for both are marked SUPERSEDED
  above with what is wrong with them.

- **2026-07-26 — WP6 done, and it found something bigger than a build graph.**
  `code/build.py` (30 declared nodes), `Makefile`, `code/tests/test_benchmark_baseline.py`,
  `.build/` sidecars and the unstamped baseline landed in `fa83999`; the
  regenerated chain in `4de0d19`. `make verify` runs thirteen suites plus the
  graph and exits 0. `params_fingerprint()` hashes `params.yaml` with
  `DOCUMENTARY_KEYS` removed, so a comment no longer restales twenty-eight
  nodes. Full account in `results/build_reconciliation.md`.

  **THE THING THAT NEEDS MATTHEW'S DECISION BEFORE D1 AND D2 RUN. The
  deposited v15 results require `eps_F_N = -0.5` in S3, and this tree runs S3
  at `eps_F_N = 0`.** The rebuilt model reproduces the v15 year-1 losses to
  five decimals in all eight regions and then diverges monotonically from year
  3 (global year-10 3.198% against the deposited 3.032%, year-30 3.309%
  against 3.081%). Setting `eps_F_N = -0.5` and changing nothing else lands
  every one of the eight regional year-10 losses on the deposited value —
  0.0002 pp mean absolute error, against 0.19 pp at zero. The code is explicit
  that S3 runs at zero (`SOIL_N_RESPONSE_ELASTICITY_CENTRAL`, the S3 docstring,
  and the comment on which F-011's DECLARED_NOT_WIRED verdict rests). F-015 is
  equally explicit that the S3 numbers were produced "with `eps_F_N` active",
  `params.yaml` gives `eps_F_N` `affects_claims: [C-040, C-050]` — and C-050 is
  the S3 calibration claim — and the surviving `s3_shock_calibration.csv` shows
  the buy-back only a nonzero value produces. Both cannot be true.

  **RESOLVED, and the answer is `eps_F_N = 0`.** `v14_sol`'s results section
  reads: "production-weighted global yield loss is 2.3% in year 1, 3.2% in year
  10 and 3.3% in year 30. Regional year-10 losses range from 1.2% in East Asia
  to 5.6% in FSU/Central Asia; South Asia is 5.2%, Sub-Saharan Africa 5.0%,
  Southeast Asia 3.8%, Europe 3.6%, Latin America 2.4% and North America 1.8%."
  The regenerated chain gives 1.21 / 1.80 / 2.51 / 3.67 / 3.84 / 4.93 / 5.12 /
  5.55 and 3.20 / 3.31 — eight of eight regions and all three globals, to the
  precision the manuscript states. At -0.5, six of the eight regions and both
  multi-year globals miss. The manuscript's SC1/SC2 figures (3.7 / 3.9 / 1.9 /
  0.04) match the regenerated 3.758 / 3.898 / 1.919 / 0.045 and not the v15
  artifact's 3.69 / 1.869, and the same sentence ends "without relying on an
  unestimated soil-N demand response." `run_canonical.py` line 97 prints
  "(SOL manuscript: 2.3 / 3.2 / 3.3)".

  **So the tree is right, the regenerated artifacts are the paper's numbers,
  and the 3.03 family is an unrecorded configuration change the v15 session
  made.** F-014's own headline is where it should have been caught: "
  `canonical_ERA5_y30.json` did not move" sits beside a quoted 2.32 / 3.03
  while the committed artifact reads 2.31 / 3.18 / 3.29. It moved by 0.15 pp at
  year 10 and was recorded as unchanged. **Nothing in the tree needs changing;
  D1 and D2 should carry the regenerated numbers.**

  Two consequences that need a human, not a test: `params.yaml` gives `eps_F_N`
  `affects_claims: [C-040, C-050]`, and a parameter held at zero in S3 cannot
  affect C-050, the S3 shock-calibration claim (the two-way index test passes
  either way — it only checks mutual declaration). And F-016's owed edit "SSA
  30-year SOC decline 2.5% -> 2.14%" was computed at -0.5; **at zero it is
  2.24%**. F-015's 0.1911 sustained mean is on the same footing and needs
  recomputing when `make_s3_shock_calibration.py` is rewritten.

  Four more things the next task should know:

  1. **Two nodes refused to run and the refusal is the point.** `build.py` has a
     `BLOCKED` state for a node whose generator is behind its artifact.
     `scenario_trajectories` would have dropped the `PULSE1_global` column that
     C-061 reads and that no surviving script writes (F-016's pulse work died
     with the tree). `mc_ensemble` would have overwritten the only surviving
     v15 ensemble — F-013's P3 = 0.998 evidence — with ninety minutes of draws
     from the configuration now in question. Both are reported with their
     reason on every status and verify; `--force` overrides.
  2. **F-009 named the wrong file.** `data/figS12_curves.json` is *not*
     unsourced: `make_table_s4_sol.py` writes it at line 63 and has since before
     the reconstruction base, and README line 60 says so. The v15 graph must
     have declared one of that script's two outputs — the same defect F-014
     found in the `prices` node. The file that really is unsourced is
     `data/crop_response_calibration_table.csv`: `make_ofra_validation.py` reads
     it for Figure S13, `MANIFEST.md` credits `make_table_s4_sol.py` with
     writing it, and nothing in the deposit ever has. **D3's owed item should be
     re-pointed at that file.** The stale duplicate F-009 deleted is confirmed
     and deleted again here (SSA year-10 14.0% against 4.92%).
  3. **`make_soc_trajectories.py` (F-016) and `make_s3_shock_calibration.py`
     (F-015) did not survive the crash and no work package rebuilds them.**
     Their outputs did survive, which is why they show as one orphan and two
     unsourced inputs. F-016's "nothing is left that the register admits it
     cannot check" is no longer true of this tree. Write them *after* the
     `eps_F_N` decision, not before.
  4. **The claim gate did not move** — 42 AGREES, 28 DRIFTED, same set, smallest
     drift still 0.118 pp — so `docs/claims_baseline.json` did **not** need
     regenerating as WP5 expected. But WP5's "free acceptance test for WP6"
     (C-010, C-060, C-061 must not move when re-pointed at the regenerated
     canonical) was not run rather than passed: those three read the two
     artifacts this package refused to regenerate. Under the present model they
     would have moved, for the reason above.

  Housekeeping: `test_wp1_registry_wiring.py` is **green** (rebaselined in
  `8573179`), against what this file records — the fourth package running where
  the git log and the status table disagree. `_transfer/` and
  `_stale_git_locks/` are now gitignored. `tar` cannot unlink on this volume,
  so extractions into the repo need `--overwrite`.

- **2026-07-26 — WP5 done.** Claim register rebuilt: `docs/claims.yaml` plus
  `code/repro/claim_resolvers.py` (`86bf0b1`), then `code/tests/test_claims.py`,
  `code/repro/make_claim_report.py`, `docs/claims_baseline.json`,
  `docs/claims_index_baseline.json`, `results/claims_report.md`,
  `outputs/claims_status.csv` and the negative-control log (`c143f2f`), then
  `code/repro/make_claim_strength.py`, `docs/claim_strength_baseline.json`,
  `results/claim_strength.{md,csv}` and `results/claims_reconciliation.md`
  (`ae3dac9`). The gate runs in about two seconds; claim strength in about eight.

  **The arithmetic half reproduces exactly, on the first run and with nothing
  tuned: 19 claims, 70 checks, 42 AGREES, 28 DRIFTED, 0 unresolved, owed
  generators 0, drifted set C-010, C-011, C-014, C-021, C-030, C-041, C-042,
  C-060, C-061.** F-012's two signature numbers land on the nose as well — the
  smallest drift is `C-060/east_asia_yr10` at 0.118 pp against a 0.1 pp
  tolerance, the largest is `C-014/ssa_margin_half_pct` at 14.98 pp — as do
  C-060's two surviving agreements (`global_yr1` and the cropland total),
  C-031's four agreements at a 0.214 minimum spread, C-014's 0.27–0.99 pp
  margin gaps, C-042's exactly four drifts, C-041's three, C-010's 2.145% SOC
  decline and C-061's 0.009% year-5 pulse. Full table in
  `results/claims_reconciliation.md` §1. Four things the next task should know:

  1. **The register transcribes the v14 submission, not v14_sol, and v14_sol has
     already made most of the edits the register says are owed.** F-012's C-060
     quotes MS [56] as "East Asia 1.3, South Asia 6.0, FSU 5.5, SSA 5.4, global
     3.4"; v14_sol's paragraph 56 reads 1.2 / 5.2 / 5.6 / 5.0 / 3.2. Anchoring
     on `resumbission/v14/…_v14-clean.docx` resolves all fifteen cited
     paragraph numbers at a single offset; anchoring on v14_sol resolves none.
     C-021 (8.4 mm), C-030 (SSA $1.40) and C-014 (2.5–4.2 pp) are all already
     corrected in v14_sol. **This needs Matthew's decision, not more code, and
     it changes what D1 and D2 do:** either start those packages by diffing v14
     against v14_sol rather than by applying HANDOFF §7's list, or re-point the
     register at v14_sol (cheap — `document_basis` is a header field and the
     checks, artifacts and tolerances are unaffected). The six other drifted
     claims are numbers v14_sol also states and the model also disagrees with,
     so the finding stands either way.
  2. **P4 and P4b do not reproduce, exactly as `RECONSTRUCTION_GAPS` G-4
     predicted.** P3 reproduces on the nose (fsu_central_asia 0.998, runner-up
     southeast_asia 0.002). P4 and P4b are scored over four of eight regions,
     because the surviving deposit prices four and `prices.n_price_usd_kg`
     raises for the rest, so they read 0.961 and 1.000 against F-013's 0.542 and
     0.958. **The verdicts are unaffected** — C-063 and C-064 still name a region
     the ensemble does not put first, and both are carried in
     `docs/claim_strength_baseline.json`. What is not recoverable is the number
     the SI should print in place of 83.7%. Closing it is the eight-region wedge
     and crop-price compilation named in G-4, then an ensemble rerun. **Do not
     interpolate the four missing wedges from the registered range.**
  3. **Eighteen of the nineteen claim identifiers are fixed by evidence, not
     chosen.** `params.yaml`'s `affects_claims` names exactly eighteen, and
     because the index is two-way all eighteen must exist or the gate fails. The
     nineteenth declares no parameter and so leaves no trace in the forward
     index; it is registered as **C-002** (the SI [167] ensemble claim). The
     identifier is a reconstruction, the claim is not. C-062, C-063 and C-064
     are deliberately outside `claims.yaml` — they are probabilities, scored by
     `make_claim_strength.py`, not numbers scored against a tolerance.
  4. **The register does not read `data/canonical_ERA5_y30.json`,** because that
     artifact is the v14 one and does not reproduce. C-010, C-060 and C-061 read
     `data/scenario_trajectories.csv` and `data/soc_trajectories.json` instead,
     which are surviving v15 artifacts carrying F-014's and F-016's numbers
     exactly. **When WP6 regenerates the canonical artifact these three claims
     can be re-pointed at it and must not move** — that is a free acceptance
     test for WP6. Conversely `figure2_panels.json`, `figS8_curves.json` and
     `food_price_response.csv` ARE pre-regeneration, so C-011, C-041 and C-042
     will keep drifting after WP6 (the document numbers are wrong either way)
     but their model column will move, and `docs/claims_baseline.json` will need
     regenerating with the WP6 FINDINGS entry.

  The gate was watched failing before it was trusted to pass: eight negative
  controls in `logs/run_130_claims_neg.log`, all fired. Two of the three
  baselines are the ones FINDINGS names; the third,
  `docs/claims_index_baseline.json`, freezes the claim/parameter index itself
  and is reasoned about in `results/claims_reconciliation.md` §3.

- **2026-07-26 — WP4 done.** Benchmark suite rebuilt: `data/benchmarks/observed_values.yaml` (`97ccc58`), then `code/repro/run_benchmarks.py`, `data/benchmarks/baseline_verdicts.json`, `outputs/benchmarks.csv` / `.json`, `results/benchmark_reconciliation.md`, `logs/run_36_benchmarks.log` (`4733bc3`). The suite runs in about 40 seconds.

  **Result 35 rows at 9 / 3 / 1 / 14 / 7 / 1 against the acceptance 41 rows at 11 / 3 / 1 / 18 / 7 / 1.** MARGINAL, FAIL, OWED and NOT_APPLICABLE reproduce exactly, and so do the row identities behind them. **`B3-europe-YR30` fails at 0.4063 against F-008's 0.406.** Nothing was tuned. Full account in `results/benchmark_reconciliation.md`. Four things the next task should know:

  1. **The model still produces F-008's numbers.** Eight quantities were checked before any row was defined. Six reproduce exactly — the europe nil-N ratio at 30 years (0.4063), the 96-year fertilized and nil SOC drifts (-0.2098 %, -27.95 %), the fert-minus-nil SOC excess at 30 years (21.0 %), and the SSA implied rate (47.59) — and two to three decimal places. Only the year-1 europe ratio moved by more than rounding, 0.763 to 0.768, which is consistent with WP2's recalibration. So WP1 and WP2 did not break what F-008 measured.
  2. **The six missing rows are 2 PASS and 4 INFORMATIVE, and their identities are not recoverable.** F-008's prose names sixteen rows; the suite implements all sixteen plus the per-region extensions its naming convention implies plus three owed observations this pass opened. Every candidate partition reaching 41 was tested against the six-way tally and rejected, including folding in the 2022 price hindcast, which would have added failures F-008 does not record. Rows were not invented to close the count. Only a surviving `outputs/benchmarks.csv` from the crashed session closes this, and it is confirmed absent everywhere.
  3. **The research half found three things the manuscript owes.** The SSA response ratio of **0.572 could not be sourced at all** — and it has the exact signature of a log response ratio read as a linear one (ln 0.572 → 1.77, inside the published median band of 1.7–1.8). That needs a decision, not more code. The **7.7–20.0 SSA MPP envelope is not published either**, and it mixes marginal products with agronomic efficiencies. **No own-price elasticity of fertilizer demand exists for fsu_central_asia in any language**, so that row is scored against a range built entirely from other regions. Prague-Ruzyně, by contrast, verified exactly against Hlisnikovský et al. 2022 *Plants* 11:1825 Table 1, with four qualifications the SI must carry (ratios are derived not published; the table is 1 d.p. so they carry ±0.02; they are ratios of period means; the yields are winter wheat after potatoes only, 9 and 14 seasons).
  4. **`B1`'s MPP triple did not reproduce** — 24.72 / 26.10 / 112.18 against F-008's 24.8 / 25.9 / 109.2. Six finite-difference conventions were tested; the central difference is step-independent to five significant figures, so it is the model's true derivative and the deviation is not a step artifact. Recorded as a reconstruction gap.

- **2026-07-26 — three things for whoever runs WP6.** (Written for WP5 or WP6; WP5
  has since run and did not close any of them.)
  - **`test_benchmark_baseline.py` is still unwritten and WP4 did not write it.** It is F-009's artifact and belongs to WP6 with the build graph and `make verify`. `baseline_verdicts.json` is frozen and waiting for it, so the gate has something to compare against the moment it exists. Until then the benchmark suite does not gate.
  - **Someone rebuilt `test_spinup_partition_independence.py` in `44d0bc8`** — the hole WP3 flagged is closed — **but they did not update this file either.** That is the third package in a row to finish without touching the status table. Check `git log` against the table, not the table.
  - **The git lock problem is worse than recorded.** `.git/HEAD.lock` strands as often as `.git/index.lock`, and both must be moved aside *between* `git add` and `git commit`, not just once before. See [[icloud-git-locks]].

- **2026-07-25 — WP3 done.** Mutation harness rebuilt: `code/tests/run_mutation_coverage.py` plus `_mutation_state_probe.py`. Sweep is 20 min on 2 cores. Outputs `results/mutation_coverage.csv`, `results/mutation_coverage_summary.txt`, `results/mutation_coverage_reconciliation.md`, `logs/run_67_mutation.log`.

  **Result 5 / 27 / 2 / 6 / 16 against the acceptance 12 / 22 / 3 / 6 / 13.** 56 leaves, 44 agree with F-011 leaf by leaf; GUARDED_AT_LOAD reproduces exactly. Nothing was tuned. Full account in `results/mutation_coverage_reconciliation.md`; the four causes are:
  1. **`test_spinup_partition_independence.py` does not exist.** F-011 says it alone catches eight of the twelve COVERED leaves, and exactly those eight now score UNTESTED. It was lost with the v15 tree and **no work package rebuilds it** — that is a hole in this plan, not just in the suite. `params.yaml` cites it twice as an authority and names it as the writer of `results/spinup_partition_characterisation.yaml`, which is therefore never written. Rebuilding it restores eight COVERED verdicts and closes a dangling citation.
  2. `eps_F_PF` rose UNTESTED → COVERED, caught by WP2's `test_seam_contracts.py`. A real gain.
  3. `bnf_potential` and `yield_min_regional` fell UNTESTED → INERT: they move no canonical field now that WP2 solves `y_max` against the FAOSTAT target. SI [65]'s "legacy fallbacks" sentence already covers `yield_max_regional` and now covers these too.
  4. `texture_class` scores INERT, not DECLARED_NOT_WIRED. F-011's boundary between those two verdicts could not be recovered; three candidate rules were tested and rejected. Recorded as a reconstruction gap rather than resolved by picking the rule that reproduces 3 and 13.

- **2026-07-25 — surprises worth the next task's attention.**
  - **WP1's acceptance gate is red and will stay red.** `test_wp1_registry_wiring.py` fails with 50 numeric differences against the `20defb2` base, because WP2's F-002 recalibration deliberately moved those numbers. It needs a decision: rebaseline to post-WP2 values, or retire it as the one-time refactor gate it was.
  - **Two more tests are genuinely red**, and were before WP3 touched anything: `test_parameter_consistency_sol.py` (SSA cost share 0.03578 vs a hardcoded 0.037 — post-WP2 drift) and `test_parameter_extremes_sol.py` (non-finite values in `structural_cases`, undiagnosed). A red test cannot catch a mutation, so all three are excluded from CATCH by name.
  - **The committed `canonical_ERA5_y30.json` no longer reproduces** — 50 of 107 fields differ, `y_base` now pinned to the FAOSTAT target, global S3 loss 2.32 / 3.20 / 3.31 % against WP6's quoted 3.03 % at year 10. WP6's regeneration debt; flagged because the harness had to fingerprint a fresh baseline rather than the committed artifact.
  - **WP1 and WP2 finished without updating this file** — their status rows still read "not started" while the code sat committed. Marked done above from the git log. Point 5 of the discipline list is the one that keeps slipping.
  - `git` in the iCloud folder leaves `.git/index.lock` behind on almost every command and the bridge cannot delete it. Workaround: `mv` the lock into `_stale_git_locks/` immediately before committing, in the same command.

- **2026-07-25** — v15 pass completed through F-016 in a session that then became unrecoverable. Working tree lost. Container rescue attempted and failed; the session dies mid-turn before executing any tool call. No local footprint under `/sessions/`. Handoff and this state file written. Nothing rebuilt yet.
