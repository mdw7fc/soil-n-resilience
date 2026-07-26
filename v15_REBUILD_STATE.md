# v15 REBUILD — STATE

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
| WP5 | Claim register (`claims.yaml`) + claim strength | F-012, F-013, F-015, F-016 | not started |
| WP6 | Build graph + Makefile + full regeneration | F-009, F-014 | not started |
| D1 | Main-text edits | HANDOFF §7 | not started |
| D2 | SI edits + regenerated Table S1 + new limitations + benchmark section | HANDOFF §7 | not started |
| D3 | Deposit docs: MANIFEST.md, `make_figure_s5.py`, `figS12_curves.json` generator | F-009 | not started |

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

### D1
> Read `paper2-soil-resilience/v15/HANDOFF_v15_model_assurance.md` sections 5 and 7 only. Make the main-text edits listed under "Main text" in the v14_sol manuscript, tracked changes as Matthew Wallenstein. Every replacement number is in §5; do not rerun the model. Deliver a tracked and a clean docx and commit both.

### D2
> Read `paper2-soil-resilience/v15/HANDOFF_v15_model_assurance.md` sections 5 and 7 only, plus `v15/outputs/table_S1_parameters.md`. Make the SI edits listed under "SI", including the three new limitations and the benchmark-suite section. Grep F-008 in FINDINGS.md for the benchmark section's content. Tracked changes as Matthew Wallenstein. Deliver tracked and clean docx and commit both.

### D3
> Read `paper2-soil-resilience/v15/HANDOFF_v15_model_assurance.md` §7 "Deposit" and grep F-009 in `v15/FINDINGS.md`. Correct MANIFEST.md's climate-swap figure, write `make_figure_s5.py`, and either write a generator for `data/figS12_curves.json` or recompute the curves. Commit as you go.

---

## Log

- **2026-07-26 — WP4 done.** Benchmark suite rebuilt: `data/benchmarks/observed_values.yaml` (`97ccc58`), then `code/repro/run_benchmarks.py`, `data/benchmarks/baseline_verdicts.json`, `outputs/benchmarks.csv` / `.json`, `results/benchmark_reconciliation.md`, `logs/run_36_benchmarks.log` (`4733bc3`). The suite runs in about 40 seconds.

  **Result 35 rows at 9 / 3 / 1 / 14 / 7 / 1 against the acceptance 41 rows at 11 / 3 / 1 / 18 / 7 / 1.** MARGINAL, FAIL, OWED and NOT_APPLICABLE reproduce exactly, and so do the row identities behind them. **`B3-europe-YR30` fails at 0.4063 against F-008's 0.406.** Nothing was tuned. Full account in `results/benchmark_reconciliation.md`. Four things the next task should know:

  1. **The model still produces F-008's numbers.** Eight quantities were checked before any row was defined. Six reproduce exactly — the europe nil-N ratio at 30 years (0.4063), the 96-year fertilized and nil SOC drifts (-0.2098 %, -27.95 %), the fert-minus-nil SOC excess at 30 years (21.0 %), and the SSA implied rate (47.59) — and two to three decimal places. Only the year-1 europe ratio moved by more than rounding, 0.763 to 0.768, which is consistent with WP2's recalibration. So WP1 and WP2 did not break what F-008 measured.
  2. **The six missing rows are 2 PASS and 4 INFORMATIVE, and their identities are not recoverable.** F-008's prose names sixteen rows; the suite implements all sixteen plus the per-region extensions its naming convention implies plus three owed observations this pass opened. Every candidate partition reaching 41 was tested against the six-way tally and rejected, including folding in the 2022 price hindcast, which would have added failures F-008 does not record. Rows were not invented to close the count. Only a surviving `outputs/benchmarks.csv` from the crashed session closes this, and it is confirmed absent everywhere.
  3. **The research half found three things the manuscript owes.** The SSA response ratio of **0.572 could not be sourced at all** — and it has the exact signature of a log response ratio read as a linear one (ln 0.572 → 1.77, inside the published median band of 1.7–1.8). That needs a decision, not more code. The **7.7–20.0 SSA MPP envelope is not published either**, and it mixes marginal products with agronomic efficiencies. **No own-price elasticity of fertilizer demand exists for fsu_central_asia in any language**, so that row is scored against a range built entirely from other regions. Prague-Ruzyně, by contrast, verified exactly against Hlisnikovský et al. 2022 *Plants* 11:1825 Table 1, with four qualifications the SI must carry (ratios are derived not published; the table is 1 d.p. so they carry ±0.02; they are ratios of period means; the yields are winter wheat after potatoes only, 9 and 14 seasons).
  4. **`B1`'s MPP triple did not reproduce** — 24.72 / 26.10 / 112.18 against F-008's 24.8 / 25.9 / 109.2. Six finite-difference conventions were tested; the central difference is step-independent to five significant figures, so it is the model's true derivative and the deviation is not a step artifact. Recorded as a reconstruction gap.

- **2026-07-26 — three things for whoever runs WP5 or WP6.**
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
