# HANDOFF — v15 model assurance pass

> **SUPERSEDED working record (marked 2026-08-29, F-028).** This file is a
> preserved process document from the v15 rebuild. Numbers quoted in it predate
> F-025 (realized-yield market clearing) and F-026 (central eps_F_N = -0.50)
> and are NOT the released results. The released headline family is: global S3
> yield loss 2.32 / 3.02 / 3.07 % at years 1/10/30; SSA y_max 3.97. Current
> truth lives in README.md, docs/claims.yaml (v17 basis) and FINDINGS.md.


**Manuscript:** ERFS-100341, *Soil organic matter buffers fertilizer supply disruptions*
**Author:** Matthew Wallenstein
**Handoff written:** 2026-07-25, 20:45 UTC (14:45 MDT)
**Reason:** the session running the v15 hardening pass became unrecoverable. This document reconstructs its state from what reached disk, so the work can resume in a new session without re-deriving it.

---

## 0. Read this first

The v15 pass **ran to completion**. All sixteen findings, F-001 through F-016, were made, measured, fixed and recorded. At the last recorded run the gate was green: `make verify` exit 0, 14 test suites, 28 build-graph nodes all OK, and the claim register carried zero owed generators.

**The reasoning survived. The implementation did not.**

`FINDINGS.md` (73 KB, complete through F-016, timestamped 19:14 today) is on your disk and is the single most valuable artifact of the whole pass. It is a full specification: every defect, every measurement, every fix, and the name of the test that fails if the fix is reversed.

What is *not* on your disk is the working tree that FINDINGS.md describes. `code/model/params.yaml`, `code/model/registry.py`, `code/model/seams.py`, `code/model/prices.py`, `code/model/mc_mapping.py`, `code/build.py`, `Makefile`, all fifteen files under `code/tests/`, `docs/claims.yaml`, `docs/claims_baseline.json`, `docs/claim_strength_baseline.json`, every file under `results/` and `logs/`, and `outputs/benchmarks.csv` were written inside the crashed session's cloud container. Seven files were committed back to your machine. Nothing else was.

The GitHub deposit at `github.com/mdw7fc/soil-n-resilience` is at `7026193c` / tag `v1.2`. No v15 work was pushed there.

**Two recovery paths, in order of value. Try the first before starting the second.**

### 0.1 Container rescue — ATTEMPTED AND FAILED, do not retry

A single mechanical instruction (zip the tree, SendUserFile, device_commit_files, reply with nothing but the filename) was sent to the crashed session at 2026-07-25 ~20:50 UTC. It returned **"Cloud session stopped responding mid-turn."** The session dies before it can execute a tool call, so no phrasing will get through.

Also checked and empty: the crashed session left no home directory under `/sessions/` on the local Cowork VM, meaning it never used the device bridge, so there is no local footprint of its container. A filesystem-wide search of the VM for `params.yaml`, `claims.yaml`, `benchmarks.csv` and `mutation_coverage.csv` returned nothing.

**The working tree is unrecoverable. Reconstruct.**

### 0.2 Reconstruct

Base: `paper2-soil-resilience/resumbission/v14_sol/reproducibility/ERFS-100341-soil-resilience_sol/`, git commit **`20defb2`** ("Complete SOL parameter and evidentiary audit"), working tree clean. That is the state the v15 pass started from. Section 6 gives the rebuild order and what each step must reproduce.

Reconstruction is real work but it is not re-discovery. FINDINGS.md contains the measured value of every quantity the rebuilt code must produce, so each rebuilt piece has an acceptance test written before it is written.

---

## 1. What is on your disk

All under `paper2-soil-resilience/v15/`:

| file | size | what it is |
|---|---|---|
| `FINDINGS.md` | 73,585 B | The complete record, F-001 to F-016. The specification for everything below. |
| `outputs/table_S1_parameters.md` | 25,993 B | Generated SI Table S1: 54 registered parameters with value, units, category, whether the ensemble drew it, source, affected claims, benchmark. **This is a full mirror of `params.yaml`'s content** and is the best available reconstruction source for the registry. |
| `results/claim_strength.md` | 877 B | P3 / P4 / P4b ordering probabilities against the 1000-draw posterior. |
| `results/s3_shock_calibration.csv` | 786 B | Solved +103.9% shock, per-region realized reductions at years 1/5/10/30. |
| `data/soc_trajectories.csv` + `.json` | 2,146 / 7,632 B | 30-year SOC stock per region per year under S3. New deposit; the canonical run never had this. |
| `data/scenario_trajectories.csv` | 2,493 B | S3 / SC1 / SC2 / **PULSE1** global plus S3 per region. The pulse column is new. |

Nothing else from the pass reached disk.

## 2. What was lost

Referenced throughout FINDINGS.md, absent from disk and from the GitHub deposit:

**Model and infrastructure**
`code/model/params.yaml` · `code/model/registry.py` · `code/model/seams.py` · `code/model/prices.py` · `code/model/mc_mapping.py` · `code/build.py` · `Makefile` · `code/era5/generate_era5_module.py` · edits to `coupled_monthly.py` (`calibrate_ym_production`, `CALIBRATION_SCHEME`, `YM_REGION_FIELDS` 9→13, `ln_cap` column), `soil_n_model.py`, `coupled_econ_biophysical.py`, `monthly_model_v3.py` (all rewired to read the registry at import)

**Tests** (fifteen suites)
`test_spinup_partition_independence.py` · `test_calibration_fingerprint.py` · `test_era5_deposit_matches_runtime.py` · `test_seam_contracts.py` · `test_registry_consistency.py` · `test_uncertainty_completeness.py` · `test_benchmark_baseline.py` · `test_claims.py` · `test_table_s1.py` · `test_soc_trajectories.py` · `run_mutation_coverage.py` · rewritten `test_cap_market_clearing.py`

**Generators**
`run_benchmarks.py` · `make_claim_strength.py` · `make_s3_shock_calibration.py` · `make_soc_trajectories.py` · `make_table_s1.py` · `measure_era5_calendar.py` · `compute_figS8_curves.py` changes

**Registers and baselines**
`docs/claims.yaml` (19 claims, 70 checks) · `docs/claims_baseline.json` · `docs/claim_strength_baseline.json` · `data/benchmarks/observed_values.yaml` · `data/benchmarks/baseline_verdicts.json` (41 rows) · `.build/unstamped_baseline.json`

**Results and logs**
`outputs/benchmarks.csv` / `.json` · `results/mutation_coverage.csv` · `results/aggregation_basis_comparison.csv` · `results/calibration_production_path.csv` · `results/era5_calendar_discrepancy.csv` · `results/seam_contract_checks.yaml` · `results/cap_market_clearing.txt` · roughly 126 numbered run logs

**Regenerated artifacts** (F-014 re-ran the whole graph: 32 artifacts changed)
Most importantly `data/figure2_panels.json`, which was the stale artifact behind the largest block of drifted claims.

---

## 3. State at the moment of the crash

| gate | last recorded state |
|---|---|
| `make verify` | exit 0, 14 test suites (`logs/run_101_verify.log`), later `run_126_verify.log` |
| Build graph | 28 nodes, all OK; 1 orphan (`figures/Figure_S5_flux_decomposition.png`), 1 unsourced input (`data/figS12_curves.json`) |
| Claim register | 19 claims, AGREES 42 / DRIFTED 28, **owed generators 0** |
| Benchmark suite | 41 rows: 11 PASS, 3 MARGINAL, **1 FAIL**, 18 INFORMATIVE, 7 OWED, 1 N/A |
| Mutation coverage | 56 leaves: COVERED 12, UNTESTED 22, DECLARED_NOT_WIRED 3, GUARDED_AT_LOAD 6, INERT 13 |
| Monte Carlo | 1000 draws, post-recalibration; `mc_summary`, `mc_probabilities`, `mc_priors` reproduce bit-identically |

Five baselines were in force, each with the rule that it may only shrink and that an unrecorded *improvement* also stops the build: `claims_baseline.json`, `claim_strength_baseline.json`, `baseline_verdicts.json`, `.build/unstamped_baseline.json`, and the mutation pre-refactor comparison.

Drifted claim set at the end: **C-010, C-011, C-014, C-021, C-030, C-041, C-042, C-060, C-061.**

---

## 4. The sixteen findings

| ID | Finding | Manuscript consequence |
|---|---|---|
| F-001 | Spin-up never reaches steady state; the passive pool (1374-yr turnover) is inherited, not equilibrated. Temperate SOC is anchored on measured stock, tropical on model kinetics. | SI limitation: absolute SOC is initialization, not prediction. Stocks are not comparable across the temperate/tropical regimes. |
| F-002 | **The calibration was fitted on a code path nobody published.** `calibrate_ym` rooted on `run_model`; every published run goes through `century_dynamic_spinup` + `MonthlyBiophysicalEngine`. Baseline yields missed FAOSTAT by −3.87% to +4.19%. | Forced full regeneration of the MC and every figure. Ranking unchanged (FSU/Central Asia still highest year-1 loss). |
| F-003 | `k_slow` cannot reach the baseline (first-order pool at steady state passes input through), which is what licenses one `yield_max` per region outside the draw loop. | None. Defensive finding. |
| F-004 | The deposited ERA5 module disagreed with what ran: three regions' maturity months differed. Moves year-1 loss by up to 0.263 pp. | **Owed:** register the twelve planting/maturity months with provenance; SI must state the growing season is fixed and unsampled. |
| F-005 | **Three aggregation bases in one paper**, two docstrings naming the wrong one. Global S3 year-1 loss reads 2.652 / 2.156 / 2.305 % on area / nitrogen / production weights. Largest spread on a reported quantity: 1.87 pp. | **Owed:** every global number in MS and SI must state its basis. The "20% reduction" is nitrogen-weighted; the "2.30% loss" is production-weighted; they are in the same sentence. |
| F-006 | The registry documented the model rather than driving it. 45 of 56 registered leaves changed no published number when perturbed. 33 of the 45 carried an `uncertainty:` block the ensemble never read. | Superseded by F-011. |
| F-007 | **Every prior that could be compared numerically disagreed** between `params.yaml` (which generates SI Table S1) and what the ensemble drew. Six of six differed, in 13 of 18 fields. Seventeen declared uncertainties are never sampled; fourteen had no recorded reason. | The SI parameter table was wrong in six rows and misleading in fourteen. **The manuscript must stop saying fixation ramps over 8–15 years** — that mechanism does not exist in the model. No published result changes. |
| F-008 | **The first external check.** 41 benchmark rows, all held out by construction. One FAIL: B3-europe-YR30, model yield ratio 0.406 at 30 years without synthetic N against 0.681–0.776 observed at Prague-Ruzyne. Corroborated at a second site in a second quantity (B2-BROADBALK-FERT-MINUS-NIL: 38.6% vs 18.6% at 96 yr). **The temperate withdrawal response runs about twice as hard as the field record.** SSA is fine once the N rate is matched. `eps_F_N` has no published analogue. | **The SI must report the suite including the failure. The abstract must not call the magnitude validated. The model must NOT be retuned to these benchmarks.** The buffering claim itself is not contradicted. |
| F-009 | Two artifacts cited as results with no live generator; one (`data/climate_swap_comparison.csv`) held pre-recalibration numbers and disagreed on every row. `MANIFEST.md`'s 0.74 pp climate-swap figure matches neither the deleted file nor the current output. | **Owed:** correct MANIFEST.md; write a generator for `data/figS12_curves.json`; write `make_figure_s5.py`. |
| F-010 | `test_cap_market_clearing.py` asserted an identity (`0 == 0`) and could not fail. Rewritten to re-solve the market with `brentq` from outside the class. A deliberately wrong denominator now drives the residual to 3.0e-03 and fails. | None. Test integrity. |
| F-011 | Direction of authority reversed: the registry now drives the model. DECLARED_NOT_WIRED fell 45 → 3. UNTESTED rose 0 → 22, which is the honest worklist. Two INERT verdicts are probe artifacts (the fingerprint does not include margin or price outcomes). | **Owed:** mark `bnf_ramp_years` and `yield_max_regional` declared-but-fixed in SI Table S1; delete the `cre_base` fallback. |
| F-012 | **Claim register's first run: 60 checks, 19 drifted across five claims.** The largest block (C-060, every year-10 regional figure in MS [56]) traces to one stale artifact, `data/figure2_panels.json`. C-021: registry says `whc_sensitivity` = 3.5, MS [31], MS [28] and AR [40] still say 8.4. C-014/C-030: v14's 2.5–4.2 pp margin gaps become 0.27–0.99 pp under derived regional prices. | **Owed:** document edits for C-010, C-014, C-021, C-030, C-060, and SI [197] (which conflicts with MS [53]; the main text is right). |
| F-013 | **Two of three asserted regional rankings are unsupported, and one names the wrong region.** P3 (FSU highest year-1 yield loss) p = 0.998, state it. P4 p = 0.542, South Asia and East Asia not separable. P4b highest N cost share is South Asia at p = 0.958, **not** SSA. The old 83.7% SSA figure measured a hardcoded cost-share dictionary. | **Owed:** rewrite SI [163] (C-063, C-064). |
| F-014 | Full regeneration: 32 artifacts changed, 46 byte-identical. `canonical_ERA5_y30.json` did not move. `figure2_panels.json` did (global year-10 3.412 → 3.032). One new drift, C-042 (Latin America year-10 output price 1.0% stated vs 3.97% modelled). The `prices` node had been permanently STALE from a defect in its own declaration. | **Owed:** C-042 sentence edit. |
| F-015 | The +104% shock reproduces (+103.90%). "Averages approximately 20%" is stated on an unnamed basis; on the paper's own production basis it is **18.7%**, and on the nitrogen basis 19.1%. | **Owed:** state the basis, and report S3 as 19%, not 20%. The gap between calibrated and realized reduction *is* the depletion feedback; rounding it away erases the mechanism S3 exists to show. |
| F-016 | Last three owed generators closed. **SSA 30-year SOC decline is 2.145%, not 2.5%.** The one-year pulse leaves 0.009% in year 5, not ~0.3% (it crosses 0.3% between years 2 and 3), and the sentence credits food-price normalization to a period in which the price has already returned to baseline. A `>=` bug had zeroed the pulse entirely, which presented as a clean column of zeros. SOC percentage decline does not order regions the way yield loss does. | **Owed:** three sentence edits. |

---

## 5. Numbers that must survive this crash

These are the load-bearing results. If reconstruction reproduces these, it has reproduced the pass.

**Canonical S3, production-weighted, post-recalibration** (`canonical_ERA5_y30.json`, unchanged by F-014)
Global yield loss: year 1 **2.32%**, year 10 **3.03%**.
Year-10 by region: FSU/Central Asia 5.126, South Asia 4.812, sub-Saharan Africa 4.749, Southeast Asia 3.675, Europe 3.429, Latin America 2.418, North America 1.726, East Asia 1.182.

**Aggregation basis spread** (area / nitrogen / production)
Year 1: 2.652 / 2.156 / 2.305 %. Year 10: 4.006 / 3.310 / 3.412 %. Year 30: 4.289 / 3.508 / 3.636 %.
Delivered year-1 fertilizer reduction: 21.42 / 20.00 / 19.56 %.

**Shock calibration** `fert_price_shock = 1.0389792148114703` (+103.90%). S1 realizes 0.2000 on nitrogen tonnage, 0.1964 on production. S3 sustained mean (years 1–10): 0.1911 nitrogen, **0.1872 production**. Per-region S3 mean spans 0.126 (North America) to 0.292 (SSA).

**SOC, year 0 → year 30, t C/ha (10-yr / 30-yr % decline)**
NA 50.69→50.39 (0.30/0.61) · EU 42.60→42.12 (0.58/1.12) · EA 35.47→35.32 (0.22/0.44) · SA 17.37→17.01 (1.14/2.09) · SEA 22.24→21.87 (0.88/1.65) · LATAM 31.26→30.91 (0.58/1.13) · **SSA 6.18→6.04 (1.09/2.14)** · FSU 35.31→34.72 (0.85/1.67).

**One-year pulse, global yield loss by year (%)**
1: 2.316 · 2: 0.492 · 3: 0.044 · 4: 0.015 · 5: **0.009** · 6: 0.007 · 7: 0.006 · 8: 0.005 · 9: 0.005 · 10: 0.004.

**Claim strength** (n = 1000) P3 fsu_central_asia 0.998 · P4 south_asia 0.542 (east_asia 0.447; pair 0.989) · P4b south_asia 0.958 (europe 0.040).

**The benchmark failure** B3-europe-YR30: model nil-N yield ratio 0.763 / **0.406** / 0.364 at 1 / 30 / 96 years, against observed 0.776 (1961–1981) and 0.681 (1983–2020) at Prague-Ruzyne, unfertilized since 1954 against NPK4 at 95 kg N/ha.

---

## 6. Rebuild order

The order matters for the same reason it did in the original plan: the registry is what makes everything after it enforceable. Each step below carries the FINDINGS entry that specifies it and the number the rebuilt piece must produce.

1. **Registry.** `params.yaml` + `registry.py`. Reconstruct the entries from `outputs/table_S1_parameters.md`, which mirrors all 54 parameters with units, category, source, `affects_claims` and benchmark links. Add the six corrected `uncertainty:` blocks from the F-007 table verbatim, the two `superseded_*` blocks on `bnf_potential` / `bnf_ramp_years`, and `som_pool_fractions.mc_exempt_reason` from F-001. Spec: F-001, F-006, F-007, F-011.
2. **Wire the registry into the model** (`soil_n_model.py`, `coupled_econ_biophysical.py`, `prices.py`, `monthly_model_v3.py`). Acceptance: a 123-field canonical diff returns zero numeric differences. Spec: F-011.
3. **Production-path calibration.** `calibrate_ym_production`, `CALIBRATION_SCHEME = 'production_path_v2'`, `YM_REGION_FIELDS` at 13 fields. Acceptance: FAOSTAT targets to 8e-3 percent worst case; `yield_max` moves −3.36% to +3.78% from the legacy path. Spec: F-002.
4. **Seam contracts.** `seams.py` with `outcome_weights()` / `intensity_weights()` / `assert_same_basis()`. Acceptance: `calibrate_price_shock(0.20)` returns `1.0389792148114703` unchanged. Spec: F-005.
5. **Rewrite `test_cap_market_clearing.py`** to re-solve with `brentq`. Acceptance: worst structural residual 1.4e-17; the dropped-gamma mutation drives it to 3.0e-03. Spec: F-010.
6. **Mutation harness.** Acceptance: 56 leaves scoring 12 / 22 / 3 / 6 / 13. Spec: F-011.
7. **Benchmark suite.** `run_benchmarks.py` + `observed_values.yaml` + `baseline_verdicts.json`. Acceptance: 41 rows, 11/3/1/18/7/1, with B3-europe-YR30 failing at 0.406. Spec: F-008. **This is the most expensive step to rebuild, because the observed-value compilation is the part that was research rather than code.**
8. **Claim register.** `claims.yaml`, 19 claims, 70 checks, two-way index against `affects_claims`. Acceptance: AGREES 42, DRIFTED 28, owed generators 0. Spec: F-012, F-015, F-016.
9. **Build graph.** `build.py` + `Makefile`, `params_fingerprint()` hashing with `DOCUMENTARY_KEYS` removed. Acceptance: 28 nodes OK, one orphan, one unsourced input. Spec: F-009, F-012, F-014.
10. **Claim strength.** `make_claim_strength.py` against the 1000-draw posterior. Acceptance: the three probabilities in `results/claim_strength.md`, which you already have. Spec: F-013.
11. **Regenerate.** Then the document edits in §7.

### What not to do

- **Do not retune the model to the benchmarks.** Fitting to them converts the only external check this project has into calibration data, and reproduces F-002 at a larger scale.
- **Do not edit the ensemble to match `params.yaml`.** The ensemble is what ran; `params.yaml` is what was wrong. Editing the ensemble makes the published interval unreproducible from the code that produced it.
- **Do not adopt the deposited ERA5 crop calendar.** There is no evidence it is the better calendar, and adopting it changes published numbers on no authority.
- **Do not widen a tolerance to make a claim agree.** Tolerances came from each sentence's own stated precision.

---

## 7. Consolidated manuscript worklist

This is the part that survives independently of the code and can be started now.

**Abstract**
- Margin gap 2.5–4.2 pp → **0.27–0.99 pp** (state as ~0.3–1.0). C-014.
- Do not describe the loss magnitude as validated. F-008.

**Main text**
- MS [56]: every year-10 regional figure is stale. East Asia 1.3→1.182, South Asia 6.0→4.812, FSU 5.5→5.126, SSA 5.4→4.749, global 3.4→3.032. C-060.
- MS [31], MS [28]: `whc_sensitivity` 8.4 → **3.5** mm per pp SOC. C-021. (Also in the author response, AR [40].)
- MS [56] / MS [65]: state the aggregation basis. Report the S3 reduction as **19%**, not 20%. F-015, C-050.
- MS [78]: fixation does not ramp. Remove or correct. F-007, F-011.
- Regional output-price indices: year-1 5.5→5.34, year-10 5.0→5.95, Latin America year-10 1.0→**3.97**, FSU 10.3→10.16. The Latin American change alters the claim about spread width, not only the number. C-042.
- One-year pulse: year-5 residual 0.3% → **0.009%**, and drop the food-price half of the attribution. C-061.

**SI**
- SI [163]: rewrite. South Asia and East Asia are not separable on year-1 net revenue (p = 0.542). The highest nitrogen cost share is **South Asia** (p = 0.958), not sub-Saharan Africa. C-063, C-064.
- SI [197]: 0.1–1.5 pp → **0.2**–1.5, to agree with MS [53], which the model supports. C-031.
- SSA 30-year SOC decline 2.5% → **2.145%**. C-010.
- Regenerate Table S1 from the registry; mark the 17 declared-but-fixed rows, including `bnf_ramp_years` and `yield_max_regional`. F-007, F-011.
- New limitation: absolute SOC is initialization, not prediction; temperate and tropical stocks are not comparable. F-001.
- New limitation: the crop calendar is fixed and unsampled, and moves the reported loss by up to 0.26 pp per month. F-004.
- New section: the benchmark suite, **including the B3-europe failure**. F-008.
- State that percentage SOC decline does not order regions the way yield loss does. F-016.
- State the aggregation basis for every global number. F-005.

**Deposit**
- `MANIFEST.md`: the 0.74 pp climate-swap figure is from a third generation of the analysis and matches nothing current (0.58 pp). F-009.
- Write `make_figure_s5.py`; give `data/figS12_curves.json` a generator or recompute the curves. F-009.

---

## 8. Decisions waiting on you

1. **B3-europe.** The model loses 59% of temperate yield without synthetic nitrogen where Prague-Ruzyne lost 22–32%. F-008 says state it and do not tune to it. The alternative is to compile Broadbalk plot 3 grain yields as a second temperate comparator before deciding (B3-OWED-BROADBALK-YIELD-RATIO). Reporting a failure against one site is defensible; a reviewer may still ask why the mechanism runs twice as hard.
2. **How much of §7 to do before the code is rebuilt.** The document edits are independently verifiable from §5 and do not need the tree back.
3. **Whether to keep going in a single session.** The pass that crashed did sixteen findings in one context. The rebuild should be split, with a commit to your disk after every step in §6.

---

## 9. Provenance

Built from: `paper2-soil-resilience/v15/FINDINGS.md` (read in full, 1,241 lines), the six surviving v15 artifacts, the v14_sol deposit tree and its git history, and project memory `project_erfs_revision.md`. Every number in §5 is quoted from FINDINGS.md or from a surviving CSV. Nothing here is inferred.

Confirmed absent: no `params.yaml`, `claims.yaml`, `build.py`, `benchmarks.csv`, `mutation_coverage.csv`, `Makefile`, `registry.py`, `seams.py` or `mc_mapping.py` anywhere under the project folder; no v15 commit on the GitHub deposit (`7026193c`, tag `v1.2`).
