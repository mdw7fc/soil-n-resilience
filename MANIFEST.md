# MANIFEST — ERFS-100341 canonical reproducibility deposit

Generated for the v14 SOL manuscript, deposit v1.5. Central model results derive
from the ERA5 climate run to year 30 (`code/repro/run_canonical.py`). Entries
for benchmarks and Figure 3 use the explicitly identified deposited datasets.
See `CHANGELOG.md`.

## Root

- `README.md` — overview, requirements, reproduce steps, expected results
- `MANIFEST.md` — this inventory
- `CHANGELOG.md` — correction history, including the two model corrections and the
  v1.4 Figure S7 regeneration
- `CITATION.cff` — citation metadata (GitHub/Zenodo)
- `.zenodo.json` — Zenodo deposit metadata
- `LICENSE` — MIT (code) + CC-BY-4.0 (data/figures)
- `requirements.txt` — Python dependencies
- `EVIDENTIARY_STANDARD_sol.md` — prospective acceptance rules
- `CLAIM_REGISTER_sol.csv` / `.md` — result-by-result retain/qualify/exclude decisions
- `PARAMETER_LEDGER_sol.csv` / `.md` — 577 semantic live-parameter entries
- `NUMERIC_LITERAL_AUDIT_sol.csv` — 2,087 audited source-line numeric entries

## Model code (`code/model/`)

- `coupled_monthly.py` — coupled monthly biophysical-economic model (main engine)
- `coupled_econ_biophysical.py` — economic module; scenario + regional econ params; SC1/SC2
- `monthly_model_v3.py` — monthly soil-N dynamics; Century spin-up; regional climate profiles
- `soil_n_model.py` — region definitions, SOM pools, default parameters
- `parameter_registry.py` — authoritative shared conversions, BNF primitives/derivation,
  regional prices and sensitivity bounds
- `scripts/validate_f_fert_broadbalk.py` — Broadbalk parameter benchmark
- `data/benchmark_broadbalk/` — observed and modeled benchmark trajectories

## Climate input pipeline (`code/era5/`)

- `fetch_era5_climate.py` — retrieval of ERA5 monthly normals (Open-Meteo archive)
- `data/era5_regional_climates.json` is the single executable ERA5 climate
  input; no second Python copy is retained.

## Reproduction scripts (`code/repro/`)

- `run_canonical.py` — CANONICAL RUN: ERA5 → year 30; writes `canonical_ERA5_y30.*`
- `make_table_s3.py` — Supplementary table 3 (pairwise Spearman correlations)
- `make_figure_s6.py` — Figure S6 (pairwise regional diagnostics)
- `climate_comparison.py` — expert vs ERA5 climate robustness (deposit diagnostic; there is
  no corresponding SI note or figure, the comparison is reported in the response letter)
- `make_sc_trajectories.py` — SC1/SC2 regional yield-loss trajectories
- `make_scenario_trajectories.py` — `data/scenario_trajectories.csv` (S3/SC1/SC2 global + S3 regional)
- `compute_figS8_curves.py` — **new in v1.3** — precomputes the Figure S8 elasticity curves
- `make_figure_s7.py` — **new in v1.4** — Figure S7 (farm SOC gradient under baseline vs
  halved fertilizer-demand elasticities); writes `figS7_farm_elasticity_gradient.json`.
  Figure S7 previously had no generator in the deposit (~2 min)
- `make_figure_s8.py` — Figure S8 (elasticity sensitivity: baseline vs halved eps_F_PF)
- `make_figure_s12.py` — Figure S12 (simulated regional N-response calibration curves)
- `run_mc_ensemble.py` — **new in v1.3** — 1,000-draw joint-prior Monte Carlo ensemble
  (Supplementary Note 6 / Figure S9); ported in this revision and now run on the ERA5 climate
- `make_figure_s9.py` — **new in v1.3** — Figure S9 (Monte Carlo ensemble box panels)
- `make_figure_s10.py` — **new in v1.3** — Figure S10 (in-season N capture efficiency as a
  buffering lever); ported in this revision and now run on the ERA5 climate
- `make_figure_s11.py` — **new in v1.3** — Figure S11 (SOC gradient vs price-shock severity)
- `make_food_price_table.py` — **new in v1.3** — regional and production-weighted global
  output-price response under S3; the SI food-price figures previously had no generator
- `make_ofra_validation.py` — Figure S13 (OFRA SSA maize-N benchmark overlay, canonical y_max)
- `make_broadbalk_benchmark.py` — Figure S2 benchmark and reported bias
- `make_hindcast_benchmark.py` — Figure S4 using official 2021–2022 FAOSTAT changes
- `run_price_shock_analysis.py` — **new in v1.3** — farm-level SOC-gradient computation for main
  Figures 1 and 2a; writes `figure1_farm_gradient.json` and `figure2_soc_gradient.json` (~90 s)
- `make_figure_1.py` — renders Figure 1 (farm yield and partial net-revenue buffering)
- `make_figure_2.py` — **new in v1.3** — renders Figure 2 (SOC gradient, direct/SOM decomposition, bubble panel)
- `test_zero_shock_invariance.py` — **new in v1.3** — asserts that a zero shock leaves every
  region at its baseline yield; writes `outputs/zero_shock_invariance.csv`
- `test_cap_market_clearing.py` — **new in v1.3** — asserts that the constrained equilibrium
  clears when the physical fertilizer ceiling binds
- `test_full_zero_shock_sol.py` — full S3-channel zero-shock baseline invariance
- `test_parameter_consistency_sol.py` — unique price/share/BNF definitions
- `test_dimensional_consistency_sol.py` — unit identities and dimensional conversions
- `test_parameter_boundaries_sol.py` — calibration, state and domain boundaries
- `test_mc_robustness_sol.py` — pre-specified 95% joint-prior direction gate
- `test_parameter_extremes_sol.py` — one-at-a-time prior bounds and WHC × εF,N grid
- `test_cross_document_consistency_sol.py` — headline values, Table S4 and embedded-figure check

## Data (`data/`)

- `canonical_ERA5_y30.csv` — FROZEN canonical output: per-region descriptors + yr 1/10/30 losses
- `canonical_ERA5_y30.json` — same, JSON, incl. production-weighted global losses
- `era5_regional_climates.json` — ERA5 monthly T/precip/PET per region (model input)
- `era5_raw/` — raw ERA5 pulls for the eight regions (provenance for the above)
- `climate_swap_comparison.csv` — expert vs ERA5 yr1/yr10 losses (max year-10 shift 0.69 pp)
- `SC1_regional_trajectory.csv` — SC1 (permanent 20% supply loss) yield loss %, yr 0–30, per region
- `SC2_regional_trajectory.csv` — SC2 (20% loss, 20-yr recovery) yield loss %, yr 0–30, per region
- `scenario_trajectories.csv` — year 0–30 global loss for S3, SC1, SC2 + per-region S3
- `figure1_farm_gradient.json` — **new in v1.3** — Figure 1 farm gradient, 10–200% SOC in 5% steps
- `figure1_soc_gradient.csv` — **new in v1.3** — same, flat CSV (yield, margin, fertilizer change)
- `figure2_soc_gradient.json` — **new in v1.3** — Figure 2a year-10 gradient, 10–200% SOC in 10% steps
- `figure2_panels.json` — **new in v1.3** — Figure 2 panel a/b/c plotted values
- `figS7_farm_elasticity_gradient.json` — **new in v1.4** — Figure S7 farm gradient,
  10–200% SOC in 5% steps, baseline and halved eps_F_PF, for four cost-structure regions
- `figS8_curves.json` — precomputed baseline/halved trajectories for Figure S8
- `figS10_nue_sensitivity.json` — **new in v1.3** — Figure S10 NUE sweep, 6 NUE levels x yr 0-10
- `figS11_severity_sweep.json` — **new in v1.3** — Figure S11 sweep, 0-300% shock x 4 SOC levels
- `food_price_response.csv` — **new in v1.3** — regional output-price index at yr 1/10/30 plus
  the production-weighted global aggregate (+5.45 / +5.01 / +5.26 %)
- `mc_ensemble/` — **new in v1.3** — Monte Carlo outputs: `mc_posterior.csv.gz`,
  `mc_summary.csv`, `mc_probabilities.csv`, `mc_summary.txt`, `mc_priors.json`, `ym_cache.json`
- `figS12_curves.json` — precomputed simulated N-response curves for Figure S12
- `ofra_maize_N_responsefunctions.csv` — OFRA maize N-response functions (tropical benchmark)
- `crop_response_calibration_table.csv` — regional crop-response calibration table (Table S4 source)
- `benchmarks/` — official 2022 hindcast observations, Broadbalk extraction and caveat table

`crop_response_calibration_table.csv` and
`outputs/Table_S4_calibration_sol.csv` are regenerated by
`code/repro/make_table_s4_sol.py`, including the live simulation-calibrated
`y_max` and simulated no-synthetic-N yield.

## Figures (`figures/`)

- `Figure_1_farm_buffering.png` / `.pdf` — farm yield and partial net-revenue buffering
- `Figure_2_regional_vulnerability.png` / `.pdf` — **new in v1.3** — regional vulnerability, three panels
- `Figure_3_mechanism_screen.png` / `.pdf` — country mechanism screen; complete pipeline below
- `Figure_S2_broadbalk_benchmark.png` / `.pdf` — contextual Broadbalk benchmark
- `Figure_S4_hindcast_sensitivity.png` / `.pdf` — directional 2022 demand benchmark
- `excluded_legacy_sol/Figure_S5_flux_decomposition_legacy_sol.png` — legacy
  exploratory figure; excluded from the revised evidentiary chain
  in the code. Illustrative only; no manuscript or SI number derives from it. This gap
  predates v1.3.
- `Figure_S6_pairwise_diagnostics.png` — pairwise regional diagnostics (canonical ERA5)
- `Figure_S7_farm_elasticity_gradient.png` / `.pdf` — **new in v1.4** — farm SOC gradient,
  baseline vs halved fertilizer-demand elasticities; regenerated from the corrected model
- `Figure_S9_mc_ensemble.png` / `.pdf` — **new in v1.3** — Monte Carlo ensemble (n = 1,000)
- `Figure_S10_nue_sensitivity.png` / `.pdf` — **new in v1.3** — N capture efficiency lever
- `Figure_S11_severity_gradient.png` / `.pdf` — **new in v1.3** — SOC gradient vs shock severity
- `Figure_S8_elasticity_sensitivity.png` — elasticity sensitivity (canonical ERA5)
- `Figure_S12_crop_response_calibration.png` — simulated crop-response calibration (canonical y_max)
- `Figure_S13_OFRA_SSA_validation.png` — OFRA contextual benchmark overlay (SSA y_max = 3.88)

## Outputs (`outputs/`)

Regenerated by the scripts:

- `global_S3_losses.txt` — production-weighted global S3 loss, yr 1/10/30
- `table_S3_correlations.csv` — Supplementary table 3
- `climate_swap_comparison.csv` — expert vs ERA5 robustness
- `zero_shock_invariance.csv` — **new in v1.3** — per-region zero-shock yield ratios
- `Table_S4_calibration_sol.csv` — regenerated regional calibration table
- `parameter_extreme_acceptance_sol.csv` — all one-at-a-time and structural-grid decisions
- `structural_sensitivity_sol.csv` — global WHC × εF,N results
- `price_convention_sensitivity_sol.csv` — four-region price convention check

## Country mechanism screen (`spatial_screen/`)

- `scripts/` — download/provenance, aggregation, threshold sensitivity, audit and render code
- `data_raw/` — deposited SoilGrids, MIRCA2000, FAOSTAT and Natural Earth inputs
- `data_processed/` — country buffer/exposure classifications, threshold treatments and QA
- `docs/SPEC.md` — pre-specified screen definitions and scope
- `scripts/16_phase2_final_audit.py` — final country and threshold audit
- `scripts/19_fig4_mechanism_screen_v10.py` — final Figure 3 render
