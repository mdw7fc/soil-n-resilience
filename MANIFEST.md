# MANIFEST — ERFS-100341 canonical reproducibility deposit

Generated for the v14 (canonical) manuscript, deposit v1.4. All results derive
from the ERA5 climate run to year 30 (`code/repro/run_canonical.py`). Entries
marked **new in v1.3** were added with the model corrections; entries marked
**new in v1.4** were added when Supplementary Figure S7 was regenerated from the
corrected model. See `CHANGELOG.md`.

## Root

- `README.md` — overview, requirements, reproduce steps, expected results
- `MANIFEST.md` — this inventory
- `CHANGELOG.md` — correction history, including the two model corrections and the
  v1.4 Figure S7 regeneration
- `CITATION.cff` — citation metadata (GitHub/Zenodo)
- `.zenodo.json` — Zenodo deposit metadata
- `LICENSE` — MIT (code) + CC-BY-4.0 (data/figures)
- `requirements.txt` — Python dependencies

## Model code (`code/model/`)

- `coupled_monthly.py` — coupled monthly biophysical-economic model (main engine)
- `coupled_econ_biophysical.py` — economic module; scenario + regional econ params; SC1/SC2
- `monthly_model_v3.py` — monthly soil-N dynamics; Century spin-up; regional climate profiles
- `soil_n_model.py` — region definitions, SOM pools, default parameters
- `scripts/validate_f_fert_broadbalk.py` — temperate crop-response validation (Broadbalk)

## Climate input pipeline (`code/era5/`)

- `fetch_era5_climate.py` — retrieval of ERA5 monthly normals (Open-Meteo archive)
- `REGIONAL_CLIMATES_era5.py` — ERA5-derived regional climate profiles

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
- `make_ofra_validation.py` — Figure S13 (OFRA SSA maize-N validation overlay, canonical y_max)
- `run_price_shock_analysis.py` — **new in v1.3** — farm-level SOC-gradient computation for main
  Figures 1 and 2a; writes `figure1_farm_gradient.json` and `figure2_soc_gradient.json` (~90 s)
- `make_figure_1.py` — renders Figure 1 (farm yield and partial net-revenue buffering)
- `make_figure_2.py` — **new in v1.3** — renders Figure 2 (SOC gradient, direct/SOM decomposition, bubble panel)
- `test_zero_shock_invariance.py` — **new in v1.3** — asserts that a zero shock leaves every
  region at its baseline yield; writes `outputs/zero_shock_invariance.csv`
- `test_cap_market_clearing.py` — **new in v1.3** — asserts that the constrained equilibrium
  clears when the physical fertilizer ceiling binds

## Data (`data/`)

- `canonical_ERA5_y30.csv` — FROZEN canonical output: per-region descriptors + yr 1/10/30 losses
- `canonical_ERA5_y30.json` — same, JSON, incl. production-weighted global losses
- `era5_regional_climates.json` — ERA5 monthly T/precip/PET per region (model input)
- `era5_raw/` — raw ERA5 pulls for the eight regions (provenance for the above)
- `climate_swap_comparison.csv` — expert vs ERA5 yr1/yr10 losses (max year-10 shift 0.74 pp)
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
- `ofra_maize_N_responsefunctions.csv` — OFRA maize N-response functions (tropical validation)
- `crop_response_calibration_table.csv` — regional crop-response calibration table (Table S4 source)

Note: `crop_response_calibration_table.csv` reports the simulation-calibrated
`y_max` and the **simulated** no-synthetic-N yields (`y_no_synth_sim_tha`),
consistent with the numerical year-2 calibration. **It has no generator in this
deposit** — the script that produced the `y_no_synth_sim_tha` column was not
deposited and was not recovered in this revision. The `ymax_calibrated_tha`
column matches the corrected canonical run to six figures, and an equivalent
no-synthetic-N simulation run against the pre-fix and the corrected model differs
by ≤ 0.05 t ha⁻¹ in every region, so the table is unaffected by the v1.3
corrections and is carried over unchanged.

## Figures (`figures/`)

- `Figure_1_farm_buffering.png` / `.pdf` — farm yield and partial net-revenue buffering
- `Figure_2_regional_vulnerability.png` / `.pdf` — **new in v1.3** — regional vulnerability, three panels
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
- `Figure_S13_OFRA_SSA_validation.png` — OFRA validation overlay (SSA y_max = 3.88)

## Outputs (`outputs/`)

Regenerated by the scripts:

- `global_S3_losses.txt` — production-weighted global S3 loss, yr 1/10/30
- `table_S3_correlations.csv` — Supplementary table 3
- `climate_swap_comparison.csv` — expert vs ERA5 robustness
- `zero_shock_invariance.csv` — **new in v1.3** — per-region zero-shock yield ratios
