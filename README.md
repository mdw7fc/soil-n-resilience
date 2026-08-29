# Soil organic matter buffers fertilizer supply disruptions

Model code and **canonical reproducibility deposit** for:

> Wallenstein, M. D. & Manning, D. T. *Soil organic matter buffers fertilizer
> supply disruptions.* Environmental Research: Food Systems (ERFS-100341).

Zenodo: https://doi.org/10.5281/zenodo.19699772

This deposit corresponds to the v17 revision. Central regional trajectories
derive from the ERA5 run produced by `code/repro/run_canonical.py`; every
generated artifact is declared in the build graph (`code/build.py`), farm
gradients, sensitivities, benchmarks and the country mechanism screen all have
named generators, and the parameter ledger records every live parameter.

**v17 is the release of the rebuilt model line** (FINDINGS.md F-018 through
F-028). Its defining changes: realized-yield market clearing (the food price
is root-found at every timestep against the monthly biophysical model's own
production response; F-025), the central soil-N fertilizer-demand feedback
restored to eps_F_N = -0.50 (F-026), financial results reported as crop
revenue net of nitrogen-fertilizer expenditure for the four regions with
audited price pairs, and the four-pool quantitative engine comparison
withdrawn pending a regenerable engine. The claim register that gates every
quoted number is `docs/claims.yaml` (19 claims, 70 checks, document basis
v17), run by `make verify`; the pre-rebuild register is archived under
`docs/archive/`.

## What the model is

A coupled monthly biophysical–economic framework for eight global agricultural
regions. The biophysical layer tracks soil nitrogen (SOM mineralization,
immobilization, leaching, denitrification), Century/RothC-style SOM pools, and
a Mitscherlich crop-N yield response.
The economic layer solves regional fertilizer, food, and land markets under a
fertilizer-price or physical-supply shock. See the manuscript Methods and SI.

## Requirements

Python ≥ 3.9 with:

```
pip install -r requirements.txt      # numpy, scipy, pandas, matplotlib
```

No network access is needed to reproduce results: ERA5 normals, benchmark data,
and the raw and processed inputs for the spatial screen are included.
Provenance/download scripts remain in the package.

## Reproduce

```
make verify        # the gate: 18 analytical test suites + the 32-node build graph
make everything    # regenerate every artifact including the Monte Carlo ensemble (~1 h)
make all           # the same, minus the ensemble
make graph         # the full topological order and per-node cost
```

Every generator, its inputs and its outputs are declared in `code/build.py`;
`python3 code/build.py status` reports per-node provenance (OK / STALE /
UNSTAMPED) and runs the orphan and unsourced-input scans. Node outputs are
canonicalized at write time (float literals at six significant digits), so a
regeneration on a different machine reproduces the committed bytes rather
than tripping the staleness gate on last-ulp libm noise (F-028).

The 18 suites include internal-consistency and robustness checks
(zero-shock invariance, dimensional consistency, market-clearing structure,
parameter boundaries and extremes, MC robustness), the claim register, and
document gates (cross-document consistency against the v17 files in
`resumbission/v17/`, repo-docs consistency). `test_parameter_extremes_sol.py`
records the declared conditional behaviors under the eps_F_N sensitivity
family (0, -0.25, -0.50 central, -1.0): year-1 buffering is positive
everywhere at every bound; year-10 low-SOC gradients are non-monotone in
Latin America, Sub-Saharan Africa and FSU/Central Asia under the central
-0.50 (a disclosed, conditional result; F-026), and the year-10 SSA gradient
reverses at -1.0. Disclosed decisions, not hidden failures.

## Expected canonical results (what the scripts print / write)

| Quantity | Value | Appears in |
|---|---|---|
| Global S3 yield loss, production-weighted (yr 1 / 10 / 30) | **2.32 / 3.02 / 3.07 %** | Abstract, Results |
| Regional year-10 loss, range | **1.18 % (EA) - 5.09 % (FSU)**; SA 4.80, SSA 4.74 | Results, Figure 2 |
| SSA year-30 loss | **5.08 %** | Results |
| SC1 (permanent supply loss) global year-10 | **3.70 %** (0.68 pp above S3) | Results |
| SC2 (20-yr recovery) global year-10 | **1.88 %** (<= 0.11 % in every region by year 30) | Results |
| Soil-N buffer ratio (%), NA->FSU | **42.9, 33.3, 19.6, 32.2, 49.6, 52.9, 45.8, 45.0** | Supplementary table 1 |
| Buffer ratio vs year-10 penalty | Spearman rho = **+0.19**, Pearson R^2 = **0.10** | Note 3, Figure S6f, table 3 |
| Fig S6 panel rho (a-f) | -0.61, +0.70, -0.90, -0.28, +0.58 (yr-1), +0.19 | Figure S6 |
| Climate robustness (expert vs ERA5) | max year-10 shift **0.62 pp**; Spearman rho = **0.98** | Response letter, `results/climate_swap_stats.txt` |
| Figure 1 net revenue at regional mean SOC | SSA **+0.0 %**, SA -6.8, LATAM -1.5, NA -1.0 | Figure 1b |
| Figure S7 halved-elasticity net-revenue penalty | **1.7-5.0 pp** deeper at regional mean SOC (largest South Asia); SSA yield improves **2.3 pp** from 10 % to 200 % SOC; 50-vs-100 % SOC net-revenue gap stays positive in all four regions | Figure S7 |
| MC ensemble median year-1 loss (n = 1,000) | **2.51 %**; 5-95 % range 3.3 pp; buffer 0.88 ppt; buffering P = 1.0 in every region; SSA worst-net-revenue in 0 of 1,000 draws | Note 6, Figure S9 |
| Figure S11 SOC spread (25 % vs 100 %) | **0.2-1.4 pp** at 100-150 % shock; 0.4-2.2 pp at 300 % | Figure S11 |
| Realized S3 fertilizer reduction (N-tonnage-weighted, sustained mean) | **18.7 %** (~19 %; +104 % calibrated shock) | Results, `results/s3_shock_calibration.csv` |
| N-capture sensitivity, global year-10 loss, capture 0.45 -> 0.95 | **3.87 % -> 2.77 %**; SSA 0.45 -> 0.65 is **5.28 % -> 4.88 % (~8 %)** | Note 5, Figure S10 |
| Regional output-price index (production-weighted, yr 1/10/30) | **+5.20 / +6.72 / +6.82 %**; regional yr-10 span 2.64 (EA) - 11.89 (FSU) | SI |
| Zero-shock invariance | PASS (all regions yr-10 yield fraction >= 0.99999; max 30-yr deviation 3.6e-05) | Note on model consistency |
| Market clearing (realized-yield) | PASS (per-step structural residuals <= 1e-8 from reported columns; the pre-F-025 linear supply relation must disagree, so the test cannot pass both clearings) | `code/repro/test_cap_market_clearing.py` |

Yield losses are reported as fractions of each region's baseline. They still vary slightly with the calibrated ceiling `y_max` (through residue return, SOM feedbacks, and yield constraints); the values above are the losses the canonical run produces with the `y_max` values it actually uses, documented here — not a claim of exact `y_max`-independence.

## Calibrated ceiling (y_max)

`data/canonical_ERA5_y30.*` reports `y_max` as the model's calibrated
Mitscherlich ceiling from `get_calibrated_ym()` (region-by-region, so that at
current N the modeled yield equals the FAOSTAT target), evaluated under the ERA5
climate (production-path calibration, F-002/D3): NA 6.20, EU 6.02, EA 6.22,
SA 3.77, SEA 4.87, LATAM 5.41, SSA 3.97, FSU 4.32 t ha⁻¹. This is the ceiling used by the coupled model and plotted in
Figure S6c.

## Directory layout

```
code/
  model/       coupled_monthly.py, coupled_econ_biophysical.py,
               monthly_model_v3.py, soil_n_model.py
  model/scripts/ validate_f_fert_broadbalk.py  (temperate parameter benchmark)
  era5/        fetch_era5_climate.py (provenance script)
  repro/       run_canonical.py, make_table_s3.py, make_figure_s6.py,
               climate_comparison.py, make_sc_trajectories.py,
               make_scenario_trajectories.py, run_price_shock_analysis.py,
               make_figure_1.py, make_figure_2.py, compute_figS8_curves.py,
               make_figure_s7.py, make_figure_s8.py, make_figure_s12.py,
               make_ofra_validation.py,
               run_mc_ensemble.py, make_figure_s9.py, make_figure_s10.py,
               make_figure_s11.py, make_food_price_table.py,
               make_broadbalk_benchmark.py, make_hindcast_benchmark.py,
               test_zero_shock_invariance.py, test_cap_market_clearing.py
data/          canonical_ERA5_y30.csv/.json, era5_regional_climates.json,
               era5_raw/ (8 regions), climate_swap_comparison.csv,
               SC1_/SC2_regional_trajectory.csv, scenario_trajectories.csv,
               figure1_farm_gradient.json, figure1_soc_gradient.csv,
               figure2_soc_gradient.json, figure2_panels.json,
               figS7_farm_elasticity_gradient.json, figS8_curves.json,
               figS10_nue_sensitivity.json,
               figS11_severity_sweep.json, figS12_curves.json,
               food_price_response.csv,
               mc_ensemble/ (posterior, summary, probabilities, priors),
               ofra_maize_N_responsefunctions.csv, crop_response_calibration_table.csv,
               benchmarks/ (Broadbalk and 2022-crisis extractions)
figures/       Figure_1_farm_buffering.png/.pdf,
               Figure_2_regional_vulnerability.png/.pdf,
               Figure_S6_pairwise_diagnostics.png,
               Figure_S7_farm_elasticity_gradient.png/.pdf,
               Figure_S8_elasticity_sensitivity.png,
               Figure_S9_mc_ensemble.png/.pdf,
               Figure_S10_nue_sensitivity.png/.pdf,
               Figure_S11_severity_gradient.png/.pdf,
               Figure_S12_crop_response_calibration.png,
               Figure_S13_OFRA_SSA_validation.png
outputs/       regenerated tables (written by the scripts)
spatial_screen/ complete Figure 3 source/processed data, scripts, audit and figure
```

**Scope decision.** The earlier microbially explicit 4-pool comparison and
Figure S5 are excluded from the revised evidentiary chain because their code
was not deposited and their quantitative ratios could not be regenerated after
the final corrections. Figure S5 was then withdrawn from the paper altogether,
so no generator is owed for it and none is deposited; the withdrawn figure is
kept under `excluded_legacy_sol/`. Table S4, its Figure S12 curves and
`data/crop_response_calibration_table.csv` are all generated by
`code/repro/make_table_s4_sol.py`.

See `MANIFEST.md` for a per-file description and `CHANGELOG.md`
(in the resubmission package) for the full correction history.

## License

Code: MIT (`LICENSE`). Data and figures: CC-BY-4.0.
