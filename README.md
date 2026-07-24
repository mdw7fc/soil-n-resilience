# Soil organic matter buffers fertilizer supply disruptions

Model code and **canonical reproducibility deposit** for:

> Wallenstein, M. D. & Manning, D. T. *Soil organic matter buffers fertilizer
> supply disruptions.* Environmental Research: Food Systems (ERFS-100341).

Zenodo: https://doi.org/10.5281/zenodo.19699772

This deposit corresponds to the **v14 (canonical) manuscript and SI**. Every
figure and table in the paper derives from a **single frozen model output** —
the ERA5 data-based climate run extended to year 30 — produced by
`code/repro/run_canonical.py`. Earlier deposits mixed an expert-climate run
(which stopped at year 10) with the ERA5 run; this version resolves that so all
artifacts trace to one output set.

**v1.4 supersedes v1.2.** Two internal-consistency corrections were applied to
the model, and the generators for main Figures 1 and 2 and for Supplementary
Figure S7 are now included in the deposit. All reported magnitudes are lower
than in v1.2; see `CHANGELOG.md` for the full list and the before/after values.

## What the model is

A coupled monthly biophysical–economic framework for eight global agricultural
regions. The biophysical layer tracks soil nitrogen (SOM mineralization,
immobilization, leaching, denitrification), SOM pools (Century/RothC and a
microbially-explicit 4-pool scheme), and a Mitscherlich crop-N yield response.
The economic layer solves regional fertilizer, food, and land markets under a
fertilizer-price or physical-supply shock. See the manuscript Methods and SI.

## Requirements

Python ≥ 3.9 with:

```
pip install -r requirements.txt      # numpy, scipy, pandas, matplotlib
```

No network access is needed to reproduce results: the ERA5 climate normals are
included in `data/`. (`code/era5/fetch_era5_climate.py` documents how they were
retrieved from the Open-Meteo ERA5 archive, 2001–2020, Hargreaves PET.)

## Reproduce (in order)

```
cd code/repro
python run_canonical.py               # -> data/canonical_ERA5_y30.csv / .json ; outputs/global_S3_losses.txt
python make_table_s3.py               # -> outputs/table_S3_correlations.csv   (Supplementary table 3)
python make_figure_s6.py              # -> figures/Figure_S6_pairwise_diagnostics.png
python climate_comparison.py          # -> outputs/climate_swap_comparison.csv (expert vs ERA5 robustness)
python make_sc_trajectories.py        # -> data/SC1_regional_trajectory.csv, SC2_regional_trajectory.csv
python make_scenario_trajectories.py  # -> data/scenario_trajectories.csv
python run_price_shock_analysis.py    # -> data/figure1_farm_gradient.json, data/figure2_soc_gradient.json  (~90 s)
python make_figure_1.py               # -> figures/Figure_1_farm_buffering.png/.pdf ; data/figure1_soc_gradient.csv
python make_figure_2.py               # -> figures/Figure_2_regional_vulnerability.png/.pdf ; data/figure2_panels.json
python compute_figS8_curves.py        # -> data/figS8_curves.json
python make_figure_s8.py              # -> figures/Figure_S8_elasticity_sensitivity.png
python make_figure_s7.py              # -> figures/Figure_S7_farm_elasticity_gradient.png/.pdf ; data/figS7_farm_elasticity_gradient.json (~2 min)
python make_figure_s12.py             # -> figures/Figure_S12_crop_response_calibration.png
python make_ofra_validation.py        # -> figures/Figure_S13_OFRA_SSA_validation.png
python run_mc_ensemble.py --n 1000 --workers 4   # -> data/mc_ensemble/ (Note 6; ~7 min)
python make_figure_s9.py              # -> figures/Figure_S9_mc_ensemble.png/.pdf
python make_figure_s10.py             # -> figures/Figure_S10_nue_sensitivity.png/.pdf ; data/figS10_nue_sensitivity.json (~4 min)
python make_figure_s11.py             # -> figures/Figure_S11_severity_gradient.png/.pdf (~3 min)
python make_food_price_table.py       # -> data/food_price_response.csv
```

Internal-consistency checks (both should print PASS):

```
python test_zero_shock_invariance.py  # -> outputs/zero_shock_invariance.csv
python test_cap_market_clearing.py
```

## Expected canonical results (what the scripts print / write)

| Quantity | Value | Appears in |
|---|---|---|
| Global S3 yield loss, production-weighted (yr 1 / 10 / 30) | **2.30 / 3.41 / 3.64 %** → 2.3 / 3.4 / 3.6 | Abstract, Results |
| Regional year-10 loss, range | **1.3 % (EA) – 6.0 % (SA)**; FSU 5.5, SSA 5.4, SEA 4.5, EU 3.5, LATAM 2.9, NA 1.8 | Results, Figure 2 |
| SSA year-30 loss | **5.9 %** | Results |
| SC1 (permanent supply loss) global year-10 | **4.0 %** (≈ 0.6 pp above S3) | Results |
| SC2 (20-yr recovery) global year-10 | 2.2 % (0.5 % by year 20) | Results |
| Soil-N buffer ratio (%), NA→FSU | **50, 36, 20, 32, 52, 62, 54, 49** | Supplementary table 1 |
| Buffer ratio vs year-10 penalty | Spearman ρ = **+0.02**, R² = **0.05** | Note 3, Figure S6f, table 3 |
| Fig S6 panel ρ (a–f) | −0.67, +0.70, −0.86, −0.45, +0.58, +0.02 | Figure S6 |
| Climate robustness (expert vs ERA5) | max year-10 shift **0.74 pp**; Spearman ρ = 0.93 | Response letter |
| Figure 1 gross margin at regional mean SOC | SSA **−11.4 %**, SA −8.7, LATAM −6.4, NA −3.4 | Figure 1b |
| Figure 2b year-10 SOM-depletion share (global) | 0.32 of 3.41 pp | Figure 2b |
| Figure S7 halved-elasticity gross-margin penalty | **2.6-10.3 pp** deeper at regional mean SOC (up to 14.1 pp on the most degraded SSA farms); SSA yield improves **2.3 pp** from 10 % to 200 % SOC | Figure S7 |
| MC ensemble median year-1 loss (n = 1,000) | **2.51 %**; 5-95 % range 3.3 pp; buffer 0.88 ppt | Note 6, Figure S9 |
| Figure S11 SOC spread (25 % vs 100 %) | **0.1-1.5 pp** at 100-150 % shock; 0.4-2.2 pp at 300 % | Figure S11 |
| Realized S3 fertilizer reduction, yr 1-10 (N-weighted) | **20 %** | Results |
| NUE lever: global year-10 loss, NUE 0.45 → 0.95 | **10.9 % → 1.2 %**; first 20 points of NUE deliver **59 %** of the reduction | Note 5, Figure S10 |
| Regional food-price response (production-weighted, yr 1/10/30) | **+5.5 / +5.0 / +5.3 %**; regional yr-10 span 1.0 (LATAM) - 10.3 (FSU) | SI, Food-price impacts |
| Zero-shock invariance | PASS (all regions yr-10 yield ratio ≥ 0.9998; minimum 0.99986, FSU) | Note on model consistency |
| Constrained market clearing | PASS (max cap residual 0.00e+00) | Note on model consistency |

Yield losses are reported as fractions of each region's baseline. They still vary slightly with the calibrated ceiling `y_max` (through residue return, SOM feedbacks, and yield constraints); the values above are the losses the canonical run produces with the `y_max` values it actually uses, documented here — not a claim of exact `y_max`-independence.

## Calibrated ceiling (y_max)

`data/canonical_ERA5_y30.*` reports `y_max` as the model's calibrated
Mitscherlich ceiling from `get_calibrated_ym()` (region-by-region, so that at
current N the modeled yield equals the FAOSTAT target), evaluated under the ERA5
climate: NA 6.28, EU 6.12, EA 6.09, SA 3.64, SEA 4.87, LATAM 5.60, SSA 3.88,
FSU 4.29 t ha⁻¹. This is the ceiling used by the coupled model and plotted in
Figure S6c.

## Directory layout

```
code/
  model/       coupled_monthly.py, coupled_econ_biophysical.py,
               monthly_model_v3.py, soil_n_model.py
  model/scripts/ validate_f_fert_broadbalk.py  (temperate crop-response validation)
  era5/        fetch_era5_climate.py, REGIONAL_CLIMATES_era5.py
  repro/       run_canonical.py, make_table_s3.py, make_figure_s6.py,
               climate_comparison.py, make_sc_trajectories.py,
               make_scenario_trajectories.py, run_price_shock_analysis.py,
               make_figure_1.py, make_figure_2.py, compute_figS8_curves.py,
               make_figure_s7.py, make_figure_s8.py, make_figure_s12.py,
               make_ofra_validation.py,
               run_mc_ensemble.py, make_figure_s9.py, make_figure_s10.py,
               make_figure_s11.py, make_food_price_table.py,
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
               ofra_maize_N_responsefunctions.csv, crop_response_calibration_table.csv
figures/       Figure_1_farm_buffering.png/.pdf,
               Figure_2_regional_vulnerability.png/.pdf,
               Figure_S5_flux_decomposition.png, Figure_S6_pairwise_diagnostics.png,
               Figure_S7_farm_elasticity_gradient.png/.pdf,
               Figure_S8_elasticity_sensitivity.png,
               Figure_S9_mc_ensemble.png/.pdf,
               Figure_S10_nue_sensitivity.png/.pdf,
               Figure_S11_severity_gradient.png/.pdf,
               Figure_S12_crop_response_calibration.png,
               Figure_S13_OFRA_SSA_validation.png
outputs/       regenerated tables (written by the scripts)
```

**Known gap.** `figures/Figure_S5_flux_decomposition.png` is carried over from
the earlier deposit and cannot be regenerated here: the microbially-explicit
4-pool SOM scheme it visualizes is not part of this code deposit. That figure
is illustrative of pool-level flux structure and no number in the manuscript
or SI is derived from it. This gap predates the v1.3 corrections and is
unaffected by them. With the addition of `make_figure_s7.py` in v1.4 it is now
the only figure in the paper that cannot be regenerated end to end from this
deposit.

**Second known gap.** `data/crop_response_calibration_table.csv` (the source
for Supplementary table 4) is carried over without a generator: the script that
produced its `y_no_synth_sim_tha` column was not deposited and has not been
recovered. Its `ymax_calibrated_tha` column was checked against the corrected
canonical run and is unchanged to six figures, and re-running an equivalent
no-synthetic-N simulation under the pre-fix and corrected models moves every
region by ≤ 0.05 t ha⁻¹, so the table is unaffected by the v1.3 corrections.
The missing generator is disclosed rather than reconstructed. This is a data
gap, not a figure gap, and is separate from the Figure S5 case above.

See `MANIFEST.md` for a per-file description and `CHANGELOG.md`
(in the resubmission package) for the full correction history.

## License

Code: MIT (`LICENSE`). Data and figures: CC-BY-4.0.
