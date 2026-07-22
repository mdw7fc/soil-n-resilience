# Soil organic matter buffers fertilizer supply disruptions

Model code and **canonical reproducibility deposit** for:

> Wallenstein, M. D. & Manning, D. T. *Soil organic matter buffers fertilizer
> supply disruptions.* Environmental Research: Food Systems (ERFS-100341).

Zenodo: https://doi.org/10.5281/zenodo.19699772

This deposit corresponds to the **v12 (canonical) manuscript and SI**. Every
figure and table in the paper derives from a **single frozen model output** —
the ERA5 data-based climate run extended to year 30 — produced by
`code/repro/run_canonical.py`. Earlier deposits mixed an expert-climate run
(which stopped at year 10) with the ERA5 run; this version resolves that so all
artifacts trace to one output set.

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
python run_canonical.py          # -> data/canonical_ERA5_y30.csv / .json ; outputs/global_S3_losses.txt
python make_table_s3.py          # -> outputs/table_S3_correlations.csv   (Supplementary table 3)
python make_figure_s6.py         # -> figures/Figure_S6_pairwise_diagnostics.png
python climate_comparison.py     # -> outputs/climate_swap_comparison.csv (expert vs ERA5 robustness)
python make_sc_trajectories.py   # -> data/SC1_regional_trajectory.csv, SC2_regional_trajectory.csv
```

## Expected canonical results (what the scripts print / write)

| Quantity | Value | Appears in |
|---|---|---|
| Global S3 yield loss, production-weighted (yr 1 / 10 / 30) | **4.33 / 5.58 / 5.95 %** → 4.3 / 5.6 / 5.9 | Abstract, Results |
| Regional year-10 loss, range | **2.2 % (NA) – 13.7 % (SSA)**; SA 10.1, LATAM 2.9, FSU 9.8 | Results, Figure 2 |
| SSA year-30 loss | **15.0 %** | Results |
| SC1 (permanent supply loss) global year-10 | **6.3 %** (≈ 0.7 pp above S3) | Results |
| SC2 (20-yr recovery) global year-10 | 4.5 % | Results |
| Soil-N buffer ratio (%), NA→FSU | **50, 37, 20, 34, 52, 63, 57, 50** | Supplementary table 1 |
| Buffer ratio vs year-10 penalty | Spearman ρ = **+0.07**, R² = **0.03** | Note 3, Figure S6f, table 3 |
| Fig S6 panel ρ (a–f) | −0.92, +0.90, −0.88, −0.75, +0.72, +0.07 | Figure S6 |
| Climate robustness (expert vs ERA5) | max year-10 shift **0.54 pp**; Spearman ρ = 0.98 | Note 8, Figure S13 |

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
               climate_comparison.py, make_sc_trajectories.py
data/          canonical_ERA5_y30.csv/.json, era5_regional_climates.json,
               era5_raw/ (8 regions), climate_swap_comparison.csv,
               SC1_/SC2_regional_trajectory.csv,
               ofra_maize_N_responsefunctions.csv, crop_response_calibration_table.csv
figures/       Figure_S5_flux_decomposition.png, Figure_S6_pairwise_diagnostics.png
outputs/       regenerated tables (written by the scripts)
```

See `MANIFEST.md` for a per-file description and `CHANGELOG.md`
(in the resubmission package) for the full correction history.

## License

Code: MIT (`LICENSE`). Data and figures: CC-BY-4.0.
