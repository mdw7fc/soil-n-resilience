# soil-n-resilience

Code and data for: **Soil health buffers nitrogen supply disruptions** (Wallenstein & Manning, 2026, *Nature Food*)

## Overview

This repository contains the coupled biophysical-economic model that simulates how soil organic carbon (SOC) and economic structure jointly determine agricultural vulnerability to fertilizer supply disruptions. The model combines:

- A three-pool SOM model (Century/RothC kinetics) with monthly nitrogen balance
- A partial equilibrium economic framework (Manning) with regional price elasticities
- Eight global agricultural regions covering ~1,230 Mha cropland and ~99 Tg N yr⁻¹

## Repository structure

```
model/
  soil_n_model.py              # Core SOM model: 3-pool dynamics, regional params, scenarios
  monthly_model_v3.py          # Hybrid annual SOM + monthly N balance engine
  coupled_econ_biophysical.py  # Manning partial equilibrium economics
  coupled_monthly.py           # Monthly biophysical-economic coupling (main model)

scripts/
  run_price_shock_analysis.py      # Farm-level SOC gradient x price shock (Figs 1-2)
  run_resilience_monthly.py        # Regional scenarios S1-S3, SC1-SC4, NUE sweeps (Figs 3-5)
  generate_publication_figures.py  # Generates all 6 manuscript figures
  generate_sensitivity_fig_s1.py   # Supplementary Figure S1

data/
  buffer_metrics.csv           # Regional soil N buffer ratios and dependencies
  scenario_trajectories.csv    # 30-year S1/S2/S3 yield and SOC trajectories
  supply_constrained.csv       # SC1-SC4 supply-constrained scenarios
  degradation_scenarios.csv    # SOC degradation gradient results
  nue_sensitivity.csv          # NUE sensitivity sweep (0.45-0.95)
  duration_comparison.csv      # 1/5/10/30-year disruption duration comparison
  price_shock_farm.csv         # Farm-level price shock x SOC analysis
  soc_gradient.csv             # Fine SOC gradient across all 8 regions
  regional_parameters.csv      # All regional model parameters
```

## Requirements

- Python 3.10+
- numpy, scipy, pandas, matplotlib

## Reproducing the analysis

```bash
# 1. Generate analysis data
python scripts/run_price_shock_analysis.py
python scripts/run_resilience_monthly.py

# 2. Generate figures
python scripts/generate_publication_figures.py
python scripts/generate_sensitivity_fig_s1.py
```

## Citation

Wallenstein, M.D. & Manning, D. (2026). Soil health buffers nitrogen supply disruptions. *Nature Food*.

## License

MIT
