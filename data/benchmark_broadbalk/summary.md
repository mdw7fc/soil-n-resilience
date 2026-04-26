# Tier-1 LTE Benchmark — Century v3 vs MEMS v1

*Run: 2026-04-24. Site-specific initial conditions (Broadbalk SOC₀=28.8 t C/ha 0-23 cm; Morrow SOC₀=85 t C/ha 0-30 cm).*

## Broadbalk SOC trajectory match (172 yr)

| Treatment | Plot | Model | n_obs | RMSE (t C/ha) | Willmott d | Bias | Pearson r | Final yield (t/ha) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Nil | P3 | CENTURY | 15 | 3.05 | 0.599 | -2.50 | +0.607 | 1.48 |
| Nil | P3 | MEMS | 15 | 2.07 | 0.709 | -1.23 | +0.608 | 1.33 |
|   |   | **Winner → mems** |  |  |  |  |  |  |
| PK | P5 | CENTURY | 15 | 4.23 | 0.373 | -3.98 | +0.676 | 1.48 |
| PK | P5 | MEMS | 15 | 2.99 | 0.482 | -2.70 | +0.679 | 1.33 |
|   |   | **Winner → mems** |  |  |  |  |  |  |
| N3PK | P8 | CENTURY | 13 | 23.96 | 0.117 | +22.78 | +0.318 | 10.65 |
| N3PK | P8 | MEMS | 13 | 28.69 | 0.101 | +27.33 | +0.350 | 9.59 |
|   |   | **Winner → century** |  |  |  |  |  |  |
| FYM1843 | P2.2 | CENTURY | 15 | 18.24 | 0.739 | +17.27 | +0.938 | 10.95 |
| FYM1843 | P2.2 | MEMS | 15 | 28.86 | 0.582 | +27.53 | +0.906 | 10.70 |
|   |   | **Winner → century** |  |  |  |  |  |  |
| FYM+N3 | P2.1 | CENTURY | 12 | 27.43 | 0.382 | +26.77 | +0.910 | 10.99 |
| FYM+N3 | P2.1 | MEMS | 12 | 40.61 | 0.277 | +39.72 | +0.760 | 10.99 |
|   |   | **Winner → century** |  |  |  |  |  |  |

## Morrow yield decadal match (Plot 3N)

*Split by era: pre-modern (1888-1967) is the clean SOM-driven test; modern era (1968+) includes cultivar/herbicide effects not captured by SOM-only models.*

| Era | Model | n_dec | RMSE (t/ha) | Willmott d | Bias | Pearson r | Mean mod / obs |
|---|---|---:|---:|---:|---:|---:|---|
| pre_modern_1888-1967 | CENTURY | 9 | 0.64 | 0.740 | +0.25 | +0.582 | 2.21 / 1.96 |
| pre_modern_1888-1967 | MEMS | 9 | 0.69 | 0.749 | -0.02 | +0.576 | 1.94 / 1.96 |
| pre_modern_1888-1967 | **Winner → century** |  |  |  |  |  |  |
| modern_1968-2021 | CENTURY | 6 | 1.71 | 0.199 | -1.67 | -0.482 | 1.24 / 2.91 |
| modern_1968-2021 | MEMS | 6 | 1.98 | 0.179 | -1.95 | -0.499 | 0.96 / 2.91 |
| modern_1968-2021 | **Winner → century** |  |  |  |  |  |  |
| full_1888-2021 | CENTURY | 15 | 1.19 | 0.364 | -0.52 | -0.171 | 1.82 / 2.34 |
| full_1888-2021 | MEMS | 15 | 1.36 | 0.388 | -0.79 | -0.109 | 1.55 / 2.34 |
| full_1888-2021 | **Winner → century** |  |  |  |  |  |  |

Final-SOC check (last 20 yr mean, 0-30 cm proxy): Century=52.1 t C/ha, MEMS=37.6. Observed modern SOM ≈ 3.2% → ~36 t C/ha at 0-15 cm (≈ 55-65 t C/ha extrapolated to 0-30 cm).


## Data files

- `lte_scoreboard_broadbalk.csv`  — per-treatment × model scores
- `lte_scoreboard_morrow.csv`    — Morrow yield scores
- `soc_trajectories_broadbalk.csv` — annual SOC modeled + observed
- `yield_decadal_morrow.csv`     — decadal yield modeled + observed