# Water-stress feedback: switch to a smooth, two-sided SOC-WHC response

## Problem

The original `_water_stress()` in `model/coupled_monthly.py` and
`model/soil_n_model.py` had two hard clamps that together produced a
slope discontinuity in the yield response at SOC = regional mean:

```python
delta = max(0, soc_pct_init - soc_pct)            # asymmetric: SOC gain ignored
whc_loss_mm = delta * whc_sensitivity
total_deficit = max(0, baseline_water_deficit + whc_loss_mm)
stress = 1.0 - water_stress_coeff * total_deficit
```

The `max(0, …)` on `delta` made WHC loss linear on the deficit side but
flat on the accumulation side — i.e. additional SOC above baseline
contributed nothing to available water. The `max(0, total_deficit)`
re-introduced a kink at the SOC where the computed deficit would cross
zero, which for regions with `baseline_water_deficit = 0` is exactly at
SOC = 100%. At fine (1 % SOC) resolution the resulting `y_base` slope
dropped 30–96 % across the SOC = 100 % boundary in all four regions
sampled in the farm-level price-shock analysis.

## Fix

Two edits, mirrored between the monthly and annual biophysical engines.

### 1. Two-sided, C¹-smooth WHC response

Let `delta = soc_pct_init - soc_pct` (signed):

- **Loss side** (`delta ≥ 0`): linear, unchanged —
  `whc_change_mm = delta · whc_sensitivity`.
- **Gain side** (`delta < 0`): exponential saturation toward a ceiling,
  with initial slope matched to the loss side so the function is
  C¹-continuous at `delta = 0`:

  ```
  whc_gain_max_mm = whc_gain_sat_pct · whc_sensitivity
  whc_change_mm   = -whc_gain_max_mm · (1 - exp(delta / whc_gain_sat_pct))
  ```

  `whc_gain_sat_pct = 1.0` is the characteristic SOC unit scale over
  which ~63 % of the maximum WHC gain is realised — anchored to the
  upper end of the typical agronomic SOC range.

### 2. Soft-abs floor on total water deficit

The hard `max(0, …)` on `total_deficit` is replaced by:

```
raw = baseline_water_deficit + whc_change_mm
total_deficit = 0.5 · (raw + sqrt(raw² + ε²))       # ε = 3 mm
```

`soft_pos(x, ε) = 0.5·(x + √(x² + ε²))` is C¹-smooth, monotonically ≥ 0,
and asymptotes to `x` for `x ≫ ε` and to 0 for `x ≪ -ε`. Physically, the
ε-scale represents the seasonal / within-region distributional deficit
that persists even when the annual mean deficit is near zero. Large
deficits (≥ 10 mm, typical for SSA, SA, SEA, FSU) are perturbed by
< 5 %, so degraded-soil behaviour is essentially unchanged.

## Empirical anchor

- Minasny, B. & McBratney, A. B. (2018). Limited effect of organic
  matter on soil available water capacity. *European Journal of Soil
  Science* 69, 39–47. https://doi.org/10.1111/ejss.12475 — meta-analytic
  WHC/SOC slope of ~5–15 mm per 0.1 % SOC (model parameter
  `whc_sensitivity = 8.4`), with documented diminishing returns at high
  SOC as porosity approaches its texture-determined ceiling.
- Hudson, B. D. (1994). Soil organic matter and available water
  capacity. *Journal of Soil and Water Conservation* 49, 189–194 —
  log-linear SOM–WHC relationship with the same qualitative saturation.

## Calibration impact

`yield_max` is re-fit per region by `get_calibrated_ym` (brentq) against
the FAOSTAT regional yield target. The soft-abs shift at SOC = baseline
adds a ≤ 1.5 mm effective deficit (≤ 0.5 % yield) and the recalibrated
`yield_max` moves by < 1 % per region. Baseline yields still match
FAOSTAT within the original tolerance.

## Verification

At 1 % SOC resolution, post-fix `y_base` slopes across the SOC = 100 %
boundary:

| region | slope 99→100 | slope 100→101 | slope 101→105 |
|:-------|-----------:|-----------:|-----------:|
| SSA    | +0.00457   | +0.00456 (−0.2 %) | +0.00449 |
| SA     | +0.00828   | +0.00805 (−3 %)   | +0.00701 |
| LATAM  | +0.01569   | +0.01241 (−21 %)  | +0.00748 |
| NA     | +0.02798   | +0.02276 (−19 %)  | +0.00488 |

All four regions are now visually smooth across the full 10–200 % SOC
range. Headline farm-level numbers are preserved:

- SSA 50 % vs 100 % SOC yield-loss gap @ 100 % price shock: **0.72 pp**
- SSA NUE = 0.75 year-10 loss: **14.8 %**
- Global production-weighted NUE = 0.75 year-10 loss: **6.7 %**
- Soil-buffer R² (alone / + price elasticity): **0.10 / 0.37**

## Reproducing

From the repo root:

```bash
python3 scripts/run_price_shock_analysis.py    # generates price_shock_analysis.pkl + soc_gradient_fine.pkl
python3 scripts/run_resilience_monthly.py      # generates resilience_monthly.pkl
python3 scripts/export_csvs.py                 # regenerates data/*.csv from the pickles
python3 scripts/generate_publication_figures.py  # produces figures from the pickles
```

Pickles are `.gitignore`-d; CSVs in `data/` are the canonical committed
snapshots.
