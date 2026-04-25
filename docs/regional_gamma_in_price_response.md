# Use regional supply elasticity in PY_hat / F_hat (not farm-level gamma)

## Problem

`run_farm_sweep_single` computed the output-price recovery `PY_hat` and the
shocked fertilizer demand `F_shocked` using the gamma evaluated at the
*individual farm's* SOC level. That conflated farm-level production
response with regional/market-level supply response: a farm at SOC = 200 %
was implicitly treated as if it were embedded in a region whose entire
cropland had SOC = 200 %, so its hypothetical region exhibited little
supply contraction under a fertilizer-price shock and therefore got
little output-price recovery.

The visible symptom: in panel b of Figure 1 (gross margin change vs
farm SOC), the curves for NA, LATAM, and SA were flat or *declined* above
SOC = 100 %. Decomposing `profit_chg` showed the cause was the
exp(PY_hat) price-cushion shrinking with SOC because gamma was being
read off the farm rather than the region:

| SOC % | NA exp(PY) | LATAM exp(PY) | SA exp(PY) | SSA exp(PY) |
|---:|---:|---:|---:|---:|
| 50  | 1.075 | 1.052 | 1.073 | 1.071 |
| 100 | 1.049 | 1.032 | 1.058 | 1.051 |
| 200 | 1.023 | 1.015 | 1.040 | 1.031 |

That's a model-aggregation artefact, not a real economic effect.

## Fix

A single farm's SOC has no influence on the regional/global market price
response. Replace `gamma` (farm) with `gamma_regional` (= gamma at the
regional-mean SOC, i.e. the unscaled equilibrium pools) when computing
`PY_hat` and `F_hat`. The farm's own production response (`y_shock`
given the regionally-determined `F_shocked`) is unchanged — only the
market-level price-clearing variables now correctly use the regional
aggregate.

Diff hub: `scripts/run_price_shock_analysis.py` `run_farm_sweep_single`,
two lines:

```python
# Before:
gamma = state_base['gamma']                      # farm-level
denom = eta - gamma * eps_F_PY
PY_hat = gamma * eps_F_PF * PF_hat / denom

# After:
gamma_regional = state_regional['gamma']         # at SOC = 100 %
denom = eta - gamma_regional * eps_F_PY
PY_hat = gamma_regional * eps_F_PF * PF_hat / denom
```

## Effect on results

- Curves are now monotonically improving with SOC across the entire
  10–200 % range for all four regions.
- Gross-margin gap between SOC = 50 % and SOC = 100 % at a 100 %
  fertilizer-price shock widens substantially:

  | region | gap before fix | gap after fix |
  |---|---:|---:|
  | SSA   | 1.9 pp | **4.6 pp** |
  | NA    | 0.3 pp | **3.1 pp** |
  | SA    | 0.9 pp | **2.7 pp** |
  | LATAM | 0.4 pp | **2.6 pp** |

- The manuscript's headline "2–3 pp of gross margin" claim is now
  correctly supported (range 2.6–4.6 pp across the four regions).
- Yield-loss numbers and SOC degradation trajectories from
  `resilience_monthly.pkl` are unchanged (this fix only touches the
  farm-level price-shock sweep).

## Reproducing

```bash
python3 scripts/run_price_shock_analysis.py    # regenerate pickle
python3 scripts/export_csvs.py                 # regenerate price_shock_farm.csv
```
