#!/usr/bin/env python3
"""
Diagnostic: source of Broadbalk Nil yield over-prediction.

Runs the Broadbalk Nil treatment under perturbations of two yield-side
parameters that could plausibly explain the +2x bias:

  - yield_min   : the Mitscherlich residual floor (current 0.8 t/ha)
  - yield_max   : the seasonal yield potential (current 11.0 t/ha,
                  applied uniformly across 1843-2015)

For each combination, it reports:
  - mean modelled grain yield 1843-2015 (the figure we report)
  - mean modelled yield in the last 20 yr
  - fraction of years that the floor binds
  - fraction of years that the stoichiometric cap binds
  - mean uptake N

The point is to distinguish (a) "the floor is forcing the bias" from
(b) "uptake/Mitscherlich is producing the bias even with no floor", so
the user can decide whether a floor sensitivity is a defensible SI
addition or whether the bias is a deeper structural feature.

Output: figures_regenerated/broadbalk_nil_diagnostic.csv (printed)
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'model' / 'scripts'))

from benchmark_broadbalk import (
    run_century_site,
    BROADBALK_INITIAL_SOC, ROTHAMSTED_CLIMATE,
    BROADBALK_CN_BULK, BROADBALK_ATM_N_DEP, BROADBALK_BNF_FREE,
)
from soil_n_model import CropParams

OUT = ROOT / 'paper2-soil-resilience' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

OBS_NIL_MEAN_YIELD_T_HA = 1.05  # Broadbalk handout long-term mean grain yield


def run_one(yield_min, yield_max, mit_c=0.025, n_years=173):
    res = run_century_site(
        soc_initial=BROADBALK_INITIAL_SOC, climate=ROTHAMSTED_CLIMATE,
        n_years=n_years,
        synth_n=0.0, fym_c_input=0.0, fym_mineralized_n=0.0,
        atm_n_dep=BROADBALK_ATM_N_DEP, bnf=BROADBALK_BNF_FREE,
        cn_bulk=BROADBALK_CN_BULK,
        yield_max=yield_max, yield_min=yield_min, mit_c=mit_c,
    )
    y = res['yield_tha']
    n_eff = res['n_uptake']
    crop = CropParams()
    n_grain_t = crop.grain_n_fraction * 1000  # kg N per t grain
    # Reconstruct the unconstrained Mitscherlich vs stoichiometric binding fractions
    y_mit = yield_max * (1 - np.exp(-mit_c * n_eff))
    y_stoich = n_eff / n_grain_t
    y_unfloored = np.minimum(y_mit, y_stoich)
    floor_binds = np.mean(y_unfloored < yield_min)
    stoich_binds = np.mean(y_stoich < y_mit)
    return {
        'yield_min': yield_min,
        'yield_max': yield_max,
        'mean_yield_full': float(np.mean(y)),
        'mean_yield_last20': float(np.mean(y[-20:])),
        'mean_yield_first20': float(np.mean(y[:20])),
        'mean_uptake_kgN': float(np.mean(n_eff)),
        'frac_yr_floor_binds': float(floor_binds),
        'frac_yr_stoich_binds': float(stoich_binds),
        'mean_y_mit': float(np.mean(y_mit)),
        'mean_y_stoich': float(np.mean(y_stoich)),
        'mean_y_unfloored': float(np.mean(y_unfloored)),
        'bias_pct_full': float((np.mean(y) - OBS_NIL_MEAN_YIELD_T_HA)
                               / OBS_NIL_MEAN_YIELD_T_HA * 100),
    }


def main():
    rows = []
    print(f'Observed Broadbalk Nil long-term mean yield: '
          f'{OBS_NIL_MEAN_YIELD_T_HA:.2f} t/ha')
    print(f'Current model defaults: yield_max=11.0, yield_min=0.8')
    print()

    grid = [
        # (yield_min, yield_max)
        (0.8, 11.0),  # current
        (0.5, 11.0),  # lower floor only
        (0.3, 11.0),
        (0.1, 11.0),
        (0.0, 11.0),  # no floor
        (0.8,  8.0),  # lower yield_max (mid-20th-century cultivar mean)
        (0.8,  5.0),  # lower yield_max (Edwardian wheat potential)
        (0.0,  5.0),  # combined: no floor + low yield potential
        (0.0,  3.0),  # combined: no floor + Victorian wheat potential
    ]
    for ym, yx in grid:
        rows.append(run_one(yield_min=ym, yield_max=yx))

    df = pd.DataFrame(rows)
    df['bias_vs_obs'] = df['mean_yield_full'] - OBS_NIL_MEAN_YIELD_T_HA
    cols = ['yield_min', 'yield_max', 'mean_yield_full', 'bias_vs_obs',
            'bias_pct_full', 'frac_yr_floor_binds', 'frac_yr_stoich_binds',
            'mean_uptake_kgN', 'mean_y_mit', 'mean_y_stoich',
            'mean_y_unfloored']
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    out_csv = OUT / 'broadbalk_nil_diagnostic.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')


if __name__ == '__main__':
    main()
