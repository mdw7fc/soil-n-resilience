#!/usr/bin/env python3
"""Regenerate Supp Table 1 (MEMS CUE downregulation decomposition) under matched
SC1 20% scenario. Produces fixed-CUE vs variable-CUE MEMS runs and computes
the fraction of the MEMS/Century SOC-loss difference attributable to CUE
downregulation under N stress.

Output: data/cue_decomposition_matched.pkl + printed table.
"""
import sys, pickle, copy
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJ = REPO_ROOT  # alias
ROOT = REPO_ROOT  # alias
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'model' / 'scripts'))

from coupled_monthly import CoupledMonthlyModel, calibrate_price_shock
from coupled_mems import CoupledMEMSModel
from coupled_econ_biophysical import EconParams
from soil_n_model import get_default_regions
from monthly_mems_v1 import MEMSPoolParams

REGIONS = ['north_america','europe','east_asia','south_asia','southeast_asia',
           'latin_america','sub_saharan_africa','fsu_central_asia']

def main():
    regions_all = get_default_regions()
    shock = calibrate_price_shock(0.20)
    econ = EconParams(fert_price_shock=shock, eps_F_N=0.0)
    t_max = 30

    # Default MEMS params (variable CUE)
    mems_variable = MEMSPoolParams()
    # Fixed-CUE variant: pin cue_min = cue_max
    mems_fixed = MEMSPoolParams(cue_min=mems_variable.cue_max, cue_max=mems_variable.cue_max)
    print(f'MEMS cue_min (variable run): {mems_variable.cue_min}')
    print(f'MEMS cue_max (both runs):    {mems_variable.cue_max}')
    print(f'MEMS fixed-run CUE:          {mems_fixed.cue_min} = {mems_fixed.cue_max}')

    decomp = {}
    print(f'\n{"region":<22}{"C_SOCloss":<12}{"M_var_loss":<12}{"M_fix_loss":<12}{"CUE_%":<8}{"NonCUE_%":<10}')
    for rn in REGIONS:
        r = regions_all[rn]
        # Century
        mc = CoupledMonthlyModel(region=r, econ=econ, region_key=rn, t_max=t_max)
        dfc = mc.run()
        c_loss = dfc['soc_total'].iloc[0] - dfc[dfc['year']==30]['soc_total'].iloc[0]

        # MEMS variable CUE
        mv = CoupledMEMSModel(region=r, econ=econ, region_key=rn, t_max=t_max,
                              mems_params=mems_variable)
        dfv = mv.run()
        mv_loss = dfv['soc_total'].iloc[0] - dfv[dfv['year']==30]['soc_total'].iloc[0]

        # MEMS fixed CUE (no downregulation)
        mf = CoupledMEMSModel(region=r, econ=econ, region_key=rn, t_max=t_max,
                              mems_params=mems_fixed)
        dff = mf.run()
        mf_loss = dff['soc_total'].iloc[0] - dff[dff['year']==30]['soc_total'].iloc[0]

        # CUE-downregulation contribution = delta (variable vs fixed)
        # Non-CUE = everything else in MEMS/Century difference = mf_loss - c_loss
        total_excess = mv_loss - c_loss
        cue_contribution = mv_loss - mf_loss
        noncue_contribution = total_excess - cue_contribution
        if abs(total_excess) > 1e-6:
            cue_pct = cue_contribution / total_excess * 100
            noncue_pct = noncue_contribution / total_excess * 100
        else:
            cue_pct = float('nan'); noncue_pct = float('nan')
        decomp[rn] = {
            'century_soc_loss': float(c_loss),
            'mems_variable_soc_loss': float(mv_loss),
            'mems_fixed_soc_loss': float(mf_loss),
            'cue_contribution_pct': float(cue_pct),
            'noncue_contribution_pct': float(noncue_pct),
        }
        print(f'  {rn:<22}{c_loss:<12.3f}{mv_loss:<12.3f}{mf_loss:<12.3f}{cue_pct:<8.1f}{noncue_pct:<10.1f}')

    out = PROJ / 'data' / 'cue_decomposition_matched.pkl'
    with open(out, 'wb') as f: pickle.dump(decomp, f)
    print(f'\nSaved {out}')

if __name__ == '__main__':
    main()
