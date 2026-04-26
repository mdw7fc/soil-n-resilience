#!/usr/bin/env python3
"""Run matched-scenario MEMS vs Century comparison.

Runs both SOM frameworks under the same 20% sustained supply reduction
scenario (SC1 from the main manuscript) for all 8 regions, decomposes
MEMS SOC loss by flux step, and saves outputs for Fig 6 + new Supp Fig S4.

Produces:
  - matched_mems_century.pkl with full time series for both frameworks
  - flux_decomposition.pkl with per-step MEMS carbon fluxes
  - Prints summary comparison tables

Scenario: SC1 (sustained 20% fertilizer supply reduction), 30-year horizon.
This replaces the earlier 100% withdrawal stress test as the primary
Fig 6 comparison. 100% withdrawal data retained in archive/ for optional
supplementary use.

Author: Matthew Wallenstein & Dale Manning (matched-mems-2026-04-15 effort)
"""
import sys, os, pickle, copy
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJ = REPO_ROOT  # alias
ROOT = REPO_ROOT  # alias
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'model' / 'scripts'))

from coupled_monthly import CoupledMonthlyModel, calibrate_price_shock, get_calibrated_ym
from coupled_mems import CoupledMEMSModel, get_calibrated_ym_mems
from coupled_econ_biophysical import EconParams, REGIONAL_ECON_PARAMS
import coupled_econ_biophysical as ceb
from soil_n_model import get_default_regions

REGIONS = ['north_america','europe','east_asia','south_asia','southeast_asia',
           'latin_america','sub_saharan_africa','fsu_central_asia']

def main():
    regions_all = get_default_regions()
    shock_sc1 = calibrate_price_shock(0.20)   # matched to main Results
    print(f'SC1 price shock (20% supply reduction): +{shock_sc1*100:.0f}%')

    # Econ params: SC1 scenario — S3 elasticities + physical supply constraint
    # For matched Century/MEMS comparison, use S3 behavioral response
    # (no physical ceiling — MEMS and Century both receive same shock)
    econ = EconParams(fert_price_shock=shock_sc1, eps_F_N=0.0)

    # Also baseline (no shock) for reference yields/SOC
    econ_base = EconParams(fert_price_shock=0.0, eps_F_N=0.0)

    t_max = 30

    century_runs = {}
    mems_runs = {}
    century_base_runs = {}
    mems_base_runs = {}

    for rn in REGIONS:
        r = regions_all[rn]
        print(f'\n--- {rn} ---')
        # Century baseline and shocked
        m_c_base = CoupledMonthlyModel(region=r, econ=econ_base, region_key=rn, t_max=t_max)
        century_base_runs[rn] = m_c_base.run()
        m_c = CoupledMonthlyModel(region=r, econ=econ, region_key=rn, t_max=t_max)
        century_runs[rn] = m_c.run()

        # MEMS baseline and shocked
        m_m_base = CoupledMEMSModel(region=r, econ=econ_base, region_key=rn, t_max=t_max)
        mems_base_runs[rn] = m_m_base.run()
        m_m = CoupledMEMSModel(region=r, econ=econ, region_key=rn, t_max=t_max)
        mems_runs[rn] = m_m.run()

        c_y30 = century_runs[rn][century_runs[rn]['year']==30]['soc_total'].iloc[0]
        c_y1 = century_runs[rn][century_runs[rn]['year']==1]['soc_total'].iloc[0]
        m_y30 = mems_runs[rn][mems_runs[rn]['year']==30]['soc_total'].iloc[0]
        m_y1 = mems_runs[rn][mems_runs[rn]['year']==1]['soc_total'].iloc[0]
        c_loss = c_y1 - c_y30
        m_loss = m_y1 - m_y30
        ratio = m_loss / c_loss if c_loss > 0 else float('nan')
        print(f'  Century 30yr SOC loss: {c_loss:.3f} t C/ha')
        print(f'  MEMS 30yr SOC loss:    {m_loss:.3f} t C/ha')
        print(f'  MEMS/Century ratio:    {ratio:.2f}x')

    # Save full time series
    out = {
        'century_shocked': century_runs,
        'century_baseline': century_base_runs,
        'mems_shocked': mems_runs,
        'mems_baseline': mems_base_runs,
        'scenario': 'SC1 (20% sustained, S3 behavioral response, +104% fert price shock)',
        'shock_fraction': shock_sc1,
        't_max': t_max,
        'regions': REGIONS,
    }
    out_path = PROJ / 'data' / 'matched_mems_century.pkl'
    with open(out_path, 'wb') as f: pickle.dump(out, f)
    print(f'\nSaved {out_path}')

    # Summary
    print('\n=== Summary: MEMS/Century 30-year SOC-loss ratios (matched 20% scenario) ===')
    print(f'{"region":<22}{"C_loss":<10}{"M_loss":<10}{"ratio":<8}')
    ratios = []
    for rn in REGIONS:
        c_y1 = century_runs[rn][century_runs[rn]['year']==1]['soc_total'].iloc[0]
        c_y30 = century_runs[rn][century_runs[rn]['year']==30]['soc_total'].iloc[0]
        m_y1 = mems_runs[rn][mems_runs[rn]['year']==1]['soc_total'].iloc[0]
        m_y30 = mems_runs[rn][mems_runs[rn]['year']==30]['soc_total'].iloc[0]
        c_loss = c_y1 - c_y30
        m_loss = m_y1 - m_y30
        r = m_loss / c_loss if c_loss > 0 else float('nan')
        ratios.append(r)
        print(f'  {rn:<22}{c_loss:<10.3f}{m_loss:<10.3f}{r:<8.2f}')
    valid = [r for r in ratios if not np.isnan(r)]
    print(f'\n  Range: {min(valid):.2f}× to {max(valid):.2f}×')
    print(f'  Median: {float(np.median(valid)):.2f}×')

    # === Flux decomposition (MEMS shocked run) ===
    # For each region, compute the cumulative 30-year respiratory losses by mechanism
    print('\n=== MEMS flux decomposition (cumulative 30-year, shocked scenario) ===')
    print(f'{"region":<22}{"CUE_resp":<10}{"necro_resp":<12}{"tot_resp":<10}{"pom_out":<10}{"CUE_%":<8}{"necro_%":<8}')

    flux_decomp = {}
    for rn in REGIONS:
        df = mems_runs[rn]
        # Integrate over years (excluding year 0 which is initial)
        df_use = df[df['year'] >= 1]
        tot_cue = df_use['resp_cue'].sum()
        tot_necro = df_use['resp_necro'].sum()
        tot_resp = df_use['total_respired'].sum()
        tot_pom_out = df_use['flux_pom_to_dom'].sum()
        cue_pct = tot_cue / tot_resp * 100 if tot_resp > 0 else 0
        necro_pct = tot_necro / tot_resp * 100 if tot_resp > 0 else 0
        print(f'  {rn:<22}{tot_cue:<10.3f}{tot_necro:<12.3f}{tot_resp:<10.3f}{tot_pom_out:<10.3f}{cue_pct:<8.1f}{necro_pct:<8.1f}')
        flux_decomp[rn] = {
            'cum_resp_cue': float(tot_cue),
            'cum_resp_necro': float(tot_necro),
            'cum_total_respired': float(tot_resp),
            'cum_pom_to_dom': float(tot_pom_out),
            'cum_dom_to_mic': float(df_use['flux_dom_to_mic'].sum()),
            'cum_dom_to_maom': float(df_use['flux_dom_to_maom_sorption'].sum()),
            'cum_mic_assimilated': float(df_use['flux_mic_assimilated'].sum()),
            'cum_mic_death': float(df_use['flux_mic_death'].sum()),
            'cum_necro_to_maom': float(df_use['flux_necro_to_maom'].sum()),
            'cum_necro_to_pom': float(df_use['flux_necro_to_pom'].sum()),
            'cum_necro_to_dom': float(df_use['flux_necro_to_dom'].sum()),
            'cue_pct_of_resp': float(cue_pct),
            'necro_pct_of_resp': float(necro_pct),
        }

    flux_path = PROJ / 'data' / 'mems_flux_decomposition.pkl'
    with open(flux_path, 'wb') as f: pickle.dump(flux_decomp, f)
    print(f'\nSaved {flux_path}')
    print('\nDone.')

if __name__ == '__main__':
    main()
