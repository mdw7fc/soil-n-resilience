#!/usr/bin/env python3
"""Supp Fig S3: Economic parameter sensitivity — baseline vs halved price elasticities.

Runs 30-year S3 scenario for all 8 regions under baseline and halved eps_F_PF.
Parallel to Fig 6's structural (SOM framework) sensitivity.
"""
import sys, copy
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJ = REPO_ROOT  # alias
ROOT = REPO_ROOT  # alias
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'model' / 'scripts'))
from coupled_monthly import CoupledMonthlyModel, calibrate_price_shock, get_calibrated_ym
from coupled_econ_biophysical import EconParams
import coupled_econ_biophysical as ceb
from soil_n_model import get_default_regions

regions = get_default_regions()
REGION_ORDER = ['sub_saharan_africa','south_asia','fsu_central_asia','southeast_asia',
                'east_asia','europe','latin_america','north_america']
LBL = {'north_america':'North America','europe':'Europe','east_asia':'East Asia',
       'south_asia':'South Asia','southeast_asia':'SE Asia','latin_america':'Latin America',
       'sub_saharan_africa':'Sub-Saharan Africa','fsu_central_asia':'FSU & Central Asia'}
PAL = {'ssa':'#C62828','sa':'#1565C0','latam':'#2E7D32','na':'#455A64',
       'eu':'#00695C','ea':'#795548','fsu':'#6A1B9A','sea':'#E65100'}
COL = {'north_america':PAL['na'],'europe':PAL['eu'],'east_asia':PAL['ea'],'south_asia':PAL['sa'],
       'southeast_asia':PAL['sea'],'latin_america':PAL['latam'],'sub_saharan_africa':PAL['ssa'],
       'fsu_central_asia':PAL['fsu']}

shock_baseline = calibrate_price_shock(0.20)
print(f'S3 price shock: +{shock_baseline*100:.0f}%')

def run_scenario(scale, t_max=30):
    orig = copy.deepcopy(ceb.REGIONAL_ECON_PARAMS)
    for rn in ceb.REGIONAL_ECON_PARAMS:
        ceb.REGIONAL_ECON_PARAMS[rn]['eps_F_PF'] = orig[rn]['eps_F_PF'] * scale
    try:
        s3 = EconParams(fert_price_shock=shock_baseline, eps_F_N=0.0)
        return {rn: CoupledMonthlyModel(region=regions[rn], econ=s3, region_key=rn, t_max=t_max).run()
                for rn in REGION_ORDER}
    finally:
        for rn in orig: ceb.REGIONAL_ECON_PARAMS[rn] = orig[rn]

dfs_b = run_scenario(1.0); dfs_h = run_scenario(0.5)

def gl(dfs):
    w = {rn: regions[rn].cropland_mha * get_calibrated_ym(rn) for rn in REGION_ORDER}
    tw = sum(w.values())
    yrs = sorted(dfs[REGION_ORDER[0]]['year'].unique())
    return np.array(yrs), np.array([sum((1-dfs[rn][dfs[rn]['year']==y]['yield_fraction'].iloc[0])*w[rn] for rn in REGION_ORDER)/tw*100 for y in yrs])

yrs_b, loss_b = gl(dfs_b); yrs_h, loss_h = gl(dfs_h)
y10_b = {rn:(1-dfs_b[rn][dfs_b[rn]['year']==10]['yield_fraction'].iloc[0])*100 for rn in REGION_ORDER}
y10_h = {rn:(1-dfs_h[rn][dfs_h[rn]['year']==10]['yield_fraction'].iloc[0])*100 for rn in REGION_ORDER}

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios':[1, 1.15]})
ax_a.fill_between(yrs_b, loss_h, loss_b, color='#888888', alpha=0.18, zorder=1, label='Elasticity range')
ax_a.plot(yrs_b, loss_b, color='#C62828', linewidth=2.2, zorder=3, label='Baseline εF,PF')
ax_a.plot(yrs_h, loss_h, color='#1565C0', linewidth=2.2, zorder=3, linestyle='--', label='Halved εF,PF (0.5×)')
ax_a.set_xlabel('Years after disruption onset'); ax_a.set_ylabel('Global yield loss (%)')
ax_a.set_xlim(0, 30); ax_a.set_ylim(0, max(loss_b.max(), loss_h.max())*1.15)
ax_a.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_a.text(-0.1, 1.05, 'a', transform=ax_a.transAxes, fontsize=11, fontweight='bold', va='top')

y_pos = np.arange(len(REGION_ORDER))
for i, rn in enumerate(REGION_ORDER):
    b, h = y10_b[rn], y10_h[rn]
    ax_b.plot([h, b], [i, i], color='#BBBBBB', linewidth=1.8, zorder=1)
    ax_b.scatter(b, i, s=80, color=COL[rn], zorder=3, edgecolors='black', linewidth=0.5, label='Baseline' if i==0 else None)
    ax_b.scatter(h, i, s=80, facecolors='white', edgecolors=COL[rn], linewidth=1.8, zorder=3, label='Halved (0.5×)' if i==0 else None)
    ax_b.text(b+0.3, i, f'{b:.1f}%', va='center', ha='left', fontsize=7, color='#333333')
    ax_b.text(h-0.3, i, f'{h:.1f}%', va='center', ha='right', fontsize=7, color='#666666')
ax_b.set_yticks(y_pos); ax_b.set_yticklabels([LBL[rn] for rn in REGION_ORDER], fontsize=8)
ax_b.invert_yaxis(); ax_b.set_xlabel('Year-10 yield loss (%)')
ax_b.set_xlim(-1, max(max(y10_b.values()), max(y10_h.values()))*1.25)
ax_b.axvline(0, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
ax_b.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
ax_b.text(-0.18, 1.05, 'b', transform=ax_b.transAxes, fontsize=11, fontweight='bold', va='top')

from scipy.stats import spearmanr
rs, _ = spearmanr([y10_b[rn] for rn in REGION_ORDER], [y10_h[rn] for rn in REGION_ORDER])
fig.text(0.98, 0.02, f'Regional ordering preserved: Spearman ρ = {rs:.2f}', ha='right', fontsize=7.5, style='italic', color='#555555')
plt.tight_layout()
out = PROJ / 'figures' / 'figureS7_econ_sensitivity'
fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out}.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}.png; Spearman ρ={rs:.3f}')
print(f'Global year-10: baseline={loss_b[yrs_b==10][0]:.2f}%, halved={loss_h[yrs_h==10][0]:.2f}%')
