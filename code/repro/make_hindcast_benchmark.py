#!/usr/bin/env python3
"""Supp Fig S2: 2022 hindcast under alternative price-elasticity parameterizations.

Holds price shock fixed at +68% and varies regional eps_F_PF by ±50%.
Tests whether regional predictions are robust to elasticity uncertainty.
"""
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'model'))
sys.path.insert(0, str(ROOT / 'code' / 'model' / 'scripts'))

from coupled_monthly import CoupledMonthlyModel, calibrate_price_shock
from coupled_econ_biophysical import EconParams
import coupled_econ_biophysical as ceb
from soil_n_model import get_default_regions

regions = get_default_regions()
hindcast_regions = ['north_america', 'europe', 'south_asia', 'sub_saharan_africa']
labels_short = ['NA', 'EU', 'SA', 'SSA']
labels_full = ['North America', 'Europe', 'South Asia', 'Sub-Saharan Africa']
colors = {'north_america': '#455A64', 'europe': '#00695C',
          'south_asia': '#1565C0', 'sub_saharan_africa': '#C62828'}
obs_df = pd.read_csv(ROOT / 'data' / 'benchmarks' / 'hindcast_observed_2022.csv')
obs_by_region = dict(zip(obs_df['manuscript_region'], -obs_df['change_2021_2022_pct']))
observed = [obs_by_region[x] for x in labels_full]
obs_err = [1.5, 2.0, 3.0, 4.0]
shock_baseline = calibrate_price_shock(0.15)

def run_hindcast(eps_scale):
    orig = copy.deepcopy(ceb.REGIONAL_ECON_PARAMS)
    for rn in ceb.REGIONAL_ECON_PARAMS:
        ceb.REGIONAL_ECON_PARAMS[rn]['eps_F_PF'] = orig[rn]['eps_F_PF'] * eps_scale
    try:
        s3 = EconParams(fert_price_shock=shock_baseline, eps_F_N=0.0)
        preds = []
        for rn in hindcast_regions:
            r = regions[rn]
            m_b = CoupledMonthlyModel(region=r, econ=EconParams(fert_price_shock=0.0, eps_F_N=0.0), region_key=rn, t_max=5.0)
            df_b = m_b.run()
            fb = df_b[df_b['year']==1]['fert_applied_kgha'].iloc[0]
            m = CoupledMonthlyModel(region=r, econ=s3, region_key=rn, t_max=5.0)
            df = m.run()
            fs = df[df['year']==1]['fert_applied_kgha'].iloc[0]
            preds.append((1 - fs/fb)*100 if fb>0 else 0)
        return preds
    finally:
        for rn in orig: ceb.REGIONAL_ECON_PARAMS[rn] = orig[rn]

scenarios = [('Halved (0.5×)', 0.5), ('Baseline (1.0×)', 1.0), ('Doubled (2.0×)', 2.0)]
all_preds = {name: run_hindcast(s) for name, s in scenarios}
records = []
for scenario_name, _ in scenarios:
    for region_name, prediction, observation in zip(
        labels_full, all_preds[scenario_name], observed
    ):
        records.append({
            "scenario": scenario_name,
            "region": region_name,
            "predicted_reduction_pct": prediction,
            "observed_reduction_pct": observation,
        })
pd.DataFrame(records).to_csv(
    ROOT / "data" / "benchmarks" / "hindcast_benchmark_sol.csv",
    index=False,
)

def r2_vs11(p,o):
    p=np.array(p); o=np.array(o)
    return 1 - np.sum((o-p)**2) / np.sum((o-o.mean())**2)
def rank_rho(p,o):
    from scipy.stats import spearmanr
    return spearmanr(p,o)[0]

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
for ax, (name, _) in zip(axes, scenarios):
    preds = all_preds[name]
    r2 = r2_vs11(preds, observed)
    rmse = float(np.sqrt(np.mean([(o-p)**2 for o,p in zip(observed, preds)])))
    rs = rank_rho(preds, observed)
    keep = (0, 1, 3)
    rs_ex_sa = rank_rho([preds[i] for i in keep], [observed[i] for i in keep])
    ax.plot([0,40], [0,40], color='gray', linewidth=1, linestyle='--', alpha=0.6, label='1:1 line')
    for i, rn in enumerate(hindcast_regions):
        ax.errorbar(preds[i], observed[i], yerr=obs_err[i], fmt='o', color=colors[rn], markersize=10,
                    capsize=4, capthick=1.5, elinewidth=1.5, markeredgecolor='white', markeredgewidth=0.8)
        off_y = -1.5 if labels_short[i]=='SA' else 0.8
        ax.text(preds[i]+0.8, observed[i]+off_y, labels_full[i], fontsize=7.5, color=colors[rn], fontweight='bold')
    ax.set_title(
        f'{name}\nR² vs 1:1 = {r2:.2f}  |  RMSE = {rmse:.1f} pp  |  '
        f'Spearman ρ = {rs:.2f} (excl. SA {rs_ex_sa:.2f})',
        fontsize=9.0,
    )
    ax.set_xlabel('Model-predicted reduction (%)')
    ax.axhline(0, color="#666666", linewidth=0.7)
    ax.set_xlim(0, 42); ax.set_ylim(-5, 22)
    ax.legend(fontsize=7, loc='upper left')
axes[0].set_ylabel('Observed fertilizer purchase reduction (%)')
plt.tight_layout()
out = ROOT / 'figures' / 'Figure_S4_hindcast_sensitivity'
fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out}.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}.png')
print(f'Saved {ROOT / "data" / "benchmarks" / "hindcast_benchmark_sol.csv"}')
