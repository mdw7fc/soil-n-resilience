#!/usr/bin/env python3
"""Supplementary Figure S6 (pairwise regional diagnostics) from the frozen
canonical run. Run run_canonical.py first."""
import os, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, '..', '..', 'data', 'canonical_ERA5_y30.json')))['regions']
AB = [r['abbr'] for r in d]
l1 = np.array([r['loss_yr1'] for r in d]);  l10 = np.array([r['loss_yr10'] for r in d])
soc = np.array([r['soc'] for r in d]);      wdef = np.array([r['water_deficit'] for r in d])
ym = np.array([r['y_max'] for r in d]);     bnf = np.array([r['bnf'] for r in d])
eps = np.abs([r['eps_F_PF'] for r in d]);   buf = np.array([r['buffer_ratio_pct'] for r in d]) / 100
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8})
BLUE = '#1f77b4'
panels = [
 (soc,   l10, 'SOC stock (t C ha$^{-1}$)', 'Year-10 yield penalty (%)'),
 (wdef,  l10, 'Baseline water deficit', 'Year-10 yield penalty (%)'),
 (ym,    l10, 'Calibrated yield ceiling, $y_\\mathrm{max}$ (t ha$^{-1}$)', 'Year-10 yield penalty (%)'),
 (bnf,   l10, 'BNF potential (kg N ha$^{-1}$)', 'Year-10 yield penalty (%)'),
 (eps,   l1,  '|Fertilizer-demand elasticity|', 'Year-1 yield penalty (%)'),
 (buf,   l10, 'Soil N buffer ratio', 'Year-10 yield penalty (%)')]
fig, axes = plt.subplots(2, 3, figsize=(12, 7.4)); axes = axes.flatten()
for i, (x, y, xl, yl) in enumerate(panels):
    ax = axes[i]; rho = stats.spearmanr(x, y)[0]
    ax.scatter(x, y, s=80, c=BLUE, edgecolors='black', linewidth=0.5, zorder=3)
    for j, ab in enumerate(AB):
        ax.annotate(ab, (x[j], y[j]), xytext=(4, 3), textcoords='offset points', fontsize=7, color='black')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if i == 5:
        b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color='#888888', lw=1.0, ls='--', zorder=2)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        ax.text(0.97, 0.95, f'$\\rho$ = {rho:+.2f}  (R$^2$ = {r2:.2f})', transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='#c0392b', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.text(0.97, 0.03, 'Weak cross-regional association reflects opposing mechanisms: LATAM\n'
                'short-run buffering and long-run buffer erosion (see Supplementary Note 3).',
                transform=ax.transAxes, fontsize=6.3, style='italic', color='#555555', ha='right', va='bottom')
    else:
        ax.text(0.97, 0.95, f'$\\rho$ = {rho:+.2f}', transform=ax.transAxes, fontsize=9,
                ha='right', va='top', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='gray', alpha=0.85))
plt.tight_layout()
outdir = os.path.join(HERE, '..', '..', 'figures'); os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, 'Figure_S6_pairwise_diagnostics.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close(); print('wrote figures/Figure_S6_pairwise_diagnostics.png')
