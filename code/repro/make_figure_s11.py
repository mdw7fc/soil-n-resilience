#!/usr/bin/env python3
"""Supplementary Figure S11 — the SOC gradient widens with price-shock severity.

Year-1 farm-level yield loss across fertilizer price increases of 0-300%, at
four within-region SOC levels (25, 50, 75 and 100% of the regional mean), for
the four focal regions used in main Figure 1.

Uses the same farm-level closure as Figure 1 (`farm_sweep_single` in
run_price_shock_analysis.py), so the 100%-price-increase column reproduces the
Figure 1 curves exactly.

Writes data/figS11_severity_sweep.json and
figures/Figure_S11_severity_gradient.png/.pdf.

New in deposit v1.3: this generator was previously outside the deposit.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_price_shock_analysis import (DATA, KEY4, patch_era5, farm_sweep_single)
from monthly_model_v3 import MonthlyNParams
from soil_n_model import get_default_regions
from coupled_monthly import get_calibrated_ym

FIGS = os.path.join(HERE, '..', '..', 'figures')

SHOCKS = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
SOC_LEVELS = [100, 75, 50, 25]
SOC_DISPLAY = ['100% (regional mean)', '75%', '50%', '25%']
SOC_BLUES = ['#1a4c6e', '#2E86AB', '#7FB3D3', '#BDD7E7']
LABELS = {'sub_saharan_africa': 'Sub-Saharan Africa', 'south_asia': 'South Asia',
          'latin_america': 'Latin America', 'north_america': 'North America'}


def compute():
    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()
    out = {}
    for rn in KEY4:
        r = regions[rn]
        ym = get_calibrated_ym(rn, mp)
        tab = {}
        for soc in SOC_LEVELS:
            tab[str(soc)] = [farm_sweep_single(r, rn, ym, mp, soc, ps)['yield_pen']
                             for ps in SHOCKS]
        out[rn] = tab
        sp = [tab['25'][i] - tab['100'][i] for i in range(len(SHOCKS))]
        print("%-20s spread(25-100%%) @100%%=%.2f @150%%=%.2f @300%%=%.2f pp"
              % (rn, sp[SHOCKS.index(1.00)], sp[SHOCKS.index(1.50)],
                 sp[SHOCKS.index(3.00)]))
    return out


def main():
    out = compute()
    json.dump(dict(shocks=SHOCKS, soc_levels=SOC_LEVELS, regions=out),
              open(os.path.join(DATA, 'figS11_severity_sweep.json'), 'w'), indent=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    ymax_global = max(max(v) for rn in KEY4 for v in out[rn].values()) * 1.05
    x = np.array(SHOCKS) * 100

    for idx, rn in enumerate(KEY4):
        ax = axes_flat[idx]
        for si, (soc, lbl) in enumerate(zip(SOC_LEVELS, SOC_DISPLAY)):
            ax.plot(x, -np.array(out[rn][str(soc)]), color=SOC_BLUES[si],
                    linewidth=2.0, zorder=3,
                    label=('SOC %s' % lbl) if idx == 0 else None)
        ax.axvspan(50, 150, alpha=0.12, color='#888888', zorder=0)
        if idx == 0:
            ax.text(100, -ymax_global * 0.95, 'Typical crisis\nrange', fontsize=7,
                    ha='center', va='bottom', color='#555555', fontstyle='italic')
        ax.set_title(LABELS[rn], fontsize=10, fontweight='bold')
        if idx >= 2:
            ax.set_xlabel('Fertilizer price increase (%)')
        if idx % 2 == 0:
            ax.set_ylabel('Yield change (%)')
        ax.set_xlim(0, 300)
        ax.set_ylim(-ymax_global, 0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.text(-0.08, 1.08, 'abcd'[idx], transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top', ha='right')

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.02), framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS, 'Figure_S11_severity_gradient.%s' % ext),
                    dpi=300, bbox_inches='tight', facecolor='white')
    print("wrote figures/Figure_S11_severity_gradient.png/.pdf")


if __name__ == '__main__':
    main()
