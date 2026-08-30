#!/usr/bin/env python3
"""Figure 1 — farm-level yield and net-revenue buffering under a 100% spike.

a. Year-1 yield change versus farm SOC (10-200% of the regional mean) for the
   four regions with farm-gate cost-structure data.
b. Year-1 change in crop revenue net of nitrogen-fertilizer expenditure.

Reads data/figure1_farm_gradient.json, written by run_price_shock_analysis.py.
Writes data/figure1_soc_gradient.csv and
figures/Figure_1_farm_buffering.png/.pdf.
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_price_shock_analysis import DATA, KEY4, LABEL

FIGS = os.path.join(HERE, '..', '..', 'figures')

PANEL_REGIONS = ['latin_america', 'north_america', 'south_asia',
                 'sub_saharan_africa']
COLOR = {'latin_america': '#1a7f37', 'north_america': '#3d3d3d',
         'south_asia': '#1f77d0', 'sub_saharan_africa': '#d62728'}


def main():
    src = os.path.join(DATA, 'figure1_farm_gradient.json')
    if not os.path.exists(src):
        raise SystemExit('missing %s — run run_price_shock_analysis.py first'
                         % src)
    d = json.load(open(src))
    x = np.array(d['soc_pcts'], dtype=float)
    res = {}
    for rk in PANEL_REGIONS:
        r = d['regions'][rk]
        res[rk] = {'yield_pct': -np.array(r['yield_pen']),
                   'margin_pct': np.array(r['margin_chg']),
                   'fert_pct': -np.array(r['fert_red'])}

    with open(os.path.join(DATA, 'figure1_soc_gradient.csv'), 'w') as f:
        f.write('region,soc_pct_of_regional_mean,yield_change_pct,'
                'margin_change_pct,fert_change_pct\n')
        for rk in PANEL_REGIONS:
            v = res[rk]
            for i, s in enumerate(x):
                f.write('%s,%.0f,%.4f,%.4f,%.4f\n'
                        % (rk, s, v['yield_pct'][i], v['margin_pct'][i],
                           v['fert_pct'][i]))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, key, ylab, tag in ((axa, 'yield_pct', 'Yield change (%)', 'a'),
                               (axb, 'margin_pct',
                                'Net revenue after N expenditure change (%)',
                                'b')):
        for rk in PANEL_REGIONS:
            ax.plot(x, res[rk][key], lw=2.6, color=COLOR[rk], zorder=3)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axvline(100, color='0.65', lw=0.9, ls=':')
        ax.axvspan(100, 200, color='#2e7d32', alpha=0.06, zorder=0)
        ax.set_xlabel('Farm SOC (% of regional mean)', fontsize=13)
        ax.set_ylabel(ylab, fontsize=13)
        ax.set_xlim(10, 232)
        ax.set_xticks([50, 100, 150, 200])
        ax.tick_params(labelsize=11)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.text(-0.11, 1.04, tag, transform=ax.transAxes, fontsize=17,
                fontweight='bold')
        lo = min(res[rk][key].min() for rk in PANEL_REGIONS)
        ax.set_ylim(lo * 1.12, max(0.35, -lo * 0.06))
        ax.text(97, lo * 1.06, 'Regional\nmean', fontsize=9, color='0.45',
                ha='right', va='bottom')
        ax.text(152, lo * 1.06, 'Restoration potential', fontsize=12,
                style='italic', color='#2e7d32', ha='center', va='bottom')
        ends = sorted(((res[rk][key][-1], rk) for rk in PANEL_REGIONS),
                      reverse=True)
        span = abs(lo) * 0.075
        prev = None
        for val, rk in ends:
            ypos = val if prev is None else min(val, prev - span)
            prev = ypos
            ax.text(204, ypos, LABEL[rk].replace(' ', '\n', 1), fontsize=11.5,
                    fontweight='bold', color=COLOR[rk], va='center')

    fig.text(0.5, 0.015,
             'Net revenue equals crop revenue minus nitrogen-fertilizer '
             'expenditure; it is not whole-farm gross margin. SOC effects are '
             'within-region contrasts; cross-region contrasts also reflect '
             'regional prices and demand elasticities.',
             ha='center', fontsize=10.5, style='italic', color='0.35')
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    os.makedirs(FIGS, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS, 'Figure_1_farm_buffering.%s' % ext),
                    dpi=300, bbox_inches='tight')
    print('wrote figures/Figure_1_farm_buffering.png/.pdf')

    i50, i100, i200 = (list(x).index(v) for v in (50.0, 100.0, 200.0))
    for rk in PANEL_REGIONS:
        v = res[rk]
        print('%-20s yield %6.2f %6.2f %6.2f   net rev %7.2f %7.2f %7.2f'
              % (rk, v['yield_pct'][i50], v['yield_pct'][i100],
                 v['yield_pct'][i200], v['margin_pct'][i50],
                 v['margin_pct'][i100], v['margin_pct'][i200]))


if __name__ == '__main__':
    main()
