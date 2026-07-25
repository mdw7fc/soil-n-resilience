#!/usr/bin/env python3
"""Figure S7 — farm-level resilience under halved fertilizer-demand elasticities.

a. Year-1 yield change versus farm SOC (10-200% of the regional mean) under a
   100% fertilizer price spike, with baseline regional price elasticities of
   fertilizer demand (solid) and with every eps_F_PF halved (dashed).
b. Year-1 change in crop revenue net of nitrogen-fertilizer expenditure over
   the same gradient and the same two elasticity treatments.

The farm-level calculation is identical to the one behind main-text Figure 1
(run_price_shock_analysis.farm_sweep_single, 100% price spike, SOC gradient
built by rescaling the spun-up Century pools). The only difference between the
two treatments is eps_F_PF, which is read from REGIONAL_ECON_PARAMS at call
time and is temporarily halved here.

Writes data/figS7_farm_elasticity_gradient.json and
figures/Figure_S7_farm_elasticity_gradient.png/.pdf.

Added to the deposit in v1.4: Figure S7 was previously produced from a working
script held outside the archive and could not be regenerated from it.
"""
import os
import sys
import json
import copy
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'model'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from monthly_model_v3 import MonthlyNParams
from coupled_monthly import get_calibrated_ym
from coupled_econ_biophysical import REGIONAL_ECON_PARAMS
from soil_n_model import get_default_regions

from run_price_shock_analysis import (DATA, KEY4, LABEL, FINE_SOC_PCTS,
                                      PRICE_SHOCK_FINE, farm_sweep_single,
                                      patch_era5)

FIGS = os.path.join(HERE, '..', '..', 'figures')

PANEL_REGIONS = ['latin_america', 'north_america', 'south_asia',
                 'sub_saharan_africa']
COLOR = {'latin_america': '#1a7f37', 'north_america': '#3d3d3d',
         'south_asia': '#1f77d0', 'sub_saharan_africa': '#d62728'}


def sweep(halved):
    """Farm gradient for the four cost-structure regions.

    halved=True halves every regional eps_F_PF for the duration of the sweep.
    """
    saved = copy.deepcopy(REGIONAL_ECON_PARAMS)
    if halved:
        for rn, rp in REGIONAL_ECON_PARAMS.items():
            rp['eps_F_PF'] = rp.get('eps_F_PF', -0.20) * 0.5
    try:
        mp = MonthlyNParams()
        regions = get_default_regions()
        out = {}
        for rn in PANEL_REGIONS:
            r = regions[rn]
            ym = get_calibrated_ym(rn, mp)
            rec = dict(yield_pen=[], fert_red=[], margin_chg=[])
            for soc_pct in FINE_SOC_PCTS:
                res = farm_sweep_single(r, rn, ym, mp, soc_pct,
                                        PRICE_SHOCK_FINE)
                rec['yield_pen'].append(float(res['yield_pen']))
                rec['fert_red'].append(float(res['fert_red']))
                rec['margin_chg'].append(float(res['margin_chg']))
            out[rn] = rec
    finally:
        REGIONAL_ECON_PARAMS.clear()
        REGIONAL_ECON_PARAMS.update(saved)
    return out


def main():
    patch_era5()
    print('sweeping baseline elasticities ...')
    base = sweep(False)
    print('sweeping halved elasticities ...')
    half = sweep(True)

    x = np.array(FINE_SOC_PCTS, dtype=float)
    payload = {
        'price_shock_frac': PRICE_SHOCK_FINE,
        'soc_pcts': [int(v) for v in FINE_SOC_PCTS],
        'regions': {rn: {'baseline': base[rn], 'halved': half[rn]}
                    for rn in PANEL_REGIONS},
    }
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, 'figS7_farm_elasticity_gradient.json')
    with open(dst, 'w') as f:
        json.dump(payload, f, indent=1)
    print('wrote data/figS7_farm_elasticity_gradient.json')

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.0, 5.9))
    series = {
        'yield': {rn: (-np.array(base[rn]['yield_pen']),
                       -np.array(half[rn]['yield_pen']))
                  for rn in PANEL_REGIONS},
        'margin': {rn: (np.array(base[rn]['margin_chg']),
                        np.array(half[rn]['margin_chg']))
                   for rn in PANEL_REGIONS},
    }

    for ax, key, ylab, tag in ((axa, 'yield', 'Yield change (%)', 'a'),
                               (axb, 'margin',
                                'Net revenue after N expenditure change (%)',
                                'b')):
        for rn in PANEL_REGIONS:
            b, h = series[key][rn]
            ax.plot(x, b, lw=2.4, color=COLOR[rn], zorder=3)
            ax.plot(x, h, lw=2.0, ls='--', color=COLOR[rn], alpha=0.85,
                    zorder=3)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axvline(100, color='0.65', lw=0.9, ls=':')
        ax.axvspan(100, 200, color='#2e7d32', alpha=0.06, zorder=0)
        ax.set_xlabel('Farm SOC (% of regional mean)', fontsize=13)
        ax.set_ylabel(ylab, fontsize=13)
        ax.set_xlim(10, 232)
        ax.set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
        ax.tick_params(labelsize=11)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.text(-0.10, 1.04, tag, transform=ax.transAxes, fontsize=17,
                fontweight='bold')
        lo = min(min(series[key][rn][0].min(), series[key][rn][1].min())
                 for rn in PANEL_REGIONS)
        ax.set_ylim(lo * 1.12, max(0.35, -lo * 0.06))
        ends = sorted(((max(series[key][rn][0][-1], series[key][rn][1][-1]),
                        rn) for rn in PANEL_REGIONS), reverse=True)
        span = abs(lo) * 0.075
        prev = None
        for val, rn in ends:
            ypos = val if prev is None else min(val, prev - span)
            prev = ypos
            ax.text(204, ypos, LABEL[rn], fontsize=11.5, fontweight='bold',
                    color=COLOR[rn], va='center')

    axa.legend(handles=[Line2D([], [], color='0.35', lw=2.4,
                               label='Baseline elasticities'),
                        Line2D([], [], color='0.35', lw=2.0, ls='--',
                               label='Halved elasticities')],
               loc='lower left', fontsize=11, frameon=True)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS,
                                 'Figure_S7_farm_elasticity_gradient.%s' % ext),
                    dpi=300, bbox_inches='tight')
    print('wrote figures/Figure_S7_farm_elasticity_gradient.png/.pdf')

    i10, i50, i100, i200 = (FINE_SOC_PCTS.index(v) for v in (10, 50, 100, 200))
    print('\n%-20s %-9s %8s %8s %8s %8s' % ('region', 'series', 'soc10',
                                            'soc50', 'soc100', 'soc200'))
    for rn in PANEL_REGIONS:
        for key in ('yield', 'margin'):
            for j, tag in ((0, 'baseline'), (1, 'halved')):
                v = series[key][rn][j]
                print('%-20s %-9s %8.2f %8.2f %8.2f %8.2f'
                      % ('%s/%s' % (rn, key), tag, v[i10], v[i50], v[i100],
                         v[i200]))
    print('\nSSA yield improvement, 10%% -> 200%% SOC (baseline): %.2f pp'
          % (series['yield']['sub_saharan_africa'][0][i200]
             - series['yield']['sub_saharan_africa'][0][i10]))
    gaps = [series['margin'][rn][0] - series['margin'][rn][1]
            for rn in PANEL_REGIONS]
    print('net-revenue penalty from halving (pp deeper), range over regions '
          'and SOC: %.2f to %.2f' % (min(g.min() for g in gaps),
                                     max(g.max() for g in gaps)))
    for rn in PANEL_REGIONS:
        g = series['margin'][rn][0] - series['margin'][rn][1]
        print('  %-20s %5.2f - %5.2f pp' % (rn, g.min(), g.max()))
    print('\nSOC 50%% vs 100%% gap (yield, pp):')
    for rn in PANEL_REGIONS:
        b, h = series['yield'][rn]
        print('  %-20s baseline %5.2f   halved %5.2f'
              % (rn, b[i100] - b[i50], h[i100] - h[i50]))


if __name__ == '__main__':
    main()
