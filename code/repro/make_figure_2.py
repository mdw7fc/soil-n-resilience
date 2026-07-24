#!/usr/bin/env python3
"""Figure 2 — sustained disruption exposes both soil buffering and buffer erosion.

a. Shock-induced year-10 yield change across the within-region SOC gradient,
   measured against a no-shock control at the same SOC level (farm-level
   closure, all eight regions).
b. Total year-10 yield change at regional mean SOC under the canonical S3
   scenario, split into the direct fertilizer-shortfall effect and the
   additional loss caused by SOM depletion.
c. Total year-10 regional yield change against regional mean SOC, bubble area
   scaled by the absolute fertilizer-demand price elasticity.

The decomposition in panel b is computed by re-running S3 with the three
Century carbon pools frozen at their year-0 values, so that mineralization
cannot decline as residue return falls. That counterfactual isolates the loss
attributable to the fertilizer shortfall itself; the residual, total minus
frozen-pool, is the additional loss from SOM depletion.

Writes data/figure2_panels.json and figures/Figure_2_regional_vulnerability.png/.pdf.
"""
import os, sys, json, copy, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_price_shock_analysis import (DATA, ALL_REGIONS, ABBR, patch_era5,
                                      GRADIENT_SOC_PCTS)
from monthly_model_v3 import MonthlyNParams
from soil_n_model import get_default_regions
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params, calibrate_price_shock, REGIONAL_ECON_PARAMS

FIGS = os.path.join(HERE, '..', '..', 'figures')
RO = ALL_REGIONS
COLOR = {'north_america': '#3d3d3d', 'europe': '#2aa198', 'east_asia': '#e08c1a',
         'south_asia': '#4a90d9', 'southeast_asia': '#9b59b6',
         'latin_america': '#2e9e4f', 'sub_saharan_africa': '#e0403a',
         'fsu_central_asia': '#8a6d3b'}
C_DIRECT = '#2c5f7c'
C_SOM = '#c8942e'


class FrozenSOMModel(CoupledMonthlyModel):
    """S3 with the Century pools held at their year-0 values.

    Fertilizer still responds to the price shock and yield still responds to
    fertilizer, but the residue-return feedback cannot draw the SOM pools down,
    so N mineralization stays at its baseline level. The gap between this run
    and the full run is the SOM-depletion contribution.
    """

    def run(self):
        pools0 = (self.bio.C_active, self.bio.C_slow, self.bio.C_passive)
        orig_step = self.bio.step

        def frozen_step(fert_applied, bnf=None):
            out = orig_step(fert_applied, bnf=bnf)
            (self.bio.C_active, self.bio.C_slow, self.bio.C_passive) = pools0
            out['soc_total'] = sum(pools0)
            out['soc_fraction'] = 1.0
            return out

        self.bio.step = frozen_step
        try:
            return super().run()
        finally:
            self.bio.step = orig_step


def canonical_s3():
    s3 = get_scenario_params()['S3']
    s3.fert_price_shock = calibrate_price_shock(0.20)
    return s3


def panel_bc(regions, mp):
    """Total and frozen-pool year-10 losses for the eight regions plus global."""
    s3 = canonical_s3()
    rows, yb = [], []
    for rk in RO:
        r = regions[rk]
        ym = get_calibrated_ym(rk, mp)
        full = CoupledMonthlyModel(region=r, econ=copy.deepcopy(s3), region_key=rk,
                                   t_max=10.0, yield_max_override=ym).run()
        froz = FrozenSOMModel(region=r, econ=copy.deepcopy(s3), region_key=rk,
                              t_max=10.0, yield_max_override=ym).run()
        tot = float((1 - full[full['year'] == 10]['yield_fraction'].iloc[0]) * 100)
        dir_ = float((1 - froz[froz['year'] == 10]['yield_fraction'].iloc[0]) * 100)
        yb.append(float(full[full['year'] == 0]['yield_tha'].iloc[0]))
        rows.append(dict(region=rk, abbr=ABBR[rk], total=tot,
                         direct=dir_, som=max(0.0, tot - dir_),
                         soc=float(r.soc_initial),
                         eps=abs(float(REGIONAL_ECON_PARAMS[rk]['eps_F_PF']))))
    W = np.array([regions[k].cropland_mha for k in RO]) * np.array(yb)
    W /= W.sum()
    g = dict(region='global', abbr='Global',
             total=float(sum(r['total'] * W[i] for i, r in enumerate(rows))),
             direct=float(sum(r['direct'] * W[i] for i, r in enumerate(rows))))
    g['som'] = max(0.0, g['total'] - g['direct'])
    return rows, g


def main():
    patch_era5()
    regions = get_default_regions()
    mp = MonthlyNParams()

    src = os.path.join(DATA, 'figure2_soc_gradient.json')
    if not os.path.exists(src):
        raise SystemExit('missing %s — run run_price_shock_analysis.py first'
                         % src)
    gj = json.load(open(src))
    x = np.array(gj['soc_pcts'], dtype=float)
    grad = {rk: {'soc_pct': x,
                 'yield_pct': -np.array(gj['regions'][rk]['ctrl_penalty'])}
            for rk in RO}
    for rk in RO:
        print("%-20s yr10 gradient 10%%/100%%/200%%: %6.2f %6.2f %6.2f"
              % (rk, grad[rk]['yield_pct'][0],
                 grad[rk]['yield_pct'][list(x).index(100.0)],
                 grad[rk]['yield_pct'][-1]))
    rows, g = panel_bc(regions, mp)
    print("\n%-8s %8s %8s %8s" % ("region", "total", "direct", "SOM"))
    for r in [g] + sorted(rows, key=lambda x: -x['total']):
        print("%-8s %8.2f %8.2f %8.2f" % (r['abbr'], r['total'], r['direct'], r['som']))

    json.dump(dict(gradient={k: dict(soc_pct=list(map(float, v['soc_pct'])),
                                     yield_pct=list(map(float, v['yield_pct'])))
                             for k, v in grad.items()},
                   regions=rows, global_=g),
              open(os.path.join(DATA, 'figure2_panels.json'), 'w'), indent=1)

    fig = plt.figure(figsize=(16.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- panel a
    for rk in RO:
        axa.plot(x, grad[rk]['yield_pct'], lw=2.4, color=COLOR[rk], zorder=3)
    axa.axhline(0, color='0.25', lw=0.9)
    axa.axvline(100, color='0.65', lw=0.9, ls=':')
    axa.axvspan(100, 200, color='#2e7d32', alpha=0.07, zorder=0)
    axa.set_xlim(10, 236)
    axa.set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axa.set_xlabel('Farm SOC (% of regional mean)', fontsize=12)
    axa.set_ylabel('Year-10 shock-induced yield change (%)', fontsize=12)
    lo = min(min(grad[rk]['yield_pct']) for rk in RO)
    axa.set_ylim(lo * 1.10, max(0.4, -lo * 0.05))
    axa.text(97, lo * 1.045, 'Regional\nmean', fontsize=8.5, color='0.45',
             ha='right', va='bottom')
    axa.text(155, lo * 1.02, 'Restoration potential', fontsize=10, style='italic',
             color='#2e7d32', ha='center', va='bottom')
    ends = sorted(((grad[rk]['yield_pct'][-1], rk) for rk in RO), reverse=True)
    span = abs(lo) * 0.062
    prev = None
    for val, rk in ends:
        ypos = val if prev is None else min(val, prev - span)
        prev = ypos
        axa.text(204, ypos, ABBR[rk], fontsize=11.5, fontweight='bold',
                 color=COLOR[rk], va='center')

    # ---- panel b
    order = ['global'] + [r['region'] for r in sorted(rows, key=lambda x: -x['total'])]
    lut = {r['region']: r for r in rows}
    lut['global'] = g
    xs = np.arange(len(order))
    d = [-lut[k]['direct'] for k in order]
    s = [-lut[k]['som'] for k in order]
    axb.bar(xs, d, color=C_DIRECT, width=0.72, label='Direct fertilizer-shortfall effect')
    axb.bar(xs, s, bottom=d, color=C_SOM, width=0.72,
            label='Additional loss from SOM depletion')
    for i, k in enumerate(order):
        axb.text(i, d[i] + s[i] - abs(lo) * 0.018, '%.1f' % (d[i] + s[i]),
                 ha='center', va='top', fontsize=9.5)
    axb.axhline(0, color='0.25', lw=0.9)
    axb.axvline(0.5, color='0.7', lw=0.9, ls=':')
    axb.set_xticks(xs)
    axb.set_xticklabels([lut[k]['abbr'] for k in order], rotation=40, ha='right', fontsize=10.5)
    axb.set_ylabel('Year-10 yield change (%)', fontsize=12)
    axb.set_ylim(min(np.array(d) + np.array(s)) * 1.22, 0.6)
    axb.legend(loc='lower left', fontsize=9.5, frameon=False)

    # ---- panel c
    for r in rows:
        axc.scatter(r['soc'], -r['total'], s=r['eps'] * 2200, color=COLOR[r['region']],
                    alpha=0.75, edgecolor='none', zorder=3)
        axc.annotate(r['abbr'], (r['soc'], -r['total']), textcoords='offset points',
                     xytext=(0, 20), ha='center', fontsize=11.5, fontweight='bold',
                     color=COLOR[r['region']])
    axc.axhline(0, color='0.25', lw=0.9)
    axc.set_xlabel('Regional mean SOC (t C ha$^{-1}$)', fontsize=12)
    axc.set_ylabel('Year-10 yield change (%)', fontsize=12)
    axc.set_xlim(0, 58)
    axc.set_ylim(-max(r['total'] for r in rows) * 1.35, 1.2)
    for e in (0.20, 0.35, 0.50):
        axc.scatter([], [], s=e * 2200, color='0.6', alpha=0.7, label='%.2f' % e)
    axc.legend(title='$|\\varepsilon_{F,P_F}|$', loc='lower right', fontsize=9,
               title_fontsize=10, frameon=True, labelspacing=1.5, borderpad=1.0)

    for ax, tag in ((axa, 'a'), (axb, 'b'), (axc, 'c')):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=10.5)
        ax.text(-0.14, 1.04, tag, transform=ax.transAxes, fontsize=17, fontweight='bold')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS, 'Figure_2_regional_vulnerability.%s' % ext),
                    dpi=300, bbox_inches='tight')
    print("wrote figures/Figure_2_regional_vulnerability.png/.pdf")


if __name__ == '__main__':
    main()
