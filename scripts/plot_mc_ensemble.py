#!/usr/bin/env python3
"""
Plot the Monte Carlo ensemble — Wallenstein-Manning coupled model.

Produces a single-panel SI figure with:

  Panel a: per-region year-1 yield loss (%) at three SOC levels — boxplots
           summarising the joint posterior over 8 priors x 1000 draws.
  Panel b: per-region year-1 gross-margin-over-fertilizer-cost change (%)
           at SOC=100% — boxplots over the joint posterior.
  Panel c: per-region soil-N buffer ratio (low-SOC yield loss minus high-SOC
           yield loss, ppt) — boxplots over the joint posterior.

Reads:  data/mc_ensemble/mc_posterior.csv.gz
Writes: figures/figS_mc_ensemble.png
        figures/figS_mc_ensemble.pdf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_DIR = REPO_ROOT  # alias for legacy refs
ROOT_DIR = REPO_ROOT  # alias for legacy refs
DATA_FILE = PROJECT_DIR / 'data' / 'mc_ensemble' / 'mc_posterior.csv.gz'
FIG_DIR = PROJECT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


REGIONS_ORDERED = [
    ('north_america', 'N America'),
    ('europe', 'Europe'),
    ('east_asia', 'E Asia'),
    ('south_asia', 'S Asia'),
    ('southeast_asia', 'SE Asia'),
    ('latin_america', 'L America'),
    ('sub_saharan_africa', 'SSA'),
    ('fsu_central_asia', 'FSU/CA'),
]
REGIONS = [r[0] for r in REGIONS_ORDERED]
LABELS = [r[1] for r in REGIONS_ORDERED]

SOC_LEVELS = [50, 100, 150]
SOC_COLORS = {50: '#c0392b', 100: '#7f8c8d', 150: '#2980b9'}
SOC_LABELS = {50: 'Low SOC (50%)', 100: 'Mean SOC (100%)', 150: 'High SOC (150%)'}


def boxprops(color):
    return dict(boxprops=dict(facecolor=color, alpha=0.55, edgecolor=color),
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker='o', markersize=2, markerfacecolor=color,
                                markeredgecolor='none', alpha=0.4))


def plot_panel_yield(ax, df):
    """Panel a: yield loss boxplots, three SOC levels per region."""
    n_regions = len(REGIONS)
    width = 0.22
    offsets = {50: -width, 100: 0.0, 150: +width}
    positions_centre = np.arange(n_regions)

    for soc in SOC_LEVELS:
        data = []
        for rn in REGIONS:
            sub = df[(df['region'] == rn) & (df['soc_pct'] == soc)]
            data.append(sub['yield_pen'].values)
        bp = ax.boxplot(
            data,
            positions=positions_centre + offsets[soc],
            widths=width * 0.9,
            patch_artist=True,
            whis=(5, 95),
            showfliers=False,
            **boxprops(SOC_COLORS[soc]),
        )

    ax.set_xticks(positions_centre)
    ax.set_xticklabels(LABELS, rotation=0, fontsize=9)
    ax.set_ylabel('Year-1 yield loss (%)')
    ax.set_title('a   Year-1 yield loss across the joint-prior ensemble',
                 loc='left', fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(10, ax.get_ylim()[1]))

    handles = [Patch(facecolor=SOC_COLORS[soc], edgecolor=SOC_COLORS[soc],
                     alpha=0.55, label=SOC_LABELS[soc])
               for soc in SOC_LEVELS]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=8)


def plot_panel_profit(ax, df):
    """Panel b: gross-margin boxplots at SOC=100%."""
    sub100 = df[df['soc_pct'] == 100]
    data = [sub100[sub100['region'] == rn]['profit_chg'].values for rn in REGIONS]
    color = '#34495e'
    bp = ax.boxplot(
        data,
        positions=np.arange(len(REGIONS)),
        widths=0.55,
        patch_artist=True,
        whis=(5, 95),
        showfliers=False,
        **boxprops(color),
    )
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(np.arange(len(REGIONS)))
    ax.set_xticklabels(LABELS, rotation=0, fontsize=9)
    ax.set_ylabel('Gross margin over fertilizer cost,\nyear-1 change (%)')
    ax.set_title('b   Year-1 gross-margin-over-fertilizer-cost change at mean SOC',
                 loc='left', fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)


def plot_panel_buffer(ax, df):
    """Panel c: per-region buffer ratio (low-SOC yield loss minus high-SOC yield loss)."""
    data = []
    for rn in REGIONS:
        s50 = df[(df['region'] == rn) & (df['soc_pct'] == 50)].set_index('draw')['yield_pen']
        s150 = df[(df['region'] == rn) & (df['soc_pct'] == 150)].set_index('draw')['yield_pen']
        common = s50.index.intersection(s150.index)
        diff = (s50.loc[common] - s150.loc[common]).values
        data.append(diff)
    color = '#27ae60'
    bp = ax.boxplot(
        data,
        positions=np.arange(len(REGIONS)),
        widths=0.55,
        patch_artist=True,
        whis=(5, 95),
        showfliers=False,
        **boxprops(color),
    )
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(np.arange(len(REGIONS)))
    ax.set_xticklabels(LABELS, rotation=0, fontsize=9)
    ax.set_ylabel('Soil-N yield buffer\n(low minus high SOC, ppt)')
    ax.set_title('c   Soil-N buffer ratio — yield loss avoided on high-SOC vs low-SOC farms',
                 loc='left', fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)


def main():
    if not DATA_FILE.exists():
        sys.exit(f'Missing posterior file: {DATA_FILE}')
    df = pd.read_csv(DATA_FILE)
    n_draws = df['draw'].nunique()

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.5))
    fig.subplots_adjust(top=0.93, hspace=0.45, left=0.10, right=0.97, bottom=0.05)
    plot_panel_yield(axes[0], df)
    plot_panel_profit(axes[1], df)
    plot_panel_buffer(axes[2], df)

    fig.suptitle(
        f'Monte Carlo ensemble (n={n_draws} joint draws over 8 priors)',
        fontsize=12, y=0.985,
    )

    out_png = FIG_DIR / 'figS_mc_ensemble.png'
    out_pdf = FIG_DIR / 'figS_mc_ensemble.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Wrote: {out_png}')
    print(f'Wrote: {out_pdf}')


if __name__ == '__main__':
    main()
