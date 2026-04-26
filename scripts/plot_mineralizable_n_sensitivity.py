#!/usr/bin/env python3
"""
Plot the mineralizable-N sensitivity (SI figure).

Reads data/mineralizable_n_sensitivity.pkl and produces a 2x3 grid:
rows = regions (SSA, NA), columns = SOM dimensions (f_active, k_slow,
cn_bulk). Each panel shows three curves of yield change vs SOC% under
the 100% fertilizer-price shock.

Outputs:
  figures_regenerated/FigureSx_mineralizable_n_sensitivity.{png,pdf}
"""

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_DIR = REPO_ROOT  # alias for legacy refs
ROOT_DIR = REPO_ROOT  # alias for legacy refs
DATA_DIR = PROJECT_DIR / 'data'
FIG_DIR = PROJECT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGION_LABELS = {
    'sub_saharan_africa': 'Sub-Saharan Africa',
    'north_america': 'North America',
}

# Group scenarios by dimension
DIMS = [
    ('Initial active-pool fraction (f$_{active}$)',
     [('f$_{active}$ = 0.02', 'f_active=0.02', '#1f77b4'),
      ('baseline (0.04)',     'baseline',       '#000000'),
      ('f$_{active}$ = 0.08', 'f_active=0.08', '#d62728')]),
    ('Slow-pool decay rate (k$_{slow}$)',
     [('k$_{slow}$ x 0.75', 'k_slow x0.75', '#1f77b4'),
      ('baseline',          'baseline',       '#000000'),
      ('k$_{slow}$ x 1.25', 'k_slow x1.25', '#d62728')]),
    ('Bulk soil C:N assumption',
     [('C:N x 0.8', 'cn_bulk x0.8', '#1f77b4'),
      ('baseline',  'baseline',     '#000000'),
      ('C:N x 1.2', 'cn_bulk x1.2', '#d62728')]),
]

REGIONS_TO_PLOT = ['sub_saharan_africa', 'north_america']


def main():
    in_path = DATA_DIR / 'mineralizable_n_sensitivity.pkl'
    with open(in_path, 'rb') as f:
        d = pickle.load(f)
    results = d['results']

    fig, axes = plt.subplots(
        nrows=len(REGIONS_TO_PLOT), ncols=len(DIMS),
        figsize=(11, 6.2), sharex=True, sharey='row',
    )

    for i, region_key in enumerate(REGIONS_TO_PLOT):
        for j, (dim_title, scenarios) in enumerate(DIMS):
            ax = axes[i, j]
            for label, key, color in scenarios:
                r = results[region_key][key]
                soc = np.array(r['soc_pct'])
                # yield_pen is positive = loss; convert to yield change (down=bad)
                yc = -np.array(r['yield_pen'])
                lw = 2.2 if key == 'baseline' else 1.4
                ls = '-' if key == 'baseline' else '--'
                ax.plot(soc, yc, color=color, lw=lw, ls=ls, label=label)

            # Reference lines
            ax.axvline(100, color='gray', ls=':', lw=0.7, alpha=0.6)
            ax.axhline(0, color='gray', lw=0.5, alpha=0.5)

            # Title (top row only)
            if i == 0:
                ax.set_title(dim_title, fontsize=10, pad=6)

            # Y label (left column only)
            if j == 0:
                ax.set_ylabel(f'{REGION_LABELS[region_key]}\nYield change (%)',
                              fontsize=9)

            # X label (bottom row only)
            if i == len(REGIONS_TO_PLOT) - 1:
                ax.set_xlabel('Farm SOC (% of regional mean)', fontsize=9)

            # Legend on each panel (small)
            ax.legend(fontsize=7, loc='lower right', frameon=True,
                      framealpha=0.95, handlelength=2.0, handletextpad=0.5)

            ax.tick_params(labelsize=8)
            ax.set_xlim(8, 202)
            ax.grid(True, alpha=0.25, lw=0.4)

    fig.suptitle('Sensitivity of the SOC-buffering result to '
                 'mineralizable-N parameter assumptions',
                 fontsize=11, y=0.995)
    caption = (
        'Supplementary Figure Sx. Curves show year-1 yield change under a 100% '
        'fertilizer-price shock across an SOC gradient from 10\u2013200% of the regional '
        'mean in Sub-Saharan Africa and North America. The direction of the SOC-buffering\n'
        'effect is preserved across all tested parameterizations. The result is effectively '
        'insensitive to the tested initial active-pool fractions and robust to \u00b125% '
        'perturbations in slow-pool turnover. Sensitivity is greatest for the bulk soil C:N\n'
        'assumption, for which a \u00b120% perturbation shifts the SOC = 50% versus '
        'SOC = 100% yield-change gap by approximately \u00b110%.'
    )
    fig.text(0.5, 0.015, caption,
             ha='center', va='bottom', fontsize=8.2, color='#444444')

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])

    png = FIG_DIR / 'FigureSx_mineralizable_n_sensitivity.png'
    pdf = FIG_DIR / 'FigureSx_mineralizable_n_sensitivity.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f'Wrote: {png}')
    print(f'Wrote: {pdf}')

    # Also write a small CSV of the SOC=50 and 100 gaps for quick reference
    rows = ['region,scenario,yp_50,yp_100,gap_50_100,gap_100_150']
    for region_key in REGIONS_TO_PLOT:
        for label, key, _ in [(s[0], s[1], s[2]) for dim in DIMS for s in dim[1]]:
            r = results[region_key][key]
            soc = np.array(r['soc_pct'])
            yp = np.array(r['yield_pen'])
            i50 = np.where(soc == 50)[0][0]
            i100 = np.where(soc == 100)[0][0]
            i150 = np.where(soc == 150)[0][0]
            rows.append(
                f'{region_key},{key},{yp[i50]:.3f},{yp[i100]:.3f},'
                f'{yp[i50]-yp[i100]:.3f},{yp[i100]-yp[i150]:.3f}'
            )
    csv_path = FIG_DIR / 'mineralizable_n_sensitivity_gaps.csv'
    csv_path.write_text('\n'.join(rows) + '\n')
    print(f'Wrote: {csv_path}')


if __name__ == '__main__':
    main()
