#!/usr/bin/env python3
"""
Broadbalk benchmarking mini-figure (SI).

Four panels:
  A. Modern-period (2000-2015) observed vs modelled grain yield for two
     reportable treatments where contemporary observed means are
     documented in the project validation set: Nil (P3) and N3PKMg (P8).
  B. Modern-period model bias as % of observed for the same two
     treatments.
  C. SOC trajectory for Nil 1843-2015 (Century, MEMS, observed) -
     long-term qualitative directionality benchmark.
  D. SOC trajectory for FYM 1843-2015 (Century, MEMS, observed) -
     long-term qualitative directionality benchmark.

Yield benchmarking is restricted to the modern reference period (Hereward
and contemporary cultivars, 2000-2015) because the model uses a
contemporary yield ceiling (yield_max = 11 t ha^-1). Comparing the model
to a 1843-2015 mean conflates yield-potential evolution with N/SOC
dynamics. The SOC trajectories are kept on the full 1843-2015 horizon
because they are diagnostic of long-term direction and asymptote rather
than of absolute level.

Data sources:
  - SOC trajectories: model/data/benchmark_broadbalk/soc_trajectories_broadbalk.csv
  - Modern observed yields: paper2-soil-resilience/submission/data/
      validation_data_extraction.csv (rows yield_nil_recent,
      yield_n3pkmg_recent), source Rothamsted Broadbalk handout 2024.
  - Modelled values: 2000-2015 means from the Century benchmark run with
      yield_max=11.0, yield_min=0.8, mit_c=0.025 (current calibration).

Outputs:
  figures_regenerated/FigureSx_broadbalk_minifigure.{png,pdf}
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_DIR = REPO_ROOT  # alias for legacy refs
ROOT_DIR = REPO_ROOT  # alias for legacy refs
SOC_CSV = REPO_ROOT / 'data' / 'benchmark_broadbalk' / 'soc_trajectories_broadbalk.csv'
FIG_DIR = PROJECT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Modern-period (2000-2015) yield comparison.
# Observed values are reference-period treatment means extracted from the
# project validation dataset (paper2-soil-resilience/submission/data/
# validation_data_extraction.csv), which summarizes Broadbalk 2000-2022
# treatment means from the Rothamsted Broadbalk handout (2024). Nil is
# reported as "slightly below 1.0" (1.0 used). N3PKMg is reported as
# "~8-9 t/ha" (midpoint 8.5 used; range labelled in panel a).
# Modelled values are 2000-2015 means from a fresh Century run with
# current calibration (yield_max=11.0, yield_min=0.8, mit_c=0.025).
YIELD = pd.DataFrame([
    # treatment label, plot, observed (2000-2015), modelled (2000-2015)
    ('Nil',    'P3', 1.00,  1.48),
    ('N3PKMg', 'P8', 8.50, 10.65),
], columns=['treatment', 'plot', 'obs', 'mod'])

# Reported observed range for N3PKMg from the validation set; used to
# render an obs error bar in panel a so that the precision of the
# observed value is not visually overstated.
OBS_RANGE = {
    'Nil':    (0.85, 1.10),   # "slightly below 1" -> conservative narrow band
    'N3PKMg': (8.0, 9.0),     # "~8-9" reported range
}
YIELD['resid_pct'] = (YIELD['mod'] - YIELD['obs']) / YIELD['obs'] * 100.0

# Treatment colours
COLORS = {
    'Nil':    '#d62728',  # red
    'N3PKMg': '#1f77b4',  # blue
}

# SOC trajectory treatment keys in the source CSV
SOC_KEY_NIL = 'Nil'
SOC_KEY_FYM = 'FYM1843'


def load_soc():
    df = pd.read_csv(SOC_CSV)
    return df


def panel_yield_bars(ax, ydf):
    x = np.arange(len(ydf))
    w = 0.36
    # Observed-range error bars: half-width = (high - low) / 2; centred on
    # the midpoint values used in YIELD['obs']
    obs_err = np.array([
        [(YIELD['obs'].iloc[i] - OBS_RANGE[t][0]) for i, t in enumerate(ydf['treatment'])],
        [(OBS_RANGE[t][1] - YIELD['obs'].iloc[i]) for i, t in enumerate(ydf['treatment'])],
    ])
    ax.bar(x - w/2, ydf['obs'], width=w, color='#888888',
           edgecolor='black', lw=0.5, label='Observed (2000\u20132015)',
           yerr=obs_err, capsize=3,
           error_kw=dict(ecolor='#222222', lw=0.8))
    ax.bar(x + w/2, ydf['mod'], width=w,
           color=[COLORS[t] for t in ydf['treatment']],
           edgecolor='black', lw=0.5, alpha=0.85,
           label='Model (2000\u20132015)')

    # Value labels above each pair
    for xi, (o, m) in enumerate(zip(ydf['obs'], ydf['mod'])):
        ax.text(xi - w/2, o + 0.45, f'{o:.2f}', ha='center', va='bottom',
                fontsize=8, color='#333333')
        ax.text(xi + w/2, m + 0.18, f'{m:.2f}', ha='center', va='bottom',
                fontsize=8, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(plot {p})' for t, p in
                        zip(ydf['treatment'], ydf['plot'])], fontsize=9)
    ax.set_ylabel('Grain yield (t ha$^{-1}$)', fontsize=9)
    ax.set_ylim(0, max(ydf[['obs', 'mod']].values.max() + 1.5, 8))
    ax.set_title('a  Modern-period yield (2000\u20132015)', loc='left',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', frameon=True, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=8)


def panel_residuals(ax, ydf):
    x = np.arange(len(ydf))
    ax.bar(x, ydf['resid_pct'],
           color=[COLORS[t] for t in ydf['treatment']],
           edgecolor='black', lw=0.5, alpha=0.9)
    for xi, v in zip(x, ydf['resid_pct']):
        offs = 2 if v >= 0 else -2
        va = 'bottom' if v >= 0 else 'top'
        ax.text(xi, v + offs, f'{v:+.0f}%', ha='center', va=va,
                fontsize=9, fontweight='bold', color='#222222')

    # Reference at 0
    ax.axhline(0, color='black', lw=0.7)
    # +/- 10% reference band
    ax.axhspan(-10, 10, color='#cccccc', alpha=0.35, lw=0)

    ax.set_xticks(x)
    ax.set_xticklabels([t for t in ydf['treatment']], fontsize=9)
    ax.set_ylabel('Model bias relative to observed (%)', fontsize=9)
    ymax = max(70, ydf['resid_pct'].max() + 18)
    ymin = min(-15, ydf['resid_pct'].min() - 10)
    ax.set_ylim(ymin, ymax)
    ax.set_title('b  Modern-period bias by treatment', loc='left',
                 fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=8)


def panel_soc(ax, df, treatment_key, title):
    sub = df[df['treatment'] == treatment_key].sort_values('year')
    yrs = sub['year'].values
    obs = sub['soc_obs'].values
    century = sub['soc_century'].values
    mems = sub['soc_mems'].values

    ax.plot(yrs, century, color='#1f77b4', lw=1.6, ls='-',
            label='Century', zorder=2)
    ax.plot(yrs, mems, color='#9467bd', lw=1.6, ls='--',
            label='MEMS', zorder=2)
    mask = ~np.isnan(obs)
    ax.scatter(yrs[mask], obs[mask], s=28, color='#222222',
               zorder=4, label='Observed', edgecolor='white', lw=0.6)

    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('SOC (t C ha$^{-1}$)', fontsize=9)
    ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
    ax.set_xlim(1840, 2020)
    ax.legend(fontsize=8, loc='best', frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=8)


def main():
    df = load_soc()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.0))

    panel_yield_bars(axes[0, 0], YIELD)
    panel_residuals(axes[0, 1], YIELD)
    panel_soc(axes[1, 0], df, SOC_KEY_NIL,
              'c  SOC trajectory, Nil (long-term qualitative)')
    panel_soc(axes[1, 1], df, SOC_KEY_FYM,
              'd  SOC trajectory, FYM (long-term qualitative)')

    fig.suptitle(
        'Broadbalk empirical benchmark: contemporary yield response and '
        'long-term SOC behavior',
        fontsize=11, y=0.995,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    png = FIG_DIR / 'FigureSx_broadbalk_minifigure.png'
    pdf = FIG_DIR / 'FigureSx_broadbalk_minifigure.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f'Wrote: {png}')
    print(f'Wrote: {pdf}')


if __name__ == '__main__':
    main()
