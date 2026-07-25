#!/usr/bin/env python3
"""Supplementary Figure S10 - monthly N capture efficiency (NUE) as a buffering lever.

Canonical S3 scenario (eps_F_N = 0 centrally), ERA5 climate, years 0-10.

  a) global production-weighted yield-loss trajectory for NUE in
     {0.45, 0.55, 0.65, 0.75 (default), 0.85, 0.95}
  b) regional year-10 yield loss, NUE = 0.45 vs NUE = 0.65

Writes figures/Figure_S10_nue_sensitivity.png/.pdf and
data/figS10_nue_sensitivity.json.

New in deposit v1.3: this generator was previously outside the deposit, and it
now runs on the ERA5 forcing used by every other result in the paper (the
earlier version ran on the model's built-in expert climate profiles).
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = SCRIPT_DIR.parent
ROOT_DIR = CODE_DIR.parent
MODEL_DIR = CODE_DIR / 'model'
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

DATA = ROOT_DIR / 'data'
FIGS = ROOT_DIR / 'figures'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from soil_n_model import get_default_regions
from run_canonical import patch_era5_climate
from coupled_econ_biophysical import get_scenario_params
from coupled_monthly import (
    CoupledMonthlyModel, get_calibrated_ym,
    MonthlyNParams, clear_ym_cache,
)

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
T_MAX = 10.0
NUE_VALUES = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
NUE_DEFAULT = 0.75

REGION_LABELS = {
    'north_america':      'North America',
    'europe':             'Europe',
    'east_asia':          'East Asia',
    'south_asia':         'South Asia',
    'southeast_asia':     'Southeast Asia',
    'latin_america':      'Latin America',
    'sub_saharan_africa': 'Sub-Saharan Africa',
    'fsu_central_asia':   'FSU/Central Asia',
}

# Palette matches Figure 1 / Figure 2 in the manuscript
REGION_COLORS = {
    'north_america':      '#3F3F3F',
    'europe':             '#4DA0A0',
    'east_asia':          '#DD8B33',
    'south_asia':         '#5A8FC4',
    'southeast_asia':     '#8B5DA0',
    'latin_america':      '#5DA56A',
    'sub_saharan_africa': '#D04545',
    'fsu_central_asia':   '#8B6B45',
}


# ─────────────────────────────────────────────────────────────────
# Run sweep
# ─────────────────────────────────────────────────────────────────
def run_sweep(regions, s3, nue_values, t_max=T_MAX):
    """Run a coupled S3 simulation at each NUE value for every region.

    Recalibrates yield_max for each NUE setting (max_uptake_frac is
    a calibration parameter, so ym depends on it).
    """
    trajectories = {nue: {} for nue in nue_values}
    for nue in nue_values:
        mp = MonthlyNParams(max_uptake_frac=nue)
        clear_ym_cache()
        for rn, r in regions.items():
            ym = get_calibrated_ym(rn, mp)
            model = CoupledMonthlyModel(
                r, s3, region_key=rn,
                t_max=t_max, yield_max_override=ym,
            )
            model.bio.mp = mp
            trajectories[nue][rn] = model.run()
    clear_ym_cache()
    return trajectories


def global_loss_trajectory(traj, regions, weights, nue):
    """Production-weighted global yield-loss trajectory at NUE level.

    `weights[rn]` is cropland × y0 (production weight). The aggregate
    is the production-weighted mean of the per-region fractional yield
    loss; do NOT multiply yield levels by `weights[rn]` again, because
    that introduces a second y0 factor and biases the loss downward
    (Wallenstein review, May 2026).
    """
    region_list = list(regions.keys())
    yrs = traj[nue][region_list[0]]['year'].values
    tw = sum(weights.values())
    losses = []
    for y in yrs:
        # Per-region fractional loss at year y
        loss_pct_per_region = []
        for rn in region_list:
            y0 = traj[nue][rn][traj[nue][rn]['year'] == 0]['yield_tha'].iloc[0]
            yt = traj[nue][rn][traj[nue][rn]['year'] == y]['yield_tha'].iloc[0]
            loss_pct_per_region.append((y0 - yt) / y0 * 100 if y0 > 0 else 0.0)
        # Production-weighted average across regions
        weighted = sum(
            loss_pct_per_region[i] * weights[rn]
            for i, rn in enumerate(region_list)
        ) / tw
        losses.append(weighted)
    return yrs, np.array(losses)


# ─────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────
def render_figure(traj, regions, weights, out_path):
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11, 4.2),
        gridspec_kw={'width_ratios': [1, 1.1]},
    )
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    })

    # Panel a: global yield-loss trajectories
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(NUE_VALUES)))
    for i, nue in enumerate(NUE_VALUES):
        yrs, loss = global_loss_trajectory(traj, regions, weights, nue)
        axA.plot(yrs, loss, color=cmap[i], linewidth=2.0,
                 label=f'NUE = {nue:.2f}', zorder=3,
                 marker='o', markersize=3.5)
    axA.set_xlabel('Years after disruption onset')
    axA.set_ylabel('Global yield loss (%)')
    axA.set_xlim(0, 10)
    axA.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
    axA.text(-0.08, 1.05, 'a', transform=axA.transAxes,
             fontsize=11, fontweight='bold', va='top')
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    axA.grid(axis='y', color='#DDDDDD', linewidth=0.5, alpha=0.7, zorder=0)

    # Panel b: regional dumbbell NUE = 0.45 vs NUE = 0.65 at year 10
    y10_low = {
        rn: (1 - traj[0.45][rn][traj[0.45][rn]['year'] == 10]['yield_tha'].iloc[0]
                  / traj[0.45][rn][traj[0.45][rn]['year'] == 0]['yield_tha'].iloc[0]) * 100
        for rn in regions
    }
    y10_high = {
        rn: (1 - traj[0.65][rn][traj[0.65][rn]['year'] == 10]['yield_tha'].iloc[0]
                  / traj[0.65][rn][traj[0.65][rn]['year'] == 0]['yield_tha'].iloc[0]) * 100
        for rn in regions
    }
    region_order = sorted(regions.keys(), key=lambda r: -y10_low[r])

    for i, rn in enumerate(region_order):
        lo, hi = y10_low[rn], y10_high[rn]
        axB.plot([hi, lo], [i, i], color='#BBBBBB',
                 linewidth=1.8, zorder=1)
        axB.scatter(lo, i, s=85, color=REGION_COLORS[rn], zorder=3,
                    edgecolors='black', linewidth=0.5,
                    label='NUE = 0.45' if i == 0 else None)
        axB.scatter(hi, i, s=85, facecolors='white',
                    edgecolors=REGION_COLORS[rn], linewidth=1.8, zorder=3,
                    label='NUE = 0.65' if i == 0 else None)
        axB.text(lo + 0.6, i, f'{lo:.1f}%', va='center', ha='left',
                 fontsize=7, color='#333333')
        axB.text(hi - 0.6, i, f'{hi:.1f}%', va='center', ha='right',
                 fontsize=7, color='#666666')

    axB.set_yticks(range(len(region_order)))
    axB.set_yticklabels([REGION_LABELS[rn] for rn in region_order], fontsize=8)
    axB.invert_yaxis()
    axB.set_xlabel('Year-10 yield loss (%)')
    axB.set_xlim(-1, max(y10_low.values()) * 1.25)
    axB.axvline(0, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
    axB.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
    axB.text(-0.18, 1.05, 'b', transform=axB.transAxes,
             fontsize=11, fontweight='bold', va='top')
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{out_path}.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(f'{out_path}.pdf',
                bbox_inches='tight', facecolor='white')

    # Console summary for SSA (the key headline)
    ssa_lo = y10_low['sub_saharan_africa']
    ssa_hi = y10_high['sub_saharan_africa']
    print(
        f"SSA: NUE = 0.45 → {ssa_lo:.2f}%; NUE = 0.65 → {ssa_hi:.2f}% "
        f"({(1 - ssa_hi / ssa_lo) * 100:.0f}% reduction)"
    )


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    patch_era5_climate()
    regions = get_default_regions()
    s3 = get_scenario_params()['S3']

    print(f"Running NUE sweep with S3 (ε_F,N = {s3.eps_F_N})...")
    traj = run_sweep(regions, s3, NUE_VALUES)
    print(f"Sweep done in {time.time() - t0:.1f}s")

    # Production weights based on default-NUE baseline
    weights = {
        rn: regions[rn].cropland_mha
            * traj[NUE_DEFAULT][rn]['yield_tha'].iloc[0]
        for rn in regions
    }

    out = FIGS / 'Figure_S10_nue_sensitivity'
    render_figure(traj, regions, weights, out)

    dump = {'nue_values': NUE_VALUES, 'global_trajectory': {}, 'regional_year10': {}}
    for nue in NUE_VALUES:
        yrs, loss = global_loss_trajectory(traj, regions, weights, nue)
        dump['global_trajectory'][str(nue)] = {
            'year': [float(y) for y in yrs], 'loss_pct': [float(v) for v in loss]}
        dump['regional_year10'][str(nue)] = {
            rn: float((1 - traj[nue][rn][traj[nue][rn]['year'] == 10]['yield_tha'].iloc[0]
                       / traj[nue][rn][traj[nue][rn]['year'] == 0]['yield_tha'].iloc[0]) * 100)
            for rn in regions}
    json.dump(dump, open(DATA / 'figS10_nue_sensitivity.json', 'w'), indent=1)
    print("Saved figures/Figure_S10_nue_sensitivity.png/.pdf and data/figS10_nue_sensitivity.json")


if __name__ == '__main__':
    main()
