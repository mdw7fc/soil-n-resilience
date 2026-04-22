#!/usr/bin/env python3
"""
Generate all 6 publication-quality figures for Nature Food Analysis manuscript.

Manuscript figure → data source mapping:
  Fig 1: Farm-level price shock resilience     → price_shock_analysis.pkl
  Fig 2: SOC gradient × shock severity         → price_shock_analysis.pkl
  Fig 3: Vulnerability gradient (econ mediation)→ resilience_monthly.pkl + soc_gradient_fine.pkl
  Fig 4: NUE sensitivity                       → resilience_monthly.pkl
  Fig 5: 2022 hindcast validation              → resilience_monthly.pkl (on-the-fly model run)
  Fig 6: Structural sensitivity (MEMS/Century)  → archive/mems-comparison CSVs

Publication fixes applied (from Figure-by-Figure Critique):
  1. New color palette eliminating SSA/SA collision
  2. Zero-line reference on Fig 1b
  3. "Model overpredicts" annotation on Fig 5
  4. ≥7pt minimum annotation text
  5. Reference lines on Fig 6e (y=1.0, y=2.0)
  6. Bold 10pt panel labels outside plot area
  7. White-background 300 DPI TIFF exports
  8. Source attribution moved from Fig 5 to caption
  9. Fig 2 crisis range shading with label
  10. Fig 4b dumbbell line weight 1.5pt, Fig 4a y-axis cropped to 14%
  11. "Structural SOC effect" annotation on Fig 3a
  12. Standardized regional color palette

Author: Matthew Wallenstein
"""

import sys, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# PATH SETUP
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = SCRIPT_DIR.parent
MODEL_DIR = REPO_DIR / 'model'
sys.path.insert(0, str(MODEL_DIR))

DATA_DIR = REPO_DIR / 'data'
FIG_DIR = REPO_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR = REPO_DIR / 'archive' / 'mems-comparison'

# ═══════════════════════════════════════════════════════════════════════
# PUBLICATION STYLE
# ═══════════════════════════════════════════════════════════════════════

# Standardized regional color palette (fixes SSA/SA collision)
PAL = {
    'ssa':   '#C62828',   # Red (deepened from #C1292E)
    'sa':    '#1565C0',   # Blue (was purple — collision fix)
    'latam': '#2E7D32',   # Green (was orange)
    'na':    '#455A64',   # Slate (was blue)
    'eu':    '#00695C',   # Teal (was light blue)
    'ea':    '#795548',   # Brown (was teal)
    'fsu':   '#6A1B9A',   # Purple (was gray)
    'sea':   '#E65100',   # Orange (was light purple)
    'global': '#4A4A4A',  # Dark gray (unchanged)
}
REGION_COLORS = {
    'north_america': PAL['na'], 'europe': PAL['eu'],
    'east_asia': PAL['ea'], 'south_asia': PAL['sa'],
    'southeast_asia': PAL['sea'], 'latin_america': PAL['latam'],
    'sub_saharan_africa': PAL['ssa'], 'fsu_central_asia': PAL['fsu'],
}

SOC_BLUES = ['#1a4c6e', '#2E86AB', '#7FB3D3', '#BDD7E7']  # dark to light: 100/75/50/25

REGION_ORDER = [
    'north_america', 'europe', 'east_asia', 'south_asia',
    'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia',
]
REGION_LABELS = {
    'north_america': 'North America', 'europe': 'Europe',
    'east_asia': 'East Asia', 'south_asia': 'South Asia',
    'southeast_asia': 'SE Asia', 'latin_america': 'Latin America',
    'sub_saharan_africa': 'Sub-Saharan\nAfrica', 'fsu_central_asia': 'FSU &\nCentral Asia',
}
REGION_LABELS_INLINE = {k: v.replace('\n', ' ') for k, v in REGION_LABELS.items()}

# 4-region subset for Figs 1, 2, 3a, 5
KEY4 = ['sub_saharan_africa', 'south_asia', 'latin_america', 'north_america']
KEY4_LABELS = {
    'sub_saharan_africa': 'Sub-Saharan\nAfrica', 'south_asia': 'South\nAsia',
    'latin_america': 'Latin\nAmerica', 'north_america': 'North\nAmerica',
}

# Nature Food rcParams
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})


def add_panel_label(ax, label, x=-0.08, y=1.08):
    """Add bold panel label (a, b, c...) outside plot area at ≥10pt."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right')


def dodge_labels(labels, min_gap):
    """Given list of (y, text_obj) tuples, adjust y values to maintain min_gap.
    Returns adjusted y values in original order."""
    if not labels:
        return []
    # Sort by y ascending, keeping original index
    idx = sorted(range(len(labels)), key=lambda i: labels[i][0])
    adj_sorted = [labels[i][0] for i in idx]
    # Adjust upward to maintain min_gap
    for k in range(1, len(adj_sorted)):
        if adj_sorted[k] - adj_sorted[k-1] < min_gap:
            adj_sorted[k] = adj_sorted[k-1] + min_gap
    # Map back
    out = [0.0] * len(labels)
    for k, orig_i in enumerate(idx):
        out[orig_i] = adj_sorted[k]
    return out


def save_figure(fig, name, dpi=300):
    """Save figure as PNG, PDF, and white-background TIFF."""
    fig.savefig(FIG_DIR / f'{name}.png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    # TIFF export with white background
    fig.savefig(FIG_DIR / f'{name}.tiff', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='tiff')
    print(f'  Saved {name} (PNG + PDF + TIFF)')


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
print("Loading data...")

with open(DATA_DIR / 'price_shock_analysis.pkl', 'rb') as f:
    psa = pickle.load(f)
farm_results = psa['farm_results']
fine_shock_results = psa['fine_shock_results']
psa_regions = psa['regions']

with open(DATA_DIR / 'resilience_monthly.pkl', 'rb') as f:
    monthly = pickle.load(f)
baseline = monthly['baseline']
buffer_metrics = monthly['buffer_metrics']
degradation = monthly['degradation']
no_shock_baseline = monthly['no_shock_baseline']
nue_sensitivity = monthly['nue_sensitivity']
duration_comparison = monthly['duration_comparison']
supply_constrained = monthly['supply_constrained']
regions = monthly['regions']

# SOC gradient fine data (for Fig 3)
with open(DATA_DIR / 'soc_gradient_fine.pkl', 'rb') as f:
    soc_fine = pickle.load(f)
fine_data = soc_fine['fine_results']

# MEMS comparison data (for Fig 6) from archive CSVs
mems_data = {}
for rn in REGION_ORDER:
    csv_path = ARCHIVE_DIR / f'mems_vs_century_{rn}.csv'
    if csv_path.exists():
        mems_data[rn] = pd.read_csv(csv_path)

mems_dep = pd.read_csv(ARCHIVE_DIR / 'mems_vs_century_dependency.csv')

print("Data loaded.\n")


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

from coupled_monthly import get_calibrated_ym

def compute_global_weighted(region_dfs, var):
    """Production-weighted global average."""
    frames = []
    for rn in REGION_ORDER:
        if rn not in region_dfs:
            continue
        df = region_dfs[rn].copy()
        ym = get_calibrated_ym(rn)
        weight = regions[rn].cropland_mha * ym
        df['weight'] = weight
        df['weighted_val'] = df[var] * weight
        frames.append(df)
    combined = pd.concat(frames)
    grouped = combined.groupby('year').agg(
        weighted_sum=('weighted_val', 'sum'),
        weight_sum=('weight', 'sum'),
    )
    return grouped['weighted_sum'] / grouped['weight_sum']


def disruption_penalty(rn, soc_label):
    """Yield loss at year 10 for given region and SOC level."""
    df_ns = no_shock_baseline[soc_label][rn]
    df_d = degradation[soc_label][rn]
    ns = df_ns[df_ns['year'] == 10]['yield_tha'].iloc[0]
    d = df_d[df_d['year'] == 10]['yield_tha'].iloc[0]
    return (1 - d / ns) * 100 if ns > 0 else 0


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Farm-level price shock resilience (2-panel)
# Panel a: Year-1 yield loss vs farm SOC under 100% price spike
# Panel b: Gross margin impact under same scenario
# ═══════════════════════════════════════════════════════════════════════
print('Generating Figure 1 (price shock farm)...')
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2),
                                  gridspec_kw={'width_ratios': [1, 1]})

_key4_colors = {
    'sub_saharan_africa': PAL['ssa'],
    'south_asia': PAL['sa'],
    'latin_america': PAL['latam'],
    'north_america': PAL['na'],
}

# Collect line endpoints first, then dodge labels to avoid overlap
_a_ends = []  # (rn, soc_pct[-1], yield_chg[-1])
_b_ends = []
for rn in KEY4:
    fs = fine_shock_results[rn]
    soc_pct = np.array(fs['soc_pct'])
    yield_pen = np.array(fs['yield_pen'])
    profit_chg = np.array(fs['profit_chg'])

    # Flip sign: plot yield change as negative (down = worse) for consistency
    yield_chg = -yield_pen
    ax_a.plot(soc_pct, yield_chg, color=_key4_colors[rn], linewidth=2.0, zorder=3)
    ax_b.plot(soc_pct, profit_chg, color=_key4_colors[rn], linewidth=2.0, zorder=3)

    _a_ends.append((rn, soc_pct[-1], yield_chg[-1]))
    _b_ends.append((rn, soc_pct[-1], profit_chg[-1]))

# Dodge labels on panel a (yield change, range ~0 to -3)
_a_y_raw = [e[2] for e in _a_ends]
_a_y_adj = dodge_labels([(y, None) for y in _a_y_raw], min_gap=0.55)
for (rn, xend, _), y_adj in zip(_a_ends, _a_y_adj):
    ax_a.text(xend + 1.5, y_adj, KEY4_LABELS[rn],
              fontsize=7, color=_key4_colors[rn], fontweight='bold', va='center')

# Dodge labels on panel b (gross margin change, larger range)
_b_y_raw = [e[2] for e in _b_ends]
_b_y_adj = dodge_labels([(y, None) for y in _b_y_raw], min_gap=1.4)
for (rn, xend, _), y_adj in zip(_b_ends, _b_y_adj):
    ax_b.text(xend + 1.5, y_adj, KEY4_LABELS[rn],
              fontsize=7, color=_key4_colors[rn], fontweight='bold', va='center')

ax_a.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.3, zorder=0)
ax_a.axvline(100, color='gray', linewidth=0.7, linestyle=':', alpha=0.4, zorder=0)
ax_a.text(101, ax_a.get_ylim()[0] * 0.95, 'Regional\nmean', fontsize=7, color='gray', va='bottom')
ax_a.set_xlabel('Farm SOC (% of regional mean)')
ax_a.set_ylabel('Yield change (%)')
ax_a.set_xlim(15, 130)

# Panel b: gross margin impact with zero-line reference
ax_b.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.3, zorder=0)
ax_b.axvline(100, color='gray', linewidth=0.7, linestyle=':', alpha=0.4, zorder=0)
ax_b.set_xlabel('Farm SOC (% of regional mean)')
ax_b.set_ylabel('Gross margin change (%)')
ax_b.set_xlim(15, 130)

add_panel_label(ax_a, 'a')
add_panel_label(ax_b, 'b')

plt.tight_layout()
save_figure(fig, 'figure_price_shock_farm')
plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: SOC gradient widens with price shock severity (4×4 grid)
# Yield loss curves across 0–300% price increases for 4 SOC levels
# ═══════════════════════════════════════════════════════════════════════
print('Generating Figure 2 (price shock by magnitude)...')

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes_flat = axes.flatten()
panel_labels = ['a', 'b', 'c', 'd']

soc_levels = [100, 75, 50, 25]
soc_display = ['100% (regional mean)', '75%', '50%', '25%']

# First pass: find global y-range across all panels
_all_yield_pens = []
for rn in KEY4:
    fr = farm_results[rn]
    for soc_key in soc_levels:
        _all_yield_pens.extend(fr[soc_key]['yield_penalties'])
ymax_global = max(_all_yield_pens) * 1.05

for idx, rn in enumerate(KEY4):
    ax = axes_flat[idx]
    fr = farm_results[rn]

    for si, (soc_key, soc_lbl) in enumerate(zip(soc_levels, soc_display)):
        data = fr[soc_key]
        price_mults = np.array(data['price_mults']) * 100  # convert to %
        yield_pens = np.array(data['yield_penalties'])
        ax.plot(price_mults, yield_pens, color=SOC_BLUES[si], linewidth=2.0,
                label=f'SOC {soc_lbl}' if idx == 0 else None, zorder=3)

    # Crisis range shading (FIX #9): increased opacity + label
    ax.axvspan(50, 150, alpha=0.12, color='#888888', zorder=0)
    if idx == 0:
        ax.text(100, 0.3, 'Typical crisis\nrange',
                fontsize=7, ha='center', va='bottom', color='#555555', fontstyle='italic')

    ax.set_title(REGION_LABELS_INLINE[rn], fontsize=10, fontweight='bold')
    if idx >= 2:
        ax.set_xlabel('Fertilizer price increase (%)')
    if idx % 2 == 0:
        ax.set_ylabel('Yield loss (%)')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, ymax_global)

    add_panel_label(ax, panel_labels[idx])

# Single legend for all panels
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_figure(fig, 'figure_price_shock_by_magnitude')
plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Vulnerability gradient (economic structure mediates)
# Panel a: Farm-level SOC curves with structural annotation
# Panel b: Buffer ratio scatter with R²
# ═══════════════════════════════════════════════════════════════════════
print('Generating Figure 3 (vulnerability gradient)...')
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2),
                                  gridspec_kw={'width_ratios': [1.15, 1]})

# Panel a: fine SOC gradient curves
_curve_colors = {
    'sub_saharan_africa': PAL['ssa'],
    'south_asia': PAL['sa'],
    'latin_america': PAL['latam'],
    'north_america': PAL['na'],
}

_fig3a_legend = []
for rn in KEY4:
    fr = fine_data[rn]
    xv = np.array(fr['soc_pct'])
    y_total = np.array(fr['total_penalty'])
    y_ctrl = np.array(fr['ctrl_penalty'])

    label_text = REGION_LABELS_INLINE.get(rn, KEY4_LABELS[rn].replace('\n',' '))
    ax_a.plot(xv, y_total, color=_curve_colors[rn], linewidth=2.0, zorder=3,
              label=label_text)
    ax_a.fill_between(xv, y_ctrl, y_total, color=_curve_colors[rn], alpha=0.10, zorder=1)
    ax_a.plot(xv, y_ctrl, color=_curve_colors[rn], linewidth=0.9, linestyle='--',
              alpha=0.5, zorder=2)

# Legend replaces inline end-of-line labels (prevents overlap at convergence)
ax_a.legend(loc='upper right', fontsize=7, framealpha=0.9,
            title='Region', title_fontsize=7, handlelength=1.4)

ax_a.axvline(100, color='gray', linewidth=0.7, linestyle=':', alpha=0.4, zorder=0)
ax_a.text(101, 19, 'Regional\nmean', fontsize=7, color='gray', va='top')

# Legend for line types — placed in lower-left where curves are empty at high SOC
ax_a.text(0.03, 0.32,
          'Solid: total penalty (vs healthy farm, no shock)\n'
          'Dashed: shock penalty only\n'
          'Shaded: structural SOC effect',
          transform=ax_a.transAxes, fontsize=6.5, va='top',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f8f8',
                    edgecolor='#cccccc', alpha=0.9))

# FIX #11: Structural SOC effect annotation
# Add an annotation arrow pointing to the shaded area
mid_x = 50
rn_annot = 'sub_saharan_africa'
fr_annot = fine_data[rn_annot]
xv_a = np.array(fr_annot['soc_pct'])
yt_a = np.array(fr_annot['total_penalty'])
yc_a = np.array(fr_annot['ctrl_penalty'])
ix50 = np.argmin(np.abs(xv_a - mid_x))
mid_y = (yt_a[ix50] + yc_a[ix50]) / 2
ax_a.annotate('Structural\nSOC effect',
              xy=(xv_a[ix50], mid_y), xytext=(35, mid_y + 4),
              fontsize=7.5, fontstyle='italic', color=PAL['ssa'],
              arrowprops=dict(arrowstyle='->', color=PAL['ssa'], lw=0.8),
              ha='center')

ax_a.set_xlabel('Farm SOC (% of regional mean)')
ax_a.set_ylabel('Yield penalty at year 10 (%)')
ax_a.set_xlim(8, 135)
ax_a.set_ylim(-0.5, 21)

# Panel b: scatter
for rn in REGION_ORDER:
    m = buffer_metrics[rn]
    pen = disruption_penalty(rn, 'SOC_100pct')
    size = m['cropland_mha'] * 0.6
    ax_b.scatter(m['soil_buffer_ratio'], pen, s=size,
                 c=REGION_COLORS[rn], edgecolors='black', linewidth=0.4, zorder=3)
    offset_x, offset_y = 0.015, 0.0
    ha = 'left'
    if rn == 'south_asia':
        offset_y = 0.4
    elif rn == 'sub_saharan_africa':
        # Place below-right of its dot
        offset_y = -0.65
        offset_x = 0.015
        ha = 'left'
    elif rn == 'fsu_central_asia':
        # Place above-left of its dot
        offset_y = 0.75
        offset_x = -0.015
        ha = 'right'
    elif rn == 'north_america':
        offset_y = -0.45
    elif rn == 'southeast_asia':
        # SE Asia sits between FSU and SSA in the data; push right+down
        offset_x = 0.020
        offset_y = 0.55
        ha = 'left'
    ax_b.text(m['soil_buffer_ratio'] + offset_x, pen + offset_y,
              REGION_LABELS_INLINE[rn], fontsize=7, ha=ha, va='center')

# South Asia annotation
sa_m = buffer_metrics['south_asia']
sa_pen = disruption_penalty('south_asia', 'SOC_100pct')
ax_b.annotate('High buffer ratio but\nhigh price elasticity\n(\u03b5_F = \u22120.40, \u03b7 = \u22120.60)',
              xy=(sa_m['soil_buffer_ratio'], sa_pen),
              xytext=(0.28, 10.5),
              fontsize=7, fontstyle='italic',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='gray', alpha=0.8),
              arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))

# R² regression
buf_ratios = np.array([buffer_metrics[rn]['soil_buffer_ratio'] for rn in REGION_ORDER])
penalties = np.array([disruption_penalty(rn, 'SOC_100pct') for rn in REGION_ORDER])
eps_vals = np.array([abs(buffer_metrics[rn]['eps_F_PF']) for rn in REGION_ORDER])

def r_squared(X, y):
    X = np.column_stack([np.ones(len(y)), X]) if X.ndim > 1 else np.column_stack([np.ones(len(y)), X.reshape(-1, 1)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

r2_simple = r_squared(buf_ratios, penalties)
r2_multi = r_squared(np.column_stack([buf_ratios, eps_vals]), penalties)

ax_b.text(0.97, 0.03,
          f'Soil buffer alone: R\u00B2 = {r2_simple:.2f}\n'
          f'+ price elasticity: R\u00B2 = {r2_multi:.2f}',
          transform=ax_b.transAxes, fontsize=7, ha='right', va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
print(f'  R\u00B2(buffer only)={r2_simple:.3f}, R\u00B2(buffer+eps)={r2_multi:.3f}')

ax_b.set_xlabel('Soil N buffer ratio\n(fraction from mineralization)')
ax_b.set_ylabel('Disruption penalty at year 10 (%)')
ax_b.set_xlim(0.22, 0.78)
ax_b.set_ylim(-1, 12)

add_panel_label(ax_a, 'a')
add_panel_label(ax_b, 'b')

plt.tight_layout()
save_figure(fig, 'figure1_vulnerability_gradient')
plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: NUE sensitivity (2-panel)
# Panel a: global weighted trajectories (y-axis cropped to 14%)
# Panel b: dumbbell plot (line weight 1.5pt)
# ═══════════════════════════════════════════════════════════════════════
print('Generating Figure 4 (NUE sensitivity)...')
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))

nue_labels = sorted(nue_sensitivity.keys(), reverse=True)  # high NUE first so legend matches line stacking
nue_values = [float(l.split('_')[1]) for l in nue_labels]
nue_colors = plt.cm.viridis(np.linspace(0.95, 0.15, len(nue_labels)))

# Panel a: global trajectories
# Positive yield loss convention (up = worse) matching Fig 4b
for i, (label, nue_val) in enumerate(zip(nue_labels, nue_values)):
    gw = compute_global_weighted(nue_sensitivity[label], 'yield_fraction')
    loss_pct = (1 - gw) * 100  # positive = loss, up = worse
    ax_a.plot(gw.index, loss_pct, color=nue_colors[i], linewidth=2,
              label=f'NUE = {nue_val:.0%}')

ax_a.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.3, zorder=0)
ax_a.set_xlabel('Years after disruption onset')
ax_a.set_ylabel('Global yield loss (%)')
ax_a.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax_a.set_ylim(-1, 14)  # positive = loss, up = worse (matches Fig 4b convention)
ax_a.set_xlim(0, 30)

add_panel_label(ax_a, 'a')

# Panel b: NUE dumbbell (FIX #10: line weight 1.5pt)
nue45 = nue_sensitivity['NUE_0.45']
nue65 = nue_sensitivity['NUE_0.65']
nue95 = nue_sensitivity['NUE_0.95']

dumbbell_order = [
    'sub_saharan_africa', 'south_asia', 'fsu_central_asia', 'southeast_asia',
    'east_asia', 'europe', 'latin_america', 'north_america',
]
dumbbell_labels = {
    'north_america': 'North America', 'europe': 'Europe',
    'east_asia': 'East Asia', 'south_asia': 'South Asia',
    'southeast_asia': 'SE Asia', 'latin_america': 'Latin America',
    'sub_saharan_africa': 'Sub-Saharan Africa', 'fsu_central_asia': 'FSU & Central Asia',
}

losses_45, losses_65, losses_95 = [], [], []
for rn in dumbbell_order:
    df45 = nue45[rn]; df65 = nue65[rn]; df95 = nue95[rn]
    losses_45.append((1 - df45[df45['year'] == 10]['yield_fraction'].iloc[0]) * 100)
    losses_65.append((1 - df65[df65['year'] == 10]['yield_fraction'].iloc[0]) * 100)
    losses_95.append((1 - df95[df95['year'] == 10]['yield_fraction'].iloc[0]) * 100)

y_pos = np.arange(len(dumbbell_order))
db_labels = [dumbbell_labels[rn] for rn in dumbbell_order]

# Connecting lines at 1.5pt (FIX #10)
for i in range(len(dumbbell_order)):
    ax_b.plot([losses_95[i], losses_45[i]], [y_pos[i], y_pos[i]],
              color='#CCCCCC', linewidth=1.5 * (96/72), zorder=1)  # 1.5pt in pixels

ax_b.scatter(losses_45, y_pos, color=PAL['ssa'], s=70, zorder=3, label='NUE = 0.45')
ax_b.scatter(losses_65, y_pos, color=PAL['sea'], s=50, zorder=3, label='NUE = 0.65', marker='D')
ax_b.scatter(losses_95, y_pos, color=PAL['sa'], s=70, zorder=3, label='NUE = 0.95')

for i in range(len(dumbbell_order)):
    # NUE=0.45 label: always to the RIGHT of the dot (outward)
    ax_b.text(losses_45[i] + 0.4, y_pos[i], f'{losses_45[i]:.1f}%',
              va='center', ha='left', fontsize=7, color=PAL['ssa'])
    # NUE=0.95 label: placed ABOVE the dot to avoid y-axis label overlap at low values
    ax_b.text(losses_95[i], y_pos[i] - 0.28, f'{losses_95[i]:.1f}%',
              va='bottom', ha='center', fontsize=7, color=PAL['sa'])

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(db_labels, fontsize=7.5)
ax_b.set_xlabel('Yield loss at year 10 (%)')
ax_b.legend(fontsize=7, loc='lower right', framealpha=0.9)
ax_b.set_xlim(-1.5, 25)
ax_b.invert_yaxis()
ax_b.axvline(0, color='gray', linewidth=0.5, linestyle=':')

add_panel_label(ax_b, 'b')

plt.tight_layout()
save_figure(fig, 'figure3_nue_sensitivity')
plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: 2022 hindcast validation
# FIX #3: overprediction annotation
# FIX #7: source attribution removed (moved to caption)
# ═══════════════════════════════════════════════════════════════════════
print('Generating Figure 5 (hindcast 2022)...')

from coupled_monthly import CoupledMonthlyModel, calibrate_price_shock
from coupled_econ_biophysical import EconParams

shock_15 = calibrate_price_shock(0.15)
s3_15 = EconParams(fert_price_shock=shock_15, eps_F_N=0.0)

hindcast_regions = ['north_america', 'europe', 'south_asia', 'sub_saharan_africa']
hindcast_labels_short = ['NA', 'EU', 'SA', 'SSA']
hindcast_labels_full = ['North America', 'Europe', 'South Asia', 'Sub-Saharan Africa']

model_fert_reductions = []
for rn in hindcast_regions:
    r = regions[rn]
    baseline_econ = EconParams(fert_price_shock=0.0, eps_F_N=0.0)
    model_base = CoupledMonthlyModel(region=r, econ=baseline_econ, region_key=rn, t_max=5.0)
    df_base = model_base.run()
    fert_base = df_base[df_base['year'] == 1]['fert_applied_kgha'].iloc[0]
    model = CoupledMonthlyModel(region=r, econ=s3_15, region_key=rn, t_max=5.0)
    df = model.run()
    fert_shock = df[df['year'] == 1]['fert_applied_kgha'].iloc[0]
    fert_red = (1 - fert_shock / fert_base) * 100 if fert_base > 0 else 0
    model_fert_reductions.append(fert_red)

observed_fert_reduction = [3, 5, 8, 14]
obs_err = [1.5, 2, 3, 4]

fig, ax = plt.subplots(figsize=(5.5, 5))

hindcast_colors = [REGION_COLORS[rn] for rn in hindcast_regions]

max_val = 28
ax.plot([0, max_val], [0, max_val], color='gray', linewidth=1, linestyle='--',
        alpha=0.6, zorder=1, label='1:1 line')

for i, (pred, obs, err, lbl, clr) in enumerate(
        zip(model_fert_reductions, observed_fert_reduction, obs_err,
            hindcast_labels_short, hindcast_colors)):
    ax.errorbar(pred, obs, yerr=err, fmt='o', color=clr, markersize=10,
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=3,
                markeredgecolor='white', markeredgewidth=0.8)
    offset_x, offset_y = 0.8, 0.8
    if lbl == 'SA':
        offset_y = -1.5
    ax.text(pred + offset_x, obs + offset_y, hindcast_labels_full[i],
            fontsize=8, color=clr, fontweight='bold')

ax.set_xlabel('Model-predicted fertilizer demand reduction (%)')
ax.set_ylabel('Observed fertilizer purchase reduction (%)')
ax.set_xlim(0, max_val)
ax.set_ylim(0, 20)
ax.legend(fontsize=8, loc='upper left')

# FIX #3: Overprediction annotation — placed in lower-right to avoid data overlap
ax.annotate('Model systematically\noverpredicts demand reduction',
            xy=(15, 6), xytext=(18, 2.5),
            fontsize=8, fontstyle='italic', color='#555555',
            arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                      edgecolor='#cccccc', alpha=0.9))

# FIX #8: Source attribution REMOVED from figure (moved to caption)
# Previously had a text box with IFPRI, FAO GIEWS sources — now in caption only

plt.tight_layout()
save_figure(fig, 'figure4_hindcast_2022')
plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Structural sensitivity (MEMS vs Century)
# 5-panel layout: 4 regional SOC trajectories + 1 summary bar
# FIX #5: reference lines at y=1.0, y=2.0 on panel e
# ═══════════════════════════════════════════════════════════════════════
print('Generating Figure 6 (structural sensitivity)...')

key4_fig6 = ['north_america', 'south_asia', 'sub_saharan_africa', 'latin_america']
key4_labels_fig6 = ['North America', 'South Asia', 'Sub-Saharan Africa', 'Latin America']

fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.30,
                       height_ratios=[1, 1.1])

colors_century = '#2166ac'
colors_mems = '#b2182b'

# Panels a-d: SOC trajectories under supply disruption
for idx, (rn, rlabel) in enumerate(zip(key4_fig6, key4_labels_fig6)):
    ax = fig.add_subplot(gs[0, idx])

    if rn in mems_data:
        df = mems_data[rn]
        c_df = df[df['model'] == 'century']
        m_df = df[df['model'] == 'mems']

        # SOC as fraction of initial
        c_soc0 = c_df['SOC_total'].iloc[0]
        m_soc0 = m_df['SOC_total'].iloc[0]

        ax.plot(c_df['year'], c_df['SOC_total'] / c_soc0, color=colors_century,
                linewidth=1.8, label='Century (3-pool)')
        ax.plot(m_df['year'], m_df['SOC_total'] / m_soc0, color=colors_mems,
                linewidth=1.8, linestyle='--', label='MEMS (4-pool)')
    else:
        ax.text(0.5, 0.5, f'No data for\n{rlabel}', transform=ax.transAxes,
                ha='center', fontsize=9, color='red')

    ax.set_title(rlabel, fontsize=10, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('SOC (fraction of initial)', fontsize=9)
    ax.set_xlabel('Years', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)

    if idx == 0:
        ax.legend(fontsize=7.5, loc='lower left')

    add_panel_label(ax, chr(ord('a') + idx))

# Panel e: amplification ratios across all 8 regions
ax_e = fig.add_subplot(gs[1, 1:3])

if len(mems_dep) > 0:
    # Compute amplification ratio = MEMS_SOC_loss / Century_SOC_loss
    dep_sorted = mems_dep.sort_values('Century_SOC_loss_pct', ascending=True)
    amp_ratios = dep_sorted['MEMS_SOC_loss_pct'] / dep_sorted['Century_SOC_loss_pct']

    y_pos = np.arange(len(dep_sorted))
    colors_bar = [REGION_COLORS.get(rn, '#888888') for rn in REGION_ORDER
                  if REGION_LABELS_INLINE.get(rn, '') in dep_sorted['Region'].values]

    # Match region names
    bar_colors = []
    for _, row in dep_sorted.iterrows():
        matched = False
        for rn, lbl in REGION_LABELS_INLINE.items():
            if lbl == row['Region'] or row['Region'].replace(' & ', ' & ').strip() == lbl:
                bar_colors.append(REGION_COLORS[rn])
                matched = True
                break
        if not matched:
            bar_colors.append('#888888')

    bars = ax_e.barh(y_pos, amp_ratios.values, color=bar_colors,
                     edgecolor='white', linewidth=0.5, height=0.7)

    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels(dep_sorted['Region'].values, fontsize=8)
    ax_e.set_xlabel('MEMS / Century amplification ratio', fontsize=9)

    # FIX #5: Reference lines at y-values on x-axis
    ax_e.axvline(1.0, color='black', linewidth=0.8, linestyle='-', alpha=0.4, zorder=0)
    ax_e.axvline(2.0, color='gray', linewidth=0.7, linestyle='--', alpha=0.4, zorder=0)
    ax_e.text(1.0, len(dep_sorted) - 0.3, '1:1', fontsize=7, ha='center', color='black', alpha=0.6)
    ax_e.text(2.0, len(dep_sorted) - 0.3, '2×', fontsize=7, ha='center', color='gray', alpha=0.6)

    # Value labels
    for i, (_, row) in enumerate(dep_sorted.iterrows()):
        ratio = row['MEMS_SOC_loss_pct'] / row['Century_SOC_loss_pct']
        ax_e.text(ratio + 0.05, i, f'{ratio:.1f}×', va='center', fontsize=7)

add_panel_label(ax_e, 'e', x=-0.06)

plt.tight_layout()
save_figure(fig, 'fig_structural_sensitivity')
plt.close()


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('ALL 6 FIGURES GENERATED')
print(f'Output directory: {FIG_DIR}')
print('Formats: PNG (300 DPI) + PDF + TIFF (white background, 300 DPI)')
print('=' * 70)

# List output files
for fmt in ['png', 'tiff']:
    files = sorted(FIG_DIR.glob(f'*.{fmt}'))
    print(f'\n{fmt.upper()} files:')
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f'  {f.name} ({size_mb:.1f} MB)')
