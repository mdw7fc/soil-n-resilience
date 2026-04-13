#!/usr/bin/env python3
"""
Supplementary Figure S1: Sensitivity to halved price elasticities.

Shows that the main results (SOC buffers yield and profit impacts of price
shocks) are qualitatively robust when all price elasticities of fertilizer
demand (eps_F_PF) are halved.

Two-panel figure matching Fig 1 layout:
  Panel a: Yield change (%) vs farm SOC  — baseline vs halved elasticities
  Panel b: Profit change (%) vs farm SOC — baseline vs halved elasticities

Author: Matthew Wallenstein
"""

import sys, os, pickle, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = SCRIPT_DIR.parent
MODEL_DIR = REPO_DIR / 'model'
sys.path.insert(0, str(MODEL_DIR))

DATA_DIR = REPO_DIR / 'data'
FIG_DIR = REPO_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

from coupled_monthly import (
    CoupledMonthlyModel, MonthlyBiophysicalEngine, get_calibrated_ym,
    clear_ym_cache,
)
from coupled_econ_biophysical import (
    EconParams, REGIONAL_ECON_PARAMS, calibrate_price_shock,
)
from soil_n_model import get_default_regions

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

KEY4 = ['sub_saharan_africa', 'south_asia', 'latin_america', 'north_america']
SOC_PCTS = list(range(20, 130, 10))  # 20% to 120% of regional mean
PRICE_SHOCK = 1.0  # 100% price increase

# Color and style
PAL = {
    'ssa': '#C62828', 'sa': '#1565C0', 'latam': '#2E7D32', 'na': '#455A64',
}
REGION_COLORS = {
    'sub_saharan_africa': PAL['ssa'], 'south_asia': PAL['sa'],
    'latin_america': PAL['latam'], 'north_america': PAL['na'],
}
REGION_LABELS = {
    'sub_saharan_africa': 'Sub-Saharan\nAfrica', 'south_asia': 'South\nAsia',
    'latin_america': 'Latin\nAmerica', 'north_america': 'North\nAmerica',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8,
})


def add_panel_label(ax, label, x=-0.08, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right')


# ═══════════════════════════════════════════════════════════════════════
# FARM-LEVEL SOC SWEEP
# ═══════════════════════════════════════════════════════════════════════

def farm_soc_sweep(regions, soc_pcts, price_shock_frac, eps_scale=1.0):
    """Run farm-level SOC sweep with optional elasticity scaling.

    For each region:
      1. Spin up model at equilibrium SOC
      2. For each SOC level (as % of regional mean):
         a. Scale SOM pools to that SOC
         b. Apply price shock via economic equilibrium (1-year solve)
         c. Record yield loss and profit change

    Args:
        regions: dict of RegionParams
        soc_pcts: list of SOC percentages (100 = regional mean)
        price_shock_frac: fractional price increase (1.0 = 100%)
        eps_scale: multiplier for eps_F_PF (0.5 = halved elasticities)

    Returns:
        dict[region_key] -> dict with soc_pct, yield_pen, fert_red arrays
    """
    results = {}

    # Regional fertilizer cost as fraction of gross revenue (FAO/IFDC estimates)
    FERT_COST_FRAC = {
        'sub_saharan_africa': 0.25,  # high relative cost, low absolute use
        'south_asia': 0.20,
        'latin_america': 0.12,
        'north_america': 0.08,
    }

    for rn in KEY4:
        r = regions[rn]
        rp = REGIONAL_ECON_PARAMS.get(rn, {})
        ym = get_calibrated_ym(rn)

        # Build baseline engine to get equilibrium SOC and N mineralization
        engine = MonthlyBiophysicalEngine(r, region_key=rn, yield_max_override=ym)
        soc_eq = engine.soc_initial
        C_a_eq, C_s_eq, C_p_eq = engine.C_active, engine.C_slow, engine.C_passive

        # Baseline (no-shock) run for 1 step to get reference yield
        base_state = engine.step(r.synth_n_current)
        base_yield = base_state['yield_tha']
        base_fert = r.synth_n_current

        yield_pens = []
        fert_reds = []
        profit_chgs = []

        # Fertilizer cost fraction for profit calc
        fcf = FERT_COST_FRAC.get(rn, 0.15)

        for soc_pct in soc_pcts:
            scale = soc_pct / 100.0

            # Fresh engine for this SOC level
            eng = MonthlyBiophysicalEngine(r, region_key=rn, yield_max_override=ym)
            eng.C_active = C_a_eq * scale
            eng.C_slow = C_s_eq * scale
            eng.C_passive = C_p_eq * scale

            # Get baseline yield at this SOC (no shock)
            state_base = eng.step(r.synth_n_current)
            y_base_soc = state_base['yield_tha']
            gamma = state_base['gamma']

            # Now compute shocked fertilizer via economic equilibrium
            eng2 = MonthlyBiophysicalEngine(r, region_key=rn, yield_max_override=ym)
            eng2.C_active = C_a_eq * scale
            eng2.C_slow = C_s_eq * scale
            eng2.C_passive = C_p_eq * scale

            eps_F_PF = rp.get('eps_F_PF', -0.20) * eps_scale
            eps_F_PY = rp.get('eps_F_PY', 0.10)
            eta = rp.get('eta', -0.30)
            PF_hat = np.log(1 + price_shock_frac)

            # Simultaneous solve
            denom = eta - gamma * eps_F_PY
            if abs(denom) > 1e-10:
                PY_hat = gamma * eps_F_PF * PF_hat / denom
            else:
                PY_hat = 0.0
            F_hat = eps_F_PF * PF_hat + eps_F_PY * PY_hat

            F_shocked = max(0.0, r.synth_n_current * np.exp(F_hat))

            # Biophysical response
            state_shock = eng2.step(F_shocked)
            y_shock = state_shock['yield_tha']

            # Yield penalty (%)
            yield_pen = (1 - y_shock / y_base_soc) * 100 if y_base_soc > 0 else 0.0

            # Fertilizer reduction (%)
            fert_red = (1 - F_shocked / base_fert) * 100 if base_fert > 0 else 0.0

            # Profit change using gross-margin-over-fertilizer:
            # profit = Y*Py - F*Pf  where Pf/Py normalized so fert cost = fcf of baseline revenue
            # Baseline: profit_b = Y_b*1 - F_b * (fcf/F_b) = Y_b - fcf*Y_b = Y_b*(1 - fcf)
            # Shocked: profit_s = Y_s * exp(PY_hat) - F_s * (fcf*Y_b/F_b) * (1+shock)
            pf_per_unit = fcf * y_base_soc / base_fert if base_fert > 0 else 0
            profit_b = y_base_soc - base_fert * pf_per_unit
            profit_s = y_shock * np.exp(PY_hat) - F_shocked * pf_per_unit * (1 + price_shock_frac)
            profit_chg = (profit_s / profit_b - 1) * 100 if abs(profit_b) > 1e-10 else 0.0

            yield_pens.append(round(yield_pen, 2))
            fert_reds.append(round(fert_red, 2))
            profit_chgs.append(round(profit_chg, 2))

        results[rn] = {
            'soc_pct': soc_pcts,
            'yield_pen': yield_pens,
            'fert_red': fert_reds,
            'profit_chg': profit_chgs,
        }
        print(f'  {rn}: yield_pen [{min(yield_pens):.1f}, {max(yield_pens):.1f}], '
              f'profit_chg [{min(profit_chgs):.1f}, {max(profit_chgs):.1f}], '
              f'fert_red [{min(fert_reds):.1f}, {max(fert_reds):.1f}]')

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    regions = get_default_regions()

    print('Running baseline elasticity sweep...')
    baseline = farm_soc_sweep(regions, SOC_PCTS, PRICE_SHOCK, eps_scale=1.0)

    print('\nRunning halved elasticity sweep...')
    halved = farm_soc_sweep(regions, SOC_PCTS, PRICE_SHOCK, eps_scale=0.5)

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE S1: Side-by-side comparison
    # ═══════════════════════════════════════════════════════════════════
    print('\nGenerating Supplementary Figure S1...')

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))

    for rn in KEY4:
        color = REGION_COLORS[rn]
        soc = np.array(baseline[rn]['soc_pct'])

        # Baseline (solid)
        yp_base = -np.array(baseline[rn]['yield_pen'])
        pc_base = np.array(baseline[rn]['profit_chg'])
        ax_a.plot(soc, yp_base, color=color, linewidth=2.0, linestyle='-', zorder=3)
        ax_b.plot(soc, pc_base, color=color, linewidth=2.0, linestyle='-', zorder=3)

        # Halved (dashed)
        yp_half = -np.array(halved[rn]['yield_pen'])
        pc_half = np.array(halved[rn]['profit_chg'])
        ax_a.plot(soc, yp_half, color=color, linewidth=1.5, linestyle='--', alpha=0.8, zorder=3)
        ax_b.plot(soc, pc_half, color=color, linewidth=1.5, linestyle='--', alpha=0.8, zorder=3)

        # Labels at end of baseline curves
        ax_a.text(soc[-1] + 1.5, yp_base[-1], REGION_LABELS[rn],
                  fontsize=7, color=color, fontweight='bold', va='center')
        ax_b.text(soc[-1] + 1.5, pc_base[-1], REGION_LABELS[rn],
                  fontsize=7, color=color, fontweight='bold', va='center')

    # Reference lines
    for ax in (ax_a, ax_b):
        ax.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.3, zorder=0)
        ax.axvline(100, color='gray', linewidth=0.7, linestyle=':', alpha=0.4, zorder=0)
        ax.set_xlabel('Farm SOC (% of regional mean)')
        ax.set_xlim(15, 130)

    ax_a.set_ylabel('Yield change (%)')
    ax_b.set_ylabel('Profit change (%)')

    # Legend for line styles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2.0, linestyle='-',
               label='Baseline elasticities'),
        Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--',
               label='Halved elasticities'),
    ]
    ax_a.legend(handles=legend_elements, loc='lower left', fontsize=8,
                framealpha=0.9)

    add_panel_label(ax_a, 'a')
    add_panel_label(ax_b, 'b')

    plt.tight_layout()

    # Save
    fig.savefig(FIG_DIR / 'figure_s1_elasticity_sensitivity.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(FIG_DIR / 'figure_s1_elasticity_sensitivity.pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'  Saved figure_s1_elasticity_sensitivity (PNG + PDF)')
    plt.close()

    print('\nDone.')
