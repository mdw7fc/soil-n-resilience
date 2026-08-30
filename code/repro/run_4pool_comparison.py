#!/usr/bin/env python3
"""Structural engine comparison: microbially-explicit 4-pool vs Century.

Supplementary Note 2, Supplementary table 2 and Supplementary Figure S5
(F-029, revised under F-030). Both coupled models run the matched SC1
scenario (20% sustained physical supply loss, no recovery) for 30 years
across the eight regions; the economic layer, realized-yield clearing, ERA5
forcing and water-stress response are shared, so the comparison isolates the
SOM stabilization mechanism. The comparison is EXPLORATORY structural
sensitivity: each engine runs from its own stationary spin-up equilibrium,
and the 4-pool equilibrium departs from the observed regional SOC (the
baseline states are reported in the output for exactly that reason).

Runs per region at the central texture assumption (clay+silt = 0.55):
  1. Century/RothC coupled SC1 run, plus a no-shock engine control.
  2. 4-pool coupled SC1 run, plus a no-shock engine control.
  3. 4-pool with CUE held at its N-replete baseline (cue_fixed=True); the
     difference from run 2 isolates the CUE-downregulation share OF THE
     ENGINE DIFFERENCE (4-pool minus Century loss), which is what
     Supplementary table 2 partitions. Regions where that difference is
     non-positive are reported as excluded.
Plus a texture sensitivity: the full 4-pool comparison rerun at clay+silt
0.35 and 0.75 (the MAOM sorption ceiling scales with texture).

The disruption-attributable carbon budget (Figure S5) is computed against
the no-shock controls: over 30 years the disruption reduces residue-C
inputs MORE than it reduces respiration, and the gap is the SOC loss
attributable to the disruption. Gross respiration composition (CUE step vs
non-recycled necromass) is reported as a diagnostic in the JSON.

Writes:
  results/fourpool_comparison.json
  outputs/Table_S2_fourpool_sol.csv
  figures/Figure_S5_fourpool_flux.png/.pdf
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file  # noqa: E402
from soil_n_model import CropParams, get_default_regions  # noqa: E402
from coupled_monthly import (  # noqa: E402
    CoupledMonthlyModel, MonthlyBiophysicalEngine, get_calibrated_ym,
)
from coupled_econ_biophysical import get_supply_constrained_scenarios  # noqa: E402
import coupled_4pool as C4  # noqa: E402
from coupled_4pool import (  # noqa: E402
    Coupled4PoolModel, FourPoolBiophysicalEngine, CLAY_SILT_DEFAULT,
)
from som_4pool_monthly import FourPoolParams  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
OUTPUTS = os.path.join(ROOT, 'outputs')
FIGURES = os.path.join(ROOT, 'figures')

RO = ['north_america', 'europe', 'east_asia', 'south_asia',
      'southeast_asia', 'latin_america', 'sub_saharan_africa',
      'fsu_central_asia']
LABEL = {'north_america': 'North America', 'europe': 'Europe',
         'east_asia': 'East Asia', 'south_asia': 'South Asia',
         'southeast_asia': 'Southeast Asia', 'latin_america': 'Latin America',
         'sub_saharan_africa': 'Sub-Saharan Africa',
         'fsu_central_asia': 'Former Soviet Union'}
SHORT = {'north_america': 'North America', 'europe': 'Europe',
         'east_asia': 'East Asia', 'south_asia': 'South Asia',
         'southeast_asia': 'SE Asia', 'latin_america': 'Latin America',
         'sub_saharan_africa': 'SSA', 'fsu_central_asia': 'FSU'}
TEXTURE_SENSITIVITY = (0.35, 0.75)


def soc_loss(df, yr):
    s0 = float(df[df.year == 0].soc_total.iloc[0])
    s = float(df[df.year == yr].soc_total.iloc[0])
    return 100.0 * (1.0 - s / s0)


def century_inputs_from_yields(yields, region, crop):
    """Residue-C inputs implied by a yield series (same formula as step())."""
    rf = (1 - crop.harvest_index) / crop.harvest_index
    res_cf = getattr(crop, 'residue_c_fraction', 0.45)
    per_y = res_cf * rf * (region.residue_retention
                           + region.root_shoot_c_ratio) * region.cre_regional
    return float(np.sum(np.asarray(yields) * per_y))


def century_control(region, rk, ym, years=30):
    eng = MonthlyBiophysicalEngine(region, region_key=rk,
                                   yield_max_override=ym)
    soc0 = eng.C_active + eng.C_slow + eng.C_passive
    ys = []
    for _ in range(years):
        st = eng.step(region.synth_n_current)
        ys.append(st['yield_tha'])
    soc_end = eng.C_active + eng.C_slow + eng.C_passive
    return soc0, soc_end, ys


def fourpool_control(region, rk, years=30, clay_silt=CLAY_SILT_DEFAULT):
    eng = FourPoolBiophysicalEngine(region, rk, clay_silt=clay_silt)
    soc0 = eng.soc_initial
    cum_in = cum_cue = cum_nec = 0.0
    for _ in range(years):
        st = eng.step(region.synth_n_current)
        cum_in += st['c_input']
        cum_cue += st['resp_cue']
        cum_nec += st['resp_necro']
    soc_end = eng.c_pom + eng.c_dom + eng.c_mbc + eng.c_maom
    return soc0, soc_end, cum_in, cum_cue, cum_nec


def direct_sorption_share(rk, clay_silt=CLAY_SILT_DEFAULT):
    """Share of equilibrium MAOM inputs arriving by direct DOM sorption."""
    m = FourPoolParams()
    ym = C4.calibrate_ym_fourpool(rk, clay_silt=clay_silt)
    eq = C4.fourpool_dynamic_spinup(rk, ym, clay_silt=clay_silt)
    maom_sat = min(eq['c_maom'] / max(eq['qmax'], 0.1), 1.0)
    sorption_rate = m.k_dom_sorption * max(0, 1.0 - maom_sat)
    total_rate = m.k_dom_uptake + sorption_rate
    f_uptake = m.k_dom_uptake / total_rate
    dom_removed = eq['c_dom'] * (1 - np.exp(-total_rate))
    dom_to_maom = dom_removed * (1 - f_uptake)
    assim = dom_removed * f_uptake * m.cue_max
    k = m.k_mbc_turnover
    mbc_new = (assim / k) * (1 - np.exp(-k)) + eq['c_mbc'] * np.exp(-k)
    death = eq['c_mbc'] + assim - mbc_new
    necro_to_maom = death * m.f_necro_to_maom * max(0, 1 - maom_sat)
    return 100.0 * dom_to_maom / max(dom_to_maom + necro_to_maom, 1e-12)


def run_texture(cs, regions, sc1):
    ratios = {}
    for rk in RO:
        r = regions[rk]
        ym_c = get_calibrated_ym(rk, MonthlyNParams())
        dfc = CoupledMonthlyModel(region=r, econ=sc1, region_key=rk,
                                  t_max=30.0, yield_max_override=ym_c).run()
        df4 = Coupled4PoolModel(r, sc1, rk, t_max=30.0, clay_silt=cs).run()
        lc, l4 = soc_loss(dfc, 30), soc_loss(df4, 30)
        ratios[rk] = l4 / lc if lc > 0 else None
    vals = sorted(v for v in ratios.values() if v is not None)
    return {'ratio_min': round(vals[0], 4), 'ratio_max': round(vals[-1], 4),
            'ratio_median': round(float(np.median(vals)), 4),
            'ssa_ratio': round(ratios['sub_saharan_africa'], 4)}


def main():
    apply_era5_climate_file(os.path.join(DATA, 'era5_regional_climates.json'))
    regions = get_default_regions()
    sc1 = get_supply_constrained_scenarios()['SC1_20pct']
    mp = MonthlyNParams()
    crop = CropParams()

    out = {}
    for rk in RO:
        r = regions[rk]
        ym_c = get_calibrated_ym(rk, mp)
        dfc = CoupledMonthlyModel(region=r, econ=sc1, region_key=rk,
                                  t_max=30.0, yield_max_override=ym_c).run()
        m4 = Coupled4PoolModel(r, sc1, rk, t_max=30.0)
        df4 = m4.run()
        df4f = Coupled4PoolModel(r, sc1, rk, t_max=30.0, cue_fixed=True).run()

        lc30, l4_30, l4f_30 = soc_loss(dfc, 30), soc_loss(df4, 30), soc_loss(df4f, 30)
        gap = l4_30 - lc30
        cue_part = l4_30 - l4f_30
        if gap <= 0:
            cue_pct, excl = None, 'engine difference non-positive (ratio < 1)'
        elif cue_part < 0:
            cue_pct, excl = None, ('fixed-CUE run loses marginally more SOC '
                                   'than the variable-CUE run')
        else:
            cue_pct, excl = 100.0 * cue_part / gap, None

        # ---- disruption-attributable carbon budget (vs no-shock controls) --
        sh = df4[df4.year >= 1]
        soc0_4, socN_4 = (float(df4[df4.year == 0].soc_total.iloc[0]),
                          float(df4[df4.year == 30].soc_total.iloc[0]))
        c0, cN, cin_ctrl, ccue, cnec = fourpool_control(r, rk)
        d_in_4 = float(sh.c_input.sum()) - cin_ctrl
        d_resp_4 = (float(sh.resp_cue.sum() + sh.resp_necro.sum())
                    - (ccue + cnec))
        d_soc_4 = (socN_4 - soc0_4) - (cN - c0)      # attributable SOC change

        soc0_c = float(dfc[dfc.year == 0].soc_total.iloc[0])
        socN_c = float(dfc[dfc.year == 30].soc_total.iloc[0])
        ys_shocked = dfc[dfc.year >= 1].yield_tha.tolist()
        cin_c_sh = century_inputs_from_yields(ys_shocked, r, crop)
        c0c, cNc, ys_ctrl = century_control(r, rk, ym_c)
        cin_c_ctrl = century_inputs_from_yields(ys_ctrl, r, crop)
        # respiration by budget closure: resp = inputs - delta_SOC
        resp_c_sh = cin_c_sh - (socN_c - soc0_c)
        resp_c_ctrl = cin_c_ctrl - (cNc - c0c)
        d_in_c = cin_c_sh - cin_c_ctrl
        d_resp_c = resp_c_sh - resp_c_ctrl
        d_soc_c = (socN_c - soc0_c) - (cNc - c0c)

        out[rk] = {
            'label': LABEL[rk],
            'century_soc_loss_yr30_pct': round(lc30, 4),
            'fourpool_soc_loss_yr30_pct': round(l4_30, 4),
            'fourpool_fixedcue_soc_loss_yr30_pct': round(l4f_30, 4),
            'ratio30': round(l4_30 / lc30, 4) if lc30 > 0 else None,
            'cue_contribution_pct_of_engine_difference': (
                round(cue_pct, 1) if cue_pct is not None else None),
            'noncue_pct_of_engine_difference': (
                round(100.0 - cue_pct, 1) if cue_pct is not None else None),
            'excluded_reason': excl,
            # baseline states (the engines equilibrate at different states;
            # reported so the exploratory framing is checkable)
            'observed_soc_tCha': r.soc_initial,
            'fourpool_eq_soc_tCha': round(m4.bio.soc_initial, 3),
            'faostat_yield_target_tha': float(
                __import__('monthly_model_v3').FAOSTAT_TARGETS[rk]),
            'fourpool_eq_yield_tha': round(m4.bio.yield_baseline, 4),
            'direct_maom_sorption_share_pct': round(direct_sorption_share(rk), 1),
            # disruption-attributable 30-yr carbon budget (t C/ha; negative =
            # reduction relative to the no-shock control)
            'budget_fourpool': {
                'd_inputs': round(d_in_4, 4), 'd_respiration': round(d_resp_4, 4),
                'd_soc': round(d_soc_4, 4),
                'closure_residual': round(d_soc_4 - (d_in_4 - d_resp_4), 5)},
            'budget_century': {
                'd_inputs': round(d_in_c, 4), 'd_respiration': round(d_resp_c, 4),
                'd_soc': round(d_soc_c, 4)},
            # gross respiration composition under SC1 (diagnostic)
            'resp_cue_cum_tCha': round(float(sh.resp_cue.sum()), 3),
            'resp_necro_cum_tCha': round(float(sh.resp_necro.sum()), 3),
            'resp_cue_share_pct': round(100.0 * float(sh.resp_cue.sum())
                                        / max(float(sh.resp_cue.sum()
                                              + sh.resp_necro.sum()), 1e-9), 1),
            'ym_century': round(ym_c, 4),
            'ym_fourpool': round(m4.bio.y_max, 4),
            'fourpool_spinup_years': m4.bio.spinup_years,
            'max_clearing_residual': float(abs(df4.clearing_residual).max()),
            'soc_frac_century': [round(v, 6) for v in dfc.soc_fraction],
            'soc_frac_fourpool': [round(v, 6) for v in df4.soc_fraction],
        }

    ratios = sorted(v['ratio30'] for v in out.values())
    n_above = sum(1 for v in out.values() if v['ratio30'] > 1.0)
    summary = {
        'scenario': 'SC1_20pct (20% sustained physical supply loss)',
        'horizon_years': 30,
        'clay_silt_central': CLAY_SILT_DEFAULT,
        'ratio_min': ratios[0], 'ratio_max': ratios[-1],
        'ratio_median': round(float(np.median(ratios)), 4),
        'regions_above_parity': n_above,
        'regions_below_parity': [rk for rk, v in out.items()
                                 if v['ratio30'] < 1.0],
        'calibration_shortfall': {
            str(k): {'plateau_tha': round(float(p), 4), 'target_tha': t}
            for k, (p, t) in C4.CALIBRATION_SHORTFALL.items()},
        'texture_sensitivity': {
            'cs%02d' % round(cs * 100): run_texture(cs, regions, sc1)
            for cs in TEXTURE_SENSITIVITY},
    }

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'fourpool_comparison.json'), 'w') as fh:
        json.dump({'summary': summary, 'regions': out}, fh, indent=2)

    # ---- Supplementary table 2 source ------------------------------------
    os.makedirs(OUTPUTS, exist_ok=True)
    import csv
    with open(os.path.join(OUTPUTS, 'Table_S2_fourpool_sol.csv'), 'w',
              newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['region', 'century_soc_loss_yr30_pct',
                    'fourpool_soc_loss_yr30_pct', 'ratio30',
                    'cue_contribution_pct_of_engine_difference',
                    'noncue_pct_of_engine_difference', 'excluded_reason'])
        for rk in RO:
            v = out[rk]
            w.writerow([v['label'],
                        round(v['century_soc_loss_yr30_pct'], 2),
                        round(v['fourpool_soc_loss_yr30_pct'], 2),
                        round(v['ratio30'], 2),
                        v['cue_contribution_pct_of_engine_difference']
                        if v['cue_contribution_pct_of_engine_difference']
                        is not None else '',
                        v['noncue_pct_of_engine_difference']
                        if v['noncue_pct_of_engine_difference']
                        is not None else '',
                        v['excluded_reason'] or ''])

    # ---- Supplementary Figure S5: disruption-attributable C budget -------
    order = sorted(RO, key=lambda k: out[k]['budget_fourpool']['d_inputs'])
    names = [SHORT[k] for k in order]
    ypos = np.arange(len(order))[::-1]
    C_IN, C_RESP = '#1f77b4', '#E8590C'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.09, 6.585), sharex=True)
    for ax, key, title in ((ax1, 'budget_fourpool',
                            'Microbially-explicit 4-pool scheme'),
                           (ax2, 'budget_century', 'Century/RothC')):
        d_in = np.array([-out[k][key]['d_inputs'] for k in order])
        d_re = np.array([-out[k][key]['d_respiration'] for k in order])
        net = d_in - d_re
        h = 0.38
        ax.barh(ypos + h / 2, d_in, height=h, color=C_IN,
                label='Reduction in residue-C inputs')
        ax.barh(ypos - h / 2, d_re, height=h, color=C_RESP,
                label='Reduction in respiration')
        for y, a, b, n in zip(ypos, d_in, d_re, net):
            ax.text(max(a, b) + 0.06, y,
                    'net SOC loss %.2f' % n, va='center', fontsize=10)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names if ax is ax1 else [''] * len(order),
                           fontsize=13)
        ax.set_xlabel('Cumulative 30-year change vs no-shock control '
                      '(t C ha$^{-1}$)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.text(-0.11, 1.03, 'a', transform=ax1.transAxes, fontsize=18,
             fontweight='bold')
    ax2.text(-0.03, 1.03, 'b', transform=ax2.transAxes, fontsize=18,
             fontweight='bold')
    lim = max(-out[k]['budget_fourpool']['d_inputs'] for k in order) * 1.45
    ax1.set_xlim(0, lim)
    fig.suptitle('Disruption-attributable carbon budget under SC1: inputs '
                 'fall more than respiration, and the gap is the SOC loss',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, 'Figure_S5_fourpool_flux.png'), dpi=200)
    fig.savefig(os.path.join(FIGURES, 'Figure_S5_fourpool_flux.pdf'))
    plt.close(fig)

    print('fourpool comparison: ratio %.2f-%.2f, median %.2f; above parity '
          '%d of 8; below: %s'
          % (summary['ratio_min'], summary['ratio_max'],
             summary['ratio_median'], n_above,
             ', '.join(summary['regions_below_parity']) or 'none'))
    for cs, tx in summary['texture_sensitivity'].items():
        print('  texture %s: %.2f-%.2f median %.2f ssa %.2f'
              % (cs, tx['ratio_min'], tx['ratio_max'], tx['ratio_median'],
                 tx['ssa_ratio']))
    for rk in RO:
        v = out[rk]
        b = v['budget_fourpool']
        print('  %-20s ratio %.2f (c %.2f%% / 4p %.2f%%)  dIn %.2f dResp %.2f '
              'dSOC %.2f  direct-sorb %.0f%%'
              % (rk, v['ratio30'], v['century_soc_loss_yr30_pct'],
                 v['fourpool_soc_loss_yr30_pct'], b['d_inputs'],
                 b['d_respiration'], b['d_soc'],
                 v['direct_maom_sorption_share_pct']))


if __name__ == '__main__':
    main()
