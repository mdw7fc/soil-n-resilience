#!/usr/bin/env python3
"""
Soil Health as Resilience Infrastructure: Monthly Model Analysis
================================================================
Replaces run_resilience_analysis.py and run_resilience_v2.py.
Uses the coupled monthly model (annual SOM + monthly N availability).

Six analysis sets:
  1. Baseline disruption (S1-S3 + SC1-SC4), 30-year horizon
  2. Soil N buffer metrics (cross-regional correlation)
  3. Degradation scenarios (pre-disruption SOC loss)
  4. No-shock baselines for degradation levels
  5. NUE sensitivity (vary max_uptake_frac in monthly model)
  6. Duration comparison: 1, 5, 10, 30-year disruptions
  7. Investment-then-disruption scenarios

Output: paper2-soil-resilience/data/resilience_monthly.pkl
"""
import sys
import os
import copy
import pickle
import numpy as np
import pandas as pd

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
model_dir = os.path.join(repo_dir, 'model')
sys.path.insert(0, model_dir)

from coupled_monthly import (
    CoupledMonthlyModel, MonthlyBiophysicalEngine, MonthlyNParams,
    EconParams, REGIONAL_ECON_PARAMS,
    calibrate_price_shock, get_scenario_params, get_supply_constrained_scenarios,
    get_calibrated_ym, clear_ym_cache, REGIONAL_CLIMATES,
)
from soil_n_model import get_default_regions

# Output
data_dir = os.path.join(repo_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

REGION_LABELS = {
    'north_america': 'North America',
    'europe': 'Europe',
    'east_asia': 'East Asia',
    'south_asia': 'South Asia',
    'southeast_asia': 'SE Asia',
    'latin_america': 'Latin America',
    'sub_saharan_africa': 'Sub-Saharan Africa',
    'fsu_central_asia': 'FSU & Central Asia',
}

T_MAX = 30.0


def compute_global_weighted(region_dfs, regions_dict, var):
    """Production-weighted global average."""
    frames = []
    for rn, r in regions_dict.items():
        if rn not in region_dfs:
            continue
        df = region_dfs[rn].copy()
        ym = get_calibrated_ym(rn)
        weight = r.cropland_mha * ym
        df['weight'] = weight
        df['weighted_val'] = df[var] * weight
        frames.append(df)
    combined = pd.concat(frames)
    grouped = combined.groupby('year').agg(
        weighted_sum=('weighted_val', 'sum'),
        weight_sum=('weight', 'sum'),
    )
    return grouped['weighted_sum'] / grouped['weight_sum']


def make_model(region, econ, region_key, t_max=T_MAX):
    """Create a CoupledMonthlyModel with cached ym calibration."""
    ym = get_calibrated_ym(region_key)
    return CoupledMonthlyModel(
        region=region, econ=econ, region_key=region_key,
        t_max=t_max, yield_max_override=ym,
    )


def run_scenarios(regions, scenarios, t_max=T_MAX):
    """Run scenarios for all regions."""
    results = {}
    for s_name, econ in scenarios.items():
        results[s_name] = {}
        for rn, r in regions.items():
            model = make_model(r, econ, rn, t_max)
            results[s_name][rn] = model.run()
    return results


# ================================================================
# ANALYSIS 1: Baseline S1-S3 + SC1-SC4
# ================================================================
def run_baseline(regions):
    print('[1/7] Baseline S1-S3 + SC scenarios (30 years)...')
    baseline = run_scenarios(regions, get_scenario_params())
    sc = run_scenarios(regions, get_supply_constrained_scenarios())
    return baseline, sc


# ================================================================
# ANALYSIS 2: Soil N buffer metrics
# ================================================================
def compute_soil_buffer_metrics(regions, baseline_results):
    """Extract soil N contribution for each region from baseline."""
    print('[2/7] Computing soil N buffer metrics...')
    metrics = {}
    for rn, r in regions.items():
        df = baseline_results['S3'][rn]
        yr0 = df[df['year'] == 0].iloc[0]
        yr10 = df[df['year'] == 10].iloc[0]
        yr30 = df[df['year'] == 30].iloc[0] if 30 in df['year'].values else df.iloc[-1]

        n_min = yr0['n_mineralized']
        synth_n = r.synth_n_current
        total_n = n_min + synth_n + 5.0 + r.atm_n_deposition

        soil_buffer_ratio = n_min / total_n
        synth_dependency = synth_n / total_n

        yield_loss_10 = (1 - yr10['yield_fraction']) * 100
        yield_loss_30 = (1 - yr30['yield_fraction']) * 100

        metrics[rn] = {
            'soc_initial': r.soc_initial,
            'synth_n': synth_n,
            'n_mineralized_baseline': n_min,
            'total_n_baseline': total_n,
            'soil_buffer_ratio': soil_buffer_ratio,
            'synth_dependency': synth_dependency,
            'yield_loss_10yr': yield_loss_10,
            'yield_loss_30yr': yield_loss_30,
            'soc_fraction_10yr': yr10['soc_fraction'],
            'soc_fraction_30yr': yr30['soc_fraction'],
            'cropland_mha': r.cropland_mha,
            'eps_F_PF': REGIONAL_ECON_PARAMS[rn]['eps_F_PF'],
            'eta': REGIONAL_ECON_PARAMS[rn]['eta'],
            'n_uptake_0': yr0['n_uptake'],
            'n_uptake_10': yr10['n_uptake'],
            'n_leached_0': yr0.get('n_leached', 0),
        }
    return metrics


# ================================================================
# ANALYSIS 3: Degradation scenarios
# ================================================================
def run_degradation_scenarios(regions):
    """Run disruption from progressively degraded starting conditions."""
    print('[3/7] Degradation scenarios (pre-disruption soil loss)...')
    s3 = get_scenario_params()['S3']
    soc_fractions = [1.0, 0.75, 0.50, 0.25]
    results = {}

    for frac in soc_fractions:
        label = f'SOC_{int(frac*100)}pct'
        results[label] = {}
        for rn, r in regions.items():
            r_deg = copy.deepcopy(r)
            r_deg.soc_initial = r.soc_initial * frac
            model = make_model(r_deg, s3, rn)
            results[label][rn] = model.run()
    return results


# ================================================================
# ANALYSIS 4: No-shock baselines for degradation
# ================================================================
def run_no_shock_baseline(regions):
    """Run without any price shock at each degradation level."""
    print('[4/7] No-shock baselines for degradation levels...')
    soc_fractions = [1.0, 0.75, 0.50, 0.25]
    results = {}
    no_shock = EconParams(fert_price_shock=0.0, eps_F_N=0.0)

    for frac in soc_fractions:
        label = f'SOC_{int(frac*100)}pct'
        results[label] = {}
        for rn, r in regions.items():
            r_deg = copy.deepcopy(r)
            r_deg.soc_initial = r.soc_initial * frac
            model = make_model(r_deg, no_shock, rn)
            results[label][rn] = model.run()
    return results


# ================================================================
# ANALYSIS 5: NUE sensitivity
# Vary max_uptake_frac in monthly model (affects crop N capture efficiency)
# ================================================================
def run_nue_sensitivity(regions):
    """NUE sensitivity by varying monthly crop uptake fraction."""
    print('[5/7] NUE sensitivity...')
    s3 = get_scenario_params()['S3']
    # max_uptake_frac range: 0.45 to 0.95 (default 0.75)
    nue_values = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    results = {}

    for nue in nue_values:
        label = f'NUE_{nue:.2f}'
        results[label] = {}
        mp = MonthlyNParams(max_uptake_frac=nue)
        # Need to recalibrate ym for each NUE setting
        clear_ym_cache()
        for rn, r in regions.items():
            ym = get_calibrated_ym(rn, mp)
            model = CoupledMonthlyModel(
                region=r, econ=s3, region_key=rn,
                t_max=T_MAX, yield_max_override=ym,
            )
            # Override the monthly params in the biophysical engine
            model.bio.mp = mp
            results[label][rn] = model.run()

    # Restore default cache
    clear_ym_cache()
    return results


# ================================================================
# ANALYSIS 6: Duration comparison
# ================================================================
def run_duration_comparison(regions):
    """1, 5, 10, 30-year disruptions."""
    print('[6/7] Duration comparison...')
    durations = [1, 5, 10, 30]
    results = {}

    for dur in durations:
        label = f'duration_{dur}yr'
        results[label] = {}

        for rn, r in regions.items():
            if dur >= T_MAX:
                s3 = get_scenario_params()['S3']
                model = make_model(r, s3, rn)
                results[label][rn] = model.run()
            else:
                # Phase 1: disrupted
                s3 = get_scenario_params()['S3']
                model1 = make_model(r, s3, rn, t_max=float(dur))
                df1 = model1.run()

                # Phase 2: recovery
                r_depleted = copy.deepcopy(r)
                last_row = df1.iloc[-1]
                r_depleted.soc_initial = last_row['soc_total']

                recovery_econ = EconParams(fert_price_shock=0.0, eps_F_N=0.0)
                remaining = T_MAX - dur
                model2 = make_model(r_depleted, recovery_econ, rn, t_max=remaining)
                df2 = model2.run()

                # Stitch together
                df2_adj = df2.copy()
                df2_adj['year'] = df2_adj['year'] + dur

                orig_baseline_yield = df1.iloc[0]['yield_tha']
                if orig_baseline_yield > 0:
                    df2_adj['yield_fraction'] = df2_adj['yield_tha'] / orig_baseline_yield
                    orig_land = df1.iloc[0]['land_mha']
                    df2_adj['carrying_capacity_fraction'] = (
                        df2_adj['yield_fraction'] * df2_adj['land_mha'] / orig_land
                    )

                orig_soc = df1.iloc[0]['soc_total']
                if orig_soc > 0:
                    df2_adj['soc_fraction'] = df2_adj['soc_total'] / orig_soc

                combined = pd.concat([df1, df2_adj.iloc[1:]], ignore_index=True)
                results[label][rn] = combined

    return results


# ================================================================
# ANALYSIS 7: Investment-then-disruption
# ================================================================
def run_investment_then_disruption(regions):
    """Two scenarios: current SOC vs degraded SOC (75%) facing disruption."""
    print('[7/7] Investment-then-disruption scenarios...')
    s3 = get_scenario_params()['S3']
    focus_regions = ['north_america', 'south_asia', 'sub_saharan_africa']
    results = {}

    for rn in focus_regions:
        r = regions[rn]
        results[rn] = {}

        # Scenario A: Current SOC, disruption at year 0
        model = make_model(r, s3, rn)
        results[rn]['no_invest_immediate'] = model.run()

        # Scenario B: Degraded (75% SOC), disruption at year 0
        r_deg = copy.deepcopy(r)
        r_deg.soc_initial = r.soc_initial * 0.75
        model = make_model(r_deg, s3, rn)
        results[rn]['degraded_immediate'] = model.run()

    return results


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('SOIL HEALTH AS RESILIENCE INFRASTRUCTURE')
    print('Monthly Coupled Model Analysis (30-year horizon)')
    print('=' * 70)

    regions = get_default_regions()
    mp = MonthlyNParams()
    shock = calibrate_price_shock(0.20)
    print(f'\nCalibrated price shock: {shock:.4f} ({shock*100:.1f}%)')
    print(f'Time horizon: {T_MAX} years')
    print(f'Regions: {len(regions)}')

    # Calibrate and cache all ym values
    print('\nCalibrating yield_max for each region...')
    for rk in REGIONAL_CLIMATES:
        ym = get_calibrated_ym(rk, mp)
        r = regions[rk]
        print(f'  {REGION_LABELS[rk]:<20} ym={ym:.3f}  SOC={r.soc_initial:.0f}  '
              f'F={r.synth_n_current:.0f}  Area={r.cropland_mha:.0f} Mha')

    # Run all analyses
    baseline, sc = run_baseline(regions)
    buffer_metrics = compute_soil_buffer_metrics(regions, baseline)
    degradation = run_degradation_scenarios(regions)
    no_shock = run_no_shock_baseline(regions)
    nue = run_nue_sensitivity(regions)
    duration = run_duration_comparison(regions)
    investment = run_investment_then_disruption(regions)

    # ---- Print key results ----
    print('\n' + '=' * 70)
    print('KEY RESULTS')
    print('=' * 70)

    # Regional yield loss at year 10 (S3)
    print('\nRegional yield loss at year 10 (S3):')
    for rn in regions:
        df = baseline['S3'][rn]
        row = df[df['year'] == 10]
        if len(row):
            loss = (1 - row['yield_fraction'].iloc[0]) * 100
            print(f'  {REGION_LABELS[rn]:<20} {loss:6.1f}%  (SOC={regions[rn].soc_initial:.0f} tC/ha)')

    # Supply-constrained results
    print('\nSupply-constrained yield loss at year 10:')
    for sc_name in sorted(sc.keys()):
        gw = compute_global_weighted(sc[sc_name], regions, 'yield_fraction')
        val = gw.loc[10] if 10 in gw.index else gw.iloc[-1]
        loss = (1 - val) * 100
        print(f'  {sc_name}: global weighted loss = {loss:.1f}%')

    # Buffer metrics
    print('\nSoil N buffer metrics:')
    print(f'{"Region":<20} {"SOC":>5} {"Buffer":>7} {"Dep":>6} {"Loss@10":>8} {"Loss@30":>8}')
    for rn in regions:
        m = buffer_metrics[rn]
        print(f'{REGION_LABELS[rn]:<20} {m["soc_initial"]:>5.0f} {m["soil_buffer_ratio"]:>7.1%} '
              f'{m["synth_dependency"]:>6.1%} {m["yield_loss_10yr"]:>8.1f}% '
              f'{m["yield_loss_30yr"]:>8.1f}%')

    # Degradation amplification
    print('\nDegradation amplification (disruption penalty at year 10):')
    print(f'{"Region":<20} {"100%":>8} {"75%":>8} {"50%":>8} {"25%":>8}')
    for rn in regions:
        line = f'{REGION_LABELS[rn]:<20}'
        for label in ['SOC_100pct', 'SOC_75pct', 'SOC_50pct', 'SOC_25pct']:
            df_ns = no_shock[label][rn]
            df_d = degradation[label][rn]
            ns_row = df_ns[df_ns['year'] == 10]
            d_row = df_d[df_d['year'] == 10]
            if len(ns_row) and len(d_row):
                ns_yield = ns_row['yield_tha'].iloc[0]
                d_yield = d_row['yield_tha'].iloc[0]
                if ns_yield > 0:
                    disruption_penalty = (1 - d_yield / ns_yield) * 100
                    line += f' {disruption_penalty:>7.1f}%'
                else:
                    line += f' {"N/A":>8}'
        print(line)

    # NUE sensitivity
    print('\nNUE sensitivity (global weighted yield loss at year 10):')
    for label in sorted(nue.keys()):
        gw = compute_global_weighted(nue[label], regions, 'yield_fraction')
        val = gw.loc[10] if 10 in gw.index else gw.iloc[-1]
        loss = (1 - val) * 100
        print(f'  {label}: {loss:.1f}%')

    # Duration comparison
    print('\nDuration comparison (global weighted yield loss at year 10):')
    for label in sorted(duration.keys()):
        gw = compute_global_weighted(duration[label], regions, 'yield_fraction')
        val = gw.loc[10] if 10 in gw.index else gw.iloc[-1]
        loss = (1 - val) * 100
        print(f'  {label}: {loss:.1f}%')

    # ---- Save ----
    all_results = {
        'baseline': baseline,
        'supply_constrained': sc,
        'buffer_metrics': buffer_metrics,
        'degradation': degradation,
        'no_shock_baseline': no_shock,
        'nue_sensitivity': nue,
        'duration_comparison': duration,
        'investment': investment,
        'regions': {rn: r for rn, r in regions.items()},
        'shock': shock,
        't_max': T_MAX,
        'model_type': 'monthly_v3',
    }
    pkl_path = os.path.join(data_dir, 'resilience_monthly.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nAll results saved: {pkl_path}')
    print('\nDone.')
