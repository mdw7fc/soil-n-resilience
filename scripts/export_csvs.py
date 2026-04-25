#!/usr/bin/env python3
"""
Export all CSV data files from the pickle outputs of
run_price_shock_analysis.py and run_resilience_monthly.py.

Reproduces the nine CSVs in ``data/`` so that the human-readable
snapshots stay in sync with the latest model runs.

Usage (from repo root):
    python3 scripts/export_csvs.py
"""

import csv
import os
import pickle
import sys
from dataclasses import fields
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR / 'model'))
DATA_DIR = REPO_DIR / 'data'

REGION_ORDER = [
    'north_america', 'europe', 'east_asia', 'south_asia',
    'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia',
]


def _round_num(x, ndigits=4):
    """Round numeric values for CSV readability (leave non-numerics as-is)."""
    try:
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return x


def export_buffer_metrics(m, out_path):
    cols = ['region', 'soc_initial', 'synth_n', 'n_mineralized_baseline',
            'total_n_baseline', 'soil_buffer_ratio', 'synth_dependency',
            'yield_loss_10yr', 'yield_loss_30yr',
            'soc_fraction_10yr', 'soc_fraction_30yr']
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rn in REGION_ORDER:
            if rn not in m['buffer_metrics']:
                continue
            b = m['buffer_metrics'][rn]
            w.writerow([rn] + [b[c] for c in cols[1:]])


def export_degradation(m, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['soc_level', 'region', 'year', 'yield_tha',
                    'soc_total', 'n_mineralized'])
        for lvl, regions in m['degradation'].items():
            for rn in REGION_ORDER:
                if rn not in regions:
                    continue
                df = regions[rn]
                for _, row in df.iterrows():
                    w.writerow([lvl, rn, int(row['year']),
                                _round_num(row['yield_tha']),
                                _round_num(row['soc_total'], 2),
                                _round_num(row['n_mineralized'], 2)])


def export_duration(m, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['duration', 'region', 'year', 'yield_tha', 'soc_total'])
        for dur, regions in m['duration_comparison'].items():
            for rn in REGION_ORDER:
                if rn not in regions:
                    continue
                df = regions[rn]
                for _, row in df.iterrows():
                    w.writerow([dur, rn, int(row['year']),
                                _round_num(row['yield_tha']),
                                _round_num(row['soc_total'], 2)])


def export_nue(m, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['nue', 'region', 'year', 'yield_tha', 'soc_total'])
        for nue_key, regions in m['nue_sensitivity'].items():
            nue = float(nue_key.split('_')[1])
            for rn in REGION_ORDER:
                if rn not in regions:
                    continue
                df = regions[rn]
                for _, row in df.iterrows():
                    w.writerow([nue, rn, int(row['year']),
                                _round_num(row['yield_tha']),
                                _round_num(row['soc_total'], 2)])


def export_scenario_trajectories(m, out_path):
    """Baseline (S1/S2/S3) scenario trajectories, production-weighted or
    per-region. The committed CSV is per-region/per-year for each scenario."""
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'region', 'year', 'yield_tha', 'soc_total',
                    'n_mineralized', 'fert_applied_kgha',
                    'carrying_capacity_fraction'])
        for scen, regions in m['baseline'].items():
            for rn in REGION_ORDER:
                if rn not in regions:
                    continue
                df = regions[rn]
                yield_0 = df[df['year'] == 0]['yield_tha'].iloc[0]
                for _, row in df.iterrows():
                    cc = (row['yield_tha'] / yield_0) if yield_0 > 0 else 0
                    w.writerow([scen, rn, int(row['year']),
                                _round_num(row['yield_tha']),
                                _round_num(row['soc_total'], 2),
                                _round_num(row['n_mineralized'], 2),
                                _round_num(row['fert_applied_kgha'], 2),
                                _round_num(cc)])


def export_supply_constrained(m, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'region', 'year', 'yield_tha',
                    'soc_total', 'fert_applied_kgha'])
        for scen, regions in m['supply_constrained'].items():
            for rn in REGION_ORDER:
                if rn not in regions:
                    continue
                df = regions[rn]
                for _, row in df.iterrows():
                    w.writerow([scen, rn, int(row['year']),
                                _round_num(row['yield_tha']),
                                _round_num(row['soc_total'], 2),
                                _round_num(row['fert_applied_kgha'], 2)])


def export_price_shock_farm(psa, out_path):
    """Fine SOC gradient under 100% price shock, per region."""
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'soc_pct', 'fert_reduction_pct',
                    'yield_penalty_pct', 'gross_margin_change_pct',
                    'fert_cost_saved', 'yield_value_lost'])
        for rn in REGION_ORDER:
            if rn not in psa['fine_shock_results']:
                continue
            fr = psa['fine_shock_results'][rn]
            for i, soc in enumerate(fr['soc_pct']):
                w.writerow([
                    rn, soc,
                    fr['fert_red'][i],
                    fr['yield_pen'][i],
                    fr['profit_chg'][i],
                    fr['fert_saved'][i],
                    fr['yield_lost_value'][i],
                ])


def export_soc_gradient(soc_fine, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'soc_pct', 'total_penalty', 'ctrl_penalty',
                    'yield_shock', 'yield_noshock'])
        for rn in REGION_ORDER:
            if rn not in soc_fine['fine_results']:
                continue
            fr = soc_fine['fine_results'][rn]
            for i, soc in enumerate(fr['soc_pct']):
                w.writerow([rn, soc,
                            fr['total_penalty'][i],
                            fr['ctrl_penalty'][i],
                            fr['yield_shock'][i],
                            fr['yield_noshock'][i]])


def export_regional_parameters(regions, out_path):
    first_rn = next(iter(regions))
    field_names = [f.name for f in fields(regions[first_rn])]
    cols = sorted(field_names)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region'] + cols)
        for rn in REGION_ORDER:
            if rn not in regions:
                continue
            r = regions[rn]
            w.writerow([rn] + [getattr(r, c) for c in cols])


def main():
    with open(DATA_DIR / 'resilience_monthly.pkl', 'rb') as f:
        monthly = pickle.load(f)
    with open(DATA_DIR / 'price_shock_analysis.pkl', 'rb') as f:
        psa = pickle.load(f)
    with open(DATA_DIR / 'soc_gradient_fine.pkl', 'rb') as f:
        soc_fine = pickle.load(f)

    export_buffer_metrics(monthly, DATA_DIR / 'buffer_metrics.csv')
    export_degradation(monthly, DATA_DIR / 'degradation_scenarios.csv')
    export_duration(monthly, DATA_DIR / 'duration_comparison.csv')
    export_nue(monthly, DATA_DIR / 'nue_sensitivity.csv')
    export_scenario_trajectories(monthly, DATA_DIR / 'scenario_trajectories.csv')
    export_supply_constrained(monthly, DATA_DIR / 'supply_constrained.csv')
    export_price_shock_farm(psa, DATA_DIR / 'price_shock_farm.csv')
    export_soc_gradient(soc_fine, DATA_DIR / 'soc_gradient.csv')
    export_regional_parameters(monthly['regions'],
                               DATA_DIR / 'regional_parameters.csv')

    for fname in sorted(DATA_DIR.glob('*.csv')):
        n = sum(1 for _ in open(fname)) - 1
        print(f'  {fname.name}: {n} rows')


if __name__ == '__main__':
    main()
