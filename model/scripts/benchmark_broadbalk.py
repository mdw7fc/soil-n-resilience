"""Tier-1 LTE benchmark: Century v3 vs MEMS v1 against Broadbalk + Morrow.

Site-specific runner. The main-paper regional model uses EU soc_initial=42
t C/ha (0-30 cm, LUCAS 2018 avg). Broadbalk Nil actual 1843 = 28.8 t C/ha
at 0-23 cm. This script reinitializes each model with Rothamsted's actual
initial conditions so model-vs-observation comparison is apples-to-apples.

Treatments benchmarked (Broadbalk, 1843-2015 = 172 yr):
  - Nil        (Plot 3)        no fertilizer, no manure
  - PK         (Plot 5)        PK only; no N — same N regime as Nil for model
  - N3PK       (Plot 8)        144 kg N/ha/yr synthetic + PK
  - FYM1843    (Plot 2.2)      35 t fresh FYM/ha/yr since 1843
  - FYM+N3     (Plot 2.1)      35 t FYM + 96 kg N/ha  (approx. since 2005;
                                approximated for full 172 yr)

Morrow benchmark: unfertilized continuous-corn yields, decadal means 1880s-2020s.

Scoring:
  RMSE(SOC), Willmott d, endpoint bias, Pearson pattern match.

Outputs: /model/data/benchmark_broadbalk/
  - lte_scoreboard.csv         per-site × treatment × model × metric
  - soc_trajectories.csv       modeled + observed SOC time series
  - yield_trajectories.csv     Morrow yield modeled + observed decadal means
  - summary.md                 text summary

Author: Matthew Wallenstein
Date: 2026-04-24
"""
from __future__ import annotations

import sys
import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from soil_n_model import (
    CropParams, get_default_regions, SOMPoolParams,
)
from monthly_model_v3 import (
    MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES, FAOSTAT_TARGETS,
    monthly_n_balance, update_som_pools,
)
from monthly_mems_v1 import (
    MEMSPoolParams, mems_annual_step, mems_spinup, monthly_n_balance_mems,
)


OUT = ROOT / 'data' / 'benchmark_broadbalk'
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Rothamsted-specific climate & site parameters
# ---------------------------------------------------------------------------
# Rothamsted Manor meteorological station long-term means (Rothamsted 2024
# handout p.3):
#   MAT ≈ 10.1 °C, annual rainfall ≈ 720 mm, winter wheat.
# EU default is NW Europe (temp 3-19, precip 660 mm). We use a slightly
# milder/wetter Rothamsted approximation.
ROTHAMSTED_CLIMATE = MonthlyClimate(
    name='Rothamsted (SE England)',
    temp=[3.5, 4.0, 6.5, 8.5, 12.0, 15.0, 17.2, 17.0, 14.2, 10.5, 6.5, 4.0],
    precip=[75, 50, 55, 55, 55, 60, 55, 60, 60, 75, 75, 75],
    pet=[12, 20, 35, 55, 85, 100, 110, 95, 60, 35, 15, 10],
    planting_month=10, maturity_month=7,
)

# Broadbalk site parameters
BROADBALK_INITIAL_SOC = 28.8   # t C/ha, 0-23 cm  (e-RA dataset; 1.00% C × 2.88e6 kg/ha)
BROADBALK_CLAY_SILT   = 0.70   # Batcombe silty clay loam, 19-39% clay + ~50% silt
BROADBALK_CN_BULK     = 10.0
BROADBALK_ATM_N_DEP   = 22.0   # kg N/ha/yr, post-1980s avg (Broadbalk handout)
BROADBALK_BNF_FREE    = 5.0    # free-living fixation (temperate cereal)


@dataclass
class BroadbalkTreatment:
    name: str
    plot: str
    synth_n: float           # kg N/ha/yr inorganic/available N equivalent
    fym_c_input: float       # t C/ha/yr additional OM input from manure
    fym_mineralized_n: float # kg N/ha/yr of organic N mineralized within season
    description: str = ''

# Treatment parameterization
#
# FYM rate: 35 t fresh/ha/yr (since 1843) ≈ 3.4 t C/ha/yr total C applied.
# Poulton et al. 2018 (GCB) report 11% of applied C retained over 157 yr →
# net SOC gain ~55 t C/ha (observed 1843→2015: 28.8→84.3 t C/ha). The model
# needs a "retention-adjusted" input that yields this net trajectory
# (not the full 3.4 t/ha gross). Calibrated value = 2.0 t C/ha/yr produces
# trajectory close to observed asymptote under Century's active-pool k.
#
# FYM N: ~240 kg total N/ha/yr, ~100-120 kg/ha plant-available within year
# of application (Poulton 2018; Johnston & Poulton 2022).

TREATMENTS = [
    BroadbalkTreatment('Nil',      'P3',   synth_n=0.0,   fym_c_input=0.0,   fym_mineralized_n=0.0,
                       description='No fertilizer, no manure; 1843-present'),
    BroadbalkTreatment('PK',       'P5',   synth_n=0.0,   fym_c_input=0.0,   fym_mineralized_n=0.0,
                       description='PK only; no N — in model identical to Nil (P+K not limiting in N-only model)'),
    BroadbalkTreatment('N3PK',     'P8',   synth_n=144.0, fym_c_input=0.0,   fym_mineralized_n=0.0,
                       description='144 kg N/ha/yr + PKMg since 1852'),
    BroadbalkTreatment('FYM1843',  'P2.2', synth_n=0.0,   fym_c_input=2.0,   fym_mineralized_n=110.0,
                       description='35 t FYM/ha/yr since 1843 (C input retention-adjusted to 2.0 t C/ha/yr)'),
    BroadbalkTreatment('FYM+N3',   'P2.1', synth_n=96.0,  fym_c_input=2.0,   fym_mineralized_n=110.0,
                       description='35 t FYM/ha/yr + 96 kg N (post-1968 approx, applied full 172 yr)'),
]


# ---------------------------------------------------------------------------
# Core annual-loop runners — SITE-SPECIFIC INITIAL CONDITIONS
# ---------------------------------------------------------------------------

def run_century_site(
    soc_initial: float, climate: MonthlyClimate, n_years: int,
    synth_n: float, fym_c_input: float, fym_mineralized_n: float,
    atm_n_dep: float, bnf: float,
    cn_bulk: float = 10.0,
    yield_max: float = 8.0,
    yield_min: float = 1.0,
    mit_c: float = 0.025,
    residue_retention: float = 0.90,
    root_shoot: float = 0.80,
    cre: float = 0.26,
    p: MonthlyNParams = None,
    som: SOMPoolParams = None,
) -> dict:
    """Run Century v3 forward n_years starting from soc_initial with specified
    site parameters and treatment inputs."""
    if p is None:
        p = MonthlyNParams()
    if som is None:
        som = SOMPoolParams()
    crop = CropParams()

    # Initial pool allocation from fractional split (will transient for ~100 yr,
    # matching actual 1843 prairie→Broadbalk transition)
    c_a = soc_initial * som.f_active
    c_s = soc_initial * som.f_slow
    c_p = soc_initial * (1 - som.f_active - som.f_slow)

    n_grain_t = crop.grain_n_fraction * 1000
    hi = crop.harvest_index
    rf = (1 - hi) / hi

    mineral_n = 12.0
    years, soc_traj, yield_traj = [], [], []
    n_min_traj, n_lch_traj, n_uptake_traj = [], [], []

    for yr in range(n_years):
        # Treatment-specific N: synth + manure-mineralized-N
        effective_synth_n = synth_n + fym_mineralized_n

        nb = monthly_n_balance(
            c_a, c_s, c_p, cn_bulk, effective_synth_n, bnf,
            atm_n_dep, climate, mineral_n, p)
        mineral_n = nb['mineral_n_end']

        # Yield
        n_eff = nb['uptake']
        y = yield_max * (1 - np.exp(-mit_c * n_eff))
        y_stoich = n_eff / n_grain_t if n_grain_t > 0 else y
        y = min(y, y_stoich)
        y = max(y, yield_min)

        # Residue C input = crop + manure
        shoot_c = y * 1000 * 0.45 * rf * residue_retention / 1000
        root_c = y * 1000 * 0.45 * rf * root_shoot / 1000
        c_in_crop = (shoot_c + root_c) * cre
        c_in = c_in_crop + fym_c_input  # manure OM enters pool directly

        c_a, c_s, c_p = update_som_pools(c_a, c_s, c_p, c_in, som)

        years.append(yr)
        soc_traj.append(c_a + c_s + c_p)
        yield_traj.append(y)
        n_min_traj.append(nb['min'])
        n_lch_traj.append(nb['leach'])
        n_uptake_traj.append(nb['uptake'])

    return {
        'year': np.array(years),
        'soc': np.array(soc_traj),
        'yield_tha': np.array(yield_traj),
        'n_min': np.array(n_min_traj),
        'n_leach': np.array(n_lch_traj),
        'n_uptake': np.array(n_uptake_traj),
    }


def run_mems_site(
    soc_initial: float, climate: MonthlyClimate, n_years: int,
    synth_n: float, fym_c_input: float, fym_mineralized_n: float,
    atm_n_dep: float, bnf: float,
    clay_silt: float = 0.55,
    yield_max: float = 8.0,
    yield_min: float = 1.0,
    mit_c: float = 0.025,
    residue_retention: float = 0.90,
    root_shoot: float = 0.80,
    cre: float = 0.26,
    p: MonthlyNParams = None,
    mems: MEMSPoolParams = None,
) -> dict:
    """Run MEMS v1 forward n_years starting from soc_initial with specified
    site parameters and treatment inputs."""
    if p is None:
        p = MonthlyNParams()
    if mems is None:
        mems = MEMSPoolParams()
    crop = CropParams()

    qmax = mems.qmax_per_claysilt * clay_silt
    pools = mems_spinup(soc_initial, qmax, mems)
    c_pom, c_dom, c_mbc, c_maom = (
        pools['c_pom'], pools['c_dom'], pools['c_mbc'], pools['c_maom']
    )
    pom_baseline = c_pom

    n_grain_t = crop.grain_n_fraction * 1000
    hi = crop.harvest_index
    rf = (1 - hi) / hi

    # Baseline N for CUE calc
    baseline_n_total = synth_n + bnf + fym_mineralized_n + 50.0
    # Seed residue input with a reasonable cereal crop year 1
    lagged_c_input = max(0.5, 2.0 * cre) + fym_c_input

    mineral_n = 12.0
    prev_n_total = baseline_n_total

    years, soc_traj, yield_traj = [], [], []
    n_min_traj, n_lch_traj, n_uptake_traj = [], [], []
    cue_traj, maom_traj, pom_traj = [], [], []

    for yr in range(n_years):
        n_frac = max(0.01, prev_n_total / max(baseline_n_total, 1.0))

        step = mems_annual_step(
            c_pom, c_dom, c_mbc, c_maom,
            c_input=lagged_c_input, qmax=qmax, mems=mems,
            n_available_frac=n_frac, pom_baseline=pom_baseline,
        )
        annual_n_min = max(0, step['net_n_mineralized'])
        c_pom, c_dom, c_mbc, c_maom = (
            step['c_pom'], step['c_dom'], step['c_mbc'], step['c_maom'])

        effective_synth_n = synth_n + fym_mineralized_n

        nb = monthly_n_balance_mems(
            annual_n_min, effective_synth_n, bnf,
            atm_n_dep, climate, mineral_n, p)
        mineral_n = nb['mineral_n_end']

        n_eff = nb['uptake']
        y = yield_max * (1 - np.exp(-mit_c * n_eff))
        y_stoich = n_eff / n_grain_t if n_grain_t > 0 else y
        y = min(y, y_stoich)
        y = max(y, yield_min)

        shoot_c = y * 1000 * 0.45 * rf * residue_retention / 1000
        root_c = y * 1000 * 0.45 * rf * root_shoot / 1000
        c_in_crop = (shoot_c + root_c) * cre
        lagged_c_input = c_in_crop + fym_c_input

        prev_n_total = effective_synth_n + bnf + annual_n_min

        soc = c_pom + c_dom + c_mbc + c_maom
        years.append(yr)
        soc_traj.append(soc)
        yield_traj.append(y)
        n_min_traj.append(nb['min'])
        n_lch_traj.append(nb['leach'])
        n_uptake_traj.append(nb['uptake'])
        cue_traj.append(step['cue'])
        maom_traj.append(c_maom)
        pom_traj.append(c_pom)

    return {
        'year': np.array(years),
        'soc': np.array(soc_traj),
        'yield_tha': np.array(yield_traj),
        'n_min': np.array(n_min_traj),
        'n_leach': np.array(n_lch_traj),
        'n_uptake': np.array(n_uptake_traj),
        'cue': np.array(cue_traj),
        'c_maom': np.array(maom_traj),
        'c_pom': np.array(pom_traj),
    }


# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------

def willmott_d(obs: np.ndarray, mod: np.ndarray) -> float:
    """Willmott (1981) index of agreement. 1 = perfect, 0 = no skill."""
    obs, mod = np.asarray(obs, dtype=float), np.asarray(mod, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(mod))
    obs, mod = obs[mask], mod[mask]
    if len(obs) < 2:
        return np.nan
    obs_mean = obs.mean()
    num = np.sum((mod - obs) ** 2)
    den = np.sum((np.abs(mod - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - num / den


def score_pair(obs: np.ndarray, mod: np.ndarray) -> dict:
    """Compute RMSE, Willmott d, bias (mod-obs mean), and Pearson pattern r."""
    obs, mod = np.asarray(obs, dtype=float), np.asarray(mod, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(mod))
    obs, mod = obs[mask], mod[mask]
    if len(obs) < 2:
        return {'n': len(obs), 'rmse': np.nan, 'willmott_d': np.nan,
                'bias': np.nan, 'pearson_r': np.nan, 'mean_obs': np.nan,
                'mean_mod': np.nan}
    rmse = float(np.sqrt(np.mean((mod - obs) ** 2)))
    d = willmott_d(obs, mod)
    bias = float(np.mean(mod - obs))
    if np.std(obs) == 0 or np.std(mod) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(obs, mod)[0, 1])
    return {'n': int(len(obs)), 'rmse': rmse, 'willmott_d': float(d) if not np.isnan(d) else np.nan,
            'bias': bias, 'pearson_r': r,
            'mean_obs': float(obs.mean()), 'mean_mod': float(mod.mean())}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark_broadbalk():
    print('=' * 100)
    print('BROADBALK BENCHMARK — Century v3 vs MEMS v1 (site-specific init)')
    print('=' * 100)
    print(f'  Initial SOC   : {BROADBALK_INITIAL_SOC} t C/ha (0-23 cm, actual 1843)')
    print(f'  Climate       : {ROTHAMSTED_CLIMATE.name}')
    print(f'  Clay+silt     : {BROADBALK_CLAY_SILT}')
    print(f'  Atm N dep     : {BROADBALK_ATM_N_DEP} kg N/ha/yr')
    print(f'  BNF           : {BROADBALK_BNF_FREE} kg N/ha/yr (free-living)')
    print(f'  Simulation    : 1843-2015 (172 yr)')
    print()

    obs_df = pd.read_csv(ROOT / 'data' / 'broadbalk_soc_all_treatments.csv')
    n_years = 2015 - 1843 + 1  # 173 years

    # Column in observed → treatment object
    treatment_cols = {
        'Nil': 'Nil_P3',
        'PK': 'PK_P5',
        'N3PK': 'N3PK_P8',
        'FYM1843': 'FYM1843_P2.2',
        'FYM+N3': 'FYM1885_P2.1',  # FYM1885 w/ +N since 2005 — imperfect proxy
    }

    scoreboard = []
    traj_rows = []

    for treat in TREATMENTS:
        print(f'--- {treat.name} (Plot {treat.plot}) ---')
        print(f'    {treat.description}')
        print(f'    synth_N={treat.synth_n}  fym_C={treat.fym_c_input}  fym_minN={treat.fym_mineralized_n}')

        # Run Century
        cent = run_century_site(
            soc_initial=BROADBALK_INITIAL_SOC, climate=ROTHAMSTED_CLIMATE,
            n_years=n_years, synth_n=treat.synth_n,
            fym_c_input=treat.fym_c_input,
            fym_mineralized_n=treat.fym_mineralized_n,
            atm_n_dep=BROADBALK_ATM_N_DEP, bnf=BROADBALK_BNF_FREE,
            cn_bulk=BROADBALK_CN_BULK, yield_max=11.0, yield_min=0.8,
        )

        # Run MEMS
        memr = run_mems_site(
            soc_initial=BROADBALK_INITIAL_SOC, climate=ROTHAMSTED_CLIMATE,
            n_years=n_years, synth_n=treat.synth_n,
            fym_c_input=treat.fym_c_input,
            fym_mineralized_n=treat.fym_mineralized_n,
            atm_n_dep=BROADBALK_ATM_N_DEP, bnf=BROADBALK_BNF_FREE,
            clay_silt=BROADBALK_CLAY_SILT, yield_max=11.0, yield_min=0.8,
        )

        # Observed
        col = treatment_cols[treat.name]
        obs = obs_df[['year', col]].dropna()
        obs_years = obs['year'].values
        obs_soc = obs[col].values

        # Sample modeled SOC at observed time points
        sim_years = 1843 + cent['year']
        idx = np.array([np.argmin(np.abs(sim_years - y)) for y in obs_years])
        cent_soc_at_obs = cent['soc'][idx]
        mems_soc_at_obs = memr['soc'][idx]

        # Score
        s_cent = score_pair(obs_soc, cent_soc_at_obs)
        s_mems = score_pair(obs_soc, mems_soc_at_obs)

        print(f'    CENTURY:  RMSE={s_cent["rmse"]:.2f}  d={s_cent["willmott_d"]:.3f}  '
              f'bias={s_cent["bias"]:+.2f}  r={s_cent["pearson_r"]:+.3f}  '
              f'mean_mod={s_cent["mean_mod"]:.1f} vs obs={s_cent["mean_obs"]:.1f}')
        print(f'    MEMS   :  RMSE={s_mems["rmse"]:.2f}  d={s_mems["willmott_d"]:.3f}  '
              f'bias={s_mems["bias"]:+.2f}  r={s_mems["pearson_r"]:+.3f}  '
              f'mean_mod={s_mems["mean_mod"]:.1f} vs obs={s_mems["mean_obs"]:.1f}')

        # Final yield
        final_y_cent = cent['yield_tha'][-20:].mean()
        final_y_mems = memr['yield_tha'][-20:].mean()
        print(f'    Final yield (last 20 yr mean):  Cent={final_y_cent:.2f}  MEMS={final_y_mems:.2f}')
        print()

        for model_name, scores, y_final in [
            ('century', s_cent, final_y_cent),
            ('mems', s_mems, final_y_mems),
        ]:
            scoreboard.append({
                'site': 'Broadbalk',
                'treatment': treat.name,
                'plot': treat.plot,
                'model': model_name,
                'n_obs': scores['n'],
                'rmse_soc': scores['rmse'],
                'willmott_d_soc': scores['willmott_d'],
                'bias_soc': scores['bias'],
                'pearson_r_soc': scores['pearson_r'],
                'mean_obs_soc': scores['mean_obs'],
                'mean_mod_soc': scores['mean_mod'],
                'final_yield_tha': y_final,
            })

        # Per-year trajectory output (annual resolution)
        for i, yr in enumerate(sim_years):
            obs_match = obs_df.loc[obs_df['year'] == yr, col]
            obs_val = float(obs_match.iloc[0]) if len(obs_match) and not pd.isna(obs_match.iloc[0]) else np.nan
            traj_rows.append({
                'site': 'Broadbalk',
                'treatment': treat.name,
                'plot': treat.plot,
                'year': int(yr),
                'soc_century': float(cent['soc'][i]),
                'soc_mems': float(memr['soc'][i]),
                'soc_obs': obs_val,
                'yield_century': float(cent['yield_tha'][i]),
                'yield_mems': float(memr['yield_tha'][i]),
                'c_pom_mems': float(memr['c_pom'][i]),
                'c_maom_mems': float(memr['c_maom'][i]),
                'cue_mems': float(memr['cue'][i]),
            })

    return scoreboard, traj_rows


def benchmark_morrow():
    """Morrow Plot 3N unfertilized continuous corn, 1888-2021."""
    print('=' * 100)
    print('MORROW BENCHMARK — Plot 3N Unfertilized Continuous Corn')
    print('=' * 100)

    # Morrow plots: Flanagan silt loam, central Illinois
    # MAT ≈ 11.0 °C, rainfall ≈ 1000 mm
    morrow_climate = MonthlyClimate(
        name='Urbana, IL (Morrow Plots)',
        temp=[-3.0, -1.0, 5.0, 11.0, 17.0, 22.0, 24.0, 23.0, 19.0, 12.0, 5.0, -1.0],
        precip=[50, 55, 75, 100, 110, 110, 100, 85, 75, 80, 75, 65],
        pet=[5, 10, 28, 60, 100, 130, 145, 125, 85, 50, 20, 8],
        planting_month=5, maturity_month=9,
    )
    initial_soc = 85.0   # t C/ha, 0-15 cm, estimated ~5.3% C pre-cultivation.
                         # Model uses 0-30 cm equivalent; for 0-15 use 5.3% × 1.3 g/cm³ × 15 cm = 103
                         # then degraded; we use 85 as 0-30 cm proxy at pre-1888 cultivation ~ Odell 1984.
    clay_silt = 0.78     # Flanagan silt loam = high silt+clay
    atm_n_dep = 10.0     # kg N/ha/yr, IL cornbelt
    bnf = 5.0
    n_years = 2021 - 1888 + 1  # 134 yr

    # Treatment: unfertilized CC, no manure
    cent = run_century_site(
        soc_initial=initial_soc, climate=morrow_climate,
        n_years=n_years, synth_n=0.0, fym_c_input=0.0, fym_mineralized_n=0.0,
        atm_n_dep=atm_n_dep, bnf=bnf,
        cn_bulk=10.0, yield_max=10.0, yield_min=0.5, mit_c=0.025,
        residue_retention=0.90, root_shoot=0.80, cre=0.28,
    )
    memr = run_mems_site(
        soc_initial=initial_soc, climate=morrow_climate,
        n_years=n_years, synth_n=0.0, fym_c_input=0.0, fym_mineralized_n=0.0,
        atm_n_dep=atm_n_dep, bnf=bnf,
        clay_silt=clay_silt, yield_max=10.0, yield_min=0.5, mit_c=0.025,
        residue_retention=0.90, root_shoot=0.80, cre=0.28,
    )

    # Observed decadal means (t/ha grain, unfertilized CC)
    morrow_obs = pd.read_csv(ROOT / 'data' / 'morrow_unfertilized_yield_summary.csv')
    dec_obs = morrow_obs[morrow_obs['decade'].str.contains('s$', na=False)].copy()
    # Parse decade start year
    dec_obs['decade_start'] = dec_obs['decade'].str.rstrip('s').astype(int)
    dec_obs = dec_obs[(dec_obs['decade_start'] >= 1880) & (dec_obs['decade_start'] <= 2020)]

    # Sample modeled decadal means
    sim_years = 1888 + cent['year']
    obs_y, cent_y, mems_y, dec_labels = [], [], [], []
    for _, row in dec_obs.iterrows():
        ds = int(row['decade_start'])
        mask = (sim_years >= ds) & (sim_years < ds + 10)
        if mask.sum() == 0:
            continue
        obs_y.append(float(row['mean_yield_t_ha']))
        cent_y.append(float(cent['yield_tha'][mask].mean()))
        mems_y.append(float(memr['yield_tha'][mask].mean()))
        dec_labels.append(row['decade'])

    obs_y = np.array(obs_y)
    cent_y = np.array(cent_y)
    mems_y = np.array(mems_y)

    # Split pre-modern (1888-1967) vs modern (1968-2021) — modern includes
    # cultivar + herbicide improvements no SOM model will capture.
    dec_arr = np.array([int(d.rstrip('s')) for d in dec_labels])
    premod = dec_arr < 1968
    modern = dec_arr >= 1968

    s_cent = score_pair(obs_y, cent_y)
    s_mems = score_pair(obs_y, mems_y)
    s_cent_pre = score_pair(obs_y[premod], cent_y[premod])
    s_mems_pre = score_pair(obs_y[premod], mems_y[premod])
    s_cent_mod = score_pair(obs_y[modern], cent_y[modern])
    s_mems_mod = score_pair(obs_y[modern], mems_y[modern])

    print(f'    FULL (1880s-2020s):')
    print(f'      CENTURY:  RMSE={s_cent["rmse"]:.2f}  d={s_cent["willmott_d"]:.3f}  '
          f'bias={s_cent["bias"]:+.2f}  r={s_cent["pearson_r"]:+.3f}')
    print(f'      MEMS   :  RMSE={s_mems["rmse"]:.2f}  d={s_mems["willmott_d"]:.3f}  '
          f'bias={s_mems["bias"]:+.2f}  r={s_mems["pearson_r"]:+.3f}')
    print(f'    PRE-MODERN (1888-1967): observed mean = {obs_y[premod].mean():.2f} t/ha')
    print(f'      CENTURY:  RMSE={s_cent_pre["rmse"]:.2f}  d={s_cent_pre["willmott_d"]:.3f}  '
          f'bias={s_cent_pre["bias"]:+.2f}  mean_mod={s_cent_pre["mean_mod"]:.2f}')
    print(f'      MEMS   :  RMSE={s_mems_pre["rmse"]:.2f}  d={s_mems_pre["willmott_d"]:.3f}  '
          f'bias={s_mems_pre["bias"]:+.2f}  mean_mod={s_mems_pre["mean_mod"]:.2f}')
    print(f'    MODERN ERA (1968-2021): observed mean = {obs_y[modern].mean():.2f} t/ha '
          f'(cultivar + herbicide effects not captured by SOM-only models)')
    print(f'      CENTURY:  RMSE={s_cent_mod["rmse"]:.2f}  bias={s_cent_mod["bias"]:+.2f}  '
          f'mean_mod={s_cent_mod["mean_mod"]:.2f}')
    print(f'      MEMS   :  RMSE={s_mems_mod["rmse"]:.2f}  bias={s_mems_mod["bias"]:+.2f}  '
          f'mean_mod={s_mems_mod["mean_mod"]:.2f}')

    # Final SOC vs Morrow SOM data (~3.2% SOM = ~1.86% C at 0-15 cm ≈ 36 t C/ha at 0-15)
    final_soc_cent = cent['soc'][-20:].mean()
    final_soc_mems = memr['soc'][-20:].mean()
    print(f'    Final SOC (last 20 yr mean, 0-30 cm equiv):  Cent={final_soc_cent:.1f}  MEMS={final_soc_mems:.1f}')
    print(f'    Observed modern SOC ≈ 36 t C/ha (0-15 cm) → ~55-65 t C/ha (0-30 cm extrap)')
    print()

    scoreboard = []
    for era_tag, sc, sm in [
        ('full_1888-2021', s_cent, s_mems),
        ('pre_modern_1888-1967', s_cent_pre, s_mems_pre),
        ('modern_1968-2021', s_cent_mod, s_mems_mod),
    ]:
        for model_name, scores, final_soc in [
            ('century', sc, final_soc_cent),
            ('mems', sm, final_soc_mems),
        ]:
            scoreboard.append({
                'site': 'Morrow',
                'treatment': 'Unfert_CC',
                'plot': 'P3N',
                'era': era_tag,
                'model': model_name,
                'n_obs': scores['n'],
                'rmse_yield': scores['rmse'],
                'willmott_d_yield': scores['willmott_d'],
                'bias_yield': scores['bias'],
                'pearson_r_yield': scores['pearson_r'],
                'mean_obs_yield': scores['mean_obs'],
                'mean_mod_yield': scores['mean_mod'],
                'final_soc': final_soc,
            })

    yield_rows = []
    for lab, o, c, m in zip(dec_labels, obs_y, cent_y, mems_y):
        yield_rows.append({
            'site': 'Morrow', 'treatment': 'Unfert_CC', 'plot': 'P3N',
            'decade': lab, 'yield_obs': o, 'yield_century': c, 'yield_mems': m,
        })
    return scoreboard, yield_rows


def write_summary(bb_board, bb_traj, mw_board, mw_yield):
    # Broadbalk SOC scoreboard
    df_bb = pd.DataFrame(bb_board)
    df_bb.to_csv(OUT / 'lte_scoreboard_broadbalk.csv', index=False)

    # Morrow yield scoreboard
    df_mw = pd.DataFrame(mw_board)
    df_mw.to_csv(OUT / 'lte_scoreboard_morrow.csv', index=False)

    # Trajectories
    pd.DataFrame(bb_traj).to_csv(OUT / 'soc_trajectories_broadbalk.csv', index=False)
    pd.DataFrame(mw_yield).to_csv(OUT / 'yield_decadal_morrow.csv', index=False)

    # Headline winner per site
    def pick_winner(sub):
        """Return 'century' / 'mems' / 'tie'. Use Willmott d as primary, RMSE tiebreak."""
        cent = sub[sub['model'] == 'century'].iloc[0]
        mems = sub[sub['model'] == 'mems'].iloc[0]
        d_col = 'willmott_d_soc' if 'willmott_d_soc' in cent.index else 'willmott_d_yield'
        rmse_col = 'rmse_soc' if 'rmse_soc' in cent.index else 'rmse_yield'
        dc, dm = cent[d_col], mems[d_col]
        if np.isnan(dc) or np.isnan(dm):
            return 'undetermined'
        if abs(dc - dm) < 0.02:
            rc, rm = cent[rmse_col], mems[rmse_col]
            if abs(rc - rm) / max(rc, 0.01) < 0.05:
                return 'tie'
            return 'century' if rc < rm else 'mems'
        return 'century' if dc > dm else 'mems'

    lines = []
    lines.append('# Tier-1 LTE Benchmark — Century v3 vs MEMS v1\n')
    lines.append(f'*Run: 2026-04-24. Site-specific initial conditions '
                 f'(Broadbalk SOC₀=28.8 t C/ha 0-23 cm; Morrow SOC₀=85 t C/ha 0-30 cm).*\n')
    lines.append('## Broadbalk SOC trajectory match (172 yr)\n')
    lines.append('| Treatment | Plot | Model | n_obs | RMSE (t C/ha) | Willmott d | Bias | Pearson r | Final yield (t/ha) |')
    lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
    for treat in ['Nil', 'PK', 'N3PK', 'FYM1843', 'FYM+N3']:
        sub = df_bb[df_bb['treatment'] == treat]
        for _, row in sub.iterrows():
            lines.append(
                f"| {treat} | {row['plot']} | {row['model'].upper()} | {row['n_obs']} | "
                f"{row['rmse_soc']:.2f} | {row['willmott_d_soc']:.3f} | "
                f"{row['bias_soc']:+.2f} | {row['pearson_r_soc']:+.3f} | "
                f"{row['final_yield_tha']:.2f} |"
            )
        winner = pick_winner(sub)
        lines.append(f"|   |   | **Winner → {winner}** |  |  |  |  |  |  |")
    lines.append('')
    lines.append('## Morrow yield decadal match (Plot 3N)\n')
    lines.append('*Split by era: pre-modern (1888-1967) is the clean SOM-driven test; '
                 'modern era (1968+) includes cultivar/herbicide effects not captured '
                 'by SOM-only models.*\n')
    lines.append('| Era | Model | n_dec | RMSE (t/ha) | Willmott d | Bias | Pearson r | Mean mod / obs |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---|')
    for era in ['pre_modern_1888-1967', 'modern_1968-2021', 'full_1888-2021']:
        sub = df_mw[df_mw['era'] == era]
        for _, row in sub.iterrows():
            lines.append(
                f"| {era} | {row['model'].upper()} | {row['n_obs']} | "
                f"{row['rmse_yield']:.2f} | {row['willmott_d_yield']:.3f} | "
                f"{row['bias_yield']:+.2f} | {row['pearson_r_yield']:+.3f} | "
                f"{row['mean_mod_yield']:.2f} / {row['mean_obs_yield']:.2f} |"
            )
        # Winner for this era
        if len(sub) == 2:
            sc = sub[sub['model'] == 'century'].iloc[0]
            sm = sub[sub['model'] == 'mems'].iloc[0]
            win = 'century' if sc['rmse_yield'] < sm['rmse_yield'] else 'mems'
            lines.append(f"| {era} | **Winner → {win}** |  |  |  |  |  |  |")
    final_soc = df_mw.iloc[0]['final_soc']
    final_soc_m = df_mw[df_mw['model'] == 'mems'].iloc[0]['final_soc']
    lines.append(f"\nFinal-SOC check (last 20 yr mean, 0-30 cm proxy): "
                 f"Century={final_soc:.1f} t C/ha, MEMS={final_soc_m:.1f}. "
                 f"Observed modern SOM ≈ 3.2% → ~36 t C/ha at 0-15 cm "
                 f"(≈ 55-65 t C/ha extrapolated to 0-30 cm).\n")

    lines.append('\n## Data files\n')
    lines.append('- `lte_scoreboard_broadbalk.csv`  — per-treatment × model scores')
    lines.append('- `lte_scoreboard_morrow.csv`    — Morrow yield scores')
    lines.append('- `soc_trajectories_broadbalk.csv` — annual SOC modeled + observed')
    lines.append('- `yield_decadal_morrow.csv`     — decadal yield modeled + observed')

    (OUT / 'summary.md').write_text('\n'.join(lines))
    print(f'Summary → {OUT / "summary.md"}')


def main():
    bb_board, bb_traj = benchmark_broadbalk()
    mw_board, mw_yield = benchmark_morrow()
    write_summary(bb_board, bb_traj, mw_board, mw_yield)

    print('\n' + '=' * 100)
    print('BENCHMARK COMPLETE')
    print('=' * 100)
    print(f'Outputs → {OUT}')


if __name__ == '__main__':
    main()
