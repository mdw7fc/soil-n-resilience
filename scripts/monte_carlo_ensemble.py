#!/usr/bin/env python3
"""
Monte Carlo ensemble for the Wallenstein-Manning coupled model
==============================================================

Joint-parameter Monte Carlo over 8 biophysical and economic priors.
Converts the manuscript's headline farm-level price-shock results from
point estimates into probabilistic statements: median, 5-95% CI, and
probability that key qualitative results survive across the joint
parameter space.

Eight perturbed parameters (truncated-normal multipliers unless noted):

  Biophysical
  -----------
  1. max_uptake_frac (MonthlyNParams, absolute) priors: 0.75 ± 0.075, [0.60, 0.90]
     (peak-month crop N capture efficiency; analogous to CropParams.nue_apparent
      in the annual model, set on MonthlyNParams in the monthly path)
  2. mitscherlich_c (regional, multiplier)      priors: 1.00 ± 0.15,  [0.70, 1.30]
  3. k_slow (SOMPoolParams, multiplier)         priors: 1.00 ± 0.20,  [0.60, 1.40]
  4. cre_regional (regional, multiplier)        priors: 1.00 ± 0.30,  [0.40, 1.80]
     (per-region carbon retention efficiency from RegionParams.cre_regional;
      replaces the unused FeedbackParams.cre_base since all default regions
      override that fallback with their own calibrated cre_regional)
  5. residue_retention (regional, multiplier)   priors: 1.00 ± 0.10,  [0.80, 1.15]

  Economic
  --------
  6. eps_F_PF (regional, multiplier on |eps|)   priors: 1.00 ± 0.30,  [0.50, 1.50]
  7. eta (regional, multiplier on |eta|)        priors: 1.00 ± 0.25,  [0.60, 1.40]
  8. fert_price_shock (multiplier)              priors: 1.00 ± 0.25,  [0.50, 1.50]

For each draw, the year-1 farm-level scenario is run for all 8 regions
at three SOC levels (50%, 100%, 150% of regional mean), reproducing the
Figure 1 / Sup Note 4 setup. Outputs:

  - Per-draw posterior CSV (regions x SOC levels x metrics)
  - Summary statistics (median + 5-95% CI per region)
  - Probability statements:
      * P(higher-SOC farm outperforms regional-mean farm on yield)
      * P(higher-SOC farm outperforms regional-mean farm on gross margin)
      * P(SSA remains the highest-vulnerability region)
      * P(soil nitrogen buffer ratio > 1 ppt at every region)
      * P(global mean yield loss within published 1-4 ppt buffering range)

Method note: This is a joint MC over both biophysical and economic
priors propagated through the full coupled model. Unlike the linear-
sensitivity superposition used for Paper 2 dependency uncertainty,
this is a direct (not approximated) propagation. Run cost is dominated
by the per-draw, per-region biophysical equilibrium evaluation.

Outputs (written to ../data/mc_ensemble/):
    mc_posterior.csv.gz    full per-draw record
    mc_summary.csv         per-region medians + 5-95% CIs
    mc_probabilities.csv   probability statements
    mc_summary.txt         human-readable text summary

Author: Matthew Wallenstein
"""
from __future__ import annotations

import os
import sys
import json
import time
import glob
import argparse
import copy
import multiprocessing as mp_proc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# Path setup
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_DIR = REPO_ROOT  # alias for legacy refs
ROOT_DIR = REPO_ROOT  # alias for legacy refs
MODEL_DIR = ROOT_DIR / 'model'
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / 'scripts'))

DATA_DIR = PROJECT_DIR / 'data' / 'mc_ensemble'
DATA_DIR.mkdir(parents=True, exist_ok=True)

from coupled_monthly import MonthlyBiophysicalEngine, get_calibrated_ym, clear_ym_cache
from coupled_econ_biophysical import REGIONAL_ECON_PARAMS
from soil_n_model import (
    SOMPoolParams, CropParams, RegionParams, FeedbackParams,
    get_default_regions, som_params_for_region,
)
from monthly_model_v3 import MonthlyNParams


# =====================================================================
# CONFIGURATION
# =====================================================================

ALL_REGIONS = [
    'north_america', 'europe', 'east_asia', 'south_asia',
    'southeast_asia', 'latin_america', 'sub_saharan_africa',
    'fsu_central_asia',
]

# SOC levels to evaluate (% of regional mean SOC equilibrium pool)
SOC_LEVELS = [50, 100, 150]

# Headline shock: 100% fertilizer-price spike (matches Figure 1)
PRICE_SHOCK_FRAC_BASE = 1.0

# Regional fertilizer cost as fraction of gross revenue
# (matches paper2-soil-resilience/scripts/run_price_shock_analysis.py)
FERT_COST_FRAC = {
    'north_america': 0.08,
    'europe': 0.10,
    'east_asia': 0.12,
    'south_asia': 0.20,
    'southeast_asia': 0.18,
    'latin_america': 0.12,
    'sub_saharan_africa': 0.25,
    'fsu_central_asia': 0.15,
}

# Joint parameter priors
PRIORS = {
    # Absolute prior on monthly peak-month uptake fraction (≈NUE)
    'max_uptake_frac':      dict(kind='truncnorm', mean=0.75, sd=0.075,
                                 lo=0.60, hi=0.90, mode='absolute'),
    # Multipliers (lo, hi expressed as multiplier of central value)
    'mitscherlich_c_mult':  dict(kind='truncnorm', mean=1.0,  sd=0.15,
                                 lo=0.70, hi=1.30, mode='multiplier'),
    'k_slow_mult':          dict(kind='truncnorm', mean=1.0,  sd=0.20,
                                 lo=0.60, hi=1.40, mode='multiplier'),
    'cre_regional_mult':    dict(kind='truncnorm', mean=1.0,  sd=0.30,
                                 lo=0.40, hi=1.80, mode='multiplier'),
    'residue_retention_mult':dict(kind='truncnorm', mean=1.0,  sd=0.10,
                                 lo=0.80, hi=1.15, mode='multiplier'),
    'eps_F_PF_mult':        dict(kind='truncnorm', mean=1.0,  sd=0.30,
                                 lo=0.50, hi=1.50, mode='multiplier'),
    'eta_mult':             dict(kind='truncnorm', mean=1.0,  sd=0.25,
                                 lo=0.60, hi=1.40, mode='multiplier'),
    'fert_price_shock_mult':dict(kind='truncnorm', mean=1.0,  sd=0.25,
                                 lo=0.50, hi=1.50, mode='multiplier'),
}


# =====================================================================
# SAMPLERS
# =====================================================================

def sample_truncnorm(prior: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    a = (prior['lo'] - prior['mean']) / prior['sd']
    b = (prior['hi'] - prior['mean']) / prior['sd']
    return truncnorm.rvs(a, b, loc=prior['mean'], scale=prior['sd'],
                         size=n, random_state=rng)


def draw_priors(n: int, seed: int) -> pd.DataFrame:
    """Draw n joint samples from the priors. Returns DataFrame indexed 0..n-1."""
    rng = np.random.default_rng(seed)
    cols = {}
    for name, prior in PRIORS.items():
        cols[name] = sample_truncnorm(prior, rng, n)
    return pd.DataFrame(cols)


# =====================================================================
# SINGLE-DRAW EVALUATOR
# =====================================================================

def evaluate_one_draw(params: pd.Series,
                      regions: Dict[str, RegionParams],
                      ym_table: Dict[str, float]) -> List[dict]:
    """Run year-1 farm scenario at three SOC levels for all regions.

    `params` has the eight sampled parameter values.
    `regions` are the standard RegionParams (deep-copied per draw and
    rescaled for the residue_retention multiplier).
    `ym_table` is the precomputed central yield_max per region (calibrated
    once at central parameter values; held fixed across draws to keep MC
    tractable -- recalibration would require re-tuning ym for every draw).
    """
    rows = []

    max_up = float(params['max_uptake_frac'])
    mit_c_mult = float(params['mitscherlich_c_mult'])
    k_slow_mult = float(params['k_slow_mult'])
    cre_mult = float(params['cre_regional_mult'])
    res_ret_mult = float(params['residue_retention_mult'])
    eps_F_PF_mult = float(params['eps_F_PF_mult'])
    eta_mult = float(params['eta_mult'])
    shock = PRICE_SHOCK_FRAC_BASE * float(params['fert_price_shock_mult'])
    PF_hat = np.log(1 + max(shock, 1e-6))

    # Per-draw MonthlyNParams: holds the perturbed peak-month uptake fraction.
    # All other monthly N kinetics held at central values.
    mp = MonthlyNParams()
    mp.max_uptake_frac = max_up

    for rn in ALL_REGIONS:
        region_central = regions[rn]
        ym = ym_table[rn]
        rp = REGIONAL_ECON_PARAMS.get(rn, {})

        # Apply per-draw multipliers to a copy of the region
        region = copy.deepcopy(region_central)
        region.residue_retention = float(np.clip(
            region_central.residue_retention * res_ret_mult, 0.0, 1.0))
        if region.mitscherlich_c_regional > 0:
            region.mitscherlich_c_regional *= mit_c_mult
        # Apply CRE multiplier to the regional cre_regional value
        # (each region carries its own calibrated cre_regional)
        if region.cre_regional > 0:
            region.cre_regional = float(np.clip(
                region_central.cre_regional * cre_mult, 0.01, 0.99))

        # Per-draw SOMPoolParams (regime-appropriate, then perturbed)
        som = som_params_for_region(rn)
        som.k_slow *= k_slow_mult

        # FeedbackParams left at default (cre_regional handles retention)
        fb = FeedbackParams()

        # CropParams: only the Mitscherlich curvature is varied; the
        # regional override above is what actually enters the engine.
        crop = CropParams()
        crop.mitscherlich_c *= mit_c_mult

        # Spin up ONCE per (draw, region). The same equilibrium is reused
        # for the regional baseline and all three SOC-scaling sub-runs
        # (initial_pools shortcut bypasses the per-engine spinup, which
        # would otherwise dominate the MC runtime).
        # `monthly_params=mp` propagates the perturbed max_uptake_frac into
        # every monthly_n_balance call.
        eng_regional = MonthlyBiophysicalEngine(
            region, region_key=rn, som_params=som,
            crop_params=crop, feedback_params=fb,
            monthly_params=mp,
            yield_max_override=ym,
        )
        # Snapshot the equilibrium state for reuse
        eq_pools = dict(
            c_active=eng_regional.C_active,
            c_slow=eng_regional.C_slow,
            c_passive=eng_regional.C_passive,
            soc=eng_regional.soc_initial,
            mineral_n=eng_regional.mineral_n,
            yield_eq=eng_regional.yield_baseline,
            n_min_eq=eng_regional.n_min_baseline,
        )
        state_regional = eng_regional.step(region.synth_n_current)
        y_regional_baseline = state_regional['yield_tha']
        gamma_regional = state_regional['gamma']
        C_a_eq = eq_pools['c_active']
        C_s_eq = eq_pools['c_slow']
        C_p_eq = eq_pools['c_passive']

        # Per-draw economic elasticities
        # Sign-preserving multipliers (multiplier is a magnitude scaler)
        eps_F_PF = rp.get('eps_F_PF', -0.30) * eps_F_PF_mult
        eps_F_PY = rp.get('eps_F_PY', 0.10)
        eta = rp.get('eta', -0.45) * eta_mult

        denom = eta - gamma_regional * eps_F_PY
        PY_hat = (gamma_regional * eps_F_PF * PF_hat / denom
                  if abs(denom) > 1e-10 else 0.0)
        F_hat = eps_F_PF * PF_hat + eps_F_PY * PY_hat
        F_shocked = max(0.0, region.synth_n_current * np.exp(F_hat))

        fcf = FERT_COST_FRAC.get(rn, 0.15)
        pf_per_unit = (fcf * y_regional_baseline / region.synth_n_current
                       if region.synth_n_current > 0 else 0.0)

        for soc_pct in SOC_LEVELS:
            scale = soc_pct / 100.0
            soc_pools = dict(eq_pools)
            soc_pools['c_active'] = eq_pools['c_active'] * scale
            soc_pools['c_slow'] = eq_pools['c_slow'] * scale
            soc_pools['c_passive'] = eq_pools['c_passive'] * scale
            soc_pools['soc'] = eq_pools['soc'] * scale

            eng_base = MonthlyBiophysicalEngine(
                region, region_key=rn, som_params=som,
                crop_params=crop, feedback_params=fb,
                monthly_params=mp,
                yield_max_override=ym,
                initial_pools=soc_pools,
            )
            state_base = eng_base.step(region.synth_n_current)
            y_base = state_base['yield_tha']

            eng_shock = MonthlyBiophysicalEngine(
                region, region_key=rn, som_params=som,
                crop_params=crop, feedback_params=fb,
                monthly_params=mp,
                yield_max_override=ym,
                initial_pools=soc_pools,
            )
            state_shock = eng_shock.step(F_shocked)
            y_shock = state_shock['yield_tha']

            yield_pen = (1 - y_shock / y_base) * 100 if y_base > 0 else 0.0

            # Gross margin over fertilizer cost
            profit_b = y_base - region.synth_n_current * pf_per_unit
            profit_s = (y_shock * np.exp(PY_hat)
                        - F_shocked * pf_per_unit * (1 + shock))
            profit_chg = ((profit_s / profit_b - 1) * 100
                          if abs(profit_b) > 1e-10 else 0.0)

            rows.append(dict(
                region=rn, soc_pct=soc_pct,
                yield_pen=yield_pen, profit_chg=profit_chg,
                y_base=y_base, y_shock=y_shock,
                F_shocked=F_shocked, PY_hat=PY_hat,
                gamma_regional=gamma_regional,
            ))

    return rows


# =====================================================================
# MC LOOP
# =====================================================================

# Worker globals (set in worker initializer to avoid re-pickling per task)
_WORKER_REGIONS = None
_WORKER_YM = None
YM_CACHE_FILE = DATA_DIR / 'ym_cache.json'


def _ensure_ym_cache() -> dict:
    """Compute the central-parameter ym calibration once, cache to disk.
    Subsequent chunked runs reuse the cached values, eliminating ~10s of
    per-chunk worker-startup overhead.
    """
    if YM_CACHE_FILE.exists():
        with open(YM_CACHE_FILE) as f:
            return json.load(f)
    print(f'Calibrating regional yield_max (one-time, cached to {YM_CACHE_FILE})...')
    clear_ym_cache()
    ym = {rn: float(get_calibrated_ym(rn)) for rn in ALL_REGIONS}
    YM_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(YM_CACHE_FILE, 'w') as f:
        json.dump(ym, f, indent=2)
    return ym


def _worker_init(ym_table: dict):
    global _WORKER_REGIONS, _WORKER_YM
    _WORKER_REGIONS = get_default_regions()
    _WORKER_YM = {rn: float(v) for rn, v in ym_table.items()}


def _worker_eval(args):
    draw_idx, params_dict = args
    params = pd.Series(params_dict)
    try:
        rows = evaluate_one_draw(params, _WORKER_REGIONS, _WORKER_YM)
    except Exception as e:  # noqa: BLE001
        return draw_idx, None, str(e)
    for r in rows:
        r['draw'] = draw_idx
        for k, v in params_dict.items():
            r[k] = float(v)
    return draw_idx, rows, None


def run_mc(n_draws: int, seed: int,
           start: int = 0, end: int = None,
           n_workers: int = 1) -> pd.DataFrame:
    """Run MC over draws[start:end]. If n_workers > 1, parallelize via Pool."""
    end = n_draws if end is None else min(end, n_draws)
    print(f'Drawing {n_draws} joint samples (seed={seed}); '
          f'evaluating draws [{start}:{end}); workers={n_workers}.')
    priors_df = draw_priors(n_draws, seed)

    sub = priors_df.iloc[start:end]
    tasks = [(start + i, sub.iloc[i].to_dict()) for i in range(len(sub))]

    ym_table = _ensure_ym_cache()

    all_rows = []
    t0 = time.time()
    n_done = 0

    if n_workers <= 1:
        # Single-process path (also used for diagnostics)
        _worker_init(ym_table)
        for draw_idx, params_dict in tasks:
            params = pd.Series(params_dict)
            try:
                rows = evaluate_one_draw(params, _WORKER_REGIONS, _WORKER_YM)
            except Exception as e:  # noqa: BLE001
                print(f'  draw {draw_idx}: FAILED ({e}); skipping')
                continue
            for r in rows:
                r['draw'] = draw_idx
                for k, v in params_dict.items():
                    r[k] = float(v)
                all_rows.append(r)
            n_done += 1
            if n_done % max(1, len(tasks) // 10) == 0:
                el = time.time() - t0
                rem = el / n_done * (len(tasks) - n_done)
                print(f'  {n_done}/{len(tasks)}  elapsed={el:.0f}s  eta={rem:.0f}s')
    else:
        with mp_proc.Pool(processes=n_workers, initializer=_worker_init,
                          initargs=(ym_table,)) as pool:
            for draw_idx, rows, err in pool.imap_unordered(
                    _worker_eval, tasks, chunksize=1):
                if rows is None:
                    print(f'  draw {draw_idx}: FAILED ({err}); skipping')
                else:
                    all_rows.extend(rows)
                n_done += 1
                if n_done % max(1, len(tasks) // 10) == 0:
                    el = time.time() - t0
                    rem = el / n_done * (len(tasks) - n_done)
                    print(f'  {n_done}/{len(tasks)}  elapsed={el:.0f}s  eta={rem:.0f}s')

    elapsed = time.time() - t0
    print(f'Completed {n_done} draws in {elapsed:.0f}s '
          f'({elapsed/max(n_done,1):.2f}s/draw).')

    df = pd.DataFrame(all_rows)
    return df


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================

def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compute per-region summaries and probability statements."""

    # Per-region, per-SOC summaries
    rows = []
    for rn in ALL_REGIONS:
        for soc_pct in SOC_LEVELS:
            sub = df[(df['region'] == rn) & (df['soc_pct'] == soc_pct)]
            if len(sub) == 0:
                continue
            for var in ['yield_pen', 'profit_chg']:
                vals = sub[var].values
                q = np.percentile(vals, [5, 25, 50, 75, 95])
                rows.append(dict(
                    region=rn, soc_pct=soc_pct, metric=var,
                    n=len(vals),
                    median=q[2], q25=q[1], q75=q[3],
                    p5=q[0], p95=q[4], mean=vals.mean(), sd=vals.std(),
                ))
    summary = pd.DataFrame(rows)

    # Probability statements
    prob_rows = []

    # P1: Soil-nitrogen-buffer ratio: P that 150%-SOC farm has smaller
    # |yield loss| than 50%-SOC farm in each region (yield_pen is positive
    # for losses under shock).
    for rn in ALL_REGIONS:
        s50 = df[(df['region'] == rn) & (df['soc_pct'] == 50)].set_index('draw')['yield_pen']
        s150 = df[(df['region'] == rn) & (df['soc_pct'] == 150)].set_index('draw')['yield_pen']
        common = s50.index.intersection(s150.index)
        if len(common) == 0:
            continue
        diff = s50.loc[common] - s150.loc[common]
        p_yield = float((diff > 0).mean())
        prob_rows.append(dict(
            statement=f'P(low-SOC yield loss > high-SOC yield loss | {rn})',
            value=p_yield, n=len(common),
        ))

    # P2: Same for gross margin
    for rn in ALL_REGIONS:
        s50 = df[(df['region'] == rn) & (df['soc_pct'] == 50)].set_index('draw')['profit_chg']
        s150 = df[(df['region'] == rn) & (df['soc_pct'] == 150)].set_index('draw')['profit_chg']
        common = s50.index.intersection(s150.index)
        if len(common) == 0:
            continue
        diff = s150.loc[common] - s50.loc[common]
        p_margin = float((diff > 0).mean())
        prob_rows.append(dict(
            statement=f'P(high-SOC gross margin > low-SOC gross margin | {rn})',
            value=p_margin, n=len(common),
        ))

    # P3: P(SSA highest yield loss at SOC=100%)
    by_draw_region = (df[df['soc_pct'] == 100]
                      .pivot(index='draw', columns='region', values='yield_pen'))
    if 'sub_saharan_africa' in by_draw_region.columns:
        ssa_top = (by_draw_region.idxmax(axis=1) == 'sub_saharan_africa').mean()
        prob_rows.append(dict(
            statement='P(SSA = highest year-1 yield loss across all 8 regions, SOC=100%)',
            value=float(ssa_top), n=len(by_draw_region),
        ))

    # P4: P(SSA worst gross margin at SOC=100%)
    by_draw_region_p = (df[df['soc_pct'] == 100]
                        .pivot(index='draw', columns='region', values='profit_chg'))
    if 'sub_saharan_africa' in by_draw_region_p.columns:
        ssa_worst = (by_draw_region_p.idxmin(axis=1) == 'sub_saharan_africa').mean()
        prob_rows.append(dict(
            statement='P(SSA = worst year-1 gross margin across all 8 regions, SOC=100%)',
            value=float(ssa_worst), n=len(by_draw_region_p),
        ))

    # P5: Buffer ratio (50% vs 150%) > 1 ppt for every region (yield)
    bufs = []
    for rn in ALL_REGIONS:
        s50 = df[(df['region'] == rn) & (df['soc_pct'] == 50)].set_index('draw')['yield_pen']
        s150 = df[(df['region'] == rn) & (df['soc_pct'] == 150)].set_index('draw')['yield_pen']
        common = s50.index.intersection(s150.index)
        if len(common) == 0:
            continue
        diff = s50.loc[common] - s150.loc[common]
        bufs.append(diff.rename(rn))
    if bufs:
        bufs_df = pd.concat(bufs, axis=1)
        all_pos = (bufs_df > 1.0).all(axis=1).mean()
        prob_rows.append(dict(
            statement='P(soil-N buffer ratio > 1 ppt in EVERY region simultaneously)',
            value=float(all_pos), n=len(bufs_df),
        ))
        # Median across regions
        median_buf_per_draw = bufs_df.median(axis=1)
        prob_rows.append(dict(
            statement='Median across-region buffer (ppt yield, low - high SOC), median across draws',
            value=float(median_buf_per_draw.median()), n=len(bufs_df),
        ))

    # P6: Global cropland-area-weighted year-1 yield loss at SOC=100%
    weights = {rn: get_default_regions()[rn].cropland_mha for rn in ALL_REGIONS}
    wsum = sum(weights.values())
    pivot100 = (df[df['soc_pct'] == 100]
                .pivot(index='draw', columns='region', values='yield_pen'))
    if all(rn in pivot100.columns for rn in ALL_REGIONS):
        glob = sum(pivot100[rn] * weights[rn] / wsum for rn in ALL_REGIONS)
        q = np.percentile(glob, [5, 25, 50, 75, 95])
        prob_rows.append(dict(
            statement='Global area-weighted year-1 yield loss at SOC=100% (median % loss)',
            value=float(q[2]), n=len(glob),
        ))
        prob_rows.append(dict(
            statement='Global area-weighted year-1 yield loss at SOC=100% (5%-95%, ppt)',
            value=float(q[4] - q[0]), n=len(glob),
        ))

    probs = pd.DataFrame(prob_rows)

    # Text summary
    lines = []
    lines.append('Monte Carlo ensemble — Wallenstein-Manning coupled model')
    lines.append('=' * 70)
    lines.append('')
    lines.append('Per-region year-1 yield loss (%) at three SOC levels')
    lines.append('-' * 70)
    lines.append(f'{"Region":<25} {"SOC%":>5} {"median":>9} {"5%":>9} {"95%":>9}')
    for rn in ALL_REGIONS:
        for soc_pct in SOC_LEVELS:
            r = summary[(summary['region'] == rn) & (summary['soc_pct'] == soc_pct)
                        & (summary['metric'] == 'yield_pen')]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            lines.append(f'{rn:<25} {soc_pct:>5d} {r["median"]:>8.2f}% '
                         f'{r["p5"]:>8.2f}% {r["p95"]:>8.2f}%')
    lines.append('')
    lines.append('Per-region year-1 gross-margin-over-fert-cost change (%) at SOC=100%')
    lines.append('-' * 70)
    for rn in ALL_REGIONS:
        r = summary[(summary['region'] == rn) & (summary['soc_pct'] == 100)
                    & (summary['metric'] == 'profit_chg')]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        lines.append(f'{rn:<25}      {r["median"]:>8.2f}% '
                     f'{r["p5"]:>8.2f}% {r["p95"]:>8.2f}%')
    lines.append('')
    lines.append('Probability statements')
    lines.append('-' * 70)
    for _, r in probs.iterrows():
        lines.append(f'  {r["statement"]:<70} {r["value"]:>8.3f}  (n={int(r["n"])})')
    return summary, probs, '\n'.join(lines)


# =====================================================================
# MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000, help='total number of MC draws (defines the prior sample)')
    ap.add_argument('--seed', type=int, default=20260424, help='RNG seed')
    ap.add_argument('--out', type=str, default=str(DATA_DIR), help='output dir')
    ap.add_argument('--start', type=int, default=0, help='inclusive start index for chunked runs')
    ap.add_argument('--end', type=int, default=None, help='exclusive end index for chunked runs')
    ap.add_argument('--chunk-out', type=str, default=None,
                    help='if set, write per-draw rows to this CSV.gz and skip summarization')
    ap.add_argument('--workers', type=int, default=1, help='multiprocessing workers')
    ap.add_argument('--merge', type=str, default=None,
                    help='glob of chunk CSVs to merge into final outputs (skips MC run)')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.merge:
        files = sorted(glob.glob(args.merge))
        print(f'Merging {len(files)} chunk files matching {args.merge}')
        if not files:
            sys.exit('No chunk files found; nothing to merge.')
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df.drop_duplicates(subset=['draw', 'region', 'soc_pct']).sort_values(
            ['draw', 'region', 'soc_pct']).reset_index(drop=True)
        n_draws = df['draw'].nunique()
        print(f'  merged {len(df)} rows across {n_draws} unique draws.')
    else:
        df = run_mc(args.n, args.seed,
                    start=args.start, end=args.end,
                    n_workers=max(1, args.workers))
        if args.chunk_out:
            chunk_path = Path(args.chunk_out)
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(chunk_path, index=False, compression='gzip'
                      if str(chunk_path).endswith('.gz') else None)
            print(f'Wrote chunk: {chunk_path} ({len(df)} rows)')
            return

    df.to_csv(out / 'mc_posterior.csv.gz', index=False, compression='gzip')

    summary, probs, text = summarize(df)
    summary.to_csv(out / 'mc_summary.csv', index=False)
    probs.to_csv(out / 'mc_probabilities.csv', index=False)
    with open(out / 'mc_summary.txt', 'w') as f:
        f.write(text + '\n')

    # Save the priors metadata
    with open(out / 'mc_priors.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in PRIORS.items()},
                  f, indent=2)

    print('\n' + text)
    print('\nWritten:')
    for fn in ['mc_posterior.csv.gz', 'mc_summary.csv', 'mc_probabilities.csv',
               'mc_summary.txt', 'mc_priors.json']:
        print(f'  {out / fn}')


if __name__ == '__main__':
    main()
