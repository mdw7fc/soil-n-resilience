#!/usr/bin/env python3
"""
Canonical generator for price_shock_analysis.pkl and soc_gradient_fine.pkl.

These two pickle files contain the precomputed model outputs used by
Figures 1, 2, and 3a of the Nature Food manuscript. Prior to this script
they were generated interactively; this script is the reproducible
replacement.

Outputs (written to ../data/):
    price_shock_analysis.pkl
        - farm_results: 4 key regions x 4 SOC levels x 8 price multipliers
          (yield penalty, profit impact, fert reduction sweeps for Figs 1-2)
        - fine_shock_results: 4 key regions x 11 SOC levels at 100% shock
          (smooth SOC gradient curves for Fig 1)
        - regions: all 8 RegionParams for reference

    soc_gradient_fine.pkl
        - fine_results: all 8 regions x 12 SOC levels under sustained 20%
          supply reduction (total penalty, shock-only penalty, absolute
          yields at year 10 for Fig 3a)
        - ref_yields: reference yield at SOC=100% for each region
        - soc_pcts: [10, 20, ..., 120]
        - regions: all 8 RegionParams for reference

Usage:
    python run_price_shock_analysis.py          # generate both
    python run_price_shock_analysis.py --check   # compare against existing pkl

Author: Matthew Wallenstein
"""

import sys, os, pickle, copy, argparse
import numpy as np
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = SCRIPT_DIR.parent
MODEL_DIR = REPO_DIR / 'model'
sys.path.insert(0, str(MODEL_DIR))

DATA_DIR = REPO_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

from coupled_monthly import (
    CoupledMonthlyModel, MonthlyBiophysicalEngine, get_calibrated_ym,
    clear_ym_cache,
)
from coupled_econ_biophysical import (
    EconParams, REGIONAL_ECON_PARAMS, calibrate_price_shock,
    get_scenario_params,
)
from soil_n_model import get_default_regions

# =====================================================================
# CONFIGURATION
# =====================================================================

KEY4 = ['sub_saharan_africa', 'south_asia', 'latin_america', 'north_america']
ALL_REGIONS = [
    'north_america', 'europe', 'east_asia', 'south_asia',
    'southeast_asia', 'latin_america', 'sub_saharan_africa', 'fsu_central_asia',
]

# Farm-level price shock analysis (Figs 1, 2)
FINE_SOC_PCTS = list(range(10, 205, 5))          # 10% to 200%, step 5
COARSE_SOC_LEVELS = [25, 50, 75, 100]           # For multi-shock sweep
PRICE_MULTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
PRICE_SHOCK_FINE = 1.0                           # 100% for fine SOC gradient

# Regional fertilizer cost as fraction of gross revenue (FAO/IFDC)
FERT_COST_FRAC = {
    'sub_saharan_africa': 0.25,
    'south_asia': 0.20,
    'latin_america': 0.12,
    'north_america': 0.08,
}

# SOC gradient for Fig 3a (sustained disruption)
GRADIENT_SOC_PCTS = list(range(10, 205, 10))     # 10% to 200%, step 10
T_MAX_GRADIENT = 10                               # Year 10 values


# =====================================================================
# PART 1: price_shock_analysis.pkl
# =====================================================================

def run_farm_sweep_single(region, rn, ym, soc_pct, price_shock_frac):
    """Run a single farm-level SOC + price shock combination.

    Returns dict with yield_pen, fert_red, profit_chg, base_yield, etc.
    """
    rp = REGIONAL_ECON_PARAMS.get(rn, {})
    fcf = FERT_COST_FRAC.get(rn, 0.15)

    # Equilibrium SOM pools
    engine_eq = MonthlyBiophysicalEngine(region, region_key=rn, yield_max_override=ym)
    C_a_eq = engine_eq.C_active
    C_s_eq = engine_eq.C_slow
    C_p_eq = engine_eq.C_passive
    base_fert = region.synth_n_current

    # Regional-baseline yield (SOC=100%) used to calibrate a SOC-invariant
    # fertilizer per-unit price. Fertilizer is a regional market price and must
    # not change with farm SOC level.
    eng_regional = MonthlyBiophysicalEngine(region, region_key=rn, yield_max_override=ym)
    state_regional = eng_regional.step(base_fert)
    y_regional_baseline = state_regional['yield_tha']

    scale = soc_pct / 100.0

    # Baseline yield at this SOC (no shock)
    eng_base = MonthlyBiophysicalEngine(region, region_key=rn, yield_max_override=ym)
    eng_base.C_active = C_a_eq * scale
    eng_base.C_slow = C_s_eq * scale
    eng_base.C_passive = C_p_eq * scale
    state_base = eng_base.step(base_fert)
    y_base_soc = state_base['yield_tha']
    n_min_base = state_base['n_mineralized']
    gamma = state_base['gamma']

    if price_shock_frac <= 0:
        return {
            'yield_pen': 0.0,
            'fert_red': 0.0,
            'profit_chg': 0.0,
            'base_yield': y_base_soc,
            'base_fert': base_fert,
            'base_nmin': n_min_base,
            'F_shocked': base_fert,
            'y_shock': y_base_soc,
        }

    # Economic equilibrium under shock
    eng_shock = MonthlyBiophysicalEngine(region, region_key=rn, yield_max_override=ym)
    eng_shock.C_active = C_a_eq * scale
    eng_shock.C_slow = C_s_eq * scale
    eng_shock.C_passive = C_p_eq * scale

    eps_F_PF = rp.get('eps_F_PF', -0.20)
    eps_F_PY = rp.get('eps_F_PY', 0.10)
    eta = rp.get('eta', -0.30)
    PF_hat = np.log(1 + price_shock_frac)

    denom = eta - gamma * eps_F_PY
    PY_hat = gamma * eps_F_PF * PF_hat / denom if abs(denom) > 1e-10 else 0.0
    F_hat = eps_F_PF * PF_hat + eps_F_PY * PY_hat
    F_shocked = max(0.0, base_fert * np.exp(F_hat))

    state_shock = eng_shock.step(F_shocked)
    y_shock = state_shock['yield_tha']

    yield_pen = (1 - y_shock / y_base_soc) * 100 if y_base_soc > 0 else 0.0
    fert_red = (1 - F_shocked / base_fert) * 100 if base_fert > 0 else 0.0

    # Profit via gross-margin-over-fertilizer
    # Fertilizer per-unit price is a regional market price — calibrate to
    # regional-baseline yield (SOC=100%), NOT the SOC-specific yield, so
    # that degraded farms still pay the full market fertilizer price.
    pf_per_unit = fcf * y_regional_baseline / base_fert if base_fert > 0 else 0
    profit_b = y_base_soc - base_fert * pf_per_unit
    profit_s = y_shock * np.exp(PY_hat) - F_shocked * pf_per_unit * (1 + price_shock_frac)
    profit_chg = (profit_s / profit_b - 1) * 100 if abs(profit_b) > 1e-10 else 0.0

    return {
        'yield_pen': yield_pen,
        'fert_red': fert_red,
        'profit_chg': profit_chg,
        'base_yield': y_base_soc,
        'base_fert': base_fert,
        'base_nmin': n_min_base,
        'F_shocked': F_shocked,
        'y_shock': y_shock,
    }


def generate_farm_results(regions):
    """Coarse farm_results: 4 regions x 4 SOC levels x 8 price mults."""
    farm_results = {}
    for rn in KEY4:
        r = regions[rn]
        ym = get_calibrated_ym(rn)
        farm_results[rn] = {}

        for soc_level in COARSE_SOC_LEVELS:
            price_mults_list = []
            fert_reds = []
            yield_pens = []
            profit_imps = []
            base_yield = None
            base_fert = None
            base_nmin = None

            for pm in PRICE_MULTS:
                res = run_farm_sweep_single(r, rn, ym, soc_level, pm)
                price_mults_list.append(pm)
                fert_reds.append(res['fert_red'])
                yield_pens.append(res['yield_pen'])
                profit_imps.append(res['profit_chg'])
                if base_yield is None:
                    base_yield = res['base_yield']
                    base_fert = res['base_fert']
                    base_nmin = res['base_nmin']

            farm_results[rn][soc_level] = {
                'price_mults': price_mults_list,
                'fert_reductions': fert_reds,
                'yield_penalties': yield_pens,
                'profit_impacts': profit_imps,
                'base_yield': base_yield,
                'base_fert': base_fert,
                'base_nmin': base_nmin,
            }
            print(f'  {rn} SOC={soc_level}%: '
                  f'yield_pen @100%={yield_pens[4]:.1f}%, '
                  f'profit @100%={profit_imps[4]:.1f}%')

    return farm_results


def generate_fine_shock_results(regions):
    """Fine SOC gradient at 100% shock: 4 regions x 11 SOC levels."""
    fine_results = {}
    for rn in KEY4:
        r = regions[rn]
        ym = get_calibrated_ym(rn)
        fcf = FERT_COST_FRAC.get(rn, 0.15)

        soc_pcts = []
        fert_reds = []
        yield_pens = []
        profit_chgs = []
        fert_saved = []
        yield_lost_value = []

        for soc_pct in FINE_SOC_PCTS:
            res = run_farm_sweep_single(r, rn, ym, soc_pct, PRICE_SHOCK_FINE)
            soc_pcts.append(soc_pct)
            fert_reds.append(res['fert_red'])
            yield_pens.append(res['yield_pen'])
            profit_chgs.append(res['profit_chg'])

            # Absolute values (kg/ha and relative value units)
            fert_diff = res['F_shocked'] - res['base_fert']
            fert_saved.append(fert_diff)
            yield_diff = res['y_shock'] - res['base_yield']
            # Value = yield_diff * baseline_price (normalized to 1) * cropland
            yield_lost_value.append(yield_diff * r.cropland_mha)

        fine_results[rn] = {
            'soc_pct': soc_pcts,
            'fert_red': fert_reds,
            'yield_pen': yield_pens,
            'profit_chg': profit_chgs,
            'fert_saved': fert_saved,
            'yield_lost_value': yield_lost_value,
        }
        print(f'  {rn}: yield_pen [{min(yield_pens):.1f}, {max(yield_pens):.1f}], '
              f'profit_chg [{min(profit_chgs):.1f}, {max(profit_chgs):.1f}]')

    return fine_results


def build_price_shock_pkl(regions):
    """Build the complete price_shock_analysis.pkl."""
    print('\n=== Generating price_shock_analysis.pkl ===')

    print('\nFarm-level results (coarse: 4 SOC levels x 8 price mults)...')
    farm_results = generate_farm_results(regions)

    print('\nFine SOC gradient (11 SOC levels x 100% shock)...')
    fine_results = generate_fine_shock_results(regions)

    data = {
        'farm_results': farm_results,
        'fine_shock_results': fine_results,
        'regions': regions,
    }

    out_path = DATA_DIR / 'price_shock_analysis.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved: {out_path} ({out_path.stat().st_size:,} bytes)')
    return data


# =====================================================================
# PART 2: soc_gradient_fine.pkl
# =====================================================================

def generate_soc_gradient_fine(regions):
    """Run CoupledMonthlyModel at varying SOC levels under sustained 20%
    price-mediated supply reduction (S3). Extract year-10 yields.

    For each region and SOC level:
        1. Run with S3 shock -> yield_shock at year T_MAX_GRADIENT
        2. Run without shock (fert_price_shock=0) -> yield_noshock
        3. Compute total_penalty = (1 - yield_shock / ref_yield) * 100
        4. Compute ctrl_penalty = (1 - yield_shock / yield_noshock) * 100
           (shock-specific penalty, independent of SOC structural effect)
    """
    print('\n=== Generating soc_gradient_fine.pkl ===')

    scenarios = get_scenario_params()
    s3 = scenarios['S3']

    fine_results = {}
    ref_yields = {}

    for rn in ALL_REGIONS:
        r = regions[rn]
        ym = get_calibrated_ym(rn)

        soc_pcts = []
        total_penalties = []
        ctrl_penalties = []
        yields_shock = []
        yields_noshock = []

        # Get equilibrium SOM pools
        engine_eq = MonthlyBiophysicalEngine(r, region_key=rn, yield_max_override=ym)
        C_a_eq = engine_eq.C_active
        C_s_eq = engine_eq.C_slow
        C_p_eq = engine_eq.C_passive

        # Reference yield: SOC=100%, no shock, year 10
        econ_ref = EconParams(fert_price_shock=0.0, eps_F_PY=0.0, eps_F_N=0.0,
                              eps_LD_PL=0.0, eps_LD_PY=0.0, eps_LS_PL=0.0)
        model_ref = CoupledMonthlyModel(
            r, econ_ref, region_key=rn, t_max=T_MAX_GRADIENT,
            yield_max_override=ym,
        )
        df_ref = model_ref.run()
        ref_yield = df_ref.loc[df_ref['year'] == T_MAX_GRADIENT, 'yield_tha'].iloc[0]
        ref_yields[rn] = ref_yield

        for soc_pct in GRADIENT_SOC_PCTS:
            scale = soc_pct / 100.0

            # Shocked run (S3)
            model_shock = CoupledMonthlyModel(
                r, copy.deepcopy(s3), region_key=rn, t_max=T_MAX_GRADIENT,
                yield_max_override=ym,
            )
            model_shock.bio.C_active = C_a_eq * scale
            model_shock.bio.C_slow = C_s_eq * scale
            model_shock.bio.C_passive = C_p_eq * scale
            df_shock = model_shock.run()
            y_shock = df_shock.loc[df_shock['year'] == T_MAX_GRADIENT, 'yield_tha'].iloc[0]

            # No-shock run at same SOC
            econ_noshock = EconParams(fert_price_shock=0.0, eps_F_PY=0.0, eps_F_N=0.0,
                                     eps_LD_PL=0.0, eps_LD_PY=0.0, eps_LS_PL=0.0)
            model_noshock = CoupledMonthlyModel(
                r, econ_noshock, region_key=rn, t_max=T_MAX_GRADIENT,
                yield_max_override=ym,
            )
            model_noshock.bio.C_active = C_a_eq * scale
            model_noshock.bio.C_slow = C_s_eq * scale
            model_noshock.bio.C_passive = C_p_eq * scale
            df_noshock = model_noshock.run()
            y_noshock = df_noshock.loc[df_noshock['year'] == T_MAX_GRADIENT, 'yield_tha'].iloc[0]

            # Penalties
            total_pen = (1 - y_shock / ref_yield) * 100 if ref_yield > 0 else 0.0
            ctrl_pen = (1 - y_shock / y_noshock) * 100 if y_noshock > 0 else 0.0

            soc_pcts.append(soc_pct)
            total_penalties.append(total_pen)
            ctrl_penalties.append(ctrl_pen)
            yields_shock.append(y_shock)
            yields_noshock.append(y_noshock)

        fine_results[rn] = {
            'soc_pct': soc_pcts,
            'total_penalty': total_penalties,
            'ctrl_penalty': ctrl_penalties,
            'yield_shock': yields_shock,
            'yield_noshock': yields_noshock,
        }
        print(f'  {rn}: ref_yield={ref_yield:.4f}, '
              f'total_pen @SOC50=[{total_penalties[4]:.1f}%], '
              f'@SOC100=[{total_penalties[9]:.1f}%]')

    data = {
        'fine_results': fine_results,
        'ref_yields': ref_yields,
        'soc_pcts': GRADIENT_SOC_PCTS,
        'regions': regions,
    }

    out_path = DATA_DIR / 'soc_gradient_fine.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved: {out_path} ({out_path.stat().st_size:,} bytes)')
    return data


# =====================================================================
# VALIDATION
# =====================================================================

def validate_against_existing():
    """Compare freshly generated pkl with existing versions."""
    print('\n=== Validation ===')
    ok = True

    # price_shock_analysis.pkl
    new_path = DATA_DIR / 'price_shock_analysis.pkl'
    old_path = DATA_DIR / 'price_shock_analysis.pkl.bak'
    if old_path.exists() and new_path.exists():
        with open(old_path, 'rb') as f:
            old = pickle.load(f)
        with open(new_path, 'rb') as f:
            new = pickle.load(f)

        for rn in KEY4:
            # Check fine_shock_results
            old_yp = old['fine_shock_results'][rn]['yield_pen']
            new_yp = new['fine_shock_results'][rn]['yield_pen']
            max_diff = max(abs(a - b) for a, b in zip(old_yp, new_yp))
            status = 'OK' if max_diff < 0.01 else f'DIFF={max_diff:.4f}'
            print(f'  price_shock fine yield_pen {rn}: {status}')
            if max_diff >= 0.01:
                ok = False

            # Check farm_results
            for soc in COARSE_SOC_LEVELS:
                old_yp = old['farm_results'][rn][soc]['yield_penalties']
                new_yp = new['farm_results'][rn][soc]['yield_penalties']
                max_diff = max(abs(a - b) for a, b in zip(old_yp, new_yp))
                status = 'OK' if max_diff < 0.01 else f'DIFF={max_diff:.4f}'
                if max_diff >= 0.01:
                    print(f'  price_shock farm {rn} SOC={soc}%: {status}')
                    ok = False

    # soc_gradient_fine.pkl
    new_path = DATA_DIR / 'soc_gradient_fine.pkl'
    old_path = DATA_DIR / 'soc_gradient_fine.pkl.bak'
    if old_path.exists() and new_path.exists():
        with open(old_path, 'rb') as f:
            old = pickle.load(f)
        with open(new_path, 'rb') as f:
            new = pickle.load(f)

        for rn in ALL_REGIONS:
            old_tp = old['fine_results'][rn]['total_penalty']
            new_tp = new['fine_results'][rn]['total_penalty']
            max_diff = max(abs(a - b) for a, b in zip(old_tp, new_tp))
            status = 'OK' if max_diff < 0.05 else f'DIFF={max_diff:.4f}'
            print(f'  soc_gradient total_penalty {rn}: {status}')
            if max_diff >= 0.05:
                ok = False

    if ok:
        print('\nAll checks passed.')
    else:
        print('\nSome checks failed. Review differences above.')
    return ok


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate price_shock_analysis.pkl and soc_gradient_fine.pkl')
    parser.add_argument('--check', action='store_true',
                        help='Back up existing pkl files and validate after regeneration')
    parser.add_argument('--price-shock-only', action='store_true',
                        help='Only regenerate price_shock_analysis.pkl')
    parser.add_argument('--gradient-only', action='store_true',
                        help='Only regenerate soc_gradient_fine.pkl')
    args = parser.parse_args()

    regions = get_default_regions()

    # Back up existing files if --check
    if args.check:
        import shutil
        for fname in ['price_shock_analysis.pkl', 'soc_gradient_fine.pkl']:
            src = DATA_DIR / fname
            if src.exists():
                shutil.copy2(src, DATA_DIR / f'{fname}.bak')
                print(f'Backed up {fname} -> {fname}.bak')

    do_price = not args.gradient_only
    do_gradient = not args.price_shock_only

    if do_price:
        build_price_shock_pkl(regions)

    if do_gradient:
        generate_soc_gradient_fine(regions)

    if args.check:
        validate_against_existing()
        # Clean up backups
        for fname in ['price_shock_analysis.pkl.bak', 'soc_gradient_fine.pkl.bak']:
            bak = DATA_DIR / fname
            if bak.exists():
                bak.unlink()
                print(f'Cleaned up {fname}')

    print('\nDone.')
