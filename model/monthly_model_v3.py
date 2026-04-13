"""
Monthly Soil N Model v3 — Hybrid Architecture
==============================================
Design principle: decouple the timescale of SOC dynamics from N availability.

SOC dynamics (annual): Reference-rate k values drive SOM pool turnover.
This represents the integrated annual decomposition (winter + summer)
and is calibrated to reproduce observed SOC decline in long-term trials.

N availability (monthly): Q10 temperature + moisture correction determines
how much of the SOM-N pool is mineralized each month. Leaching,
denitrification, and crop uptake operate at monthly resolution.
Crops only access N released during the growing season.

This asymmetry is physically correct: SOC integrates ALL decomposition
(including slow winter breakdown), but crops can only capture N released
when they're actively growing. Winter-mineralized N is largely lost to
leaching before the growing season.

Author: Matthew Wallenstein
Date: April 3, 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.optimize import brentq
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soil_n_model import SOMPoolParams, CropParams, get_default_regions


# ============================================================================
# Climate data
# ============================================================================

@dataclass
class MonthlyClimate:
    name: str
    temp: List[float]
    precip: List[float]
    pet: List[float]
    planting_month: int    # 1-indexed
    maturity_month: int

REGIONAL_CLIMATES = {
    'north_america': MonthlyClimate('US Corn Belt',
        [-6, -4, 3, 10, 17, 22, 24, 23, 18, 11, 3, -4],
        [25, 30, 55, 90, 115, 120, 105, 100, 85, 65, 50, 30],
        [5, 8, 25, 55, 95, 125, 140, 120, 80, 45, 18, 7],
        5, 9),
    'europe': MonthlyClimate('NW Europe',
        [3, 4, 7, 10, 14, 17, 19, 19, 15, 11, 6, 3],
        [55, 45, 50, 50, 60, 55, 55, 55, 50, 55, 60, 60],
        [10, 15, 30, 55, 80, 100, 110, 95, 60, 35, 15, 8],
        10, 7),   # winter wheat
    'east_asia': MonthlyClimate('N China Plain',
        [-3, 0, 7, 15, 21, 26, 27, 26, 21, 14, 6, -1],
        [5, 8, 15, 25, 40, 75, 150, 130, 55, 30, 12, 5],
        [8, 12, 30, 60, 100, 130, 140, 120, 75, 40, 15, 8],
        6, 10),
    'south_asia': MonthlyClimate('Indo-Gangetic',
        [15, 18, 24, 30, 33, 34, 31, 30, 29, 26, 20, 16],
        [20, 15, 10, 10, 20, 100, 250, 250, 150, 30, 5, 10],
        [40, 55, 90, 130, 160, 150, 120, 110, 100, 80, 50, 35],
        11, 4),   # rabi wheat
    'southeast_asia': MonthlyClimate('SE Asia rice',
        [26, 27, 29, 30, 29, 28, 28, 28, 27, 27, 26, 25],
        [10, 20, 40, 70, 150, 150, 160, 170, 200, 160, 50, 15],
        [90, 95, 120, 130, 120, 110, 110, 110, 100, 100, 90, 85],
        6, 11),
    'latin_america': MonthlyClimate('Cerrado/Pampas',
        [25, 25, 24, 22, 19, 17, 17, 19, 21, 23, 24, 25],
        [200, 180, 160, 80, 40, 25, 20, 25, 50, 100, 150, 190],
        [130, 120, 120, 90, 60, 50, 50, 70, 90, 110, 120, 130],
        10, 3),
    'sub_saharan_africa': MonthlyClimate('E/W Africa',
        [25, 26, 26, 25, 24, 22, 21, 22, 23, 24, 24, 25],
        [30, 35, 80, 130, 150, 100, 60, 50, 80, 140, 100, 40],
        [110, 110, 120, 110, 100, 80, 70, 80, 100, 110, 110, 110],
        4, 9),
    'fsu_central_asia': MonthlyClimate('Ukraine/S Russia',
        [-6, -5, 1, 10, 16, 20, 22, 21, 15, 8, 1, -4],
        [35, 30, 30, 35, 45, 55, 55, 40, 35, 35, 40, 40],
        [5, 8, 20, 50, 90, 120, 130, 110, 65, 30, 12, 5],
        4, 8),
}

FAOSTAT_TARGETS = {
    'north_america': 5.50, 'europe': 5.00, 'east_asia': 6.00,
    'south_asia': 3.20, 'southeast_asia': 4.20, 'latin_america': 4.50,
    'sub_saharan_africa': 1.50, 'fsu_central_asia': 2.80
}


# ============================================================================
# Parameters
# ============================================================================

@dataclass
class MonthlyNParams:
    q10: float = 2.0
    t_ref: float = 25.0
    t_min: float = -5.0

    # Moisture: decomposition factor = f(P/PET)
    moist_opt_lo: float = 0.5
    moist_opt_hi: float = 1.5
    moist_min: float = 0.2       # Very dry
    moist_waterlog: float = 0.5  # Saturated

    # N losses
    leach_coeff: float = 0.10    # Per mm drainage, on mineral N pool
    leach_base: float = 0.025    # Base monthly diffusion/matrix loss
    denitrif_base: float = 0.008
    denitrif_wet_mult: float = 4.0

    # Immobilization
    immob_frac: float = 0.15

    # Crop — max_uptake_frac is the peak-month uptake fraction; demand scaling
    # reduces it for non-peak months. Increased from 0.55 to compensate for
    # demand-profile modulation (keeps effective NUE near 0.75 at current fert).
    max_uptake_frac: float = 0.75
    min_n_pool: float = 2.0


# ============================================================================
# Abiotic factors
# ============================================================================

def temp_factor(t: float, p: MonthlyNParams) -> float:
    if t <= p.t_min:
        return 0.0
    return p.q10 ** ((t - p.t_ref) / 10.0)

def moist_factor(pr: float, pe: float, p: MonthlyNParams) -> float:
    if pe <= 0:
        return p.moist_waterlog
    r = pr / pe
    if r < p.moist_opt_lo:
        return p.moist_min + (1 - p.moist_min) * (r / p.moist_opt_lo)
    elif r <= p.moist_opt_hi:
        return 1.0
    else:
        return max(p.moist_waterlog, 1.0 - (r - p.moist_opt_hi) * 0.3)


# ============================================================================
# Crop profiles
# ============================================================================

def growing_months(clim: MonthlyClimate) -> List[int]:
    pm = clim.planting_month - 1
    mm = clim.maturity_month - 1
    if mm >= pm:
        return list(range(pm, mm + 1))
    return list(range(pm, 12)) + list(range(0, mm + 1))

def demand_profile(clim: MonthlyClimate) -> List[float]:
    gm = growing_months(clim)
    d = [0.0] * 12
    n = len(gm)
    if n == 0:
        return [1/12]*12
    for i, m in enumerate(gm):
        pos = i / max(n-1, 1)
        d[m] = np.exp(-((pos-0.6)**2)/(2*0.15**2))
    s = sum(d)
    return [x/s for x in d] if s > 0 else [1/12]*12

def fert_profile(clim: MonthlyClimate) -> List[float]:
    gm = growing_months(clim)
    p = [0.0]*12
    if gm:
        p[gm[0]] = 0.33
        p[gm[len(gm)//2]] += 0.67
    return p


# ============================================================================
# Core: monthly N availability for one year
# ============================================================================

def monthly_n_balance(
    c_active: float, c_slow: float, c_passive: float,
    cn: float, synth_n: float, bnf_annual: float,
    atm_dep: float, climate: MonthlyClimate,
    mineral_n_start: float,
    p: MonthlyNParams,
) -> Dict:
    """
    Run 12 monthly N balance steps for one year.
    Returns crop N uptake, losses, and ending mineral N pool.
    """
    som = SOMPoolParams()
    ref_tf = temp_factor(p.t_ref, p)
    dem = demand_profile(climate)
    fp = fert_profile(climate)
    monthly_bnf = bnf_annual / 12
    monthly_atm = atm_dep / 12

    mineral_n = mineral_n_start
    ann = {'min': 0, 'leach': 0, 'den': 0, 'uptake': 0, 'immob': 0}
    peak_demand = max(dem) if max(dem) > 0 else 1.0

    for month in range(12):
        t = climate.temp[month]
        pr = climate.precip[month]
        pe = climate.pet[month]

        tf = temp_factor(t, p)
        mf = moist_factor(pr, pe, p)
        abiotic = tf * mf / ref_tf / 12.0 if ref_tf > 0 else 0.0

        # Gross mineralization
        n_min = ((som.k_active * c_active + som.k_slow * c_slow +
                  som.k_passive * c_passive) / cn) * 1000 * abiotic

        # Inputs
        n_fert = synth_n * fp[month]
        mineral_n += n_min + n_fert + monthly_bnf + monthly_atm

        # Immobilization
        n_immob = n_min * p.immob_frac
        mineral_n -= n_immob
        mineral_n = max(mineral_n, p.min_n_pool)

        # Leaching
        drainage = max(pr - pe, 0)
        lf = min(p.leach_coeff * drainage / 100 + p.leach_base, 0.60)
        n_leach = mineral_n * lf
        mineral_n -= n_leach

        # Denitrification
        wet = pr > pe * 0.8
        dr = p.denitrif_base * (p.denitrif_wet_mult if wet else 1.0)
        n_den = mineral_n * dr
        mineral_n -= n_den

        # Crop uptake — scaled by demand profile
        if dem[month] > 0.01:
            rel_demand = dem[month] / peak_demand
            uptake_frac = p.max_uptake_frac * rel_demand
            n_up = min(mineral_n * uptake_frac, mineral_n - p.min_n_pool)
            n_up = max(n_up, 0)
        else:
            n_up = 0.0
        mineral_n -= n_up

        ann['min'] += n_min
        ann['leach'] += n_leach
        ann['den'] += n_den
        ann['uptake'] += n_up
        ann['immob'] += n_immob

    ann['mineral_n_end'] = mineral_n
    return ann


# ============================================================================
# SOM pool update (shared by run_model and coupled_monthly)
# ============================================================================

def update_som_pools(c_a: float, c_s: float, c_p: float,
                     c_in: float, som: SOMPoolParams) -> tuple:
    """Annual SOM pool update with humification transfers.

    Returns (c_a_new, c_s_new, c_p_new).
    """
    dec_a = som.k_active * c_a
    dec_s = som.k_slow * c_s
    dec_p = som.k_passive * c_p
    c_a_new = c_a - dec_a + c_in * 0.90
    c_s_new = c_s - dec_s + c_in * 0.10 + dec_a * som.h_active_to_slow
    c_p_new = c_p - dec_p + dec_s * som.h_slow_to_passive
    return c_a_new, c_s_new, c_p_new


# ============================================================================
# Dynamic spinup: iterate Century 3-pool to true steady state
# ============================================================================

def century_dynamic_spinup(
    region_key: str,
    n_spinup: int = 2000,
    tol: float = 0.002,
    p: MonthlyNParams = None,
    synth_n: float = None,
    bnf_annual: float = None,
    yield_max_override: float = None,
    verbose: bool = False,
) -> Dict:
    """Iterate Century 3-pool SOM to true steady state.

    Runs the full annual loop (monthly N balance + yield + residue C +
    SOM pool update) until SOC stabilises over a 50-year window.
    Eliminates the initialization transient caused by the fraction-based
    pool allocation not matching the true equilibrium for the given
    fertilizer/yield/residue regime.

    Convergence criterion: fractional SOC change over a 50-year window
    < tol, with a minimum of 100 years.

    Parameters
    ----------
    region_key : str
        Region identifier (e.g. 'north_america').
    n_spinup : int
        Maximum spinup years (default 2000).
    tol : float
        Convergence tolerance on 50-year SOC drift (default 0.002 = 0.2%).
    p : MonthlyNParams
        Monthly N balance parameters.
    synth_n : float
        Fertilizer rate; defaults to region's current rate.
    bnf_annual : float
        BNF; defaults to region-specific value from get_regional_bnf.
    yield_max_override : float
        Calibrated Ymax; if None, uses region default.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        c_active, c_slow, c_passive : equilibrium pool sizes (t C/ha)
        soc : total SOC at equilibrium
        mineral_n : equilibrium mineral N pool
        yield_eq : equilibrium yield (t/ha)
        n_min_eq : equilibrium N mineralization (kg N/ha/yr)
        c_input_eq : equilibrium residue C input (t C/ha/yr)
        converged : bool
        years_to_converge : int
    """
    if p is None:
        p = MonthlyNParams()

    regions = get_default_regions()
    region = regions[region_key]
    som = SOMPoolParams()
    crop = CropParams()
    climate = REGIONAL_CLIMATES[region_key]

    if synth_n is None:
        synth_n = region.synth_n_current
    if bnf_annual is None:
        bnf_annual = get_regional_bnf(region_key)

    # Initial pool allocation (fraction-based, same as run_model)
    soc = region.soc_initial
    cn = region.cn_bulk
    c_a = soc * som.f_active
    c_s = soc * som.f_slow
    c_p = soc * (1 - som.f_active - som.f_slow)

    # Yield parameters
    mit_c = (region.mitscherlich_c_regional
             if region.mitscherlich_c_regional > 0 else crop.mitscherlich_c)
    ym = yield_max_override if yield_max_override else region.yield_max_regional
    n_grain_t = crop.grain_n_fraction * 1000
    hi = crop.harvest_index
    rf = (1 - hi) / hi
    rr = region.residue_retention

    mineral_n = 12.0

    converged = False
    years_to_converge = n_spinup
    conv_window = 50
    soc_history = []

    for yr in range(n_spinup):
        # Monthly N balance
        nb = monthly_n_balance(
            c_a, c_s, c_p, cn, synth_n, bnf_annual,
            region.atm_n_deposition, climate, mineral_n, p)
        mineral_n = nb['mineral_n_end']

        # Yield
        n_eff = nb['uptake']
        y = ym * (1 - np.exp(-mit_c * n_eff))
        y_stoich = n_eff / n_grain_t if n_grain_t > 0 else y
        y = min(y, y_stoich)
        y = max(region.yield_min_regional if region.yield_min_regional > 0 else 0.0, y)

        # Residue C
        shoot_c = y * 1000 * 0.45 * rf * rr / 1000
        root_c = y * 1000 * 0.45 * rf * region.root_shoot_c_ratio / 1000
        c_in = (shoot_c + root_c) * region.cre_regional

        # Update SOM pools
        c_a, c_s, c_p = update_som_pools(c_a, c_s, c_p, c_in, som)

        soc = c_a + c_s + c_p
        soc_history.append(soc)

        # Window-based convergence
        if yr >= conv_window:
            soc_old = soc_history[yr - conv_window]
            frac_drift = abs(soc - soc_old) / max(soc_old, 0.1)
            if frac_drift < tol and yr >= 100:
                converged = True
                years_to_converge = yr + 1
                if verbose:
                    print(f'  Century spinup yr {yr:>3d}: SOC={soc:.2f} '
                          f'Ca={c_a:.2f} Cs={c_s:.2f} Cp={c_p:.2f} '
                          f'y={y:.2f} Nmin={nb["min"]:.1f} '
                          f'Δ50yr={frac_drift:.6f}')
                break
        else:
            frac_drift = 1.0

        if verbose and (yr < 5 or yr % 50 == 0):
            print(f'  Century spinup yr {yr:>3d}: SOC={soc:.2f} '
                  f'Ca={c_a:.2f} Cs={c_s:.2f} Cp={c_p:.2f} '
                  f'y={y:.2f} Nmin={nb["min"]:.1f} '
                  f'Δ50yr={frac_drift:.6f}')

    soc_eq = c_a + c_s + c_p

    if verbose:
        print(f'  {"CONVERGED" if converged else "NOT CONVERGED"} after '
              f'{years_to_converge} years.  SOC={soc_eq:.2f}')

    return {
        'c_active': c_a, 'c_slow': c_s, 'c_passive': c_p,
        'soc': soc_eq,
        'mineral_n': mineral_n,
        'yield_eq': y,
        'n_min_eq': nb['min'],
        'c_input_eq': c_in,
        'converged': converged,
        'years_to_converge': years_to_converge,
    }


# ============================================================================
# Full simulation: annual SOM + monthly N
# ============================================================================

def run_model(
    region_key: str,
    synth_n: float = None,
    n_years: int = 200,
    yield_max_override: float = None,
    bnf_annual: float = None,
    residue_ret_override: float = None,
    p: MonthlyNParams = None,
    verbose: bool = False,
) -> Dict:
    regions = get_default_regions()
    region = regions[region_key]
    som = SOMPoolParams()
    crop = CropParams()
    if p is None:
        p = MonthlyNParams()
    climate = REGIONAL_CLIMATES[region_key]

    if synth_n is None:
        synth_n = region.synth_n_current
    if bnf_annual is None:
        bnf_annual = get_regional_bnf(region_key)
    rr = residue_ret_override if residue_ret_override is not None else region.residue_retention

    soc = region.soc_initial
    cn = region.cn_bulk
    c_a = soc * som.f_active
    c_s = soc * som.f_slow
    c_p = soc * (1 - som.f_active - som.f_slow)

    c_mits = crop.mitscherlich_c
    ym = yield_max_override if yield_max_override else region.yield_max_regional
    n_grain_t = crop.grain_n_fraction * 1000

    res = {k: [] for k in ['year','soc','yield_tha','n_min','n_leach','n_den','n_uptake','n_immob']}
    mineral_n = 12.0

    for year in range(n_years):
        # --- Monthly N balance (temperature/moisture corrected) ---
        nb = monthly_n_balance(
            c_a, c_s, c_p, cn, synth_n, bnf_annual,
            region.atm_n_deposition, climate, mineral_n, p
        )
        mineral_n = nb['mineral_n_end']

        # --- Yield from crop N uptake ---
        n_eff = nb['uptake']
        y = ym * (1 - np.exp(-c_mits * n_eff))
        y_stoich = n_eff / n_grain_t if n_grain_t > 0 else y
        y = min(y, y_stoich)
        y = max(y, region.yield_min_regional)

        # --- Annual SOM dynamics (reference-rate k values) ---
        hi = crop.harvest_index
        rf = (1 - hi) / hi
        shoot_c = y * 1000 * 0.45 * rf * rr / 1000
        root_c = y * 1000 * 0.45 * rf * region.root_shoot_c_ratio / 1000
        c_in = (shoot_c + root_c) * region.cre_regional

        c_a, c_s, c_p = update_som_pools(c_a, c_s, c_p, c_in, som)
        soc = c_a + c_s + c_p

        res['year'].append(year)
        res['soc'].append(soc)
        res['yield_tha'].append(y)
        res['n_min'].append(nb['min'])
        res['n_leach'].append(nb['leach'])
        res['n_den'].append(nb['den'])
        res['n_uptake'].append(nb['uptake'])
        res['n_immob'].append(nb['immob'])

        if verbose and (year < 5 or year % 50 == 0 or year == n_years-1):
            print(f"  Yr {year:>3d}: SOC={soc:.1f} y={y:.2f} Nmin={nb['min']:.1f} "
                  f"Nup={nb['uptake']:.1f} Nlch={nb['leach']:.1f} Nden={nb['den']:.1f}")

    return res


# ============================================================================
# Calibration
# ============================================================================

def calibrate_ym(region_key: str, target: float, p: MonthlyNParams = None) -> float:
    def obj(ym):
        r = run_model(region_key, n_years=5, yield_max_override=ym, p=p)
        return r['yield_tha'][2] - target
    try:
        return brentq(obj, 1.0, 50.0, xtol=0.01)
    except ValueError:
        best, best_e = 10.0, 999
        for ym in np.arange(2.0, 40.0, 0.5):
            r = run_model(region_key, n_years=5, yield_max_override=ym, p=p)
            e = abs(r['yield_tha'][2] - target)
            if e < best_e:
                best, best_e = ym, e
        return best


# ============================================================================
# Managed transition: empirically-constrained BNF
# ============================================================================
#
# The old managed transition applied bnf_potential (15-35 kg/ha/yr) uniformly
# to ALL cropland. This overstates system-level N inputs for three reasons:
#
# 1. ROTATION FRACTION: Only 25-50% of cropland is in legumes in any year.
#    The rest grows cereals and gets zero BNF.
#
# 2. GRAIN N EXPORT: Grain legumes (soybean, chickpea) export more N in grain
#    than they fix via BNF in ~80% of cases (Salvagiotti et al. 2008;
#    Ciampitti & Vyn 2014). Net N balance for soybean averages -40 kg N/ha.
#    The N credit to subsequent cereals comes from root/nodule residues only:
#    typically 30-80 kg N/ha (Peoples et al. 2009; Ladha et al. 2022).
#
# 3. CEREAL AREA PENALTY: Land in legume rotation doesn't grow cereals that
#    year. Total cereal production = yield × (1 - legume_fraction) × area.
#    Legumes produce lower caloric yield (soybean ~1.8-2.5 t/ha vs corn 5-6).
#
# Corrected approach: net landscape-level BNF = legume_frac × net_n_credit,
# where net_n_credit accounts for grain export. Cereal area reduced by
# legume fraction. Legume production added at reduced caloric equivalence.
#
# Literature values for net_n_credit (N remaining in soil for subsequent crop):
#   Peoples et al. 2009 (global meta-analysis): 25-80 kg N/ha
#   Ladha et al. 2022 (tropical rice-legume): 30-60 kg N/ha
#   Salvagiotti et al. 2008 (soybean): net balance typically negative,
#       but root+nodule residue contributes 40-60 kg N/ha to next crop
#   Preissel et al. 2015 (European grain legumes): 25-50 kg N/ha credit
#   Hungria & Mendes 2015 (Brazil soybean-maize): 30-40 kg N/ha residual

MANAGED_TRANSITION_PARAMS = {
    # legume_frac: fraction of cropland in legumes in any given year
    # net_n_credit: kg N/ha delivered to subsequent cereal (after grain export)
    # legume_yield_cereal_equiv: legume yield in cereal-equivalent t/ha
    #   (soy ~2.2 t/ha × 1.5 caloric adjustment ≈ 1.5 cereal-equiv;
    #    pulses ~1.0-1.5 t/ha × 1.3 ≈ 1.3 cereal-equiv)
    # free_living_bnf: non-symbiotic BNF (cyanobacteria, free-living fixers), kg/ha/yr
    #   Applied to all cropland. Typically 3-8 kg/ha (Herridge et al. 2008)
    'north_america':        {'legume_frac': 0.35, 'net_n_credit': 50, 'legume_yield_ceq': 1.8, 'free_living_bnf': 5},
    'europe':               {'legume_frac': 0.25, 'net_n_credit': 40, 'legume_yield_ceq': 1.3, 'free_living_bnf': 5},
    'east_asia':            {'legume_frac': 0.20, 'net_n_credit': 35, 'legume_yield_ceq': 1.2, 'free_living_bnf': 5},
    'south_asia':           {'legume_frac': 0.30, 'net_n_credit': 40, 'legume_yield_ceq': 1.0, 'free_living_bnf': 5},
    'southeast_asia':       {'legume_frac': 0.25, 'net_n_credit': 45, 'legume_yield_ceq': 1.2, 'free_living_bnf': 8},  # rice-paddy BNF adds to free-living
    'latin_america':        {'legume_frac': 0.45, 'net_n_credit': 40, 'legume_yield_ceq': 1.5, 'free_living_bnf': 5},  # strong soy tradition
    'sub_saharan_africa':   {'legume_frac': 0.25, 'net_n_credit': 30, 'legume_yield_ceq': 0.8, 'free_living_bnf': 5},  # P limitation constrains BNF
    'fsu_central_asia':     {'legume_frac': 0.20, 'net_n_credit': 35, 'legume_yield_ceq': 1.0, 'free_living_bnf': 5},
}


def get_regional_bnf(region_key: str) -> float:
    """Compute landscape-level BNF for a region from managed transition params.

    Returns kg N ha⁻¹ yr⁻¹ averaged across cropland, accounting for legume
    rotation N credits spread over cereal hectares plus free-living fixation.

    Sources: Herridge et al. (2008) New Phytologist for symbiotic + free-living;
    regional legume fractions from FAOSTAT crop area shares.
    """
    if region_key not in MANAGED_TRANSITION_PARAMS:
        return 5.0  # fallback
    mt = MANAGED_TRANSITION_PARAMS[region_key]
    landscape_bnf = (mt['legume_frac'] * mt['net_n_credit']
                     / (1 - mt['legume_frac']))
    return landscape_bnf + mt['free_living_bnf']


# ============================================================================
# Dependency
# ============================================================================

def compute_dep(region_key, ym, n_years=300, managed=False, p=None):
    regions = get_default_regions()
    region = regions[region_key]

    rc = run_model(region_key, n_years=5, yield_max_override=ym, p=p)
    yc = rc['yield_tha'][2]

    kw = dict(synth_n=0.0, n_years=n_years, yield_max_override=ym, p=p)
    if managed:
        mt = MANAGED_TRANSITION_PARAMS[region_key]
        # Net landscape BNF: rotation N credit spread over cereal hectares,
        # plus free-living BNF on all land.
        # The net_n_credit accrues on legume-year hectares but benefits the
        # subsequent cereal year, so landscape-average = frac × credit.
        # We divide by (1 - legume_frac) to express per cereal-hectare,
        # since the model runs per-hectare of cereal.
        landscape_bnf = (mt['legume_frac'] * mt['net_n_credit']
                         / (1 - mt['legume_frac']))
        kw['bnf_annual'] = landscape_bnf + mt['free_living_bnf']
        kw['residue_ret_override'] = min(region.residue_retention + 0.15, 0.95)

    rnf = run_model(region_key, **kw)
    yss_per_ha = np.mean(rnf['yield_tha'][-20:])
    soc_ss = np.mean(rnf['soc'][-20:])

    # For managed transition, total production includes cereal area penalty
    # and legume caloric offset
    if managed:
        mt = MANAGED_TRANSITION_PARAMS[region_key]
        lf = mt['legume_frac']
        # Effective yield: weighted average of cereal and legume production
        yss_effective = yss_per_ha * (1 - lf) + mt['legume_yield_ceq'] * lf
    else:
        yss_effective = yss_per_ha

    return {
        'yc': yc, 'yss': yss_effective,
        'yss_cereal_only': yss_per_ha,  # for diagnostics
        'soc_i': region.soc_initial, 'soc_ss': soc_ss,
        'soc_loss': (region.soc_initial - soc_ss)/region.soc_initial*100,
        'dep': (yc - yss_effective)/yc*100 if yc > 0 else 0,
        'n_up_c': rc['n_uptake'][2],
        'n_up_ss': np.mean(rnf['n_uptake'][-20:]),
        'n_lch_ss': np.mean(rnf['n_leach'][-20:]),
        'n_den_ss': np.mean(rnf['n_den'][-20:]),
        'n_min_ss': np.mean(rnf['n_min'][-20:]),
        'ts': rnf,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    regions = get_default_regions()
    p = MonthlyNParams()

    print("=" * 110)
    print("MONTHLY MODEL v3 — Hybrid: Annual SOM + Monthly N Availability")
    print("=" * 110)

    # 1. Calibrate
    print("\n--- Calibrating yield_max ---\n")
    cal = {}
    for rk in REGIONAL_CLIMATES:
        t = FAOSTAT_TARGETS[rk]
        ym = calibrate_ym(rk, t, p)
        cal[rk] = ym
        r = run_model(rk, n_years=5, yield_max_override=ym, p=p)
        print(f"  {regions[rk].name:>35}: target={t:.2f}, ym={ym:.3f}, "
              f"actual={r['yield_tha'][2]:.2f}, Nup={r['n_uptake'][2]:.1f}, "
              f"Nlch={r['n_leach'][2]:.1f}, Nden={r['n_den'][2]:.1f}")

    # 2. Abrupt
    print("\n\n--- Abrupt Withdrawal ---\n")
    hdr = f"{'Region':>35} {'Y_now':>6} {'Y_SS':>6} {'SOC_i':>6} {'SOC_SS':>6} {'SOC%':>6} {'Dep%':>7} {'Nlch':>6} {'Nden':>6} {'SS_yr':>6}"
    print(hdr)
    print("-" * 108)

    tp_c = tp_s = 0
    abrupt = {}
    for rk in REGIONAL_CLIMATES:
        reg = regions[rk]
        d = compute_dep(rk, cal[rk], n_years=300, managed=False, p=p)
        abrupt[rk] = d
        pc = d['yc'] * reg.cropland_mha
        ps = d['yss'] * reg.cropland_mha
        tp_c += pc; tp_s += ps
        # Find when yield stabilizes (within 2% of SS)
        ts = d['ts']
        ss_yr = 300
        for i, y in enumerate(ts['yield_tha']):
            if abs(y - d['yss']) / max(d['yss'], 0.01) < 0.02:
                ss_yr = i; break
        print(f"  {reg.name:>33} {d['yc']:>5.2f} {d['yss']:>5.2f} "
              f"{d['soc_i']:>5.1f} {d['soc_ss']:>5.1f} {d['soc_loss']:>5.1f}% "
              f"{d['dep']:>6.1f}% {d['n_lch_ss']:>5.1f} {d['n_den_ss']:>5.1f} {ss_yr:>6d}")

    gdep_a = (1 - tp_s/tp_c)*100
    pop = sum(r.pop_supported for r in regions.values())
    print(f"\n  GLOBAL DEPENDENCY (abrupt): {gdep_a:.1f}%")
    print(f"  Production loss: {tp_c - tp_s:.0f} Mt of {tp_c:.0f} Mt")
    print(f"  Population at risk: {gdep_a/100*pop:.0f} M of {pop:.0f} M")

    # 3. Managed (empirically-constrained BNF)
    print("\n\n--- Managed Transition (empirically-constrained BNF) ---\n")
    print(f"{'Region':>35} {'Y_now':>6} {'Y_cer':>6} {'Y_eff':>6} {'LegFr':>6} {'BNF':>6} {'SOC%':>6} {'Dep%':>7}")
    print("-" * 105)

    tp_c2 = tp_s2 = 0
    managed = {}
    for rk in REGIONAL_CLIMATES:
        reg = regions[rk]
        d = compute_dep(rk, cal[rk], n_years=300, managed=True, p=p)
        managed[rk] = d
        mt = MANAGED_TRANSITION_PARAMS[rk]
        landscape_bnf = (mt['legume_frac'] * mt['net_n_credit']
                         / (1 - mt['legume_frac'])) + mt['free_living_bnf']
        pc = d['yc'] * reg.cropland_mha
        ps = d['yss'] * reg.cropland_mha
        tp_c2 += pc; tp_s2 += ps
        print(f"  {reg.name:>33} {d['yc']:>5.2f} {d['yss_cereal_only']:>5.2f} "
              f"{d['yss']:>5.2f} {mt['legume_frac']:>5.2f} {landscape_bnf:>5.1f} "
              f"{d['soc_loss']:>5.1f}% {d['dep']:>6.1f}%")

    gdep_m = (1 - tp_s2/tp_c2)*100
    print(f"\n  GLOBAL DEPENDENCY (managed): {gdep_m:.1f}%")
    print(f"  Population at risk: {gdep_m/100*pop:.0f} M")
    print(f"\n  Key: Y_cer = cereal yield on cereal hectares")
    print(f"       Y_eff = area-weighted effective yield (cereal + legume cereal-equiv)")
    print(f"       LegFr = fraction of cropland in legumes")
    print(f"       BNF = net landscape BNF per cereal hectare (kg N/ha/yr)")

    # 4. Empirical validation
    print("\n\n--- Empirical Validation ---")

    res_bb = run_model('europe', synth_n=0.0, n_years=180,
                       yield_max_override=cal['europe'], p=p)
    ybb = np.mean(res_bb['yield_tha'][-20:])
    socbb = np.mean(res_bb['soc'][-20:])
    print(f"\n  Broadbalk (180yr): yield={ybb:.2f} (target ~1.0), SOC={socbb:.1f} (target ~22)")

    res_mw = run_model('north_america', synth_n=0.0, n_years=150,
                       yield_max_override=cal['north_america'], p=p)
    ymw = np.mean(res_mw['yield_tha'][-20:])
    socmw = np.mean(res_mw['soc'][-20:])
    print(f"  Morrow (150yr):    yield={ymw:.2f} (target ~1.9), SOC={socmw:.1f} (target ~35)")

    # Bad Lauchstädt: uses same 'europe' params as Broadbalk (not independent).
    # TODO: add site-specific climate/soil params for true independent validation.
    res_bl = run_model('europe', synth_n=0.0, n_years=120,
                       yield_max_override=cal['europe'], p=p)
    ybl = np.mean(res_bl['yield_tha'][-20:])
    socbl = np.mean(res_bl['soc'][-20:])
    print(f"  Bad Lauchstädt (120yr): yield={ybl:.2f}, SOC={socbl:.1f}  [* same params as Broadbalk]")

    # 5. Summary
    print("\n\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"\n  {'Model':>45} {'Abrupt':>10} {'Managed':>10} {'BB yield':>10} {'BB SOC':>10}")
    print(f"  {'-'*85}")
    print(f"  {'Annual v6 (NUE=0.75, no losses)':>45} {'31.4%':>10} {'13.6%':>10} {'3.85':>10} {'36.4':>10}")
    print(f"  {'Monthly v1 (Q10, no moisture, low leach)':>45} {'48.8%':>10} {'n/a':>10} {'1.54':>10} {'23.6':>10}")
    print(f"  {'Monthly v3 (hybrid: ann SOM + monthly N)':>45} {f'{gdep_a:.1f}%':>10} {f'{gdep_m:.1f}%':>10} {f'{ybb:.2f}':>10} {f'{socbb:.1f}':>10}")
    print(f"  {'Erisman et al. (2008)':>45} {'48%':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
    print(f"  {'Smil (2001)':>45} {'~40%':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10}")

    # 6. N budget diagnostic
    print("\n\n--- N Budget: North America (Year 2, current fert) ---")
    rd = run_model('north_america', n_years=5, yield_max_override=cal['north_america'],
                   verbose=True, p=p)
    i = 2
    na_reg = regions['north_america']
    na_fert = na_reg.synth_n_current
    na_atm = na_reg.atm_n_deposition
    na_bnf = 5.0
    total_in = rd['n_min'][i] + na_fert + na_atm + na_bnf
    total_out = rd['n_leach'][i] + rd['n_den'][i] + rd['n_immob'][i] + rd['n_uptake'][i]
    print(f"\n  Gross min={rd['n_min'][i]:.1f}, Fert={na_fert}, Atm={na_atm}, BNF={na_bnf} -> Total in={total_in:.1f}")
    print(f"  Leach={rd['n_leach'][i]:.1f}, Denitrif={rd['n_den'][i]:.1f}, "
          f"Immob={rd['n_immob'][i]:.1f}, Uptake={rd['n_uptake'][i]:.1f} -> Total out={total_out:.1f}")
    print(f"  Effective NUE = uptake/total_in = {rd['n_uptake'][i]/total_in:.3f}")

    # 7. Save data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    for scenario, results_dict, label in [
        ('no_adaptation', abrupt, 'abrupt'),
        ('managed', managed, 'managed')
    ]:
        fname = os.path.join(data_dir, f'monthly_v3_dependency_{scenario}.csv')
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['region','region_key','yield_current','yield_ss','soc_initial',
                        'soc_ss','soc_loss_pct','dependency_pct','n_uptake_current',
                        'n_uptake_ss','n_leach_ss','n_denitrif_ss','n_min_ss',
                        'cropland_mha','production_current_mt','production_ss_mt'])
            for rk in REGIONAL_CLIMATES:
                reg = regions[rk]
                d = results_dict[rk]
                w.writerow([reg.name, rk, f"{d['yc']:.4f}", f"{d['yss']:.4f}",
                           f"{d['soc_i']:.1f}", f"{d['soc_ss']:.1f}",
                           f"{d['soc_loss']:.2f}", f"{d['dep']:.2f}",
                           f"{d['n_up_c']:.1f}", f"{d['n_up_ss']:.1f}",
                           f"{d['n_lch_ss']:.1f}", f"{d['n_den_ss']:.1f}",
                           f"{d['n_min_ss']:.1f}", f"{reg.cropland_mha}",
                           f"{d['yc']*reg.cropland_mha:.1f}",
                           f"{d['yss']*reg.cropland_mha:.1f}"])
        print(f"\n  Data saved: {fname}")
