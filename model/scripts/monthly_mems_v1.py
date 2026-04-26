"""
Monthly MEMS Model v1 — Hybrid Architecture (MEMS pools + Monthly N)
====================================================================
Structural sensitivity test for monthly_model_v3.py.

Uses the SAME monthly N availability framework (Q10, moisture, leaching,
denitrification, demand-scaled crop uptake, Mitscherlich yield, grain-N
stoichiometric cap) but replaces Century's 3-pool SOM dynamics
(active/slow/passive) with MEMS 4-pool dynamics (POM/DOM/MBC/MAOM).

Key structural differences from Century v3:
  1. Explicit microbial biomass pool (MBC) mediating all transformations
  2. MAOM formed via microbial necromass (in-vivo pathway), not humification
  3. Carbon use efficiency (CUE) governs growth vs respiration split
  4. DOM as intermediate pool between POM decomposition and microbial uptake
  5. Mineral sorption capacity (Qmax) limits MAOM accumulation
  6. Priming: fresh C availability modulates MAOM desorption rate
  7. N mineralization from stoichiometric balance of microbial consumption,
     NOT from pool turnover × C:N ratio

Monthly N availability is computed identically to monthly_model_v3:
  - MEMS annual N mineralization is distributed across 12 months using
    Q10 temperature and P/PET moisture corrections
  - Same leaching, denitrification, immobilization, crop uptake logic
  - Same demand-scaled Gaussian uptake profile
  - Same fertilizer application timing

Author: Matthew Wallenstein
Date: April 3, 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.optimize import brentq
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from soil_n_model import CropParams, get_default_regions

# Import shared infrastructure from Century monthly model
from monthly_model_v3 import (
    MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES, FAOSTAT_TARGETS,
    MANAGED_TRANSITION_PARAMS,
    temp_factor, moist_factor,
    growing_months, demand_profile, fert_profile,
    get_regional_bnf,
)


# ============================================================================
# MEMS Pool Parameters
# ============================================================================

@dataclass
class MEMSPoolParams:
    """MEMS pool structure and kinetic parameters."""

    # --- Pool fractions of total SOC (initial partitioning) ---
    f_pom: float = 0.25         # POM: ~20-30% of SOC (Lavallee et al. 2020)
    f_maom: float = 0.70        # MAOM: ~60-75% (dominant fraction; Cotrufo et al. 2019)
    f_mbc: float = 0.03         # MBC: ~2-3% (Anderson & Domsch 1989)
    f_dom: float = 0.02         # DOM: ~1-2% (fast-cycling intermediate)

    # --- C:N ratios ---
    cn_pom: float = 25.0        # Plant-derived, range 15-40
    cn_dom: float = 20.0        # Leached from POM + desorbed from MAOM
    cn_mbc: float = 8.0         # Microbial biomass (bacteria ~5, fungi ~10)
    cn_maom: float = 10.0       # Necromass + sorbed; range 8-12

    # --- Rate constants (yr^-1) ---
    k_pom_to_dom: float = 0.08      # POM depolymerization; MRT ~12 yr (Heckman et al. 2022)
    k_maom_desorption: float = 0.007 # MAOM desorption; MRT ~140 yr (Georgiou et al. 2022)
    k_mbc_turnover: float = 3.0      # MBC turnover; ~4 month MRT
    k_dom_uptake: float = 10.0       # DOM uptake by microbes (fast)
    k_dom_sorption: float = 2.0      # DOM sorption to mineral surfaces

    # --- Carbon Use Efficiency (CUE) ---
    cue_max: float = 0.40       # N-replete (Wieder et al. 2015)
    cue_min: float = 0.20       # Severe N limitation (Manzoni et al. 2012)
    cue_km_n: float = 0.5       # Half-saturation for CUE-N response

    # --- Necromass partitioning ---
    f_necro_to_maom: float = 0.55   # -> MAOM (in-vivo pathway; Liang et al. 2017)
    f_necro_to_pom: float = 0.20    # -> POM (structural: cell walls, chitin, fungal hyphae)
    f_necro_to_dom: float = 0.10    # -> DOM (labile: cytoplasm, metabolites)
    # Remainder (0.15) respired during turnover

    # --- Mineral sorption capacity ---
    qmax_per_claysilt: float = 65.0  # t C/ha at 100% clay+silt (Georgiou et al. 2022; range 30-80)

    # --- Priming ---
    priming_sensitivity: float = 0.5  # Modulates MAOM desorption by fresh C input


# ============================================================================
# Annual MEMS pool dynamics
# ============================================================================

def mems_annual_step(
    c_pom: float, c_dom: float, c_mbc: float, c_maom: float,
    c_input: float, qmax: float, mems: MEMSPoolParams,
    n_available_frac: float = 1.0,
    pom_baseline: float = None,
) -> Dict:
    """
    Advance MEMS pools by one annual timestep.

    Parameters
    ----------
    c_pom, c_dom, c_mbc, c_maom : current pool sizes (t C/ha)
    c_input : residue C input (t C/ha/yr) -> all goes to POM
    qmax : mineral sorption capacity (t C/ha)
    mems : MEMSPoolParams
    n_available_frac : N availability as fraction of baseline (for CUE)
    pom_baseline : baseline POM for priming calculation

    Returns
    -------
    dict with new pool sizes and N mineralization (kg N/ha/yr)
    """
    m = mems
    soc_before = c_pom + c_dom + c_mbc + c_maom

    # --- CUE from N availability ---
    f = max(n_available_frac, 0.0)
    cue = m.cue_min + (m.cue_max - m.cue_min) * f / (f + m.cue_km_n)

    # --- MAOM saturation ---
    maom_sat = min(c_maom / max(qmax, 0.1), 1.0)

    # --- Priming: fresh C from POM decomposition stimulates MAOM desorption ---
    pom_decomp_rate = m.k_pom_to_dom * c_pom
    if pom_baseline and pom_baseline > 0:
        priming = 1.0 + m.priming_sensitivity * (
            pom_decomp_rate / (m.k_pom_to_dom * pom_baseline) - 1.0)
        priming = max(0.5, min(2.0, priming))
    else:
        priming = 1.0

    # --- Step 1: Fluxes into and out of DOM pool ---
    pom_decomp_c = m.k_pom_to_dom * c_pom
    maom_desorb_c = m.k_maom_desorption * c_maom * priming

    # DOM removal: microbial uptake + mineral sorption, applied to existing stock
    sorption_rate = m.k_dom_sorption * max(0, 1.0 - maom_sat)
    total_rate = m.k_dom_uptake + sorption_rate
    f_uptake = m.k_dom_uptake / total_rate if total_rate > 0 else 0.5

    dom_removed_frac = 1.0 - np.exp(-total_rate)
    dom_removed_c = c_dom * dom_removed_frac

    dom_to_mic_c = dom_removed_c * f_uptake
    dom_to_maom_c = dom_removed_c * (1.0 - f_uptake)

    # --- Step 2: Microbial assimilation ---
    mic_assimilated_c = dom_to_mic_c * cue
    mic_respired_uptake = dom_to_mic_c * (1.0 - cue)

    # --- Step 3: MBC as true state variable ---
    # Analytical solution for dMBC/dt = assimilation - k*MBC over one year,
    # assuming constant assimilation rate. This avoids Euler instability
    # (k_turnover=3/yr requires dt < 0.33 yr for Euler stability).
    # MBC(1) = (assimilated/k)(1 - exp(-k)) + MBC(0)*exp(-k)
    k = m.k_mbc_turnover
    exp_k = np.exp(-k)
    c_mbc_new = max(0.001, (mic_assimilated_c / k) * (1.0 - exp_k) + c_mbc * exp_k)
    # Effective death over the year = what left MBC
    mic_death_c = c_mbc + mic_assimilated_c - c_mbc_new

    # --- Step 5: Necromass partitioning ---
    necro_to_maom_c = mic_death_c * m.f_necro_to_maom * max(0, 1.0 - maom_sat)
    necro_to_pom_c = mic_death_c * m.f_necro_to_pom
    necro_to_dom_c = mic_death_c * m.f_necro_to_dom
    necro_respired_c = max(0, mic_death_c - necro_to_maom_c - necro_to_pom_c - necro_to_dom_c)
    total_respired = mic_respired_uptake + necro_respired_c

    # --- Step 6: N mineralization from stoichiometric balance ---
    # N in DOM consumed by microbes. DOM C:N reflects the mixture of
    # existing DOM stock plus fresh inputs from POM and MAOM this year.
    # But removal acts on the existing stock (before inputs arrive).
    # Use the existing DOM C:N for the consumed fraction.
    mic_n_consumed = dom_to_mic_c / m.cn_dom * 1000  # kg N/ha
    mic_n_demand = mic_assimilated_c / m.cn_mbc * 1000       # kg N/ha
    net_from_consumption = mic_n_consumed - mic_n_demand

    # N from microbial turnover overhead (fraction NOT recycled to organic pools)
    mic_death_n = mic_death_c / m.cn_mbc * 1000
    f_recycled = (m.f_necro_to_maom * max(0, 1.0 - maom_sat)
                  + m.f_necro_to_pom + m.f_necro_to_dom)
    net_from_turnover = mic_death_n * (1.0 - f_recycled)

    net_n_mineralized = net_from_consumption + net_from_turnover  # kg N/ha/yr

    # --- Step 7: Update pools ---
    c_pom_new = c_pom + c_input + necro_to_pom_c - pom_decomp_c
    c_pom_new = max(0.01, c_pom_new)

    c_maom_new = c_maom + necro_to_maom_c + dom_to_maom_c - maom_desorb_c
    c_maom_new = max(0.01, min(c_maom_new, qmax))

    # DOM: existing stock - removed + inputs (POM decomp, MAOM desorb, necromass)
    c_dom_new = max(0.001, c_dom - dom_removed_c + pom_decomp_c + maom_desorb_c + necro_to_dom_c)

    # --- C balance check ---
    # Pool floor clamps (max(0.01,...)) can inject small amounts of C when
    # pools would otherwise go negative. Track but don't assert on small
    # violations from clamping.
    soc_after = c_pom_new + c_dom_new + c_mbc_new + c_maom_new
    c_balance = (soc_before + c_input) - (soc_after + total_respired)

    return {
        'c_pom': c_pom_new, 'c_dom': c_dom_new,
        'c_mbc': c_mbc_new, 'c_maom': c_maom_new,
        'net_n_mineralized': net_n_mineralized,
        'cue': cue, 'maom_sat': maom_sat, 'priming': priming,
        'total_respired': total_respired,
        'c_balance': c_balance,
        # Per-step flux decomposition (t C/ha/yr). Added 2026-04-14 for
        # matched-mems-2026-04-15 Fig 6 rework and Supp Fig S4 flux decomposition.
        # See matched-mems-2026-04-15/PARAMETERS.md.
        'flux_pom_to_dom': pom_decomp_c,            # POM decomposition → DOM
        'flux_maom_desorb_to_dom': maom_desorb_c,   # MAOM desorption → DOM
        'flux_dom_removed': dom_removed_c,          # Total DOM removal
        'flux_dom_to_mic': dom_to_mic_c,            # DOM → MBC (pre-CUE)
        'flux_dom_to_maom_sorption': dom_to_maom_c, # DOM → MAOM (direct sorption)
        'flux_mic_assimilated': mic_assimilated_c,  # DOM → MBC after CUE
        'resp_cue': mic_respired_uptake,            # CUE respiration on DOM uptake
        'flux_mic_death': mic_death_c,              # MBC → {MAOM, POM, DOM, respired}
        'flux_necro_to_maom': necro_to_maom_c,
        'flux_necro_to_pom': necro_to_pom_c,
        'flux_necro_to_dom': necro_to_dom_c,
        'resp_necro': necro_respired_c,             # Necromass not recycled
    }


def mems_spinup(soc: float, qmax: float, mems: MEMSPoolParams) -> Dict:
    """Initialize MEMS pools from total SOC, equilibrating fast pools.

    Sets POM and MAOM from fractional allocation, then computes DOM and MBC
    at quasi-steady-state consistent with the dynamics in mems_annual_step:
      dom_supply = pom_decomp + maom_desorb (no existing DOM added, since
      we're solving for the steady-state DOM that balances supply and removal).
      MBC at QSS = mic_assimilated / k_mbc_turnover.
    """
    m = mems
    c_pom = soc * m.f_pom
    c_maom = min(soc * m.f_maom, qmax * 0.95)

    # DOM supply rate from slow pools (steady-state: no existing DOM stock yet)
    pom_c_rate = m.k_pom_to_dom * c_pom
    maom_c_rate = m.k_maom_desorption * c_maom
    dom_supply_rate = pom_c_rate + maom_c_rate

    # DOM removal partitioning
    maom_sat = c_maom / max(qmax, 0.1)
    sorption_rate = m.k_dom_sorption * max(0, 1.0 - maom_sat)
    total_rate = m.k_dom_uptake + sorption_rate
    f_uptake = m.k_dom_uptake / total_rate if total_rate > 0 else 0.5

    # DOM at steady state: inputs / (removal_fraction per year)
    # At QSS: dom_input_rate = dom * removal_frac, so dom = input / removal_frac
    dom_removal_frac = 1.0 - np.exp(-total_rate)
    c_dom = max(0.001, dom_supply_rate / max(dom_removal_frac, 0.01))

    # MBC at steady state (using CUE at full N availability for spinup)
    dom_removed = c_dom * dom_removal_frac
    dom_to_mic = dom_removed * f_uptake
    mic_assimilated = dom_to_mic * m.cue_max
    c_mbc = max(0.001, mic_assimilated / m.k_mbc_turnover)

    # Conserve total SOC: QSS DOM+MBC may differ from allocated fractions.
    # Distribute residual to MAOM first (stable pool), then POM.
    total = c_pom + c_dom + c_mbc + c_maom
    if total > 0 and abs(total - soc) > 0.01:
        residual = soc - total
        maom_headroom = qmax - c_maom
        to_maom = min(max(residual, 0), maom_headroom)
        c_maom += to_maom
        c_pom += (residual - to_maom)
        c_pom = max(0.01, c_pom)

    return {'c_pom': c_pom, 'c_dom': c_dom, 'c_mbc': c_mbc, 'c_maom': c_maom}


def mems_dynamic_spinup(
    region_key: str,
    n_spinup: int = 2000,
    tol: float = 0.002,
    p: MonthlyNParams = None,
    mems_p: MEMSPoolParams = None,
    synth_n: float = None,
    bnf_annual: float = None,
    yield_max_override: float = None,
    verbose: bool = False,
) -> Dict:
    """Iterate MEMS pools to true steady state at current fertilizer and yield.

    Unlike mems_spinup (which allocates pools from fractions and only
    equilibrates fast pools analytically), this function runs the full
    annual MEMS step + monthly N balance + yield + residue loop until
    all pools converge.  This eliminates the initialization transient
    where MAOM desorbs from an oversaturated starting condition.

    Convergence criterion: fractional SOC change over a 50-year window
    < tol.  This avoids premature convergence when slow pools (MAOM)
    drift at rates below the year-over-year detection threshold.

    Parameters
    ----------
    region_key : str
        Region identifier (e.g. 'north_america').
    n_spinup : int
        Maximum spinup years (default 500).
    tol : float
        Convergence tolerance on max fractional pool change (default 0.01 = 1%).
    p : MonthlyNParams
        Monthly N balance parameters.
    mems_p : MEMSPoolParams
        MEMS pool parameters.
    synth_n : float
        Fertilizer rate; defaults to region's current rate.
    bnf_annual : float
        BNF; defaults to region-specific value from get_regional_bnf.
    yield_max_override : float
        Calibrated Ymax; if None, uses region default.
    verbose : bool
        Print progress every 50 years.

    Returns
    -------
    dict with keys:
        c_pom, c_dom, c_mbc, c_maom : equilibrium pool sizes (t C/ha)
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
    if mems_p is None:
        mems_p = MEMSPoolParams()

    regions = get_default_regions()
    region = regions[region_key]
    crop = CropParams()
    climate = REGIONAL_CLIMATES[region_key]

    if synth_n is None:
        synth_n = region.synth_n_current
    if bnf_annual is None:
        bnf_annual = get_regional_bnf(region_key)

    clay_silt = getattr(region, 'clay_silt_fraction', 0.55)
    qmax = mems_p.qmax_per_claysilt * clay_silt

    # Initial pool allocation (analytical spinup as starting point)
    pools = mems_spinup(region.soc_initial, qmax, mems_p)
    c_pom = pools['c_pom']
    c_dom = pools['c_dom']
    c_mbc = pools['c_mbc']
    c_maom = pools['c_maom']
    pom_baseline = c_pom

    # Yield parameters
    mit_c = (region.mitscherlich_c_regional
             if region.mitscherlich_c_regional > 0 else crop.mitscherlich_c)
    ym = yield_max_override if yield_max_override else region.yield_max_regional
    n_grain_t = crop.grain_n_fraction * 1000
    hi = crop.harvest_index
    rf = (1 - hi) / hi
    rr = region.residue_retention

    # Initial residue estimate from FAOSTAT target
    target_y = FAOSTAT_TARGETS.get(region_key, 3.0)
    shoot_c = target_y * 1000 * 0.45 * rf * rr / 1000
    root_c = target_y * 1000 * 0.45 * rf * region.root_shoot_c_ratio / 1000
    lagged_c_input = (shoot_c + root_c) * region.cre_regional

    mineral_n = 12.0
    baseline_n_total = synth_n + bnf_annual + 50.0  # rough initial estimate

    converged = False
    years_to_converge = n_spinup
    conv_window = 50  # years over which to measure SOC drift
    soc_history = []

    for yr in range(n_spinup):
        # CUE from N availability
        if yr == 0:
            n_frac = 1.0
        else:
            n_frac = max(0.01, prev_n_total / max(baseline_n_total, 1.0))

        # MEMS annual step
        step = mems_annual_step(
            c_pom, c_dom, c_mbc, c_maom,
            c_input=lagged_c_input, qmax=qmax, mems=mems_p,
            n_available_frac=n_frac, pom_baseline=pom_baseline)
        annual_n_min = max(0, step['net_n_mineralized'])

        c_pom = step['c_pom']
        c_dom = step['c_dom']
        c_mbc = step['c_mbc']
        c_maom = step['c_maom']

        # Monthly N balance
        nb = monthly_n_balance_mems(
            annual_n_min, synth_n, bnf_annual,
            region.atm_n_deposition, climate, mineral_n, p)
        mineral_n = nb['mineral_n_end']

        # Yield
        n_eff = nb['uptake']
        y = ym * (1 - np.exp(-mit_c * n_eff))
        y_stoich = n_eff / n_grain_t if n_grain_t > 0 else y
        y = min(y, y_stoich)
        y = max(region.yield_min_regional if region.yield_min_regional > 0 else 0.0, y)

        # Residue C for next year
        shoot_c = y * 1000 * 0.45 * rf * rr / 1000
        root_c = y * 1000 * 0.45 * rf * region.root_shoot_c_ratio / 1000
        lagged_c_input = (shoot_c + root_c) * region.cre_regional

        prev_n_total = synth_n + bnf_annual + annual_n_min

        # Update baseline_n_total after first year to track actual
        if yr == 0:
            baseline_n_total = prev_n_total

        # Window-based convergence: SOC change over last conv_window years
        soc = c_pom + c_dom + c_mbc + c_maom
        soc_history.append(soc)

        if yr >= conv_window:
            soc_old = soc_history[yr - conv_window]
            frac_drift = abs(soc - soc_old) / max(soc_old, 0.1)
            if frac_drift < tol and yr >= 100:  # minimum 100 years
                converged = True
                years_to_converge = yr + 1
                if verbose:
                    print(f'  Spinup yr {yr:>3d}: SOC={soc:.2f} POM={c_pom:.2f} '
                          f'MAOM={c_maom:.2f} y={y:.2f} Nmin={nb["min"]:.1f} '
                          f'Δ50yr={frac_drift:.6f}')
                break
        else:
            frac_drift = 1.0  # placeholder

        if verbose and (yr < 5 or yr % 50 == 0):
            print(f'  Spinup yr {yr:>3d}: SOC={soc:.2f} POM={c_pom:.2f} '
                  f'MAOM={c_maom:.2f} y={y:.2f} Nmin={nb["min"]:.1f} '
                  f'Δ50yr={frac_drift:.6f}')

    soc_eq = c_pom + c_dom + c_mbc + c_maom

    if verbose:
        print(f'  {"CONVERGED" if converged else "NOT CONVERGED"} after '
              f'{years_to_converge} years.  SOC={soc_eq:.2f}')

    return {
        'c_pom': c_pom, 'c_dom': c_dom, 'c_mbc': c_mbc, 'c_maom': c_maom,
        'soc': soc_eq,
        'mineral_n': mineral_n,
        'yield_eq': y,
        'n_min_eq': nb['min'],
        'c_input_eq': lagged_c_input,
        'pom_baseline': pom_baseline,
        'converged': converged,
        'years_to_converge': years_to_converge,
    }


# ============================================================================
# Monthly N balance — MEMS version
# ============================================================================
# This uses the SAME monthly framework as monthly_model_v3, but substitutes
# MEMS annual N mineralization for Century-based mineralization.

def monthly_n_balance_mems(
    annual_n_min: float,
    synth_n: float, bnf_annual: float,
    atm_dep: float, climate: MonthlyClimate,
    mineral_n_start: float,
    p: MonthlyNParams,
) -> Dict:
    """
    Run 12 monthly N balance steps using MEMS-derived annual mineralization.

    The annual N mineralization from MEMS stoichiometric balance is distributed
    across months using the same Q10 + moisture corrections as Century v3.
    All loss/uptake processes are identical to monthly_model_v3.
    """
    ref_tf = temp_factor(p.t_ref, p)
    dem = demand_profile(climate)
    fp = fert_profile(climate)
    monthly_bnf = bnf_annual / 12
    monthly_atm = atm_dep / 12

    # Compute monthly abiotic weights to distribute annual mineralization
    weights = []
    for month in range(12):
        t = climate.temp[month]
        pr = climate.precip[month]
        pe = climate.pet[month]
        tf = temp_factor(t, p)
        mf = moist_factor(pr, pe, p)
        w = tf * mf / ref_tf if ref_tf > 0 else 1/12
        weights.append(w)
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum for w in weights]
    else:
        weights = [1/12] * 12

    mineral_n = mineral_n_start
    ann = {'min': 0, 'leach': 0, 'den': 0, 'uptake': 0, 'immob': 0}
    peak_demand = max(dem) if max(dem) > 0 else 1.0

    for month in range(12):
        t = climate.temp[month]
        pr = climate.precip[month]
        pe = climate.pet[month]

        # Monthly mineralization from MEMS annual total
        n_min = annual_n_min * weights[month]

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

        # Crop uptake — demand-scaled
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
# Full simulation: annual MEMS pools + monthly N
# ============================================================================

def run_model(
    region_key: str,
    synth_n: float = None,
    n_years: int = 200,
    yield_max_override: float = None,
    bnf_annual: float = None,
    residue_ret_override: float = None,
    p: MonthlyNParams = None,
    mems: MEMSPoolParams = None,
    verbose: bool = False,
) -> Dict:
    regions = get_default_regions()
    region = regions[region_key]
    crop = CropParams()
    if p is None:
        p = MonthlyNParams()
    if mems is None:
        mems = MEMSPoolParams()
    climate = REGIONAL_CLIMATES[region_key]

    if synth_n is None:
        synth_n = region.synth_n_current
    if bnf_annual is None:
        bnf_annual = 5.0
    rr = residue_ret_override if residue_ret_override is not None else region.residue_retention

    soc = region.soc_initial
    clay_silt = getattr(region, 'clay_silt_fraction', 0.55)
    qmax = mems.qmax_per_claysilt * clay_silt

    # Initialize MEMS pools
    pools = mems_spinup(soc, qmax, mems)
    c_pom = pools['c_pom']
    c_dom = pools['c_dom']
    c_mbc = pools['c_mbc']
    c_maom = pools['c_maom']
    pom_baseline = c_pom  # For priming reference

    c_mits = crop.mitscherlich_c
    ym = yield_max_override if yield_max_override else region.yield_max_regional
    n_grain_t = crop.grain_n_fraction * 1000

    res = {k: [] for k in ['year','soc','yield_tha','n_min','n_leach','n_den',
                            'n_uptake','n_immob','cue','maom_sat','c_pom','c_maom']}
    mineral_n = 12.0

    # Estimate baseline residue C input from FAOSTAT target yield
    hi = crop.harvest_index
    rf = (1 - hi) / hi
    target_y = FAOSTAT_TARGETS.get(region_key, 3.0)
    baseline_shoot_c = target_y * 1000 * 0.45 * rf * rr / 1000
    baseline_root_c = target_y * 1000 * 0.45 * rf * region.root_shoot_c_ratio / 1000
    baseline_c_input = (baseline_shoot_c + baseline_root_c) * region.cre_regional

    # Baseline N for CUE calculation
    baseline_pools = mems_spinup(region.soc_initial, qmax, mems)
    baseline_step = mems_annual_step(
        baseline_pools['c_pom'], baseline_pools['c_dom'],
        baseline_pools['c_mbc'], baseline_pools['c_maom'],
        c_input=baseline_c_input, qmax=qmax, mems=mems, n_available_frac=1.0,
        pom_baseline=baseline_pools['c_pom'])
    baseline_n_total = region.synth_n_current + 5.0 + max(0, baseline_step['net_n_mineralized'])

    # Lagged residue C: year 0 uses baseline estimate, subsequent years use
    # previous year's actual residue (physically correct: crop residue from
    # harvest decomposes the following year).
    lagged_c_input = baseline_c_input

    for year in range(n_years):
        # --- N availability fraction for CUE ---
        n_frac = 1.0 if year == 0 else max(0.01, prev_n_total / max(baseline_n_total, 1.0))

        # --- Single MEMS step with lagged residue input ---
        step = mems_annual_step(
            c_pom, c_dom, c_mbc, c_maom,
            c_input=lagged_c_input,
            qmax=qmax, mems=mems,
            n_available_frac=n_frac,
            pom_baseline=pom_baseline,
        )
        annual_n_min = max(0, step['net_n_mineralized'])

        # Update pools
        c_pom = step['c_pom']
        c_dom = step['c_dom']
        c_mbc = step['c_mbc']
        c_maom = step['c_maom']
        soc_now = c_pom + c_dom + c_mbc + c_maom

        # --- Monthly N balance (using MEMS mineralization) ---
        nb = monthly_n_balance_mems(
            annual_n_min, synth_n, bnf_annual,
            region.atm_n_deposition, climate, mineral_n, p
        )
        mineral_n = nb['mineral_n_end']

        # --- Yield from crop N uptake ---
        n_eff = nb['uptake']
        y = ym * (1 - np.exp(-c_mits * n_eff))
        y_stoich = n_eff / n_grain_t if n_grain_t > 0 else y
        y = min(y, y_stoich)
        y = max(y, region.yield_min_regional)

        # --- Residue C input (becomes next year's lagged input) ---
        shoot_c = y * 1000 * 0.45 * rf * rr / 1000
        root_c = y * 1000 * 0.45 * rf * region.root_shoot_c_ratio / 1000
        lagged_c_input = (shoot_c + root_c) * region.cre_regional

        prev_n_total = synth_n + bnf_annual + annual_n_min

        res['year'].append(year)
        res['soc'].append(soc_now)
        res['yield_tha'].append(y)
        res['n_min'].append(nb['min'])
        res['n_leach'].append(nb['leach'])
        res['n_den'].append(nb['den'])
        res['n_uptake'].append(nb['uptake'])
        res['n_immob'].append(nb['immob'])
        res['cue'].append(step['cue'])
        res['maom_sat'].append(step['maom_sat'])
        res['c_pom'].append(c_pom)
        res['c_maom'].append(c_maom)

        if verbose and (year < 5 or year % 50 == 0 or year == n_years-1):
            print(f"  Yr {year:>3d}: SOC={soc_now:.1f} POM={c_pom:.1f} MAOM={c_maom:.1f} "
                  f"y={y:.2f} CUE={step['cue']:.3f} Nmin={nb['min']:.1f} "
                  f"Nup={nb['uptake']:.1f} Nlch={nb['leach']:.1f}")

    return res


# ============================================================================
# Calibration
# ============================================================================

def calibrate_ym(region_key: str, target: float, p: MonthlyNParams = None,
                 mems: MEMSPoolParams = None) -> float:
    def obj(ym):
        r = run_model(region_key, n_years=5, yield_max_override=ym, p=p, mems=mems)
        return r['yield_tha'][2] - target
    try:
        return brentq(obj, 1.0, 50.0, xtol=0.01)
    except ValueError:
        best, best_e = 10.0, 999
        for ym in np.arange(2.0, 40.0, 0.5):
            r = run_model(region_key, n_years=5, yield_max_override=ym, p=p, mems=mems)
            e = abs(r['yield_tha'][2] - target)
            if e < best_e:
                best, best_e = ym, e
        return best


# ============================================================================
# Dependency computation
# ============================================================================

def compute_dep(region_key, ym, n_years=300, managed=False, p=None, mems=None):
    regions = get_default_regions()
    region = regions[region_key]

    rc = run_model(region_key, n_years=5, yield_max_override=ym, p=p, mems=mems)
    yc = rc['yield_tha'][2]

    kw = dict(synth_n=0.0, n_years=n_years, yield_max_override=ym, p=p, mems=mems)
    if managed:
        mt = MANAGED_TRANSITION_PARAMS[region_key]
        landscape_bnf = (mt['legume_frac'] * mt['net_n_credit']
                         / (1 - mt['legume_frac']))
        kw['bnf_annual'] = landscape_bnf + mt['free_living_bnf']
        kw['residue_ret_override'] = min(region.residue_retention + 0.15, 0.95)

    rnf = run_model(region_key, **kw)
    yss_per_ha = np.mean(rnf['yield_tha'][-20:])
    soc_ss = np.mean(rnf['soc'][-20:])

    if managed:
        mt = MANAGED_TRANSITION_PARAMS[region_key]
        lf = mt['legume_frac']
        yss_effective = yss_per_ha * (1 - lf) + mt['legume_yield_ceq'] * lf
    else:
        yss_effective = yss_per_ha

    return {
        'yc': yc, 'yss': yss_effective,
        'yss_cereal_only': yss_per_ha,
        'soc_i': region.soc_initial, 'soc_ss': soc_ss,
        'soc_loss': (region.soc_initial - soc_ss)/region.soc_initial*100,
        'dep': (yc - yss_effective)/yc*100 if yc > 0 else 0,
        'n_up_c': rc['n_uptake'][2],
        'n_up_ss': np.mean(rnf['n_uptake'][-20:]),
        'n_lch_ss': np.mean(rnf['n_leach'][-20:]),
        'n_den_ss': np.mean(rnf['n_den'][-20:]),
        'n_min_ss': np.mean(rnf['n_min'][-20:]),
        'cue_ss': np.mean(rnf['cue'][-20:]),
        'maom_sat_ss': np.mean(rnf['maom_sat'][-20:]),
        'ts': rnf,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    regions = get_default_regions()
    p = MonthlyNParams()
    mems = MEMSPoolParams()

    print("=" * 115)
    print("MONTHLY MEMS v1 — Hybrid: Annual MEMS Pools + Monthly N Availability")
    print("=" * 115)

    # 1. Calibrate
    print("\n--- Calibrating yield_max ---\n")
    cal = {}
    for rk in REGIONAL_CLIMATES:
        t = FAOSTAT_TARGETS[rk]
        ym = calibrate_ym(rk, t, p, mems)
        cal[rk] = ym
        r = run_model(rk, n_years=5, yield_max_override=ym, p=p, mems=mems)
        print(f"  {regions[rk].name:>35}: target={t:.2f}, ym={ym:.3f}, "
              f"actual={r['yield_tha'][2]:.2f}, Nup={r['n_uptake'][2]:.1f}, "
              f"CUE={r['cue'][2]:.3f}")

    # 2. Abrupt
    print("\n\n--- Abrupt Withdrawal ---\n")
    hdr = (f"{'Region':>35} {'Y_now':>6} {'Y_SS':>6} {'SOC_i':>6} {'SOC_SS':>6} "
           f"{'SOC%':>6} {'Dep%':>7} {'CUE':>6} {'MAOM%':>6}")
    print(hdr)
    print("-" * 110)

    tp_c = tp_s = 0
    abrupt = {}
    for rk in REGIONAL_CLIMATES:
        reg = regions[rk]
        d = compute_dep(rk, cal[rk], n_years=300, managed=False, p=p, mems=mems)
        abrupt[rk] = d
        pc = d['yc'] * reg.cropland_mha
        ps = d['yss'] * reg.cropland_mha
        tp_c += pc; tp_s += ps
        print(f"  {reg.name:>33} {d['yc']:>5.2f} {d['yss']:>5.2f} "
              f"{d['soc_i']:>5.1f} {d['soc_ss']:>5.1f} {d['soc_loss']:>5.1f}% "
              f"{d['dep']:>6.1f}% {d['cue_ss']:>5.3f} {d['maom_sat_ss']:>5.2f}")

    gdep_a = (1 - tp_s/tp_c)*100
    pop = sum(r.pop_supported for r in regions.values())
    print(f"\n  GLOBAL DEPENDENCY (MEMS, abrupt): {gdep_a:.1f}%")
    print(f"  Production loss: {tp_c - tp_s:.0f} Mt of {tp_c:.0f} Mt")
    print(f"  Population at risk: {gdep_a/100*pop:.0f} M of {pop:.0f} M")

    # 3. Managed
    print("\n\n--- Managed Transition (empirically-constrained BNF) ---\n")
    print(f"{'Region':>35} {'Y_now':>6} {'Y_cer':>6} {'Y_eff':>6} {'LegFr':>6} "
          f"{'BNF':>6} {'SOC%':>6} {'Dep%':>7} {'CUE':>6}")
    print("-" * 115)

    tp_c2 = tp_s2 = 0
    managed = {}
    for rk in REGIONAL_CLIMATES:
        reg = regions[rk]
        d = compute_dep(rk, cal[rk], n_years=300, managed=True, p=p, mems=mems)
        managed[rk] = d
        mt = MANAGED_TRANSITION_PARAMS[rk]
        landscape_bnf = (mt['legume_frac'] * mt['net_n_credit']
                         / (1 - mt['legume_frac'])) + mt['free_living_bnf']
        pc = d['yc'] * reg.cropland_mha
        ps = d['yss'] * reg.cropland_mha
        tp_c2 += pc; tp_s2 += ps
        print(f"  {reg.name:>33} {d['yc']:>5.2f} {d['yss_cereal_only']:>5.2f} "
              f"{d['yss']:>5.2f} {mt['legume_frac']:>5.2f} {landscape_bnf:>5.1f} "
              f"{d['soc_loss']:>5.1f}% {d['dep']:>6.1f}% {d['cue_ss']:>5.3f}")

    gdep_m = (1 - tp_s2/tp_c2)*100
    print(f"\n  GLOBAL DEPENDENCY (MEMS, managed): {gdep_m:.1f}%")
    print(f"  Population at risk: {gdep_m/100*pop:.0f} M")

    # 4. Empirical validation
    print("\n\n--- Empirical Validation ---")

    res_bb = run_model('europe', synth_n=0.0, n_years=180,
                       yield_max_override=cal['europe'], p=p, mems=mems)
    ybb = np.mean(res_bb['yield_tha'][-20:])
    socbb = np.mean(res_bb['soc'][-20:])
    print(f"\n  Broadbalk (180yr): yield={ybb:.2f} (target ~1.0), SOC={socbb:.1f} (target ~22)")

    res_mw = run_model('north_america', synth_n=0.0, n_years=150,
                       yield_max_override=cal['north_america'], p=p, mems=mems)
    ymw = np.mean(res_mw['yield_tha'][-20:])
    socmw = np.mean(res_mw['soc'][-20:])
    print(f"  Morrow (150yr):    yield={ymw:.2f} (target ~1.9), SOC={socmw:.1f} (target ~35)")

    # 5. Comparison summary
    print("\n\n" + "=" * 115)
    print("COMPARISON SUMMARY")
    print("=" * 115)
    print(f"\n  {'Model':>45} {'Abrupt':>10} {'Managed':>10} {'BB yield':>10} {'BB SOC':>10}")
    print(f"  {'-'*85}")
    print(f"  {'Century v3 (monthly_model_v3)':>45} {'63.7%':>10} {'44.2%':>10} {'1.06':>10} {'26.2':>10}")
    print(f"  {'MEMS v1 (monthly_mems_v1)':>45} {f'{gdep_a:.1f}%':>10} {f'{gdep_m:.1f}%':>10} {f'{ybb:.2f}':>10} {f'{socbb:.1f}':>10}")
    print(f"  {'Erisman et al. (2008)':>45} {'48%':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
    print(f"  {'Smil (2001)':>45} {'~40%':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10}")

    # 6. Save data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    for scenario, results_dict, label in [
        ('no_adaptation', abrupt, 'abrupt'),
        ('managed', managed, 'managed')
    ]:
        fname = os.path.join(data_dir, f'monthly_mems_v1_dependency_{scenario}.csv')
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['region','region_key','yield_current','yield_ss','soc_initial',
                        'soc_ss','soc_loss_pct','dependency_pct','n_uptake_current',
                        'n_uptake_ss','n_leach_ss','n_denitrif_ss','n_min_ss',
                        'cue_ss','maom_sat_ss',
                        'cropland_mha','production_current_mt','production_ss_mt'])
            for rk in REGIONAL_CLIMATES:
                reg = regions[rk]
                d = results_dict[rk]
                w.writerow([reg.name, rk, f"{d['yc']:.4f}", f"{d['yss']:.4f}",
                           f"{d['soc_i']:.1f}", f"{d['soc_ss']:.1f}",
                           f"{d['soc_loss']:.2f}", f"{d['dep']:.2f}",
                           f"{d['n_up_c']:.1f}", f"{d['n_up_ss']:.1f}",
                           f"{d['n_lch_ss']:.1f}", f"{d['n_den_ss']:.1f}",
                           f"{d['n_min_ss']:.1f}", f"{d['cue_ss']:.3f}",
                           f"{d['maom_sat_ss']:.3f}", f"{reg.cropland_mha}",
                           f"{d['yc']*reg.cropland_mha:.1f}",
                           f"{d['yss']*reg.cropland_mha:.1f}"])
        print(f"\n  Data saved: {fname}")
