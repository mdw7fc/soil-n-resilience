"""Microbially-explicit 4-pool SOM scheme: pools, kinetics, N balance.

POM / DOM / MBC / MAOM with CUE that downregulates under N limitation
(Wieder et al. 2015; Manzoni et al. 2012), MAOM sorption saturation
(Georgiou et al. 2022), necromass partitioning (Liang et al. 2017) and a
priming term on MAOM desorption. SOM formation follows two routes: DOM is
either assimilated into microbial biomass (with necromass partitioned to
MAOM, POM and DOM on turnover) or sorbs directly from DOM to MAOM; at the
model's equilibrium roughly half of MAOM formation follows each route.
N mineralization follows from the stoichiometric balance of DOM
consumption against microbial demand rather than from pool turnover x C:N.

Adopted into the deposit under F-029 from the project's April four-pool
line (recovered from tropical-reparam-2026-04-14/backups and
matched-mems-2026-04-15/backups, byte-identical copies); the pool dynamics
are that code unchanged except that the annual step now also reports the
respiration split (uptake-step vs non-recycled necromass) used by
Supplementary Figure S5. The engine that couples this scheme to the
economic layer is `coupled_4pool.py`.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict

from monthly_model_v3 import (
    MonthlyClimate, MonthlyNParams,
    temp_factor, moist_factor, demand_profile, fert_profile,
)


@dataclass
class FourPoolParams:
    """4-pool structure and kinetic parameters."""

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
# Annual microbially-explicit 4-pool scheme pool dynamics
# ============================================================================

def fourpool_annual_step(
    c_pom: float, c_dom: float, c_mbc: float, c_maom: float,
    c_input: float, qmax: float, mems: FourPoolParams,
    n_available_frac: float = 1.0,
    pom_baseline: float = None,
) -> Dict:
    """
    Advance microbially-explicit 4-pool scheme pools by one annual timestep.

    Parameters
    ----------
    c_pom, c_dom, c_mbc, c_maom : current pool sizes (t C/ha)
    c_input : residue C input (t C/ha/yr) -> all goes to POM
    qmax : mineral sorption capacity (t C/ha)
    mems : FourPoolParams
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
        'resp_cue': mic_respired_uptake,    # (1-CUE) loss at DOM uptake
        'resp_necro': necro_respired_c,     # non-recycled necromass
        'c_balance': c_balance,
    }


def fourpool_analytic_init(soc: float, qmax: float, mems: FourPoolParams) -> Dict:
    """Initialize microbially-explicit 4-pool scheme pools from total SOC, equilibrating fast pools.

    Sets POM and MAOM from fractional allocation, then computes DOM and MBC
    at quasi-steady-state consistent with the dynamics in fourpool_annual_step:
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



# ============================================================================
# Monthly N balance — microbially-explicit 4-pool scheme version
# ============================================================================
# This uses the SAME monthly framework as monthly_model_v3, but substitutes
# microbially-explicit 4-pool scheme annual N mineralization for Century-based mineralization.

def monthly_n_balance_4pool(
    annual_n_min: float,
    synth_n: float, bnf_annual: float,
    atm_dep: float, climate: MonthlyClimate,
    mineral_n_start: float,
    p: MonthlyNParams,
) -> Dict:
    """
    Run 12 monthly N balance steps using microbially-explicit 4-pool scheme-derived annual mineralization.

    The annual N mineralization from microbially-explicit 4-pool scheme stoichiometric balance is distributed
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

        # Monthly mineralization from microbially-explicit 4-pool scheme annual total
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
