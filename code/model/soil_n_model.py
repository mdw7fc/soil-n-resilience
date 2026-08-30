"""
Dynamic Soil Nitrogen Carrying Capacity Model
==============================================

A system dynamics model tracking soil organic matter pool depletion,
nitrogen mineralization, crop yield response, and carrying capacity
following synthetic nitrogen withdrawal.

Framework: Three-pool SOM model (active, slow, passive) informed by
Century/RothC logic, with coupled feedback loops for residue return,
soil physical degradation, BNF substitution, and marginal land expansion.

Author: Matthew Wallenstein
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import registry as _reg
from parameter_registry import (
    BASELINE_BNF_KG_N_HA_YR,
    RESIDUE_C_FRACTION,
    SOC_T_C_HA_PER_PERCENT_30CM,
    WATER_STRESS_GAIN_SAT_SOC_PCT,
    WATER_STRESS_MIN_FACTOR,
    WATER_STRESS_SOFTPLUS_EPS_MM,
    WHC_MM_PER_SOC_PCT_30CM,
)

# v15 (F-011). Everything below that used to be a literal is now read from
# code/model/params.yaml through registry.py at import. The registry drives the
# model; it no longer documents it. Perturbing an entry in params.yaml changes
# the number this module hands the engine.
_SOM_F = _reg.value('som_pool_fractions')      # guarded at load: sums to 1
_SOM_K = _reg.value('som_decay_rates')
_SOM_CN = _reg.value('som_pool_cn')
_SOM_H = _reg.value('som_humification')
_LAUB = _reg.value('laub_tropical_ratios')
_CRE_ALLOC = _reg.value('cre_allocation')      # guarded at load: sums to 1

# The eight regions' seventeen quantitative fields. Order is the order the
# fields appear on RegionParams, and it is the order the equality log and the
# mutation harness both walk, so it is stated once, here.
REGISTRY_REGION_FIELDS = (
    'soc_initial',
    'cn_bulk',
    'cropland_mha',
    'synth_n_current',
    'pop_supported',
    'texture_class',
    'whc_sensitivity',
    'water_stress_coeff',
    'baseline_water_deficit',
    'atm_n_deposition',
    'bnf_potential',
    'bnf_ramp_years',
    'residue_retention',
    'yield_max_regional',
    'yield_min_regional',
    'root_shoot_c_ratio',
    'cre_regional',
)

REGION_DISPLAY_NAMES = {
    'north_america': 'North America',
    'europe': 'Europe',
    'east_asia': 'East Asia',
    'south_asia': 'South Asia',
    'southeast_asia': 'Southeast Asia',
    'latin_america': 'Latin America',
    'sub_saharan_africa': 'Sub-Saharan Africa',
    'fsu_central_asia': 'Former Soviet Union & Central Asia',
}


# ============================================================
# MODEL PARAMETERS
# ============================================================

@dataclass
class SOMPoolParams:
    """Three-pool SOM structure (Century/RothC-informed).

    Defaults are Century/RothC temperate-biome parameters. For tropical
    regions, use ``SOMPoolParams.tropical()`` which applies Laub et al.
    (2024, *Biogeosciences* 21:3691–3716) Kenya-calibrated DayCent posterior
    ratios to k_slow and k_passive. See
    paper2-soil-resilience/tropical-reparam-2026-04-14/PARAMETERS.md
    for full documentation of the mapping.
    """
    # Pool fractions of total SOC. The sum-to-one constraint is enforced by
    # registry.py at load, not here: a partition that does not sum to one
    # cannot reach this dataclass.
    f_active: float = _SOM_F['f_active']
    f_slow: float = _SOM_F['f_slow']
    f_passive: float = _SOM_F['f_passive']

    # Decay constants (yr^-1) — reciprocal of turnover time
    k_active: float = _SOM_K['k_active']      # ~3 yr turnover
    k_slow: float = _SOM_K['k_slow']          # ~27 yr turnover
    k_passive: float = _SOM_K['k_passive']    # ~1,373 yr turnover (F-001)

    # C:N ratios by pool
    cn_active: float = _SOM_CN['cn_active']
    cn_slow: float = _SOM_CN['cn_slow']
    cn_passive: float = _SOM_CN['cn_passive']

    # Fraction of decomposed C transferred to next pool (humification)
    h_active_to_slow: float = _SOM_H['h_active_to_slow']
    h_slow_to_passive: float = _SOM_H['h_slow_to_passive']

    # Tag identifying the parameterization regime (for traceability in
    # saved outputs and figures)
    regime: str = "temperate_century"

    @classmethod
    def tropical(cls) -> "SOMPoolParams":
        """Tropical-biome calibration applying Laub et al. 2024 ratios.

        Laub et al. 2024 Biogeosciences 21:3691–3716, Table 1.
        Bayesian posterior (best single parameter set, all 4 Kenya sites):
        - dec5(2): DayCent default 0.10 → Kenya posterior 0.13 (ratio 1.30)
        - dec4:   DayCent default 0.0035 → Kenya posterior 0.0060 (ratio 1.71)

        We apply the Kenya/DayCent-default ratio to our framework's
        calibrated baselines rather than substituting Laub's absolute
        values, because our baselines are calibrated to our model's
        specific SOC equilibrium framework. Applying the ratios preserves
        the framework while incorporating the empirically grounded
        tropical adjustment.

        Scope: Sub-Saharan Africa, South Asia, Southeast Asia, Latin America.
        Not applied to: North America, Europe, East Asia, FSU/Central Asia.
        """
        LAUB_K_SLOW_RATIO = _LAUB['k_slow_ratio']        # 0.13 / 0.10
        LAUB_K_PASSIVE_RATIO = _LAUB['k_passive_ratio']  # 0.0060 / 0.0035

        base = cls()  # pull temperate defaults
        return cls(
            f_active=base.f_active,
            f_slow=base.f_slow,
            f_passive=base.f_passive,
            k_active=base.k_active,
            k_slow=base.k_slow * LAUB_K_SLOW_RATIO,
            k_passive=base.k_passive * LAUB_K_PASSIVE_RATIO,
            cn_active=base.cn_active,
            cn_slow=base.cn_slow,
            cn_passive=base.cn_passive,
            h_active_to_slow=base.h_active_to_slow,
            h_slow_to_passive=base.h_slow_to_passive,
            regime="tropical_laub2024",
        )


# Region → regime mapping (used wherever SOMPoolParams is instantiated
# per-region). Temperate default; tropical regions listed explicitly.
TROPICAL_REGIONS = frozenset({
    "sub_saharan_africa",
    "south_asia",
    "southeast_asia",
    "latin_america",
})


def som_params_for_region(region_key: str) -> SOMPoolParams:
    """Return regime-appropriate SOMPoolParams for a given region.

    See tropical-reparam-2026-04-14/PARAMETERS.md for regional assignment
    rationale and limitations.
    """
    if region_key in TROPICAL_REGIONS:
        return SOMPoolParams.tropical()
    return SOMPoolParams()


@dataclass
class CropParams:
    """Crop yield and nitrogen response parameters."""
    # Maximum yield under optimal N (t/ha grain, global average mix)
    # Now a default; overridden by region-specific yield_max if present
    yield_max: float = 5.0

    # Yield-N response: Mitscherlich function y = y_max * (1 - exp(-c * N_avail))
    # c calibrated so that at N_avail = 150 kg/ha, yield ~ 0.90 * y_max
    # Now a default; overridden by region-specific mitscherlich_c if present
    mitscherlich_c: float = 0.015

    # Minimum yield floor REMOVED in v2 revision. The yield floor is now
    # an emergent property of the regionally calibrated Mitscherlich function
    # evaluated at steady-state N availability, not an imposed constant.
    # Retained only as a physiological absolute minimum (plant cannot produce
    # negative grain). Set to 0.0 by default.
    yield_min: float = 0.0      # t/ha — effectively disabled

    # Residue-to-grain ratio (IPCC)
    residue_grain_ratio: float = 1.0

    # Carbon content of residue (fraction)
    residue_c_fraction: float = RESIDUE_C_FRACTION

    # C:N ratio of crop residues
    # Wheat straw: 80-100, rice straw: 60-80, corn stover: 50-70.
    # Production-weighted global cereal blend: ~75.
    residue_cn: float = 75.0

    # N content of grain (fraction)
    grain_n_fraction: float = 0.018  # ~1.8% N in grain

    # Harvest index: fraction of above-ground biomass that is grain
    harvest_index: float = 0.45

    # Apparent N recovery: fraction of gross mineral N pool taken up by the
    # crop over the growing season. This is NOT fertilizer N recovery (NRE,
    # ~0.40-0.55) but total apparent N uptake from the gross mineral N supply
    # including mineralized N, applied N, BNF, and atmospheric deposition.
    # Literature: 0.60-0.85 depending on management (Cassman et al. 2002;
    # Ladha et al. 2005). Higher than NRE because mineralized N is released
    # in the root zone with high spatial and temporal coincidence with demand.
    # Calibrated so that at current N supply, plant N uptake ≈ total crop N
    # at FAOSTAT yields (stoichiometric consistency).
    nue_apparent: float = 0.75


@dataclass
class RegionParams:
    """Region-specific soil and agricultural parameters."""
    name: str

    # Initial SOC stock (t C/ha, 0-30 cm)
    soc_initial: float

    # Initial C:N ratio of bulk SOM
    cn_bulk: float = 10.0

    # Cropland area (million ha)
    cropland_mha: float = 100.0

    # Current synthetic N application (kg N/ha/yr)
    synth_n_current: float = 120.0

    # Current population supported (millions) — proportional to crop production
    pop_supported: float = 500.0

    # Soil texture class (affects water-holding capacity feedback)
    # 0 = sand, 1 = loam, 2 = clay
    texture_class: int = 1

    # Water holding capacity sensitivity to SOC (mm per SOC percentage point
    # in the modeled 0-30 cm layer)
    whc_sensitivity: float = _reg.value('whc_sensitivity')

    # Yield penalty per mm of water deficit (fraction per mm)
    water_stress_coeff: float = 0.004

    # Baseline water deficit without SOC effect (mm)
    baseline_water_deficit: float = 0.0

    # Atmospheric N deposition (kg N/ha/yr). Wet + dry deposition of reactive N
    # from industrial emissions, agricultural volatilization, and lightning.
    # Dentener et al. 2006; Vet et al. 2014. Ranges: 2-5 (remote), 5-15
    # (agricultural), 15-30 (heavily industrialized). Included as an N source
    # that was previously missing from the model.
    atm_n_deposition: float = 8.0

    # Baseline landscape BNF (kg N/ha/yr NET available to crops, expressed per
    # cereal hectare). Values are derived from the single BNF component ledger
    # in parameter_registry rather than specified independently here.
    # This is the net contribution after accounting for N removal in legume grain.
    # Grain legumes (soybean): fix 150-200 but export 130-170 in seed; net ~0-40 kg/ha
    # Cover crop legumes (vetch, clover): 50-80 kg net N/ha but no food production
    # Free-living + associative BNF: 5-15 kg/ha/yr
    # Landscape average with 25-30% legume rotation: 15-30 kg/ha/yr total
    bnf_potential: float = 25.0

    # Years to reach full BNF potential
    bnf_ramp_years: float = 10.0

    # Fraction of residue retained (vs. harvested for fuel/feed)
    residue_retention: float = 0.85

    # Region-specific crop response parameters (override CropParams defaults)
    # yield_max: Maximum attainable yield under optimal N for this region (t/ha)
    # mitscherlich_c: Yield-response curvature. Higher c = steeper initial response,
    #   meaning the crop extracts more yield per unit N at low N availability.
    #   Calibrated so that at equilibrium N availability (no synthetic N, depleted SOM),
    #   the Mitscherlich function produces historically plausible unfertilized yields.
    yield_max_regional: float = 0.0    # 0 = use CropParams default
    mitscherlich_c_regional: float = 0.0  # 0 = use CropParams default

    # Empirical yield floor (t/ha) — minimum sustainable yield on depleted soils,
    # calibrated from long-term unfertilized experiments (Rothamsted, Morrow, etc.)
    # and pre-industrial yield records. Represents yields on soils that have
    # already lost ~50-60% of SOC through decades of cultivation without N inputs.
    yield_min_regional: float = 0.0     # 0 = use CropParams.yield_min (0.0)

    # Root:shoot C ratio for below-ground C inputs to SOM.
    # Root C is NOT subject to residue_retention (roots stay in the soil).
    # Literature: Bolinder et al. 1999, 2007: 0.5-1.5 for cereals
    # Katterer et al. 2011: ~0.8-1.0 for Nordic cereals
    # Johnson et al. 2006: 0.3-0.6 (more conservative)
    root_shoot_c_ratio: float = 0.80

    # Region-specific carbon retention efficiency (fraction of total C input
    # entering SOM pools). Calibrated so that initial SOC is approximately at
    # equilibrium under current management. Includes both above-ground residue
    # and root C inputs.
    # Literature range (total-input basis): 0.10-0.30
    # Variation driven by clay content (MAOM stabilization), temperature,
    # and tillage system.
    # Required. There is no global fallback; see region_cre().
    cre_regional: float = 0.0


def region_cre(region):
    """The carbon retention efficiency a region runs at. No fallback.

    Until v15 every call site read
    `region.cre_regional if region.cre_regional > 0 else fb.cre_base`, and
    `cre_base` was a registered, provenanced, Monte-Carlo-excluded parameter
    sitting behind that guard. F-011's mutation sweep scored it INERT: all
    eight regions set `cre_regional`, so the branch was reached by nothing and
    perturbing the value moved no published number. A registered parameter the
    model can never read is a documented assumption that is not in the model,
    which is one of the shapes this rebuild exists to remove.

    So the fallback is deleted and an unset value is an error. The old
    behaviour substituted a pooled cross-site mean for a regional one and left
    no trace of having done so: nothing in any output distinguishes a region
    running at its own measured efficiency from a region running at 0.11
    because its entry was blank.
    """
    cre = float(region.cre_regional)
    if cre <= 0:
        raise ValueError(
            'region %r has cre_regional=%r; it is required and there is no '
            'global fallback. Register the regional value in params.yaml '
            'under cre_regional.' % (getattr(region, 'name', region), cre))
    return cre


@dataclass
class ScenarioParams:
    """Withdrawal scenario parameters."""
    name: str

    # Withdrawal schedule: years over which synthetic N goes to zero
    withdrawal_years: float = 0.0   # 0 = abrupt

    # Whether BNF substitution is actively managed
    bnf_managed: bool = False

    # Whether residue retention is optimized
    residue_optimized: bool = False

    # Whether crop mix shifts to include more legumes
    legume_expansion: bool = False

    # Fraction of cropland shifted to legumes (if managed)
    legume_fraction_target: float = 0.0

    # Years to reach legume target
    legume_ramp_years: float = 10.0


@dataclass
class FeedbackParams:
    """Feedback loop strength parameters."""
    # Residue feedback: enabled
    residue_feedback: bool = True

    # Physical degradation feedback: enabled
    physical_feedback: bool = True

    # Physical feedback strength multiplier (0-1)
    physical_strength: float = _reg.value('physical_feedback_strength')

    # Marginal land expansion feedback: enabled
    expansion_feedback: bool = False  # Off by default (regional, not always relevant)

    # C-N coupling feedback: enabled
    cn_coupling_feedback: bool = True

    # Fraction of CRE going to active vs. slow pool. Guarded at load: sums to 1.
    cre_to_active: float = _CRE_ALLOC['cre_to_active']
    cre_to_slow: float = _CRE_ALLOC['cre_to_slow']


# ============================================================
# DEFAULT REGIONS
# ============================================================

def get_default_regions() -> Dict[str, RegionParams]:
    """Return eight regions covering all global cropland.

    Calibrated against:
    - FAO FAOSTAT 2023: global arable land ~1,400 Mha, synthetic N ~110 Tg/yr
    - ISRIC SoilGrids for SOC stocks (cropland-specific where available)
    - Regional fertilizer intensity data (IFA, FAO)
    - Population supported proportional to regional share of global crop calories

    Total cropland: ~1,230 Mha (sum across regions)
    Total synthetic N: ~98.5 Tg/yr (sum: cropland_mha * synth_n_current / 1000)
    Total population supported: ~7,650 M (global food system, excluding fisheries/pasture)
    """
    return {
        region_key: RegionParams(
            name=REGION_DISPLAY_NAMES[region_key],
            **_reg.region_fields(region_key, REGISTRY_REGION_FIELDS)
        )
        for region_key in _reg.REGIONS
    }


def get_default_scenarios() -> Dict[str, ScenarioParams]:
    """Return three default scenarios."""
    return {
        'abrupt': ScenarioParams(
            name='Abrupt Withdrawal',
            withdrawal_years=0.0,
            bnf_managed=False,
            residue_optimized=False,
            legume_expansion=False,
        ),
        'gradual': ScenarioParams(
            name='Gradual Phase-Out (20 yr)',
            withdrawal_years=20.0,
            bnf_managed=False,
            residue_optimized=False,
            legume_expansion=False,
        ),
        'managed': ScenarioParams(
            name='Managed Agronomic Transition',
            withdrawal_years=20.0,
            bnf_managed=True,
            residue_optimized=True,
            legume_expansion=True,
            legume_fraction_target=0.25,
            legume_ramp_years=15.0,
        ),
    }


# ============================================================
# CORE MODEL
# ============================================================

class SoilNCarryingCapacityModel:
    """
    System dynamics model of agricultural carrying capacity under
    synthetic nitrogen withdrawal.

    State variables (per hectare):
        C_active: Carbon in active SOM pool (t C/ha)
        C_slow: Carbon in slow SOM pool (t C/ha)
        C_passive: Carbon in passive SOM pool (t C/ha)

    Derived quantities:
        N_mineralized: Annual N mineralization from all pools (kg N/ha/yr)
        N_available: Total plant-available N (mineralization + BNF + residual synthetic)
        yield_actual: Crop yield given N_available and water stress
        carrying_capacity: Population supportable from this region's production
    """

    def __init__(
        self,
        region: RegionParams,
        scenario: ScenarioParams,
        som_params: SOMPoolParams = None,
        crop_params: CropParams = None,
        feedback_params: FeedbackParams = None,
        dt: float = 1.0,          # Time step (years)
        t_max: float = 100.0,     # Simulation length (years)
    ):
        self.region = region
        self.scenario = scenario
        self.som = som_params or SOMPoolParams()
        self.crop = crop_params or CropParams()
        self.fb = feedback_params or FeedbackParams()
        self.dt = dt
        self.t_max = t_max

        # Initialize state
        self._initialize_state()

    def _initialize_state(self):
        """Set initial conditions from region and SOM parameters."""
        soc = self.region.soc_initial  # t C/ha total

        self.C_active = soc * self.som.f_active
        self.C_slow = soc * self.som.f_slow
        self.C_passive = soc * self.som.f_passive

        # Initial SOC for reference
        self.soc_initial = soc

        # Track reference WHC (at initial SOC)
        soc_pct_initial = self._soc_to_percent(soc)
        self.whc_initial = soc_pct_initial * self.region.whc_sensitivity

    def _soc_to_percent(self, soc_tha: float) -> float:
        """Convert t C/ha (0-30 cm) to approximate % SOC.

        Assumes bulk density ~1.3 g/cm3, 30 cm depth.
        1% SOC = 1.3 * 30 * 0.01 * 10000 / 1000 = 39 t C/ha
        """
        return soc_tha / SOC_T_C_HA_PER_PERCENT_30CM

    def _n_mineralization(self, C_pool: float, cn_ratio: float, k: float) -> float:
        """Annual N mineralized from a single SOM pool (kg N/ha/yr).

        N_min = k * C_pool / CN * 1000 (convert t to kg)
        """
        return k * C_pool / cn_ratio * 1000.0

    def _synthetic_n(self, t: float) -> float:
        """Synthetic N application at time t (kg N/ha/yr)."""
        if self.scenario.withdrawal_years <= 0:
            # Abrupt: zero after t=0
            return self.region.synth_n_current if t < 0 else 0.0
        else:
            # Linear phase-out
            frac = max(0.0, 1.0 - t / self.scenario.withdrawal_years)
            return self.region.synth_n_current * frac

    def _bnf_supply(self, t: float) -> float:
        """Biological nitrogen fixation at time t (kg N/ha/yr).

        Baseline BNF exists even without management; managed transition
        ramps up to region's BNF potential.
        """
        baseline_bnf = self.region.bnf_potential

        if not self.scenario.bnf_managed:
            return baseline_bnf

        # Managed BNF ramps up over bnf_ramp_years
        if t <= 0:
            return baseline_bnf

        ramp_frac = min(1.0, t / self.region.bnf_ramp_years)
        managed_bnf = baseline_bnf + ramp_frac * (self.region.bnf_potential - baseline_bnf)

        # Legume expansion adds NET N contribution to subsequent crops.
        # Grain legumes fix 150-200 kg N/ha but export 130-170 in seed;
        # net residual for next crop ~20-40 kg N/ha (grain legumes) or
        # 50-80 kg N/ha (cover crop legumes, but no food calories).
        # We use 30 kg net N/ha as a blended average across grain + cover legumes.
        if self.scenario.legume_expansion:
            legume_ramp = min(1.0, t / self.scenario.legume_ramp_years)
            legume_n = legume_ramp * self.scenario.legume_fraction_target * 30.0
            managed_bnf += legume_n

        return managed_bnf

    def _yield_from_n(self, n_available: float, water_stress_factor: float = 1.0) -> float:
        """Crop yield from available N: Mitscherlich with stoichiometric cap.

        y = min(
            y_max * (1 - exp(-c * N)) * water_stress,   # response curve
            N_available / n_cost_per_tonne                # mass balance
        )

        The Mitscherlich curve governs at high N (current fertilized conditions
        where the curve is near saturation). The stoichiometric cap governs at
        low N (post-withdrawal), ensuring yield never implies more crop N than
        the plant absorbed. At the crossover, the response transitions from
        diminishing-returns to linear, consistent with first-principles N
        limitation (Lassaletta et al. 2014; Mueller et al. 2012).
        """
        y_max = self.region.yield_max_regional if self.region.yield_max_regional > 0 else self.crop.yield_max
        mit_c = self.region.mitscherlich_c_regional if self.region.mitscherlich_c_regional > 0 else self.crop.mitscherlich_c

        n_eff = max(0.0, n_available)
        y = y_max * (1.0 - np.exp(-mit_c * n_eff))
        y *= water_stress_factor

        # Stoichiometric cap: grain N export per tonne of grain.
        # Uses grain N only (not total crop N) because in the annual model,
        # gross mineralization already includes N from decomposing previous
        # years' residue. The residue N cycle is self-sustaining at steady
        # state; the binding constraint is whether external + mineralized N
        # can replace what's permanently removed in grain.
        n_grain_per_tonne = self.crop.grain_n_fraction * 1000  # kg N / t grain
        y_n_limited = n_eff / n_grain_per_tonne if n_grain_per_tonne > 0 else y
        y = min(y, y_n_limited)

        # Apply yield floor: regional empirical floor if set, else CropParams default (0.0)
        y_floor = self.region.yield_min_regional if self.region.yield_min_regional > 0 else self.crop.yield_min
        return max(y_floor, y)

    def _water_stress(self, soc_current: float) -> float:
        """Water-stress factor (0-1) from a smooth, two-sided SOC-WHC response.

        See coupled_monthly.MonthlyBiophysicalEngine._water_stress for the
        empirical anchor (Minasny & McBratney 2018; Hudson 1994). Loss
        side is linear; gain side is an exponential saturation that is
        C¹-continuous with the loss side at SOC = baseline. Replaces the
        previous one-sided ``max(0, …)`` clamp.
        """
        if not self.fb.physical_feedback:
            return 1.0

        soc_pct = self._soc_to_percent(soc_current)
        soc_pct_init = self._soc_to_percent(self.soc_initial)

        delta_soc_pct = soc_pct_init - soc_pct  # +degraded, -accumulated
        whc_sens = self.region.whc_sensitivity * self.fb.physical_strength
        whc_gain_sat_pct = WATER_STRESS_GAIN_SAT_SOC_PCT
        if delta_soc_pct >= 0.0:
            whc_change_mm = delta_soc_pct * whc_sens
        else:
            whc_gain_max_mm = whc_gain_sat_pct * whc_sens
            whc_change_mm = -whc_gain_max_mm * (
                1.0 - np.exp(delta_soc_pct / whc_gain_sat_pct)
            )

        # Soft-abs floor on total water deficit (see coupled_monthly.py for
        # rationale). ε = 3 mm.
        raw = self.region.baseline_water_deficit + whc_change_mm
        eps_mm = WATER_STRESS_SOFTPLUS_EPS_MM
        total_deficit = 0.5 * (raw + np.sqrt(raw * raw + eps_mm * eps_mm))
        stress = 1.0 - self.region.water_stress_coeff * total_deficit
        return max(WATER_STRESS_MIN_FACTOR, min(1.0, stress))

    def _residue_c_input(self, yield_actual: float) -> float:
        """Total carbon input from above-ground residue + root C (t C/ha/yr).

        Above-ground residue is subject to residue_retention (fraction not
        harvested for fuel/feed). Root C stays in the soil regardless.

        Root:shoot C ratio from Bolinder et al. 1999, 2007; Katterer et al. 2011.
        """
        # Above-ground residue (subject to retention)
        above_ground = yield_actual * self.crop.residue_grain_ratio
        above_ground *= self.region.residue_retention
        if self.scenario.residue_optimized:
            above_ground *= min(1.0, 1.1)  # 10% improvement in retention

        # Below-ground root C (not subject to retention)
        root_c = yield_actual * self.crop.residue_grain_ratio * self.region.root_shoot_c_ratio

        return (above_ground + root_c) * self.crop.residue_c_fraction

    def _n_immobilization(self, residue_c: float) -> float:
        """Net N immobilized when residue C enters SOM pools (kg N/ha/yr).

        When residue (C:N ~60) is incorporated into SOM pools with much lower
        C:N ratios (8-12), the additional N must be drawn from the mineral N
        pool. This is the stoichiometric immobilization demand that Century-type
        models must account for to avoid overestimating plant-available N.

        Net immobilization = N needed for new SOM - N supplied by residue.

        At equilibrium (constant pool sizes), gross mineralization equals
        gross immobilization, so net N from SOM cycling ≈ 0. Plants then
        depend on external N inputs (synthetic, BNF, deposition).

        References:
            Parton et al. 1987 (Century model); Manzoni & Porporato 2009
            (stoichiometric constraints); Robertson et al. 2019 (MEMS).
        """
        cre = region_cre(self.region)

        # N required to maintain pool C:N ratios as residue C enters SOM
        c_to_active = residue_c * cre * self.fb.cre_to_active
        c_to_slow = residue_c * cre * self.fb.cre_to_slow
        n_demand = (c_to_active / self.som.cn_active +
                    c_to_slow / self.som.cn_slow) * 1000  # kg N/ha

        # N supplied by the residue itself
        n_supply = residue_c / self.crop.residue_cn * 1000  # kg N/ha

        # Net immobilization (positive = N drawn from mineral pool)
        return max(0.0, n_demand - n_supply)

    def _legume_displacement_factor(self, t: float) -> float:
        """Food production penalty from cropland shifted to legumes.

        When legumes replace cereal crops, food calorie production declines
        even though N supply improves. Assumes:
        - 50% of legume area is grain legumes (soybean, chickpea) producing
          ~40% of cereal calories per hectare
        - 50% is cover crop legumes / green manures producing zero food calories
        - Blended: each hectare shifted to legumes produces 20% of cereal calories
        - So effective food area = (1 - legume_frac) + legume_frac * 0.20
        """
        if not self.scenario.legume_expansion or self.scenario.legume_fraction_target <= 0:
            return 1.0

        legume_ramp = min(1.0, max(0.0, t) / self.scenario.legume_ramp_years)
        current_legume_frac = legume_ramp * self.scenario.legume_fraction_target

        # Effective food-producing fraction of total cropland
        legume_calorie_fraction = 0.20  # blended: 50% grain legumes at 40% caloric equiv
        effective = (1.0 - current_legume_frac) + current_legume_frac * legume_calorie_fraction
        return effective

    def _cn_coupling_factor(self, soc_current: float) -> float:
        """Modifier to N mineralization efficiency based on C-N coupling.

        As SOC declines and residue C:N is high, mineralization becomes
        less efficient. Returns a multiplier 0.5-1.0.
        """
        if not self.fb.cn_coupling_feedback:
            return 1.0

        # Fraction of initial SOC remaining
        frac_remaining = soc_current / self.soc_initial

        # Below 60% of initial, coupling effects begin
        if frac_remaining > 0.60:
            return 1.0
        elif frac_remaining < 0.30:
            return 0.6  # 40% reduction at severe depletion
        else:
            # Linear interpolation
            return 1.0 - 0.4 * (0.60 - frac_remaining) / 0.30

    def run(self) -> pd.DataFrame:
        """Run the simulation and return time series as DataFrame."""
        n_steps = int(self.t_max / self.dt) + 1
        times = np.arange(0, self.t_max + self.dt/2, self.dt)

        # Output arrays
        results = {
            'year': times[:n_steps],
            'C_active': np.zeros(n_steps),
            'C_slow': np.zeros(n_steps),
            'C_passive': np.zeros(n_steps),
            'SOC_total': np.zeros(n_steps),
            'SOC_pct': np.zeros(n_steps),
            'N_mineralized': np.zeros(n_steps),      # gross
            'N_immobilized': np.zeros(n_steps),      # net immobilization from residue->SOM
            'N_net_mineralized': np.zeros(n_steps),  # gross - immobilization
            'N_synthetic': np.zeros(n_steps),
            'N_bnf': np.zeros(n_steps),
            'N_available': np.zeros(n_steps),
            'yield_tha': np.zeros(n_steps),
            'yield_fraction': np.zeros(n_steps),
            'water_stress': np.zeros(n_steps),
            'cn_coupling': np.zeros(n_steps),
            'residue_c_input': np.zeros(n_steps),
            'carrying_capacity_fraction': np.zeros(n_steps),
        }

        # Initial state
        C_a = self.C_active
        C_s = self.C_slow
        C_p = self.C_passive

        # Calculate initial yield (Year 0, full synthetic N)
        soc_0 = C_a + C_s + C_p
        n_min_0 = (
            self._n_mineralization(C_a, self.som.cn_active, self.som.k_active) +
            self._n_mineralization(C_s, self.som.cn_slow, self.som.k_slow) +
            self._n_mineralization(C_p, self.som.cn_passive, self.som.k_passive)
        )
        # N_available from gross mineral N pool (plant and microbes compete;
        # nue_apparent captures the plant's share). Immobilization is tracked
        # in the C budget but NOT deducted from plant-available N, because in
        # an annual model the gross mineralization flux and plant uptake are
        # concurrent processes drawing from the same mineral pool.
        n_supply_0 = (n_min_0 + self.region.synth_n_current
                      + self.region.bnf_potential +
                      self.region.atm_n_deposition)
        ws_0 = self._water_stress(soc_0)
        n_avail_0 = n_supply_0 * self.crop.nue_apparent
        yield_0 = self._yield_from_n(n_avail_0, ws_0)

        for i in range(n_steps):
            t = times[i]
            soc = C_a + C_s + C_p

            # N mineralization from each pool (gross)
            cn_factor = self._cn_coupling_factor(soc)
            n_min_active = self._n_mineralization(C_a, self.som.cn_active, self.som.k_active) * cn_factor
            n_min_slow = self._n_mineralization(C_s, self.som.cn_slow, self.som.k_slow) * cn_factor
            n_min_passive = self._n_mineralization(C_p, self.som.cn_passive, self.som.k_passive) * cn_factor
            n_mineralized = n_min_active + n_min_slow + n_min_passive

            # External N inputs
            n_synth = self._synthetic_n(t)
            n_bnf = self._bnf_supply(t)

            # Water stress
            ws = self._water_stress(soc)

            # N_available from gross mineral N supply × apparent NUE.
            # Immobilization tracked for diagnostics but NOT deducted from
            # plant-available N (see initial yield calculation for rationale).
            n_supply = n_mineralized + n_synth + n_bnf + self.region.atm_n_deposition
            n_available = n_supply * self.crop.nue_apparent
            y = self._yield_from_n(n_available, ws)

            # Compute immobilization for diagnostic output (and C budget below)
            res_c = self._residue_c_input(y)
            n_immob = self._n_immobilization(res_c)
            n_net_from_som = n_mineralized - n_immob

            # Legume displacement: less cropland producing food calories
            legume_disp = self._legume_displacement_factor(t)

            # Residue C input (legume residues also contribute, so use full yield)
            res_c = self._residue_c_input(y)

            # Carrying capacity accounts for both yield per ha AND food-producing area
            cc_frac = (y / yield_0 * legume_disp) if yield_0 > 0 else 0

            # Store results
            results['C_active'][i] = C_a
            results['C_slow'][i] = C_s
            results['C_passive'][i] = C_p
            results['SOC_total'][i] = soc
            results['SOC_pct'][i] = self._soc_to_percent(soc)
            results['N_mineralized'][i] = n_mineralized
            results['N_immobilized'][i] = n_immob
            results['N_net_mineralized'][i] = n_net_from_som
            results['N_synthetic'][i] = n_synth
            results['N_bnf'][i] = n_bnf
            results['N_available'][i] = n_available
            results['yield_tha'][i] = y
            results['yield_fraction'][i] = y / yield_0 if yield_0 > 0 else 0
            results['water_stress'][i] = ws
            results['cn_coupling'][i] = cn_factor
            results['residue_c_input'][i] = res_c
            results['carrying_capacity_fraction'][i] = cc_frac

            # Update state (Euler integration)
            if i < n_steps - 1:
                # Decomposition losses
                d_active = self.som.k_active * C_a * self.dt
                d_slow = self.som.k_slow * C_s * self.dt
                d_passive = self.som.k_passive * C_p * self.dt

                # Humification transfers
                h_a_to_s = d_active * self.som.h_active_to_slow
                h_s_to_p = d_slow * self.som.h_slow_to_passive

                # Residue input allocation
                cre = region_cre(self.region)

                if self.fb.residue_feedback:
                    c_in_active = res_c * cre * self.fb.cre_to_active * self.dt
                    c_in_slow = res_c * cre * self.fb.cre_to_slow * self.dt
                else:
                    # Fixed residue input (no feedback from yield)
                    fixed_res = self._residue_c_input(yield_0)
                    c_in_active = fixed_res * cre * self.fb.cre_to_active * self.dt
                    c_in_slow = fixed_res * cre * self.fb.cre_to_slow * self.dt

                # Pool updates
                C_a += (-d_active + c_in_active + 0) * 1.0  # No input to active from other pools
                C_s += (-d_slow + h_a_to_s + c_in_slow) * 1.0
                C_p += (-d_passive + h_s_to_p) * 1.0

                # Floor at zero
                C_a = max(0.0, C_a)
                C_s = max(0.0, C_s)
                C_p = max(0.0, C_p)

        return pd.DataFrame(results)

    def to_population(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add absolute population columns based on carrying capacity fraction."""
        df = df.copy()
        df['pop_supported_millions'] = df['carrying_capacity_fraction'] * self.region.pop_supported
        return df


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def run_sensitivity(
    region: RegionParams,
    scenario: ScenarioParams,
    param_name: str,
    param_values: List[float],
    param_target: str = 'som',  # 'som', 'crop', 'feedback', 'region'
    t_max: float = 100.0,
) -> Dict[str, pd.DataFrame]:
    """Run model across a range of values for one parameter.

    Returns dict mapping param_value -> DataFrame.
    """
    results = {}
    for val in param_values:
        som_p = SOMPoolParams()
        crop_p = CropParams()
        fb_p = FeedbackParams()
        reg = RegionParams(**{k: v for k, v in region.__dict__.items()})

        if param_target == 'som':
            setattr(som_p, param_name, val)
        elif param_target == 'crop':
            setattr(crop_p, param_name, val)
        elif param_target == 'feedback':
            setattr(fb_p, param_name, val)
        elif param_target == 'region':
            setattr(reg, param_name, val)

        model = SoilNCarryingCapacityModel(
            region=reg, scenario=scenario,
            som_params=som_p, crop_params=crop_p,
            feedback_params=fb_p, t_max=t_max,
        )
        df = model.run()
        results[val] = model.to_population(df)

    return results


# ============================================================
# GLOBAL AGGREGATION
# ============================================================

def run_global_scenarios(t_max: float = 100.0) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run all regions x all scenarios, return nested dict."""
    regions = get_default_regions()
    scenarios = get_default_scenarios()

    results = {}
    for s_name, scenario in scenarios.items():
        results[s_name] = {}
        for r_name, region in regions.items():
            model = SoilNCarryingCapacityModel(
                region=region, scenario=scenario, t_max=t_max,
            )
            df = model.run()
            results[s_name][r_name] = model.to_population(df)

    return results


def aggregate_global(scenario_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sum population across regions for one scenario."""
    dfs = list(scenario_results.values())
    agg = dfs[0][['year']].copy()
    agg['pop_total_millions'] = 0.0

    for df in dfs:
        agg['pop_total_millions'] += df['pop_supported_millions'].values

    # Also compute global average yield fraction
    total_area = sum(
        get_default_regions()[k].cropland_mha
        for k in scenario_results.keys()
    )
    agg['yield_fraction_weighted'] = 0.0
    for r_name, df in scenario_results.items():
        weight = get_default_regions()[r_name].cropland_mha / total_area
        agg['yield_fraction_weighted'] += df['yield_fraction'].values * weight

    return agg


# ============================================================
# PARAMETER EXPORT
# ============================================================

def export_parameters(filepath: str):
    """Export all model parameters to JSON for auditability."""
    params = {
        'som_pools': SOMPoolParams().__dict__,
        'crop': CropParams().__dict__,
        'feedback': FeedbackParams().__dict__,
        'regions': {k: v.__dict__ for k, v in get_default_regions().items()},
        'scenarios': {k: v.__dict__ for k, v in get_default_scenarios().items()},
    }
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)


if __name__ == '__main__':
    # Quick test run
    regions = get_default_regions()
    scenarios = get_default_scenarios()

    # Run North America, abrupt withdrawal
    model = SoilNCarryingCapacityModel(
        region=regions['north_america'],
        scenario=scenarios['abrupt'],
        t_max=100.0,
    )
    df = model.run()
    df = model.to_population(df)

    print("North America - Abrupt Withdrawal")
    for yr in [0, 10, 50, 100]:
        row = df.iloc[yr]
        print(f"Year {yr:3d}: SOC={row['SOC_total']:.1f} t/ha, "
              f"Yield={row['yield_tha']:.2f} t/ha, "
              f"CC={row['carrying_capacity_fraction']:.2%}")
