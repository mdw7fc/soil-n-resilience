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


# ============================================================
# MODEL PARAMETERS
# ============================================================

@dataclass
class SOMPoolParams:
    """Three-pool SOM structure (Century/RothC-informed)."""
    # Pool fractions of total SOC (must sum to 1.0)
    f_active: float = 0.04      # 2-5% of total SOM
    f_slow: float = 0.38        # 25-50% of total SOM
    f_passive: float = 0.58     # 45-73% of total SOM

    # Decay constants (yr^-1) — reciprocal of turnover time
    k_active: float = 0.33      # ~3 yr turnover
    k_slow: float = 0.03705     # ~27 yr turnover (calibrated for SOC equilibrium)
    k_passive: float = 0.000728 # ~1,373 yr turnover (calibrated for SOC equilibrium)

    # C:N ratios by pool
    cn_active: float = 8.0
    cn_slow: float = 12.0
    cn_passive: float = 12.0

    # Fraction of decomposed C transferred to next pool (humification)
    h_active_to_slow: float = 0.40
    h_slow_to_passive: float = 0.03


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
    residue_c_fraction: float = 0.42

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

    # Water holding capacity sensitivity to SOC (mm per % SOC in top 20 cm)
    whc_sensitivity: float = 8.4

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

    # BNF potential under managed transition (kg N/ha/yr NET available to crops,
    # averaged across all cropland, not per hectare of legume).
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
    # If 0, uses FeedbackParams.cre_base as default.
    cre_regional: float = 0.0


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
    physical_strength: float = 1.0

    # Marginal land expansion feedback: enabled
    expansion_feedback: bool = False  # Off by default (regional, not always relevant)

    # C-N coupling feedback: enabled
    cn_coupling_feedback: bool = True

    # Carbon retention efficiency of residue (fraction of residue C entering SOM)
    cre_base: float = 0.11  # 11% long-term average (Lehtinen et al. 2014)

    # Fraction of CRE going to active vs. slow pool
    cre_to_active: float = 0.60
    cre_to_slow: float = 0.40


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
        'north_america': RegionParams(
            name='North America',
            soc_initial=50.0,       # t C/ha, 0-30 cm; blended US+Canada
            cn_bulk=10.0,
            cropland_mha=170.0,     # US ~155 + Canada ~35 Mha arable (FAO)
            synth_n_current=76.0,   # ~13 Tg N / 170 Mha (USDA ERS, IFA)
            pop_supported=900.0,    # ~12% of global crop calories (high yields, major exporter)
            texture_class=1,        # Loam dominant
            whc_sensitivity=8.4,
            water_stress_coeff=0.003,
            baseline_water_deficit=0.0,
            atm_n_deposition=10.0,  # NADP monitoring: 8-12 kg N/ha in Corn Belt (Vet et al. 2014)
            bnf_potential=25.0,     # Net landscape avg BNF under managed transition
            bnf_ramp_years=8.0,
            residue_retention=0.90,
            yield_max_regional=5.926,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=1.1,     # Morrow Plots/Sanborn Field unfertilized: ~1.0-1.1 t/ha
            root_shoot_c_ratio=0.80,    # Bolinder et al. 2007; temperate cereals
            cre_regional=0.280,         # Calibrated: active pool equilibrium at initial SOC
        ),
        'europe': RegionParams(
            name='Europe',
            soc_initial=42.0,       # t C/ha; LUCAS 2018, Lugato et al. 2014
            cn_bulk=10.5,
            cropland_mha=130.0,     # EU27 + UK + non-EU Europe (FAO)
            synth_n_current=85.0,   # ~11 Tg N / 130 Mha (Eurostat, IFA)
            pop_supported=900.0,    # ~12% of global crop calories
            texture_class=1,
            whc_sensitivity=8.4,
            water_stress_coeff=0.003,
            baseline_water_deficit=0.0,
            atm_n_deposition=12.0,  # EMEP: 10-15 kg N/ha in W. Europe (Simpson et al. 2014)
            bnf_potential=25.0,
            bnf_ramp_years=8.0,
            residue_retention=0.90,
            yield_max_regional=5.448,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=1.0,     # Rothamsted Broadbalk unfertilized: ~1.0 t/ha
            root_shoot_c_ratio=0.80,    # Bolinder et al. 2007; temperate cereals
            cre_regional=0.259,         # Calibrated: active pool equilibrium at initial SOC
        ),
        'east_asia': RegionParams(
            name='East Asia',
            soc_initial=35.0,       # t C/ha; China SOC variable, much degraded
            cn_bulk=10.0,
            cropland_mha=120.0,     # China ~120 Mha arable (NBS, FAO)
            atm_n_deposition=20.0,  # Very high in E. China: 15-30 kg N/ha (Liu et al. 2013)
            synth_n_current=250.0,  # ~30 Tg N / 120 Mha; China is world's largest N consumer
            pop_supported=1875.0,   # ~25% of global crop calories (China dominates)
            texture_class=1,
            whc_sensitivity=8.4,
            water_stress_coeff=0.004,
            baseline_water_deficit=5.0,
            bnf_potential=20.0,     # Dense cropping limits rotation options
            bnf_ramp_years=10.0,
            residue_retention=0.75, # Significant residue burning despite bans
            yield_max_regional=6.090,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=0.9,     # China 1949 avg ~1.0; depleted dryland blended: 0.9
            root_shoot_c_ratio=0.60,    # Lower for rice-dominated systems (Katterer 2011)
            cre_regional=0.226,         # Calibrated: active pool equilibrium at initial SOC
        ),
        'south_asia': RegionParams(
            name='South Asia',
            soc_initial=25.0,       # t C/ha; severely depleted (Lal 2004, SoilGrids)
            cn_bulk=9.5,
            cropland_mha=200.0,     # India ~155 + Pakistan ~22 + Bangladesh ~8 (FAO)
            synth_n_current=110.0,  # ~22 Tg N / 200 Mha (FAI India, IFA)
            pop_supported=1350.0,   # ~18% of global crop calories
            texture_class=1,
            whc_sensitivity=8.4,
            water_stress_coeff=0.005,  # Monsoon dependence, high evaporative demand
            baseline_water_deficit=10.0,
            atm_n_deposition=12.0,  # Indo-Gangetic Plain: 10-15 (Dentener et al. 2006)
            bnf_potential=20.0,     # Heat limits BNF; limited cover crop adoption
            bnf_ramp_years=12.0,
            residue_retention=0.50, # Widespread harvesting for fuel/fodder
            yield_max_regional=3.584,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=0.5,     # ICRISAT Vertisol trials: 0.3-0.6; pre-GR wheat: 0.66
            root_shoot_c_ratio=0.70,    # Moderate; rice+wheat systems (Johnson et al. 2006)
            cre_regional=0.341,         # Calibrated: active pool equilibrium at initial SOC
        ),
        'southeast_asia': RegionParams(
            name='Southeast Asia',
            soc_initial=32.0,       # t C/ha; tropical soils, variable
            cn_bulk=10.0,
            cropland_mha=90.0,      # Indonesia, Vietnam, Thailand, Myanmar, etc.
            synth_n_current=89.0,   # ~8 Tg N / 90 Mha (IFA)
            pop_supported=750.0,    # ~10% of global crop calories (rice-dominant)
            texture_class=1,
            whc_sensitivity=8.4,
            water_stress_coeff=0.004,
            baseline_water_deficit=5.0,
            atm_n_deposition=8.0,   # Moderate tropical: 5-10 (Vet et al. 2014)
            bnf_potential=25.0,     # Rice paddy BNF + legume potential
            bnf_ramp_years=10.0,
            residue_retention=0.70, # Some residue burning in rice systems
            yield_max_regional=4.737,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=1.2,     # Wetland rice BNF advantage: 30-60 kg N/ha/yr
            root_shoot_c_ratio=0.60,    # Lower for rice systems (Katterer 2011)
            cre_regional=0.307,         # Calibrated: active pool equilibrium at initial SOC
        ),
        'latin_america': RegionParams(
            name='Latin America',
            soc_initial=45.0,       # t C/ha; blended; Cerrado degraded, Pampas higher
            cn_bulk=11.0,
            cropland_mha=160.0,     # Brazil ~55, Argentina ~30, Mexico ~20, etc. (FAO)
            atm_n_deposition=5.0,   # Low: relatively clean atmosphere (Vet et al. 2014)
            synth_n_current=50.0,   # ~8 Tg N / 160 Mha (IFA); soybean BNF offsets
            pop_supported=900.0,    # ~12% of global crop calories (major exporter)
            texture_class=1,
            whc_sensitivity=8.4,
            water_stress_coeff=0.003,
            baseline_water_deficit=0.0,
            bnf_potential=35.0,     # Strong soybean/legume tradition; highest BNF potential
            bnf_ramp_years=8.0,
            residue_retention=0.80,
            yield_max_regional=5.112,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=0.9,     # Pampas ~1.0-1.5, Cerrado degraded; blended: 0.9
            root_shoot_c_ratio=0.90,    # High; diverse cropping, deep-rooted tropical systems
            cre_regional=0.308,         # Calibrated: active pool equilibrium at initial SOC
        ),
        'sub_saharan_africa': RegionParams(
            name='Sub-Saharan Africa',
            soc_initial=9.0,        # t C/ha, 0-30 cm; cropland-specific on degraded
                                    # Oxisols/Ultisols. AfSIS landscape avg ~35 includes
                                    # forest/woodland. Cultivated soils 5-15 t/ha typical
                                    # (Batjes 2001; Vågen et al. 2005)
            cn_bulk=11.0,
            cropland_mha=230.0,     # Largest cropland area, low intensity (FAO)
            synth_n_current=7.0,    # ~1.5 Tg N / 230 Mha (IFA); extremely low
            pop_supported=600.0,    # ~8% of global crop calories (low yields)
            texture_class=0,        # Sandy dominant in many regions
            whc_sensitivity=8.4,
            water_stress_coeff=0.005,
            baseline_water_deficit=15.0,
            atm_n_deposition=5.0,   # Low: 3-7 kg/ha (Dentener et al. 2006)
            bnf_potential=15.0,     # Limited institutional capacity for rapid BNF adoption
            bnf_ramp_years=15.0,
            residue_retention=0.55, # Fuel, construction, livestock feed
            yield_max_regional=3.931,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=0.4,     # TSBF network controls: 0.3-0.6; Oxisol/Ultisol baseline
            root_shoot_c_ratio=1.0,     # High; grassland-origin soils, deep roots (Bolinder 2007)
            cre_regional=0.20,          # Within literature range; equilibrium at SOC=9
        ),
        'fsu_central_asia': RegionParams(
            name='Former Soviet Union & Central Asia',
            soc_initial=35.0,       # t C/ha; cultivated chernozems (lower than native ~50+)
                                    # Reflects ~30% loss from decades of cultivation
                                    # (Mikhailova et al. 2000; Torn et al. 2002)
            cn_bulk=10.0,
            cropland_mha=130.0,     # Russia ~80, Ukraine ~33, Kazakhstan ~12, etc.
            synth_n_current=38.0,   # ~5 Tg N / 130 Mha (IFA)
            pop_supported=375.0,    # ~5% of global crop calories
            atm_n_deposition=5.0,   # Low: continental interior (Vet et al. 2014)
            texture_class=1,
            whc_sensitivity=8.4,
            water_stress_coeff=0.004,
            baseline_water_deficit=10.0,
            bnf_potential=20.0,     # Cold climate limits BNF season
            bnf_ramp_years=10.0,
            residue_retention=0.85,
            yield_max_regional=3.453,   # Recalibrated (v6: NUE=0.75, gross mineral N, grain-N cap)
            yield_min_regional=0.9,     # Pryanishnikov Institute trials: 0.8-1.0 t/ha
            root_shoot_c_ratio=1.0,     # High; steppe-origin soils, deep roots (Bolinder 2007)
            cre_regional=0.35,          # Slightly elevated; includes manure/organic amendments
                                        # common in FSU mixed crop-livestock systems
        ),
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
        return soc_tha / 39.0

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
        baseline_bnf = 5.0  # kg N/ha/yr from free-living/associative fixation

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
        """Water stress factor (0-1) based on SOC-driven water holding capacity loss."""
        if not self.fb.physical_feedback:
            return 1.0

        soc_pct = self._soc_to_percent(soc_current)
        soc_pct_init = self._soc_to_percent(self.soc_initial)

        # WHC loss relative to initial
        delta_soc_pct = soc_pct_init - soc_pct  # positive when SOC has declined
        whc_loss_mm = delta_soc_pct * self.region.whc_sensitivity * self.fb.physical_strength

        # Total water deficit
        total_deficit = self.region.baseline_water_deficit + max(0, whc_loss_mm)

        # Yield multiplier (1.0 = no stress)
        stress = 1.0 - self.region.water_stress_coeff * total_deficit
        return max(0.3, min(1.0, stress))  # Floor at 30% of potential

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
        cre = self.region.cre_regional if self.region.cre_regional > 0 else self.fb.cre_base

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
        n_supply_0 = (n_min_0 + self.region.synth_n_current + 5.0 +
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
                # Use region-specific CRE if provided, else fall back to global
                cre = self.region.cre_regional if self.region.cre_regional > 0 else self.fb.cre_base

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
