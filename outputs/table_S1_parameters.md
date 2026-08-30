# Table S1. Model parameters

Generated from `code/model/params.yaml` by `code/repro/make_table_s1.py`. Do not edit by hand: `code/tests/test_table_s1.py` fails if this file and the registry disagree.

`category` states what kind of number each value is. `measured` reproduces a published quantity, `calibrated` was chosen so the model matches a stated observation, `judgment` was chosen by a modeller, and `derived` is computed from other entries and cannot be set independently.

`varied in MC` is what the reported ensemble did, not what the registry declared. `declared, held fixed` marks a parameter whose uncertainty is registered and whose value the ensemble held constant; the credible intervals reported in this paper do not contain it. `exempt` marks a unit conversion, an accounting convention, or a quantity that defines a scenario rather than varying within it. Every reason is printed below the table.

| parameter | value | units | category | varied in MC | source | claims | benchmark |
|---|---|---|---|---|---|---|---|
| alpha | 0.08 to 0.15 | dimensionless | judgment | declared, held fixed |  |  |  |
| atm_n_deposition | 5 to 20 | kg_N per ha per year | measured | declared, held fixed | Regional total reactive nitrogen deposition to cropland | C-070 |  |
| baseline_water_deficit | 0 to 15 | mm | judgment | declared, held fixed |  | C-014 |  |
| bnf_potential | 15 to 35 | kg_N per ha per year | judgment | declared, held fixed |  | C-071 |  |
| bnf_ramp_years | 8 to 15 | year | judgment | declared, held fixed |  | C-071 |  |
| cm2_per_ha | 1e+08 | cm2 per ha | measured | exempt | Exact by definition of the unit |  |  |
| cn_bulk | 9.5 to 11 | g_C per g_N | measured | declared, held fixed | Bulk soil C:N ratios for cultivated soils, regional means |  |  |
| cost_share_band | [0.01, 0.4] | dimensionless | judgment | exempt |  |  |  |
| cre_allocation | 0.4 to 0.6 | dimensionless | judgment | exempt |  |  |  |
| cre_base | 0.11 | dimensionless | measured | declared, held fixed | Lehtinen et al. (2014) Soil Use and Management 30:524-538; doi:10.1111/sum.12134 | C-010 |  |
| cre_regional | 0.2 to 0.35 | dimensionless | calibrated | yes | Regionalised carbon retention efficiency | C-010, C-011 | B2 |
| crop_price_bounds | north_america [0.88, 1.15], europe [0.92, 1.1], east_asia [0.8, 1.25], south_asia [0.85, 1.2], southeast_asia [0.8, 1.25], latin_america [0.85, 1.2], sub_saharan_africa [0.8, 1.25], fsu_central_asia [0.75, 1.35] | dimensionless | judgment | exempt |  |  |  |
| crop_price_usd_t | 200 to 385 | USD per t_grain | measured | yes | Farm-gate producer prices for the regionally dominant cereal mix, 2019-2023 | C-030, C-031, C-032 |  |
| cropland_mha | 90 to 230 | Mha | measured | exempt | FAOSTAT arable land and permanent crops, 2019-2021 mean, aggregated to the eight model regions; https://www.fao.org/faostat/en/#data/RL | C-060 |  |
| eps_F_N | -0.5 | dimensionless | judgment | declared, held fixed | No clean regional estimates exist; chosen for the S4 feedback channel | C-040, C-050 | B5 |
| eps_F_PF | -0.5 to -0.2 | dimensionless | judgment | yes | Fertilizer Demand Model Estimates; Ethiopia and sub-Saharan Africa evidence; Roberts & Schlenker (2013) for the United States
 | C-040, C-041, C-061 | B5 |
| eps_F_PY | 0.03 to 0.1 | dimensionless | judgment | declared, held fixed | Manning calibration; Huang & Khanna (2010) |  |  |
| eps_LD_PL | -0.3 | dimensionless | judgment | exempt |  |  |  |
| eps_LD_PY | 0.15 to 0.25 | dimensionless | judgment | exempt |  |  |  |
| eps_LS_PL | 0.3 to 0.7 | dimensionless | judgment | exempt |  |  |  |
| eta | -0.7 to -0.3 | dimensionless | judgment | yes |  | C-042 |  |
| faostat_yield_target | 1.5 to 6 | t_grain per ha | measured | exempt | FAOSTAT cereal yield, 2019-2021 mean, aggregated to the eight model regions; https://www.fao.org/faostat/en/#data/QCL | C-070 |  |
| fert_cost_frac | derived: n_price_usd_kg * synth_n_current / (crop_price_usd_t * yield_baseline) | dimensionless | derived | not applicable | derived: n_price_usd_kg * synth_n_current / (crop_price_usd_t * yield_baseline) | C-030, C-031, C-032 |  |
| fert_reduction_target | 0.2 | dimensionless | judgment | exempt | Scenario definition | C-050 |  |
| g_per_t | 1e+06 | g per t_C | measured | exempt | Exact by definition of the unit |  |  |
| global_aggregation_weights | 0.0731707 to 0.186992 | dimensionless | derived | not applicable | derived: cropland_mha / sum(cropland_mha) | C-060 |  |
| laub_tropical_ratios | 1.3 to 1.71 | dimensionless | measured | exempt | Laub et al. (2024) Biogeosciences 21:3691-3716; doi:10.5194/bg-21-3691-2024 |  |  |
| n_benchmark_usd_kg | 0.876 | USD per kg_N | measured | exempt | World Bank Pink Sheet, urea, East Europe bulk FOB, 2019-2023 mean, 46% N; https://www.worldbank.org/en/research/commodity-markets |  |  |
| n_price_usd_kg | 0.7008 to 2.30388 | USD per kg_N | derived | not applicable | derived: n_benchmark_usd_kg * n_price_wedge | C-030, C-031, C-032 |  |
| n_price_usd_kg_farmer_paid | south_asia 0.35 | USD per kg_N | measured | exempt | Indian urea maximum retail price, subsidised, 2019-2023 | C-033 |  |
| n_price_wedge | 0.8 to 2.63 | dimensionless | judgment | yes | Regional delivered-price premia over the world urea benchmark | C-030, C-031, C-032 |  |
| n_price_wedge_bounds | north_america [0.9, 1.15], europe [0.9, 1.12], east_asia [0.7, 1.35], south_asia [0.8, 1.25], southeast_asia [0.7, 1.3], latin_america [0.85, 1.2], sub_saharan_africa [0.85, 1.2], fsu_central_asia [0.65, 1.4] | dimensionless | judgment | exempt |  |  |  |
| pct_to_fraction | 0.01 | dimensionless per pct_SOC | measured | exempt | Exact by definition of the unit |  |  |
| physical_feedback_strength | 1 | dimensionless | judgment | exempt |  |  |  |
| pop_supported | 375 to 1875 | million_people | measured | exempt | Population supported by each region's cereal production at current per-capita availability |  |  |
| price_benchmark_max_factor | 3 | dimensionless | judgment | exempt |  |  |  |
| residue_c_to_active_fraction | 0.9 | dimensionless | judgment | declared, held fixed | Century-family structural partition of fresh residue carbon | C-010, C-011 |  |
| residue_retention | 0.5 to 0.9 | dimensionless | judgment | yes | Regional crop-residue retention fractions | C-010, C-011 | B2 |
| root_shoot_c_ratio | 0.6 to 1 | dimensionless | judgment | declared, held fixed |  | C-010 |  |
| soc_bulk_density | 1.3 | g per cm3 | judgment | declared, held fixed | Representative cultivated-topsoil bulk density |  |  |
| soc_initial | 9 to 50 | t_C per ha | measured | declared, held fixed | SoilGrids 250m cropland SOC stock, 0-30 cm, regional medians; https://soilgrids.org/ | C-001, C-014 |  |
| soc_profile_depth_cm | 30 | cm | judgment | exempt | Model convention, stated once here |  |  |
| soc_tha_per_pct | 39 | t_C per ha per pct_SOC | derived | not applicable | derived: soc_bulk_density * soc_profile_depth_cm * cm2_per_ha / g_per_t * pct_to_fraction | C-001, C-014 |  |
| som_decay_rates | 0.000728 to 0.33 | per year | calibrated | yes | Century-family first-order decay constants, temperate regime | C-010, C-011, C-061 | B2 |
| som_humification | 0.03 to 0.4 | dimensionless | judgment | declared, held fixed | Century-family humification efficiencies | C-010, C-011 |  |
| som_pool_cn | 8 to 12 | g_C per g_N | measured | exempt | Century-family pool C:N ratios |  |  |
| som_pool_fractions | 0.04 to 0.58 | dimensionless | measured | exempt | Century-family pool partitioning for temperate cultivated soils |  |  |
| synth_n_current | 7 to 250 | kg_N per ha | measured | declared, held fixed | IFA regional nitrogen consumption divided by FAOSTAT cropland area, 2019-2021 | C-030, C-031, C-070 |  |
| texture_class | 0 to 1 | dimensionless | judgment | exempt |  |  |  |
| urea_n_fraction | 0.46 | dimensionless | measured | exempt | Stoichiometric nitrogen content of urea, CO(NH2)2 |  |  |
| water_stress_coeff | 0.003 to 0.005 | dimensionless per mm | judgment | declared, held fixed |  | C-014 |  |
| whc_sensitivity | 3.5 | mm_water per pct_SOC per 30cm_profile | measured | yes | Minasny & McBratney (2018) Eur J Soil Sci 69:39-47; doi:10.1111/ejss.12475 | C-014, C-021 | B4 |
| yield_max_regional | 3.453 to 6.09 | t_grain per ha | calibrated | exempt | Fallback ceiling used only when no monthly calibration is available | C-070 |  |
| yield_min_regional | 0.4 to 1.2 | t_grain per ha | judgment | declared, held fixed |  | C-072 | B3 |

## Parameters not varied in the ensemble (42)

**alpha** — Enters the price solution only as the product alpha*lambda_L in the denominator of the reduced-form price equation (coupled_econ_biophysical:731), and the ensemble's reduced form omits that term entirely (run_mc_ensemble:365) because cropland area is held fixed over the horizon. Drawing alpha would multiply a zero. The SI limitation this implies is about lambda_L rather than about alpha: land does not respond to price in this model, so the land-share elasticity is unsampled by construction, and any extension permitting cropland expansion reopens it.

**atm_n_deposition** — Added to mineral nitrogen every month in both the baseline and the disrupted run (monthly_model_v3:260), so it cancels to first order in the reported differences and what survives is second-order through the curvature of the yield-nitrogen response. Held fixed because the deposition field is a spatially interpolated product, and resampling it as eight independent regional multipliers would treat a spatially structured error as independent noise. SI limitation: the reported losses are conditional on fixed deposition, and the residual sensitivity is largest where deposition is the largest share of total nitrogen supply, which is sub-Saharan Africa (5 of about 40 kg N/ha) rather than east Asia, where deposition is largest in absolute terms (20 of about 340).

**baseline_water_deficit** — A regional constant offset added inside coupled_monthly.MonthlyBiophysicalEngine._water_stress before the SOC-dependent water-holding term. It sets the level of water stress; the paper reports the change in stress that follows a change in SOC, which is governed by whc_sensitivity and water_stress_coeff. Held fixed because it is a judgment-coded stand-in for a climatology the model does not otherwise carry, so a prior on it would express a precision the underlying quantity does not have. SI limitation: absolute water-stress levels carry no interval; only the SOC-driven change does, through whc_sensitivity.

**bnf_potential** — Superseded. The published model takes biological fixation from monthly_model_v3.get_regional_bnf, which computes a landscape average from legume rotation fraction and net nitrogen credit (MANAGED_TRANSITION_PARAMS) and never reads bnf_potential; the only surviving consumer is run_canonical.py, which copies it into the canonical CSV as a reported column. Drawing it would perturb a reported column and no modelled quantity. SI limitation: the parameter table must not present bnf_potential as an input to the transition, and fixation carries no sampled uncertainty at all, because MANAGED_TRANSITION_PARAMS is neither registered nor drawn. Recorded in F-007 as owed work rather than resolved here.

**bnf_ramp_years** — Superseded with bnf_potential, and more completely: no module reads it. Fixation in the published model is a constant landscape average with no ramp, so there is no transition length in the code that ran. SI limitation: the manuscript must not describe fixation as ramping in over 8 to 15 years. That is a description-versus-code discrepancy to fix in the text, not a prior to widen.

**cm2_per_ha** — Exact by definition; not an uncertain quantity.

**cn_bulk** — The bulk soil C:N ratio converts mineralized carbon to mineralized nitrogen (monthly_model_v3.update_som_pools via seams.SeamB). Not drawn because the ensemble already samples k_slow, and delivered nitrogen is to first order the quotient of the two, so a 10% error in C:N and a 10% error in decay are the same 10% error in the nitrogen the crop sees. Drawing both would report one uncertain number as two. SI limitation: the mineralization interval should be read as covering decay and stoichiometry jointly through the k_slow prior; the carbon stock trajectory, which responds to k_slow alone, is the one place the two are separable.

**cost_share_band** — A contract bound, not a model parameter.

**cre_allocation** — The split redistributes a fixed carbon input between two pools whose combined stock is what the reported SOC change measures. Perturbation moves the partition of a small flux, not its magnitude. Flagged for the mutation harness to confirm rather than assumed.


**cre_base** — The ensemble perturbs the regional carbon-response efficiency (cre_regional_mult), and cre_base is the global fallback those regional values override in every region the paper reports. Sampling both would double-count the same uncertainty.

**crop_price_bounds** — These are the prior bounds themselves, not a modelled quantity.

**cropland_mha** — Reported areas are administrative totals with far smaller uncertainty than any behavioural parameter, and they enter only as aggregation weights. Perturbing them within their reported uncertainty moves the global mean by less than the reporting precision of the results.


**eps_F_N** — eps_F_N is not a random variable in this design. It is the scenario dial: S1, S2 and S3 are defined by the value it takes, so drawing it would sample across the scenarios the paper reports separately.

**eps_F_PY** — Read by the ensemble at its regional central value (run_mc_ensemble:362) and entering the price solution only through the same denominator as eta, which is drawn (denom = eta - gamma*eps_F_PY). The two are not separately identified by any quantity the paper reports, so drawing both would report as independent uncertainty what is one uncertain number. SI limitation: the reported price interval is a statement about that denominator, not about eps_F_PY.

**eps_LD_PL** — The land market is inactive in the S1-S3 experiments the paper reports; cropland_expansion feedback is off. Enters only the S4 supplementary scenario, whose claims are qualitative.


**eps_LD_PY** — As eps_LD_PL: land market inactive in the reported experiments.

**eps_LS_PL** — As eps_LD_PL: land market inactive in the reported experiments.

**faostat_yield_target** — An observation the model is fitted to, not a parameter. Its uncertainty propagates through the fitted ceiling, and the ensemble recalibrates the ceiling every draw.


**fert_reduction_target** — A scenario definition. The uncertainty of interest is what a given reduction implies, not how large a reduction to consider; alternative magnitudes are reported as separate scenarios rather than as a prior.


**g_per_t** — Exact by definition; not an uncertain quantity.

**laub_tropical_ratios** — A ratio between two published parameterisations, varied implicitly through the som_decay_rates multiplier which scales the temperate and tropical rates together.


**n_benchmark_usd_kg** — The benchmark is a common factor on all eight regional nitrogen prices. Scaling it scales every price together, which the model absorbs into the same shock calibration; the quantity that matters for the regional comparison is the wedge, and that is what the ensemble varies. Recorded rather than varied.


**n_price_usd_kg_farmer_paid** — Used only in a labelled one-region sensitivity reported in the SI, not in the main experiment or the ensemble.


**n_price_wedge_bounds** — These are the prior bounds themselves, not a modelled quantity.

**pct_to_fraction** — Exact by definition; not an uncertain quantity.

**physical_feedback_strength** — Perfectly collinear with whc_sensitivity, which does enter the ensemble. Varying both would double-count the same uncertainty.


**pop_supported** — A presentational conversion applied after the model has run, not a model parameter.


**price_benchmark_max_factor** — A contract bound, not a model parameter.

**residue_c_to_active_fraction** — Wired but not drawn. Outside the SOC unit conversions this is the only registered parameter the published canonical run reads at all (F-006), and the mutation sweep measured a 10% perturbation moving southeast Asia year-1 loss by 2.5e-2 percentage points, the largest reach of any registered leaf. It was a bare literal at two use sites until 2026-07-25, so no ensemble that has run could have drawn it. SI limitation: the structural residue-carbon split is fixed at 0.90/0.10; its measured reach is small against the reported interval, but it is unsampled and the next ensemble should draw it.

**root_shoot_c_ratio** — Scales the below-ground half of the annual residue-carbon input, which the model forms as a sum rather than a product: c_in is proportional to (residue_retention + root_shoot_c_ratio) at monthly_model_v3:505-507. The ensemble draws residue_retention, so only the shoot term varies. This is an understatement rather than a cancellation, and naming it as one is the point of this entry. SI limitation: input-side carbon uncertainty is propagated on roughly half the input, so the reported SOC interval is narrower than an input-complete ensemble would give.

**soc_bulk_density** — Enters only through the derived soc_tha_per_pct, which is an accounting convention for converting a reported stock to a reported percentage. Varying it would change the units of the answer rather than the answer. Its effect is measured instead by test_soc_conversion_invariance.

**soc_initial** — Not a free parameter of the run. It is the anchor the spin-up targets, and engine_at_soc rescales the pools to it, so a draw would move the starting point and the calibration target together. The paper reports fractional losses relative to each region's own baseline, which removes the level but not the nitrogen-supply effect of a larger stock, and that residual effect is unsampled. SI limitation: absolute SOC stocks are initialisation anchored, on measurement in temperate regions and on model kinetics in tropical ones, and the measurement uncertainty of the source product is not propagated.

**soc_profile_depth_cm** — A convention, not an uncertain quantity. Changing it changes what the model means rather than how uncertain the model is. It is registered so that whc_sensitivity's units string can be checked against it.


**som_humification** — The active-to-slow and slow-to-passive transfer efficiencies. Not drawn because the k_slow prior already moves the slow pool's turnover, and over a 30-year horizon a humification error and a decay error are close to indistinguishable in the reported stock trajectory. SI limitation: pool-transfer efficiencies are fixed at Century-family values and the SOC interval covers turnover uncertainty through k_slow alone; structural sensitivity here is tested by the four-pool comparison rather than by the ensemble.

**som_pool_cn** — Pool C:N enters only the mineralization stoichiometry, which is constrained by the Seam B mass balance; the mass balance holds for any declared C:N, and the paper reports no quantity sensitive to it at the precision the ensemble could resolve. Flagged for review by the mutation harness rather than assumed harmless.


**som_pool_fractions** — The reported outcomes are invariant to this partition, which is a different and weaker statement than the one recorded here before 2026-07-25. The earlier reason claimed the dynamic spin-up overwrites the initial partition entirely. It does not, and test_spinup_partition_independence.py was written to check that claim and falsified it: with k_passive at 0.000728 per year the passive pool has a 1374-year turnover time, while the spin-up's convergence criterion (fractional SOC drift below 0.002 over a 50-year window) is met after about a century. The active and slow pools do reach their equilibrium in that time and are partition-independent to within the asserted fast-pool tolerance; the passive pool does not, and the total SOC the spin-up delivers therefore still carries the assumed passive fraction. Run to a true fixed point instead (n_spinup 20000, tol 1e-6) every starting partition lands on the same SOC to within 1e-3, so the equilibrium itself is partition-independent; the shipped spin-up simply does not run that far, and should not, because the model's own equilibrium would replace a measured stock with an inferred one.
Point values are deliberately not restated here. Quoting a measured number in prose that no test regenerates is how one fact ends up specified in two places with two values, which is what happened to this entry between 2026-07-25 and the production-path recalibration: the numbers written here shifted by up to 0.7 t C/ha and one percent of relaxation while the prose stayed put. The measured characterisation is written by the test to results/spinup_partition_characterisation.yaml on every run and is cited from there. What this entry states are the bands the test asserts, so the two cannot disagree.
How far the spin-up runs is regime-dependent. Measured against the analytic fixed point of the pool cascade, c_p* = h_slow_to_passive * k_slow * c_slow* / k_passive, the passive pool covers a few percent of the distance from its initialisation to its equilibrium in the four temperate regions (asserted band 3 to 15 percent) and almost all of it in the four regions on the Laub tropical parameterisation, whose k_passive is 0.001245 rather than 0.000728 per year (asserted band 85 to 97 percent). Temperate SOC is therefore close to the measured stock it was initialised from and tropical SOC is close to the model's own equilibrium, which for Sub-Saharan Africa is well below the registered soc_initial. That split is a real limitation and belongs in the SI: the absolute SOC level the model runs at is anchored on measurement in the temperate regions and on the model's kinetics in the tropical ones, so absolute stocks are not comparable across regimes even though relative changes are.
What licenses the exemption is the measured sensitivity of the published quantities rather than an argument about the spin-up. Over f_passive from 0.45 to 0.73, the range the Century literature supports for cultivated temperate soils, the S3 year-1 and year-10 yield losses move by less than 0.05 percentage points in every region (the asserted tolerance, against a reporting precision of 0.1), while the absolute SOC level moves by more than 8 t C/ha in every temperate region, which the test asserts as a lower bound because an invariance measured over a level that had stopped moving would be vacuous. The measured spans are in the results file. The reason is structural: the water-stress term responds to SOC change measured against each run's own equilibrium, not to the absolute stock, and the passive pool's contribution to N mineralization is C_passive x k_passive / cn_passive, about 2 percent of the total. A larger inert stock moves the level and leaves the response alone. Sampling this parameter would therefore add draws that differ in a quantity the paper does not report and agree in every quantity it does.
The consequence for the paper is that absolute SOC stocks are an initialisation anchored on the measured stock and the assumed passive fraction, not a model prediction, and the manuscript must not present them as one. Relative SOC and yield changes are what the model supports. See test_spinup_partition_independence.py, which asserts every band stated above so that a change to k_passive, to the convergence tolerance, to the calibration path, or to the water-stress formulation cannot quietly invalidate this reasoning.


**synth_n_current** — The observed regional application rate, which also defines the scenario: the disruption is specified as a fractional reduction from it, so a draw would rescale the shock rather than express uncertainty about it. Its measurement uncertainty reaches the reported outcomes through the cost share, where the drawn nitrogen-price multiplier carries it. SI limitation: FAOSTAT application rates are treated as known, so a region whose reported rate is biased has a proportionally biased baseline and shock.

**texture_class** — A categorical label with no continuous perturbation, and it does not currently modify any coefficient. If the mutation harness confirms it is unused it should be deleted rather than exempted.


**urea_n_fraction** — A stoichiometric constant, not an uncertain quantity.

**water_stress_coeff** — Converts a water deficit into a yield-fraction penalty inside _water_stress, multiplying whc_sensitivity along the SOC-to-yield water pathway. whc_sensitivity is drawn across a factor of 3.65 (0.657x to 2.40x), which is the wider of the two and dominates the product. SI limitation: the water pathway carries one multiplier of uncertainty; how much water a percentage point of SOC holds and how much yield a millimetre of water buys are not separately identified.

**yield_max_regional** — Recalibrated within every Monte Carlo draw rather than perturbed: each draw's ceiling is solved against that draw's own parameters, so the calibration is re-established rather than carried over. Perturbing the stored fallback in addition would double-count.


**yield_min_regional** — The floor of the Mitscherlich response, applied as a max() at monthly_model_v3:502. It is not on the path the reported runs take: at year 30 of S3 every region sits at least 3.5 times its floor, the closest being sub-Saharan Africa at 1.42 against 0.40 t/ha. A prior on a parameter that never binds would widen nothing and would suggest the floor had been tested. SI limitation: the floor binds only under complete withdrawal sustained past the exhaustion of mineralization, outside the reported scenario set; a harsher scenario makes it a live parameter.

