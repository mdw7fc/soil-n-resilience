# CHANGELOG — ERFS-100341

## v14 / deposit v1.4 — Figure S7 regenerated and its generator deposited

The v1.3 pass recovered the generators for main-text figures 1 and 2, the Monte
Carlo ensemble, the in-season N capture sweep and the severity sweep, but
Supplementary Figure S7 (the halved-elasticity farm gradient) was still carried
over from the pre-correction working script. It has now been regenerated from
the corrected model and its generator deposited as
`code/repro/make_figure_s7.py`, together with
`data/figS7_farm_elasticity_gradient.json`. The baseline curves reproduce
`data/figure1_farm_gradient.json` exactly, confirming that the ported farm
calculation is the same one behind main-text Figure 1.

Three claims in the Figure S7 caption were wrong against the corrected output
and have been corrected in the SI:

- The Sub-Saharan African yield improvement from low to high farm SOC was given
  as "~3.5 percentage points"; the corrected model gives **2.3 percentage
  points** (10 % to 200 % of regional mean SOC).
- The extra gross-margin loss from halving the fertilizer-demand elasticities
  was given as "~7-12 percentage points deeper"; the corrected values are
  **2.6 to 10.3 percentage points deeper at the regional mean SOC**, and up to
  **14.1 percentage points** on the most degraded Sub-Saharan African farms.
- The robustness claim ("the gap between SOC = 50 % and SOC = 100 % is
  comparable in both scenarios") held for gross margins but not for yields. It
  now reads that the gross-margin gap is comparable or larger under halved
  elasticities in all four regions (2.5-4.2 pp at baseline against 3.0-5.9 pp
  when halved), while the corresponding yield gap scales with the elasticity
  itself and is close to half as large, so the ordering is preserved and the
  magnitude is not.

The panel b axis label was harmonized from "Profit change (%)" to "Gross margin
change (%)", matching the quantity the model computes and the wording used
elsewhere in the paper.

### Other corrections in v14

- **Zero-shock tolerance.** SI Note 8, the README and this file stated year-10
  zero-shock yield ratios "≥ 0.9999". The true minimum is **0.99986** (FSU), so
  all three now state "≥ 0.9998".
- **Supplementary table 1, soil-N buffer ratio row.** The row was regenerated at
  full precision from the corrected canonical run: 37 → **36** (EU), 34 → **32**
  (SA), 63 → **62** (LATAM), 57 → **54** (SSA), 50 → **49** (FSU). NA (50) and
  EA (20) are unchanged. The corresponding SI sentence "to 63 % in Latin
  America" becomes **62 %**.
- **Spatial screen focal class.** "from 17.14 % to 1.41 %" was a transcription
  of the pre-correction value and becomes **17.4 %**.
- **Figure S11 SOC spread.** "0.2-1.5 percentage points across the typical
  crisis range" becomes **0.1-1.5**.
- **SI organization.** The opening paragraph said the SI is organized into five
  parts and did not list Supplementary Note 8. It now says six parts and adds a
  clause describing Note 8.
- **Author response, SC1 versus SC2.** The letter said SC1 "produces roughly
  twice the year-30 yield loss" of SC2; the corrected trajectories give 4.376 %
  against 0.418 %, a factor of **10.5**, so it now reads "roughly ten times",
  matching the manuscript.
- Supplementary Note 8 and the author response now list Figure S7 among the
  generators recovered and deposited in this revision.

## v13 / deposit v1.3 — two internal-consistency corrections; Figure 1 and 2 generators deposited

Two defects were found in the model during preparation of this revision. Both
are internal-consistency problems rather than parameter changes: in each case
the code was not doing what the Methods describe. Both are now fixed, and every
number in the manuscript, SI, figures and deposit has been regenerated from the
corrected code. **All reported magnitudes fall.** The direction and ordering of
the paper's central results are preserved, with two regional exceptions noted
below.

### Correction 1 — stationary Century spin-up

`century_dynamic_spinup` solved for equilibrium carbon pools **without** the
water-stress multiplier that the simulation itself applies at ΔSOC = 0. The
spin-up therefore handed the simulation a pool configuration that was not a
fixed point of the simulation's own dynamics, so every run began with a
spurious transient. The spin-up now applies the same baseline water-stress
multiplier (`apply_water_stress: bool = True`), making the equilibrium a
genuine fixed point. Relatedly, the mineralization baseline N̂ is normalized to
the **year-0 recorded** mineralization (`self.N_min_baseline = n_min_init`)
rather than to the spin-up's analytic value, so N̂ = 0 at t = 0 by construction.

A zero-shock invariance test was added (`test_zero_shock_invariance.py`): with
the shock set to zero, every region's yield must stay at its baseline. Before
the fix this test failed; it now passes (all regions, year-10 ratio ≥ 0.9998,
year-30 ≥ 0.9995).

### Correction 2 — re-solved rather than clipped equilibrium under a binding cap

When the physical fertilizer supply ceiling bound, the code clipped the
unconstrained equilibrium quantity to the ceiling and left the associated
prices untouched. The result did not clear the constrained market. The
equilibrium is now **re-solved** subject to the ceiling
(`_solve_equilibrium_capped`). A market-clearing test was added
(`test_cap_market_clearing.py`); maximum cap residual is now 0.00e+00.

### Effect on reported results

| Quantity | v12 | v13 |
|---|---|---|
| Global S3 loss, yr 1 / 10 / 30 (%) | 4.33 / 5.58 / 5.95 | **2.30 / 3.41 / 3.64** |
| SC1 global yr 1 / 10 / 30 (%) | 4.77 / 6.27 / 6.79 | **2.72 / 4.05 / 4.38** |
| SC2 global yr 1 / 10 / 30 (%) | 4.63 / 4.49 / 3.05 | **2.58 / 2.23 / 0.42** |
| NA yr1 / yr10 / yr30 (%) | 2.00 / 2.24 / 2.36 | **1.56 / 1.77 / 1.87** |
| EU | 2.94 / 3.95 / 4.10 | **2.51 / 3.50 / 3.64** |
| EA | 3.24 / 3.50 / 3.71 | **1.11 / 1.28 / 1.37** |
| SA | 7.43 / 10.10 / 10.55 | **2.96 / 5.98 / 6.26** |
| SEA | 4.70 / 6.14 / 6.59 | **3.13 / 4.48 / 4.83** |
| LATAM | 1.73 / 2.89 / 3.11 | **1.62 / 2.90 / 3.12** |
| SSA | 10.34 / 13.74 / 14.98 | **3.56 / 5.41 / 5.92** |
| FSU | 8.26 / 9.76 / 10.59 | **4.49 / 5.46 / 5.96** |
| Climate robustness, max yr-10 shift | 0.54 pp (ρ = 0.98) | **0.74 pp (ρ = 0.93)** |

The year-10 vulnerability ranking changes: South Asia (5.98) > FSU (5.46) >
SSA (5.41) > SEA (4.48) > EU (3.50) > LATAM (2.90) > NA (1.77) > EA (1.28).
Sub-Saharan Africa moves from first to third and the top three now cluster
within 0.6 pp, so the paper no longer identifies a single most-exposed region.

Table S3 rank correlations weaken materially. Year-1 / year-10 Spearman ρ:
SOC stock −0.61 / −0.67 (was −0.87 / −0.92); buffer ratio +0.29 / +0.02 (was
0.00 / +0.07); water deficit +0.69 / +0.70 (was +0.96 / +0.90); y_max
−0.71 / −0.86 (was −0.79 / −0.88); BNF −0.43 / −0.45 (was −0.88 / −0.75);
synthetic N −0.55 / −0.26 (was −0.24 / −0.21); |ε_F,PF| +0.58 / +0.60 (was
+0.72 / +0.81); |ε_food| +0.51 / +0.60 (was +0.61 / +0.75); λ_L +0.53 / +0.47
(was +0.30 / +0.41).

### Figure 1 and Figure 2 generators added to the deposit

Main Figures 1 and 2 were previously produced outside the deposit and could not
be regenerated from it. The original analysis code has been recovered and
ported in as `code/repro/run_price_shock_analysis.py` (computation) with
`make_figure_1.py` and `make_figure_2.py` (rendering). Running the ported code
against the **pre-fix** model reproduces the published panels, which establishes
that the differences reported here are attributable to the two corrections and
not to reconstruction error.

Figure 1 is barely moved: gross-margin change at the regional mean SOC is
SSA −11.4 %, SA −8.7 %, LATAM −6.4 %, NA −3.4 % (v12 published: −12.0, −9.2,
≈−6.5, ≈−2.9), and no curve shifts by more than 0.5 pp.

Figure 2a changes shape in two regions:

- **East Asia is no longer an outlier.** Its year-10 gradient was mildly
  inverted (−0.7 % at low SOC to −1.2 % at high SOC); it is now steeply
  monotone (−4.06 % at 10 % of regional mean SOC to −0.62 % at 200 %).
- **Sub-Saharan Africa is now non-monotone.** The curve runs −3.32 % at 10 %
  SOC, reaches a maximum loss of −4.80 % near 70–80 %, and recovers to −4.24 %
  at 200 %. Latin America and FSU show slight versions of the same hump.

The SSA hump was diagnosed and is not a bug: no supply cap binds anywhere along
the gradient. A farm rescaled to 10 % of regional mean SOC is far from SOM
equilibrium, so over ten years its pools rebuild, its mineralization rises above
its own year-0 baseline, and the fertilizer-demand response to mineral N
(ε_F,N = −0.5) cuts fertilizer use. By year 10 the 10 %-SOC SSA farm draws
11.3 kg N ha⁻¹ from mineralization against 2.07 (shocked) / 2.91 (control)
kg ha⁻¹ of fertilizer, versus a 7.0 kg ha⁻¹ baseline, so fertilizer is a small
share of its nitrogen supply and a proportional cut costs proportionally less.
**The low-SOC end of Figure 2a is dominated by ten-year SOM relaxation rather
than by the shock**, and this is now stated in the SI and the response letter.

### Monte Carlo ensemble and Figure S11 generators added to the deposit

The Supplementary Note 6 / Figure S9 Monte Carlo ensemble and the Figure S11
severity sweep were also produced outside the earlier deposit. Both generators
are now included (`code/repro/run_mc_ensemble.py`, `code/repro/make_figure_s9.py`,
`code/repro/make_figure_s11.py`) and both figures were regenerated.

One substantive change beyond the two corrections: the published ensemble ran on
the model's built-in expert climate profiles, whereas every other result in the
paper derives from the ERA5 forcing. The ported ensemble patches ERA5 in, so
Note 6 is now consistent with the canonical run. Ensemble results (n = 1,000,
seed 20260424):

| Quantity | published | v1.3 |
|---|---|---|
| Global area-weighted year-1 loss, ensemble median | 2.32 % | **2.51 %** |
| 5-95 % range | 3.2 pp | **3.3 pp** |
| Cross-region soil-N buffer, ensemble median | 0.91 ppt | **0.88 ppt** |
| P(SSA worst year-1 gross margin) | 89.2 % | **83.7 %** |
| P(SSA largest year-1 yield loss) | 0 % | **0 %** |

The three robustness findings of Note 6 are unchanged: SOC buffering holds in
all 1,000 draws in every region on both yield and gross margin; the ensemble
median brackets the main-text central estimate; and the sign of the buffer is
preserved throughout. The regional ranking sentence was corrected: FSU/Central
Asia carries the largest year-1 yield loss in 99.8 % of draws, and Southeast
Asia now exceeds Sub-Saharan Africa in only 25.4 % of draws (previously
described as consistently larger).

Figure S11 changes materially. The 25 %-versus-100 % SOC spread is 0.2-1.5 pp
at 100-150 % price increases, widening to 0.4-2.2 pp at 300 %, against the
2.5-4 pp reported previously. The steepest separation is now North America
(2.18 pp at 300 %) rather than Sub-Saharan Africa (1.89 pp).

### Figure S10 and the food-price table added to the deposit

Two further items were produced outside the earlier deposit and are now
included. `code/repro/make_figure_s10.py` regenerates Supplementary Figure S10
(in-season N capture efficiency as a buffering lever) and, like the Monte Carlo
ensemble, now runs on the ERA5 forcing rather than the model's built-in expert
climate profiles. Global year-10 yield loss across the NUE sweep is 10.92 %
(NUE 0.45), 7.57 (0.55), 5.18 (0.65), 3.41 (0.75, default), 2.09 (0.85) and
1.16 (0.95); the first 20 points of NUE deliver **59 %** of the total reduction
(published: 55 %). The Figure S10 caption gains the corresponding Sub-Saharan
African figures, whose year-10 loss falls from 21.0 % to 9.4 % across the same
NUE span, a 55 % reduction, and its statement of the realized S3 fertilizer
reduction moves from "approximately 18 %" to "approximately 20 %".

The regional food-price impacts quoted in the SI could not be reproduced from
the deposited model at all. They are now recomputed by
`code/repro/make_food_price_table.py` and written to
`data/food_price_response.csv`. The production-weighted global output-price
response is **+5.45 / +5.01 / +5.26 %** at years 1 / 10 / 30, with a year-1
regional span of 2.04 % (East Asia) to 9.21 % (FSU/Central Asia) and a year-10
span of 1.01 % (Latin America) to 10.27 % (FSU/Central Asia). The ε_F,PY channel
contributes **approximately 0.5 pp** production-weighted. The SI now states
explicitly that in unconstrained runs this is a reduced-form price index implied
by the linearized clearing condition, conditional on the assumed elasticities,
rather than a calibrated food-price forecast.

### Realized fertilizer reduction

The N-tonnage-weighted realized S3 fertilizer reduction over years 1-10 is
**20.3 %** under the corrected model; the manuscript's "approximately 18 %"
becomes 20 %.

### Supplementary figure numbering

`Figure_S14_OFRA_SSA_validation.png` was renamed `Figure_S13_...`: the v13 SI
carries Supplementary Notes 1-8 and Figures S1-S13, with the OFRA validation as
Figure S13. (Supplementary Note 8, added in this revision, documents the two
model corrections above and the reproducibility status of every figure.) The expert-versus-ERA5 climate comparison has no SI note or figure
of its own; `climate_comparison.py` is a deposit diagnostic reported in the
response letter. The earlier README and MANIFEST cited a nonexistent "Note 8 /
Figure S13" for it.

### Other deposit changes

- `run_canonical.py` docstring and console output updated to the corrected
  global values (were 4.33 / 5.58 / 5.95).
- Added `data/figure1_farm_gradient.json`, `data/figure1_soc_gradient.csv`,
  `data/figure2_soc_gradient.json`, `data/figure2_panels.json`,
  `outputs/zero_shock_invariance.csv`.
- Figures S6, S8, S9 and S11 regenerated. Figures S12 and S13 verified unchanged.
- Added `data/mc_ensemble/` (posterior, summary, probabilities, priors),
  `data/figS11_severity_sweep.json`, `data/figS10_nue_sensitivity.json` and
  `data/food_price_response.csv`.
- Figure S10 regenerated on the ERA5 forcing.
- `data/crop_response_calibration_table.csv` (Supplementary table 4 source) is
  carried over **without a generator**: the script producing its
  `y_no_synth_sim_tha` column was never deposited and was not recovered here.
  Its `ymax_calibrated_tha` column matches the corrected canonical run to six
  figures, and an equivalent no-synthetic-N run under the pre-fix versus the
  corrected model differs by ≤ 0.05 t ha⁻¹ in every region, so the table is
  unaffected by the two corrections. Disclosed rather than reconstructed.
- `figures/Figure_S5_flux_decomposition.png` cannot be regenerated from this
  deposit: the microbially-explicit 4-pool SOM scheme it illustrates is not
  included in the code. This gap predates v1.3 and is now disclosed in the
  README and SI.

## v12b — y_max reconciled to the simulation-calibrated ceiling
The reported calibrated ceiling y_max is now the value the coupled model actually
uses: `get_calibrated_ym`, a **Brent's-method root-finder** that adjusts y_max until
the model's **year-2 simulated yield** (under current N inputs and
the ERA5 climate) equals the FAOSTAT target — not a closed-form Mitscherlich inversion.
Canonical y_max (t/ha): NA 6.28, EU 6.12, EA 6.09, SA 3.64, SEA 4.87, LATAM 5.60,
SSA 3.88, FSU 4.29.

- **Table S4**: y_max → canonical; the no-synthetic-N column is now the **simulated**
  year-2 yield (`y_no_synth_sim`), e.g. NA 3.71, EU 2.29, SSA 1.26 (was closed-form
  5.08/4.42/1.28). Column relabeled pred → sim.
- **SI Methods**: closed-form `y_max = y_obs/(1−e^(−cN))` replaced with the numerical
  year-2 procedure; `yield_max_regional` noted as a legacy fallback parameter.
- **SI emergent-yield text**: "≈5.1 (NA), 4.4 (EU)" → "≈3.7 (NA), 2.3 (EU)" (SSA ≈1.3
  unchanged); still well above the empirical floors.
- **Figure S12**: regenerated as simulated N-response curves (FAOSTAT points now lie on
  the curves; canonical y_max asymptotes).
- **Figure S14 (OFRA)**: regenerated with SSA y_max=3.88 (was 3.47); conclusion revised
  from "low edge of the OFRA envelope" to "below the observed median but within the
  interquartile range." Observed OFRA envelope (n=364, B_gain>0) unchanged.
- **Response letter**: SSA ceiling 3.47→3.88; OFRA conclusion reworded to match.
- **Manuscript**: SSA SOC decline 6.2%→6.0% (canonical total-SOC value).
- Losses are reported as fractions of baseline; they vary slightly with y_max through
  residue return and SOM feedbacks and were NOT claimed to be y_max-independent — the
  reported losses use the y_max the canonical run actually applied.
