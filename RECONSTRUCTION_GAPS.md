# v15 reconstruction gaps — opened by WP1

**Written 2026-07-25.** Every item here is a place where a surviving v15
artifact records a value the reconstruction base does not contain, or contains
differently. Nothing below has been guessed. Where a number could not be
recovered, the registry carries the code's value and the artifact is marked as
owing an edit, per the standing rule: **where the code and a document disagree,
the code is right and the document owes the edit.**

Each gap names what would close it.

---

## G-1 — `whc_sensitivity`: 3.48 in code, 3.5 in Table S1

`outputs/table_S1_parameters.md` prints this row as `3.5`. FINDINGS F-008
records it as "1.16 mm per 100 mm soil per percentage point SOC over a 300 mm
profile is 3.48, **registered as 3.5**", and F-012's C-021 says "the registry
carries `whc_sensitivity` = 3.5". The reconstruction base carries 3.48
(`parameter_registry.WHC_MM_PER_SOC_PCT_30CM`), which is the exact arithmetic.

**Measured.** Registering 3.5 and wiring it moves sixteen of the 123 canonical
fields — every region's `loss_yr10` and `loss_yr30`, at the 1e-5 relative level.
`loss_yr1` does not move, because at year 0 the SOC change the sensitivity
multiplies is zero. The global losses still round to 2.31 / 3.18 / 3.29.

**Resolved as:** `value: 3.48`. Registering 3.5 would have failed WP1's own
acceptance test, and WP1 is defined as a refactor that changes no number.

**Consequence for the manuscript.** C-021's correction to MS [31], MS [28] and
AR [40] is 8.4 → 3.48, which is 3.5 at the two significant figures the
manuscript states. Table S1 will print 3.48 on regeneration.

**Closes it:** deciding whether the v15 registry genuinely held 3.5. If it did,
the change belongs in WP2 alongside the production-path recalibration, where
canonical numbers are expected to move and the move is measured, not in a
refactor that claims to move nothing.

---

## G-2 — `bnf_potential`: Table S1's range is the superseded mechanism's

Table S1 prints the range `15 to 35`. The live per-region values, which
`RegionParams` carries and which `run_canonical.py` copies into the canonical
CSV, are the `BNF_COMPONENTS` derivation and span 13.75 to 37.73.
`monthly_model_v3:647` names `15-35 kg/ha/yr` explicitly as **the old managed
transition**, so Table S1's range is the superseded mechanism's, not the live
one.

**Resolved as:** the live derived values, with `superseded_by`,
`superseded_note` and a `documented_as` recording the 15–35 range and where it
came from. The canonical `bnf` column is unchanged.

**Closes it:** nothing. This one is understood; Table S1 owes the edit.

---

## G-3 — `yield_max_regional`: Table S1's eight legacy fallbacks are lost

Table S1 prints `3.453 to 6.09` and describes the row as a "fallback ceiling
used only when no monthly calibration is available". The base sets
`yield_max_regional=0.0` in all eight regions — the fallback is never populated,
and the reported ceiling is the one `coupled_monthly.get_calibrated_ym` solves
against the FAOSTAT target (3.636 to 6.277 in the canonical run, which is a
different set of numbers again).

**Resolved as:** eight zeros, matching the code, with a `documented_as` note.
The eight legacy values are not reconstructed, because inventing them would put
a number in the registry that no code produces.

**Closes it:** `git log -p` on `soil_n_model.py` in the deposit repo, which
still has its `.git`. The values predate `20defb2`. Worth one command in WP2.

---

## G-4 — `prices.py`: four of eight regions are unrecovered — THE REAL GAP

This is the only gap that blocks later work.

Table S1 registers an eight-region price system:

| entry | Table S1 | recoverable from the base |
|---|---|---|
| `n_price_wedge` | 0.8 to 2.63, eight regions | four regions |
| `crop_price_usd_t` | 200 to 385, eight regions | four regions |
| `n_price_wedge_bounds` | eight regions, printed in full | recovered verbatim |
| `crop_price_bounds` | eight regions, printed in full | recovered verbatim |
| `n_benchmark_usd_kg` | 0.876 | recovered |
| `urea_n_fraction` | 0.46 | recovered |

The base carries `parameter_registry.REGIONAL_PRICES` with **four** audited
regions only: north_america, south_asia, latin_america, sub_saharan_africa.
`prices.py` itself was created in v15 and did not survive.

**Resolved as:** the four audited regions are registered, with wedges computed
as `n_price / n_benchmark_usd_kg` so that
`prices.n_price_usd_kg` reproduces the v14 values bit-identically (1.10, 1.20,
1.15, 2.30). Europe, east_asia, southeast_asia and fsu_central_asia are absent
from the registry, and `prices.n_price_usd_kg` **raises** for them rather than
guessing. `PRICED_REGIONS` names the four.

**Why it does not break WP1.** No price appears in the canonical artifact. The
123-field diff is unaffected, and F-011 already records that the six price
leaves score INERT "by construction, not because they are irrelevant", because
the probe's fingerprint excludes the margin outcomes.

**Why it blocks later work.** F-013's P4b — "the highest nitrogen cost share is
**South Asia** at p = 0.958, **not** sub-Saharan Africa", with europe at 0.040 —
is a statement over regions the four-region table cannot price. WP5 cannot
reproduce that probability, and WP4's benchmark rows that touch cost shares
cannot be evaluated for the four missing regions.

**Closes it, in order of cost:**

1. `git log -p` on the deposit repo for a commit that touched the price table.
   The v15 tables were new, so this likely fails, but it is cheap.
2. The eight-region wedges and crop prices are a *research* compilation — World
   Bank Pink Sheet urea plus regional delivered-price premia, and FAOSTAT
   producer prices for the dominant cereal mix, 2019–2023. This is the same
   kind of work WP4's observed-value compilation is, and it should be done
   there, once, with sources recorded, rather than twice.

**Do not** interpolate the four missing wedges from the range `0.8 to 2.63`.
A delivered-price premium is a market fact, not an interpolation, and a
fabricated one would propagate into a reported cost share.

---

## G-5 — `n_price_usd_kg_farmer_paid`: 0.35 in Table S1, 0.39 in code

Table S1 prints `south_asia 0.35`; the base carries
`SOUTH_ASIA_FARMER_PAID_N_PRICE = 0.39`. The parameter is used only in a
labelled one-region SI sensitivity, so nothing reported in the main experiment
depends on it.

**Resolved as:** 0.39, the code's value, with a `documented_as` note.

**Closes it:** whether the v15 pass had a source for 0.35 (the Indian urea MRP
is a published, dated number, so this is a lookup rather than a judgment). Fold
it into WP4's compilation.

---

## G-6 — four model constants are still unregistered

`RESIDUE_C_FRACTION` (0.45), `WATER_STRESS_GAIN_SAT_SOC_PCT` (1.0),
`WATER_STRESS_SOFTPLUS_EPS_MM` (3.0) and `WATER_STRESS_MIN_FACTOR` (0.30)
remain literals in `parameter_registry.py`, and `BNF_COMPONENTS` — which is what
fixation is actually computed from (F-007) — is not registered either.

They are **not** registered in WP1 on purpose: adding an entry changes the leaf
count, and WP3's mutation harness is calibrated against exactly 56 leaves.
Registering them is a deliberate decision with a measured consequence, which
belongs after WP3 has a baseline to compare against.

**Note.** `WHC_MM_PER_SOC_PCT_LOW` moved from 2.32 to 2.3 as a side effect of
reading the bound from the registry's `declared_absolute_bounds`, which F-007
gives as `[2.3, 8.4]`. It is used only in two prose strings in
`make_parameter_ledger_sol.py` and reaches no model number.
