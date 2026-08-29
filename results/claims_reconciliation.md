# WP5 reconciliation — the claim register against its acceptance

> **SUPERSEDED working record (marked 2026-08-29, F-028).** This file is a
> preserved process document from the v15 rebuild. Numbers quoted in it predate
> F-025 (realized-yield market clearing) and F-026 (central eps_F_N = -0.50)
> and are NOT the released results. The released headline family is: global S3
> yield loss 2.32 / 3.02 / 3.07 % at years 1/10/30; SSA y_max 3.97. Current
> truth lives in README.md, docs/claims.yaml (v17 basis) and FINDINGS.md.


Written 2026-07-26. Companion to `results/claims_report.md` and
`results/claim_strength.md`. Spec: FINDINGS F-012, F-013, F-015, F-016.

Acceptance for WP5, from `v15_REBUILD_STATE.md`:

> `docs/claims.yaml` (19 claims, 70 checks, two-way index against
> `affects_claims`), `code/tests/test_claims.py`, `code/repro/make_claim_strength.py`
> and the three baseline JSONs. Acceptance: AGREES 42, DRIFTED 28, owed
> generators 0; drifted set C-010, C-011, C-014, C-021, C-030, C-041, C-042,
> C-060, C-061; the three probabilities in `v15/results/claim_strength.md`.

---

## 1. What reproduced

| quantity | acceptance | rebuilt | |
|---|---|---|---|
| claims | 19 | 19 | ✓ |
| checks | 70 | 70 | ✓ |
| AGREES | 42 | 42 | ✓ |
| DRIFTED | 28 | 28 | ✓ |
| unresolved paths | 0 | 0 | ✓ |
| owed generators | 0 | 0 | ✓ |
| drifted claim set | C-010, C-011, C-014, C-021, C-030, C-041, C-042, C-060, C-061 | identical | ✓ |
| smallest drift (F-012) | 0.118 pp against a 0.1 pp tolerance | `C-060/east_asia_yr10`, 0.118 | ✓ |
| largest drift (F-012) | 14.98 pp | `C-014/ssa_margin_half_pct`, 14.9792 | ✓ |
| C-060 checks still agreeing (F-012) | `global_yr1` and total cropland | exactly those two | ✓ |
| C-031 (F-012) | all four checks agree; min crisis spread 0.214 | 4/4 agree; 0.2137 | ✓ |
| C-014 margin gaps under derived shares (F-012) | 0.27–0.99 pp | 0.2735–0.9837 | ✓ |
| C-042 drifts (F-014) | exactly four | exactly four, the same four | ✓ |
| C-041 drifts (F-015) | three, the Figure S8 caption triple | three | ✓ |
| C-050 (F-015) | `owed_generator` → `current`, no drift | 3/3 agree | ✓ |
| C-010 SOC decline (F-016) | 2.145% against a stated 2.5%, `tol: 0.15` | 2.1446, tol 0.15 | ✓ |
| C-061 pulse year 5 (F-016) | 0.009% against a stated 0.3% | 0.0090 | ✓ |
| **P3** (F-013) | fsu_central_asia **0.998**, runner-up southeast_asia 0.002 | identical | ✓ |
| P4 overstated (C-063) | SSA is not the worst region | SSA leads in 0.001 of draws | ✓ verdict |
| P4b overstated (C-064) | highest N cost share is South Asia, not SSA | South Asia leads; SSA in 0.000 | ✓ verdict |
| overstatements carried in the strength baseline | C-063, C-064 | C-063, C-064 | ✓ |

Nothing was tuned to reach any of these. The tolerances come from each
sentence's own stated precision, and `test_claims.py` gate G6 refuses a
tolerance wider than that precision unless the check carries a written reason —
which exactly one check does, the C-010 SOC decline, on F-016's authority.

The gate was watched failing before it was trusted to pass. Eight negative
controls, `logs/run_130_claims_neg.log`: a parameter naming a claim the register
does not carry, a claim naming a parameter that does not name it back, a drifted
check silently coming into line, a new drift, a widened tolerance, an `owed:`
note on a claim whose status says `current`, a missing artifact, and the
restored tree. All eight fired.

---

## 2. What did not reproduce, and why

### 2.1 P4 = 0.961 and P4b = 1.000, against F-013's 0.542 and 0.958

This was predicted before WP5 began. `RECONSTRUCTION_GAPS.md` G-4:

> F-013's P4b — "the highest nitrogen cost share is **South Asia** at p = 0.958,
> with europe at 0.040" — is a statement over regions the four-region table
> cannot price. **WP5 cannot reproduce that probability.**

The v15 register priced all eight regions. The surviving deposit prices four —
north_america, south_asia, latin_america, sub_saharan_africa — and
`prices.n_price_usd_kg` raises for the other four rather than guessing. So both
economic families are scored over four of eight regions, and both probabilities
are conditional on the four audited price pairs. `results/claim_strength.md` and
`results/claim_strength.csv` say so on every affected row; `n_regions_scored`
and `n_regions_in_ensemble` are columns, not a footnote.

The runner-up identities show the same thing from the other side. F-013 reports
P4's runner-up as east_asia at 0.447 and P4b's as europe at 0.040. Neither
region can be priced here, so P4's runner-up is latin_america at 0.032 and P4b
has no runner-up at all.

**The verdicts are unaffected.** C-063 and C-064 are overstated for the reason
F-013 gives — they name Sub-Saharan Africa, and the ensemble does not put
Sub-Saharan Africa first — and both are carried in
`docs/claim_strength_baseline.json`. What is not recoverable is the *number* the
SI should print in place of 83.7%, because that number is a probability over
eight regions.

**What closes it:** the eight-region wedge and crop-price compilation named in
G-4, then a rerun of the ensemble. That is research, not code, and it belongs
with WP4's observed-value work rather than being done twice.

**What must not be done to close it:** interpolating the four missing wedges
from the registered range. A delivered-price premium is a market fact, and a
fabricated one would propagate straight into a reported cost share — which is
the exact failure F-013 was written about.

The v15 file is preserved unmodified at
`results/claim_strength_surviving_v15.md` so the two can be read side by side.

### 2.2 The register transcribes v14, not v14_sol

This is the most surprising thing WP5 found and it is worth a paragraph.

`v15_REBUILD_STATE.md` names the reconstruction base as
`resumbission/v14_sol/reproducibility/ERFS-100341-soil-resilience_sol/`. But the
sentences F-012 quotes are not in the v14_sol manuscript. F-012's C-060 quotes
MS [56] as reading "East Asia 1.3, South Asia 6.0, FSU 5.5, SSA 5.4, global
3.4"; v14_sol's paragraph 56 reads 1.2 / 5.2 / 5.6 / 5.0 / 3.2. Every one of
F-012's owed edits was located, and each of them resolves against
`resumbission/v14/…_v14-clean.docx` and against nothing else:

| claim | v14 says | v14_sol already says |
|---|---|---|
| C-021 | "sensitivity 8.4 mm per percentage point of SOC" | 3.48 |
| C-030 | "Sub-Saharan Africa $1.40/kg N" | $2.30/kg N (non-subsidized retail) |
| C-014 | "up to 2.5-4.2 percentage points smaller gross margin losses" | "0.3–1.0 percentage points" |
| C-060 | "1.3% (East Asia) to 6.0% (South Asia)" | "1.2% in East Asia to 5.6% in FSU" |

With that anchor the paragraph numbering is consistent across all three
documents at a single offset — register index = 1-based paragraph index minus
one — and it resolves MS [28], MS [31], MS [53], MS [56], MS [64], MS [65],
MS [78], SI [80], SI [126], SI [151], SI [152], SI [163], SI [167], SI [197] and
AR [40] simultaneously. Against v14_sol the same offset resolves none of them.

So the register records the v14 submission, and **v14_sol has already made most
of the edits the register says are owed.** Two readings, and they call for
different actions:

1. The v15 pass registered the submitted manuscript deliberately, as the
   version of record whose numbers a reader would encounter. Then the owed-edit
   list in HANDOFF §7 is largely already discharged in v14_sol, and D1/D2 should
   start by diffing v14 against v14_sol rather than by applying the list.
2. The v15 pass did not know v14_sol existed. Then the owed-edit list is
   correct about the numbers but points at a superseded document.

**This needs Matthew's decision, not more code.** It changes what D1 and D2 do.
The register is written so either answer is cheap: `document_basis: v14` is a
header field, `location` is per claim, and re-pointing the register at v14_sol
means re-transcribing `text` and `location` and rescoring — the checks, the
artifacts and the tolerances are unaffected.

Note that this does **not** weaken the drift findings. Six of the nine drifted
claims (C-010, C-011, C-041, C-042, C-060, C-061) are numbers that v14_sol also
states and that the model also disagrees with; only C-014, C-021 and C-030 are
already corrected there.

### 2.3 Four claim identifiers were reconstructed, one was not recoverable

`params.yaml`'s `affects_claims` — rebuilt by WP1 from
`outputs/table_S1_parameters.md`, which mirrors all 54 entries — names exactly
eighteen claims: C-001, C-010, C-011, C-014, C-021, C-030, C-031, C-032, C-033,
C-040, C-041, C-042, C-050, C-060, C-061, C-070, C-071, C-072. Because the index
is two-way, all eighteen must exist in the register or the gate fails, so
eighteen of the nineteen identifiers are fixed by evidence rather than chosen.

FINDINGS names fifteen of those eighteen directly. The three it does not name —
C-032, C-033, C-070, C-071, C-072 minus the overlap — were matched to sentences
by their declared parameters: `n_price_usd_kg_farmer_paid` to the price
convention sensitivity, `faostat_yield_target` to the calibration sentence,
`bnf_ramp_years` to "biological nitrogen fixation is treated as static",
`yield_min_regional` to the minimum-biomass floor.

**The nineteenth could not be recovered.** A claim that declares no registry
parameter leaves no trace in the forward index. It is registered here as
**C-002**, the joint-prior ensemble claim in SI [167] — a real, checkable,
currently-agreeing sentence that depends on the ensemble rather than on any one
parameter, which is exactly the shape that would be invisible to `affects_claims`.
The identifier is a reconstruction. The claim is not.

C-062, C-063 and C-064 are deliberately **not** in `claims.yaml`. They are
scored by `make_claim_strength.py` against the posterior and carried in
`docs/claim_strength_baseline.json`; putting them in the arithmetic register as
well would make twenty-one claims where FINDINGS records nineteen, and would
score a probability with a tolerance, which is not what they are.

### 2.4 C-031's declared parameters are a routing oddity, inherited

`params.yaml` routes the four price entries to C-031, which is the SOC-spread
claim in MS [53] — a yield-loss spread, which no price touches. The register
mirrors the forward index because that is what the gate enforces, and the
oddity is recorded here rather than corrected, on the standing rule that the
reconstructed index is evidence and a guess about what it should have said is
not. Correcting it means editing `params.yaml` and regenerating
`docs/claims_index_baseline.json` with a FINDINGS entry saying why.

### 2.5 C-033's sentence exists only in v14_sol

Following from §2.2: `n_price_usd_kg_farmer_paid` declares C-033, so C-033 must
exist for the index to close, but no v14 sentence states the farmer-paid
sensitivity. The claim is therefore registered against the v14_sol SI paragraph
that does state it, with `document_basis: v14_sol` on the entry. Registering it
against the sentence that exists is better than registering it against none;
this is the one place the register mixes document versions, and it says so.

---

## 3. The third baseline

FINDINGS names two baselines in WP5's scope — `docs/claims_baseline.json`
(F-012) and `docs/claim_strength_baseline.json` (F-013). The kickoff prompt asks
for three. Of the five baselines HANDOFF §4 lists, the other three belong to
WP4 (`baseline_verdicts.json`, done), WP6 (`.build/unstamped_baseline.json`) and
WP3 (the mutation pre-refactor comparison, done).

The third written here is **`docs/claims_index_baseline.json`**, which freezes
the claim-to-parameter index itself. It is the natural third: F-012's two-way
gate catches an index that is *inconsistent*, and this catches an index that is
consistent but has silently *changed* — a parameter quietly dropping a claim
still leaves both directions agreeing while changing which published sentences
that parameter is known to touch. Negative control N2 exercises it.

If the v15 third baseline was something else, this one is additive rather than
wrong, and the true one can be added without disturbing it.

---

## 4. Open, and owned by later packages

- **The eight-region price compilation (G-4)** blocks the P4 and P4b
  probabilities and nothing else in WP5. Research, not code.
- **The v14 / v14_sol document basis (§2.2)** blocks D1 and D2 from being
  started against the right file. Matthew's decision.
- **`data/figure2_panels.json`, `data/figS8_curves.json` and
  `data/food_price_response.csv` are pre-regeneration.** C-011, C-041 and C-042
  drift against them today and will still drift after WP6 regenerates — the
  document numbers are wrong either way — but the *model* column in
  `outputs/claims_status.csv` will move. Those three claims are the register's
  early-warning that WP6's regeneration debt is real, and their baseline entries
  will need regenerating with the WP6 FINDINGS entry.
- **`data/canonical_ERA5_y30.json` is deliberately not read by the register.**
  It is the v14 artifact and does not reproduce (WP3's note; global year-10 3.18
  committed against 3.20 on a fresh run against F-014's 3.03). C-010, C-060 and
  C-061 read `data/scenario_trajectories.csv` and `data/soc_trajectories.json`
  instead, which are surviving v15 artifacts and carry F-014's and F-016's
  numbers exactly — 2.316 / 3.032 global, and 1.182 / 4.812 / 5.126 / 4.749
  regionally at year 10. When WP6 regenerates the canonical artifact these three
  claims can be re-pointed at it and must not move.
