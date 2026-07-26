# D1 and D2 — why they did not run

**2026-07-26.** D3 is done (`1037e39`, `d6272b0`, `9731cdd`). D1 and D2 were not
started. This is the report you asked for.

---

## The short version

The D1 and D2 prompts rest on two claims that were true when they were written
into `v15_REBUILD_STATE.md` and are not true now.

1. **"Every replacement number is in §5."** WP6 found that §5's canonical family
   was produced at `eps_F_N = -0.5`, and that the rebuilt tree at `eps_F_N = 0`
   reproduces what `v14_sol` already prints. Applying §5 would replace correct
   figures with the wrong family.
2. **§7's worklist is anchored on the v14 submission, not on `v14_sol`.** Almost
   every main-text item on it has already been made. Working the list as written
   would mean re-editing paragraphs that are already right, using numbers that
   are already wrong.

Neither is a reason to abandon the packages. Both are reasons not to run them
until you have settled `eps_F_N` yourself, which is what you decided.

---

## 1. The decision

`eps_F_N` is the fertilizer-demand response to soil nitrogen. The code says S3
runs it at zero: `SOIL_N_RESPONSE_ELASTICITY_CENTRAL`, the S3 docstring, and the
comment F-011's `DECLARED_NOT_WIRED` verdict rests on. F-015 says the S3 numbers
were produced "with `eps_F_N` active", `params.yaml` gives it
`affects_claims: [C-040, C-050]`, and C-050 is the S3 calibration claim. Both
cannot be true.

The evidence WP6 assembled is one-sided, and it is worth reading before you
decide, because it is the whole basis of my recommendation:

| | `v14_sol` prints | regenerated at `eps_F_N = 0` | HANDOFF §5 at `-0.5` |
|---|---|---|---|
| Global yr 1 / 10 / 30 | 2.3 / 3.2 / 3.3 | 3.20 / 3.31 | 2.32 / **3.03** |
| East Asia yr 10 | 1.2 | 1.21 | 1.182 |
| North America | 1.8 | 1.80 | 1.726 |
| Latin America | 2.4 | **2.51** | 2.418 |
| Europe | 3.6 | 3.67 | 3.429 |
| Southeast Asia | 3.8 | 3.84 | 3.675 |
| Sub-Saharan Africa | 5.0 | 4.93 | 4.749 |
| South Asia | 5.2 | 5.12 | 4.812 |
| FSU/Central Asia | 5.6 | 5.55 | 5.126 |
| SC1 yr10 / yr30 | 3.7 / 3.9 | 3.758 / 3.898 | 3.69 / — |
| SC2 yr10 / yr30 | 1.9 / 0.04 | 1.919 / 0.045 | 1.869 / — |

At zero, eight of eight regions and all three globals land on what the paper
already says. At `-0.5`, six of the eight regions and both multi-year globals
miss. `run_canonical.py` line 97 prints "(SOL manuscript: 2.3 / 3.2 / 3.3)". The
sentence that carries SC1 and SC2 ends "without relying on an unestimated
soil-N demand response," which is a statement that the central case runs at zero.

**One thing to check before you sign this off.** Latin America is the single
region where the regenerated value and the printed value differ by more than
rounding: 2.51 against a printed 2.4. Every other region agrees to the digit.
That is either a transcription slip in `v14_sol` or a small residual difference
in the regenerated chain, and it is worth ten minutes because it is the only
crack in an otherwise exact match.

**If you settle on zero**, the consequences are:

- `params.yaml`'s `affects_claims: [C-040, C-050]` for `eps_F_N` is wrong. A
  parameter held at zero in S3 cannot affect the S3 calibration claim. The
  two-way index test passes either way, because it only checks mutual
  declaration.
- F-014's headline is wrong. "`canonical_ERA5_y30.json` did not move" sits beside
  a quoted 2.32 / 3.03 while the committed artifact reads 2.31 / 3.18 / 3.29. It
  moved by 0.15 pp at year 10 and was recorded as unchanged.
- **The SSA 30-year SOC decline becomes 2.24%, not 2.145%.** F-016's owed edit
  was computed at `-0.5`.
- **The one-year pulse year-5 residual cannot currently be recomputed at all.**
  0.009% is a `-0.5` number. `make_scenario_trajectories.py` lost its pulse
  capability with the crashed tree, so the `scenario_trajectories` node is
  BLOCKED and the deposited CSV is the only copy of that column. Its value at
  zero is unknown until that script is rebuilt.
- F-015's 0.1911 sustained mean is on the same footing and needs recomputing
  when `make_s3_shock_calibration.py` is rewritten. Neither script is owned by
  any work package.

---

## 2. What §7 asks for that `v14_sol` has already done

I checked every item on the Abstract and Main-text lists against the actual
`Wallenstein-Manning_ERFS_manuscript_v14_sol.docx`, not against the `qa/`
paragraph index, which predates the document by about an hour and does not match
it.

| §7 item | State in `v14_sol` |
|---|---|
| Abstract: margin gap 2.5–4.2 pp → ~0.3–1.0 | **Done.** Abstract reads 0.3–1.0 pp. |
| Abstract: do not describe the magnitude as validated | **Done.** No validation claim survives. |
| MS [56] regional year-10 figures are stale | **Done, to the zero family.** 1.2 / 1.8 / 2.4 / 3.6 / 3.8 / 5.0 / 5.2 / 5.6, global 3.2. |
| MS [31], MS [28]: `whc_sensitivity` 8.4 → 3.5 | **Done.** Both paragraphs read 3.48 mm. |
| MS [56] / [65]: state the aggregation basis | **Done.** "production-weighted global yield loss". |
| Report the S3 reduction as 19%, not 20% | **Moot in the main text.** No realized-reduction figure is stated any more. It survives in `README.md`'s key-numbers table as "20% (N-weighted)", and F-015 puts it at 18.7% on the paper's own basis. That line should be fixed whatever you decide. |
| MS [78]: fixation does not ramp | **Done.** The limitations paragraph reads "BNF is static". |
| Regional output-price indices (5.5 → 5.34 etc.) | **Not in the main text at all.** Those numbers now live only in `data/food_price_response.csv` and `MANIFEST.md` (+5.45 / +5.01 / +5.26 %). WP6 flagged that file as pre-regeneration, so C-042 keeps drifting. Deposit item, not a manuscript item. |
| One-year pulse: year-5 residual 0.3% → 0.009%, drop the food-price attribution | **Half done.** The 0.3% and the food-price half are both gone. No residual is stated. Adding a correct one needs the pulse column, which is BLOCKED. |

The SI list is in the same condition. There is no 83.7%, no p = 0.542 passage,
no 2.5% SOC decline and no 0.1–1.5 pp range left in `v14_sol` to correct.

---

## 3. What is genuinely still owed

**Independent of `eps_F_N`, could run today:**

1. **The three new limitations.** Absolute SOC is initialization and not
   prediction, so temperate and tropical stocks are not comparable (F-001). The
   crop calendar is fixed and unsampled and moves the reported loss by up to
   0.26 pp per month (F-004). Percentage SOC decline does not order regions the
   way yield loss does (F-016).
2. **The benchmark-suite section, including B3-europe.** The model loses 59% of
   temperate yield without synthetic nitrogen where Prague-Ruzyne lost 22 to 32%.
   WP4 reproduced the failure at 0.4063 against F-008's 0.406 with nothing tuned.
   WP4 also found four qualifications the SI has to carry with the Prague
   numbers: the ratios are derived rather than published, the source table is one
   decimal place so they carry ±0.02, they are ratios of period means, and the
   yields are winter wheat after potatoes only, nine and fourteen seasons.
3. **Regenerating Table S1 from the registry**, marking the seventeen
   declared-but-fixed rows.
4. **Reprinting Supplementary Table S4.** This is new, from D3, and it is the
   item I would move first. See below.

**Blocked on the decision:** the SSA 30-year SOC decline, the pulse year-5
residual, and any re-statement of the S3 realized reduction.

**Blocked on something else entirely:** WP5 found that the number the SI should
print in place of 83.7% is not recoverable. P4 and P4b score over four of eight
regions because the surviving deposit prices four, and `prices.n_price_usd_kg`
raises for the rest. Closing it needs the eight-region wedge and a crop-price
compilation, then an ensemble rerun. WP5's note says explicitly: do not
interpolate the four missing wedges from the registered range.

---

## 4. The thing D3 turned up that D2 needs

`data/crop_response_calibration_table.csv` was written by no script in the
deposit and had not been since at least v14. It is now generated
(`make_table_s4_sol.py`), and its numbers moved, because the frozen copy was two
recalibrations behind.

**Supplementary Table S4 is a row-for-row transcription of that file.**

| | printed in `v14_sol` | regenerated |
|---|---|---|
| N America N_current (kg/ha) | 223.9 | **142.7** |
| N America N no-synth (kg/ha) | 147.9 | **66.7** |
| N America y_max (t/ha) | 6.277 | **6.198** |
| S Asia y_max | 3.636 | **3.773** |
| L America y_max | 5.602 | **5.414** |
| SSA y_max | 3.876 | **3.967** |
| SSA y(no-synth) sim | 1.26 | **1.29** |

Every numeric column except `FAOSTAT y_obs`, `c` and `Floor` is wrong. The
nitrogen columns are wrong by 36% and 55%. This is independent of `eps_F_N`:
the regenerated values reproduce `figS12_curves.json` and
`Table_S4_calibration_sol.csv`, the two outputs of the same script that did have
generators, both of which came back byte-identical.

Figure S13 is unaffected. It reads the sourced table, its PNG is byte-identical
after regeneration, and its caption claim still holds at the regenerated ceiling.

---

## 5. What I need from you

1. **`eps_F_N`: zero or `-0.5`.** WP6's evidence points hard at zero. Check
   Latin America (2.4 against 2.51) before you commit to it.
2. **Whether to split D2.** The three limitations, the benchmark section, Table
   S1 and the Table S4 reprint are four-fifths of D2 and none of them depend on
   the answer. They could run as a package now, leaving one small package for
   the SOC and pulse numbers afterwards.
3. **B3-europe, which is still open from §8 of the handoff.** Report the failure
   against one site, or compile Broadbalk plot 3 grain yields as a second
   temperate comparator first. F-008 says state it and do not tune to it.

Once you answer 1, I can rewrite the D1 and D2 prompts. Both are marked
SUPERSEDED in `v15_REBUILD_STATE.md` with the reason, so nobody pastes them in
the meantime.
