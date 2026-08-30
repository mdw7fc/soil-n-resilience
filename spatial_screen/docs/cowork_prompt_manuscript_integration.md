# Cowork prompt — Figure 4 manuscript integration

> Copy everything below the rule into Claude cowork.

---

I'm integrating a new Figure 4 (a global soil-buffer / fertilizer-shock screen) into a Nature Food manuscript. The figure has been built and audited externally; my task here is purely the manuscript edit, with all decisions already locked.

## Files in this conversation

**Manuscript and SI** (the live docx files I have open):
- Main text: `Wallenstein-Manning_Nature-Food_manuscript_v5.docx` (or whichever is the current main-text version — I'll tell you which one to edit)
- SI: `Wallenstein-Manning_Nature-Food_SI_v5.docx` (or current SI version)

**Figure assets** (already produced; insert as a new Figure 4):
- 4-panel assembled: `figure4-resilience-map/figures/fig4_phase2_assembled.png`
- Individual panels (if the journal needs them split):
  - `figure4-resilience-map/figures/fig4_phase2_panel_a.png` — cropland-weighted soil organic N buffer proxy
  - `figure4-resilience-map/figures/fig4_phase2_panel_b.png` — fertilizer-shock exposure
  - `figure4-resilience-map/figures/fig4_phase2_panel_c.png` — combined classification
  - `figure4-resilience-map/figures/fig4_phase2_panel_d.png` — cropland area by class × region

**Methods reference text** (auto-generated; don't paste verbatim into the manuscript, but use as a fact-check source for any methods detail):
- `figure4-resilience-map/data_processed/map_methods_summary.txt`
- `figure4-resilience-map/data_processed/area_accounting_note.txt`

---

## Locked decisions — apply verbatim, do not re-derive

1. **Buffer classification:** unweighted-across-cropland-bearing-country terciles. Low buffer = bottom tercile of cropland-bearing countries by cropland-weighted soil organic N buffer proxy. Frame as a country-level policy/resilience screen, not hectare-level vulnerability. **Do not write** "17.4% of global cropland is vulnerable." **Do write** "17.4% of global FAOSTAT cropland lies in countries classified as low buffer / high exposure."

2. **Cropland-area-weighted terciles → SI sensitivity only.** Report focal class drops 17.4% → 4.85% under cropland-area-weighted terciles, driven primarily by China shifting from low buffer / high exposure to intermediate. Frame this as threshold sensitivity in the country-level screen, not as invalidation.

3. **Buffer metric = SOC/C:N (cropland-weighted soil organic N buffer proxy).** Do not call it a direct mineralizable-N estimate. SOC-only is an SI sensitivity (focal drops to ~3–4%, demonstrating the C:N adjustment is consequential).

4. **Exposure rule = modified two-pathway max-rule:**
   - High exposure: N intensity ≥ 150 kg N ha⁻¹ **OR** (import reliance ≥ 0.70 AND N intensity ≥ 25 kg N ha⁻¹).
   - Low exposure: N intensity < 50 kg N ha⁻¹ AND import reliance < 0.30.
   - Otherwise intermediate.
   - Keep the 25 kg N ha⁻¹ material-stake floor. SI reports stake-floor sensitivity at 10 / 25 / 50.

5. **Map framed narrowly:** "mechanism-specific resilience screen" / "soil-buffer / fertilizer-exposure screen" — **not** "food-security vulnerability index." Explicitly state the figure does not include poverty, irrigation, conflict, crop mix, fiscal capacity, food import dependence, or adaptive policy response.

---

## A. Main manuscript edits

### A1. Replace current main-text Figure 4 with the new map figure

Insert `fig4_phase2_assembled.png` as the new Figure 4. Replace the existing Figure 4 in place; update its caption (use draft below).

### A2. Move the current monthly N uptake capture efficiency figure (current Figure 4) to the SI

Place it under the SI sensitivity / robustness section. Renumber as Supplementary Figure (next available number — see audit at end).

### A3. Insert a new Results subsection after Figure 3 and before the section that previously held Figure 4 (the monthly N uptake capture efficiency content)

**Use this draft text. Lightly revise for flow with surrounding paragraphs but do not change the locked numerical claims, the framing, or the country lists.**

> **Global geography of soil buffering and fertilizer-shock exposure**
>
> To visualize the geography of this mechanism, we developed a country-level soil-buffer / exposure screen combining a cropland-weighted soil organic N buffer proxy with fertilizer-shock exposure, defined by synthetic N intensity and fertilizer import reliance. This screen is not a food-security vulnerability index; it isolates the overlap between limited soil organic N buffering capacity and structural exposure to fertilizer shocks. Under the country-level classification, 17.4% of global FAOSTAT cropland lies in countries classified as low buffer / high exposure. This class includes high-input systems with limited soil organic N buffering, including China and parts of Central Asia, and import-dependent systems with moderate fertilizer use, including Brazil, Australia, Thailand and South Africa. A cropland-area-weighted buffer-threshold sensitivity reduces the focal class to 4.85%, primarily by shifting China to intermediate, indicating that the country-level screen identifies a concentrated but threshold-sensitive geography of fertilizer-shock exposure.

### A4. Figure 4 caption (verbatim draft; lightly polish punctuation only)

> **Figure 4. Global geography of soil organic N buffering and fertilizer-shock exposure.** **a,** Country-level cropland-weighted soil organic N buffer proxy, calculated from SoilGrids SOC and C:N and weighted by MIRCA2000 cropped-area distribution. **b,** Fertilizer-shock exposure based on synthetic N application intensity and fertilizer import reliance. **c,** Country-level classification of soil-buffer / exposure classes. Low buffer is defined as the bottom tercile of cropland-bearing countries by cropland-weighted buffer proxy; high exposure is defined as synthetic N intensity ≥ 150 kg N ha⁻¹ or import reliance ≥ 0.70 with N intensity ≥ 25 kg N ha⁻¹. **d,** FAOSTAT cropland area in each country class, stacked by region. The figure is a mechanism-specific resilience screen, not a food-security vulnerability index.

---

## B. SI edits

### B1. Add a new SI subsection — title "**Spatial soil-buffer / fertilizer-exposure screen**"

Place it under the existing "Empirical benchmarking and sensitivity checks" section, or as a new Supplementary Note if that section doesn't exist. Use the seven elements below verbatim; combine into flowing paragraphs as needed but preserve all numerical claims and threshold definitions.

**B1.1 Data sources.**
- SoilGrids SOC and C:N for soil organic N buffer proxy.
- MIRCA2000 maximum annual cropped-area distribution as spatial weights for cropland-weighted buffer aggregation.
- FAOSTAT *Fertilizers by Nutrient* for synthetic N use, fertilizer production / import / export, and apparent consumption.
- FAOSTAT cropland area for the N-intensity denominator and panel d area accounting.
- Natural Earth boundaries.

**B1.2 Area accounting note (verbatim):**
> MIRCA2000 cropped-area weights were used only to compute cropland-weighted country means of the soil organic N buffer proxy. FAOSTAT cropland areas were used for fertilizer-intensity calculations and panel d area summaries to maintain consistency with the fertilizer-use denominator. MIRCA cropped area and FAOSTAT cropland area therefore need not match exactly because they represent different accounting systems and temporal/spatial definitions.

**B1.3 Classification methods.**
- *Soil buffer:* Low = bottom tercile of cropland-bearing countries by cropland-weighted soil organic N buffer proxy. High = top tercile. Middle tercile = intermediate unless an exposure/buffer combination rule overrides.
- *Exposure:* High = N intensity ≥ 150 OR (import reliance ≥ 0.70 AND N intensity ≥ 25). Low = N intensity < 50 AND import reliance < 0.30. Other = intermediate.
- *Combined classes:* High buffer / low exposure; high buffer / high exposure; low buffer / low exposure; **low buffer / high exposure** (focal); intermediate; data missing.

**B1.4 Buffer-tercile sensitivity (verbatim):**
> We used unweighted country terciles for the main buffer classification because the figure is designed as a country-level policy screen: each cropland-bearing country receives one classification based on its cropland-weighted soil organic N buffer proxy. We separately evaluated cropland-area-weighted buffer thresholds to test whether the classification was sensitive to the cropland distribution among countries. Under the main country-tercile rule, 17.4% of global FAOSTAT cropland fell in low-buffer / high-exposure countries. Under cropland-area-weighted terciles, the focal class decreased to 4.85%, largely because China shifted from low buffer / high exposure to intermediate. We therefore interpret the mapped focal area as a country-level screening result rather than a hectare-level estimate of global cropland vulnerability.

**B1.5 SOC-only sensitivity (verbatim):**
> Replacing the soil organic N proxy with SOC alone reduced the focal class to approximately 3–4%, indicating that stoichiometric adjustment is consequential for a nitrogen-buffering screen.

**B1.6 Stake-floor sensitivity** (insert as one paragraph or table):
- 10 kg N ha⁻¹ floor: 19.2% focal cropland.
- 25 kg N ha⁻¹ floor (locked): 17.4% focal cropland.
- 50 kg N ha⁻¹ floor: 14.0% focal cropland.

Conclude: the focal-class pattern is stable across the tested material-use floors.

**B1.7 Buffer–exposure correlation** (one or two sentences):
> Buffer and exposure are not collinear. Unweighted Pearson r = +0.24, cropland-weighted Pearson r = −0.40 across 174 cropland-bearing countries. The screen therefore captures distinct mechanistic axes rather than a single underlying gradient.

### B2. NUE figure move

Move the current main Figure 4 (monthly N uptake capture efficiency) to the SI under the sensitivity / robustness section.
- Renumber as the next available Supplementary Figure (audit at end).
- Rename: **"Supplementary Figure Sx. Monthly N uptake capture efficiency as a buffering lever."**
- Caption fixes:
  - Title says "monthly N uptake capture efficiency," not generic "nitrogen use efficiency."
  - **Delete the stray "T" currently at the end of the caption.**
  - Add a clarifying clause: "This parameter represents monthly N uptake capture efficiency and is not equivalent to fertilizer recovery efficiency."

---

## C. Cleanup edits — apply throughout main manuscript and SI

Apply each as a precise find-and-replace where the wording matches; otherwise revise locally to satisfy the rule. Confirm each in the changelog.

| # | Rule | Before → After |
|---|---|---|
| C1 | Disambiguate "gross margin" | "gross margin" → "**gross margin over fertilizer cost**" wherever it refers to the modeled metric (not where it's used generically about agricultural economics). |
| C2 | Reframe n=8 R² language as descriptive | "In a univariate regression…" → "**As a descriptive diagnostic across the eight modeled regions, …**" (main text). In SI, **avoid foregrounding "R² = 1.00"**; phrase as "remaining modeled variance is absorbed by base-yield and structural economic terms." |
| C3 | Replace "Broadbalk validation" → "**Broadbalk benchmarking**" everywhere. |
| C4 | Remove "uniquely" from "building SOM uniquely combines three functions." → "building SOM combines three functions." |
| C5 | Monte Carlo wording | (a) "posterior" → "**joint-prior ensemble**" or "**ensemble distribution**" (whichever fits the sentence — the simulation has no Bayesian update). (b) "P = 1.000" → "**in all 1,000 draws**". |

---

## D. Cross-reference audit

After inserting the new Figure 4 and moving the NUE figure to the SI:

1. Search the main text for every "Figure 4" reference and confirm it now points to the new map figure.
2. Search for every reference to the monthly N uptake capture efficiency content and update it to the new Supplementary Figure number.
3. Renumber any later "Supplementary Figure Sx" references that shifted because of the NUE figure insertion.
4. Search for "Supplementary Note" cross-references and verify each still resolves correctly after the new Spatial Screen subsection is added.
5. Build a list of every cross-reference touched and include it in the changelog.

---

## E. Outputs to produce

1. **Updated main manuscript** as a new docx (next version in the lineage — e.g. `Wallenstein-Manning_Nature-Food_manuscript_v6.docx` if v5 is current). Track-changes ON.
2. **Updated SI** as a new docx (next version in the lineage). Track-changes ON.
3. **Changelog** as a separate short document listing:
   1. New Figure 4 inserted (note source path of PNG used).
   2. NUE figure moved to SI as Supplementary Figure Sx (specify number).
   3. Spatial Screen methods added to SI (specify section/page).
   4. Sensitivity results added (buffer tercile, SOC-only, stake-floor, correlation).
   5. Cleanup edits applied (one bullet per item C1–C5 with count of replacements).
   6. Cross-references audited (list of every "Figure" / "Supplementary Figure" / "Supplementary Note" reference checked, with old → new where it changed).

---

## Style guardrails

- Minimize em-dashes; use only when clearly the best option.
- Do not use "uniquely" or "vulnerability" in the new figure language unless explicitly licensed above.
- Do not call the screen a food-security index anywhere.
- Preserve the manuscript's existing reference style; if SoilGrids, MIRCA2000, FAOSTAT, or Natural Earth need to be cited and aren't in the bibliography yet, add citations and flag them in the changelog.
