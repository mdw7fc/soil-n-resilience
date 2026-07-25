# Figure 4 — Global geography of soil N buffering and fertilizer-shock exposure

**Status:** Signed off, with revisions. Phase 0 prototype next.
**Owner:** Wallenstein
**Date:** 2026-04-25 (signed off after Wallenstein revision pass)
**Replaces:** current main-text Figure 4 (NUE → moved to SI)

---

## 0. Scope statement (must appear verbatim in main text and SI)

> This figure maps a mechanism-specific resilience screen. It does not include poverty, irrigation, crop mix, conflict, fiscal capacity, food-import dependence, or adaptive policy response, and should not be interpreted as a food-security vulnerability index.

---

## 1. Figure-level decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Layout | 4-panel small multiples (a, b, c, d) | Avoids bivariate-map cartographic traps; preserves combined message via panel c |
| Spatial unit (panel a) | Cropland raster, masked to non-cropland — **only if Phase 0 prototype passes** | Visualizes within-country heterogeneity in soil organic N buffer |
| Spatial unit (panels b, c) | Country choropleth | Exposure data is country-level; mixing units in panel c is incoherent |
| Country aggregation rule | **Cropland-area-weighted** mean for any soil variable summarized to country | Unweighted means overstate the role of vast, low-cropland soil regions (boreal, desert) |
| Projection | Equal Earth | Area-faithful; standard for global agricultural figures |
| Year window | **2018–2020 mean** (label: "Pre-shock structural exposure, 2018–2020 mean") | Dampens single-year FAOSTAT noise; anchors figure as pre-shock structural screen, not a 2022-shock postmortem |
| Classification (panels a, b) | Quintiles within cropland-bearing countries (panel b) and within cropland pixels (panel a) | Robust to skewed distributions; understandable |
| Classification (panel c) | 2×2 pre-specified classification with explicit middle-tercile/middle-exposure handling (see §4) | "Pre-specified" replaces "absolute" — the buffer threshold is data-driven tercile, the exposure threshold is absolute index |
| Panel c class labels | **Descriptive only** ("Low buffer / high exposure" etc.); avoid "resilient" / "at risk" | Avoids food-security-vulnerability claims |
| Panel d | Stacked bar: global cropland area in each of the 4 extreme classes + intermediate, **stacked by region** | Embeds geography into the bar chart |

---

## 2. Variable definitions

### 2.1 Soil organic N buffer proxy (panel a)

**Default formula:**
```
buffer_proxy_raw = SOC_stock_0_30cm / soil_C_to_N
```
Inputs: SoilGrids 2.0 SOC stock (t C ha⁻¹) and SoilGrids total soil N (t N ha⁻¹), with `C:N = SOC / total_N`.

**Interpretation language (use everywhere):**
> Approximates **cropland soil organic N stock**. It is a **resilience-stock proxy** consistent with the manuscript framing that SOC is the modeled stock and mineralizable organic N is the mechanism. **It does not estimate mineralization rate directly.**

**Panel label:** "Cropland soil organic N buffer proxy"

**Rendered fields:**
- `buffer_proxy_raw` — continuous, units of `t N ha⁻¹` (the C:N division collapses to total N, but we keep the explicit form so the SOC vs total-N decomposition is auditable)
- `buffer_percentile` — within-cropland percentile (0–100), used for visual mapping in panel a
- `buffer_country_cwm` — country-level cropland-weighted mean of `buffer_proxy_raw`

**Sensitivity (SI only):** SOC-only buffer (drop the `/ C:N` term).

### 2.2 Fertilizer-shock exposure (panel b)

**Sub-index 1: Synthetic N application intensity**
```
N_intensity_raw    = N_fertilizer_use_tonnes × 1000 / cropland_area_ha    [kg N ha⁻¹]
N_intensity_scaled = pmin(N_intensity_raw, 300) / 300                     [0–1]
```
Cap at 300 kg N/ha to prevent outliers from compressing the scale. Documented in SI.

**Sub-index 2: Net N-fertilizer import reliance**
```
Apparent_consumption = Production + Imports − Exports              [tonnes N]
Net_imports          = max(0, Imports − Exports)                   [tonnes N]
Import_reliance      = clip(Net_imports / Apparent_consumption, 0, 1)
```

**Re-export hubs (tightened per Phase 0 finding):** flagged where ALL three hold:
- `(Imports + Exports) / Apparent_consumption > 3`
- `Imports + Exports > 50 kt N`           (absolute trade volume floor)
- `Apparent_consumption > 10 kt N`        (avoid noisy ratios on tiny markets)

**Treatment:** flag in `exposure_country.csv` + caption footnote. **No hatching** unless the map is materially misleading.

**Phase-0-locked classification rule — Modified Two-Pathway Max-Rule (replaces the
combined-index rule originally specified):**

```
HIGH exposure if either pathway holds:
   (A) input-intensity pathway:
       N_intensity_raw ≥ 150 kg N ha⁻¹
   (B) import-dependence pathway:
       Import_reliance ≥ 0.70  AND  N_intensity_raw ≥ 25 kg N ha⁻¹
                                          ─────────────────────────
                                          "material fertilizer stake"

LOW exposure if BOTH:
       N_intensity_raw < 50 kg N ha⁻¹  AND  Import_reliance < 0.30

INTERMEDIATE: everything else
DATA_MISSING: N_intensity_raw or Import_reliance unavailable
```

The 25 kg N/ha "material stake" floor on pathway B is what distinguishes the
modified max-rule from the raw max-rule. Without it, countries with negligible
N use (Niger 0.55 kg/ha, DR Congo 1.5) get classified high simply because their
tiny supply is fully imported. Phase 0 confirmed this artifact; the floor
removes it while retaining genuinely import-dependent intensified systems
(Brazil 81 kg/ha + 92% reliance, Australia 49 kg/ha + 78% reliance).

**Sensitivity (always reported in SI):**
- Stake-floor variants: 10 / 25 / 50 kg N/ha
- Rule comparison: original combined ≥0.66 · combined ≥0.50 · raw max-rule ·
  modified max-rule · exposure terciles
- Intensity cap: 200 / 300 / 400 kg N/ha
- Combined-index weights {0.7/0.3, 0.5/0.5, 0.3/0.7} (kept for reviewer cross-check)

### 2.3 Combined classification (panel c)

**Buffer (3 levels — pre-specified terciles of cropland-bearing countries):**
- Low: bottom tercile
- Moderate: middle tercile
- High: top tercile

**Exposure (3 levels — absolute index thresholds):**
- Low: Exposure < 0.33
- Moderate: 0.33 ≤ Exposure < 0.66
- High: Exposure ≥ 0.66

**Panel c displays 5 classes:**

| Class | Buffer | Exposure | Color (suggested) | Label |
|---|---|---|---|---|
| 1 | High | Low | desaturated blue | High buffer / low exposure |
| 2 | High | High | purple | High buffer / high exposure |
| 3 | Low | Low | light grey-blue | Low buffer / low exposure |
| 4 | Low | High | warm red | **Low buffer / high exposure** |
| 5 | any moderate | any moderate / any moderate | neutral grey | Intermediate |

**Captions and Results text refer to the focal class as "low-buffer / high-exposure cropland."** Avoid "resilient" and "at risk."

### 2.4 Panel d — cropland area by class, stacked by region

Regions (FAO grouping):
- Sub-Saharan Africa
- North Africa & West Asia
- South Asia
- East & Southeast Asia
- Europe
- Northern America
- Latin America & Caribbean
- Oceania

Output: stacked bar chart. x = vulnerability class (5 bars). Bar height = global cropland area (Mha). Stack fill = region.

**Companion table (always produced):** `top_countries_by_class.csv` — top 15 countries by cropland area within each of the 5 classes. Used for Results-text writing and for sanity-checking the classification.

---

## 3. Data sources & access

| Layer | Source | Resolution | Phase 0 use | Phase 1 use |
|---|---|---|---|---|
| SOC stock 0–30 cm | SoilGrids 2.0 | 250 m | coarsened country-level pull (5 km or pre-aggregated) | full 250 m raster for panel a |
| Total soil N 0–30 cm | SoilGrids 2.0 | 250 m | coarsened country-level pull | full 250 m raster for panel a |
| Cropland mask (country aggregation) | EarthStat harvested area 2020 | 5 arcmin | for cropland-weighted soil aggregation | same |
| Cropland mask (raster visual) | ESA WorldCover 2021 (class 40) | 10 m | not used | for panel a visual |
| N fertilizer use | FAOSTAT "Inputs / Fertilizers by Nutrient" → element "Agricultural Use", item "Nutrient nitrogen N (total)" | country-year | core input | same |
| N fertilizer trade & production | FAOSTAT "Inputs / Fertilizers by Nutrient" → elements "Production", "Import quantity", "Export quantity" | country-year | core input | same |
| Cropland area (country) | FAOSTAT "Land Use" → item "Cropland" | country-year | core input | same |
| Country boundaries | Natural Earth 1:50m admin-0 | vector | regions / labels only | choropleth geometry |
| Country code crosswalk | manually curated `countries_master.csv` (M49 ↔ ISO3 ↔ FAO codes) | — | core input | core input |

**Year handling:** all FAOSTAT layers averaged over 2018–2020. Soil layers are time-invariant within the analytical window.

---

## 4. Threshold sources to verify before code finalization

- [ ] Mueller et al. (2012) *Nature* — N intensity yield-gap-closure thresholds
- [ ] Lassaletta et al. (2014) *ERL* — country-level N use intensity distributions
- [ ] Erisman et al. (2008) *Nat. Geosci.* — global N synthesis baseline
- [ ] FAO (2022) *World Fertilizer Trends and Outlook to 2025* — import-reliance language

If thresholds shift after the lit pass, update §2 before any classification runs.

---

## 5. Data manifest (traceability requirement)

`data_processed/00_data_manifest.csv` columns:

| col | meaning |
|---|---|
| layer_id | short ID, e.g. `soc_stock_0_30` |
| source | dataset name |
| source_url | direct download URL |
| version | dataset version / vintage |
| access_date | YYYY-MM-DD |
| native_units | as-shipped units |
| native_resolution | spatial / temporal |
| transformations | comma-separated (e.g. `clip_to_cropland_mask, country_cwm`) |
| output_file | path to processed file |
| output_units | post-transformation units |
| notes | gotchas, missing-data handling |

Every layer used in the figure must have a row. Every value in the figure must trace through this manifest to a file on disk. Per project standards.

**Auto-generated companion:** `data_processed/map_methods_summary.txt` — a plain-text dump of the exact formulas, thresholds, year window, caps, weights, and per-region data coverage used to produce the published figure. Generated at the end of the pipeline. Will be pasted into the SI methods section.

---

## 6. Country handling — gotchas

- **FAOSTAT vs ISO3:** FAO uses M49 + custom codes; harmonize via curated crosswalk with explicit handling of: Sudan/South Sudan (post-2011), Serbia/Montenegro, China/Taiwan/Hong Kong (FAO splits "China, mainland" vs "China, Taiwan Province of"), USSR successors.
- **Missing fertilizer data:** countries with no FAOSTAT N fertilizer record → flagged as `data_missing` and shown in light grey in panels b, c. Listed in SI table.
- **Small island states:** flagged separately; small cropland area means they will not register in panel d but should not be silently dropped.
- **Re-export hubs:** flagged per §2.2.

**Coverage check (kill criterion):** if `data_missing` countries collectively hold >20% of any continental cropland area, flag and revisit imputation strategy.

---

## 7. Outputs (files we will produce)

```
data_processed/
  00_data_manifest.csv
  countries_master.csv                 # ISO3, name, fao_code, region, cropland_ha, missing flags
  buffer_country.csv                   # iso3, buffer_cwm, buffer_tercile, buffer_class
  exposure_country.csv                 # iso3, n_intensity_raw, n_intensity_scaled,
                                       # production, imports, exports, apparent_consumption,
                                       # net_imports, import_reliance,
                                       # exposure_combined, exposure_class, reexport_flag
  vulnerability_country.csv            # iso3, buffer_class, exposure_class, panel_c_class
  panel_d_summary.csv                  # panel_c_class × region × cropland_Mha
  top_countries_by_class.csv           # top 15 countries by cropland area within each class
  threshold_sensitivity_summary.csv    # ± 20 % threshold shifts → reclassification rates
  map_methods_summary.txt              # auto-generated methods text

outputs/
  panel_a_buffer_raster.tif            # cropland-masked buffer proxy (Phase 1 only)
  panel_b_exposure_country.gpkg
  panel_c_vuln_country.gpkg

figures/
  fig4_panel_a.pdf                     # Phase 1 only
  fig4_panel_b.pdf
  fig4_panel_c.pdf
  fig4_panel_d.pdf
  fig4_assembled.pdf                   # patchwork; pre-Illustrator polish
```

---

## 8. Kill criteria (commit to these now)

1. Combined map is visually muddy / not intuitive at the assembled stage.
2. `data_missing` countries hold >20% of cropland in any continental region.
3. Buffer–exposure correlation > 0.6 globally (collapsed story).
4. Threshold sensitivity: shifting "high" thresholds by ±20 % reclassifies >25 % of cropland area → fragile; revert to 2-panel fallback or add SI sensitivity panel.
5. Scope creep into poverty / irrigation / crop mix / conflict.
6. **Data-artifact dominance:** if the low-buffer / high-exposure class is dominated by countries with poor FAOSTAT fertilizer-data quality or re-export artifacts, **do not publish panel c** — drop to 2-panel fallback.

Fallback: 2-panel figure (cropland SOC buffer map; fertilizer import-reliance map) + short text bridge.

---

## 9. Build phases (prototype-first)

### Phase 0 — country-level prototype, no heavy raster

Build:
1. `countries_master.csv`
2. `exposure_country.csv` (full)
3. `buffer_country.csv` — coarsened country-level SOC + total-N pull (e.g. SoilGrids 5 km aggregated), cropland-weighted; this is provisional but informative
4. `vulnerability_country.csv`
5. Draft panel b (country exposure ranked + class colors), panel c (country vulnerability classes), panel d (cropland area × class × region) — all renderable without geopandas via tables + matplotlib bars in Phase 0; geopandas choropleths come in Phase 1
6. `top_countries_by_class.csv`
7. `threshold_sensitivity_summary.csv`

**Phase 0 checkpoint (decision gate before Phase 1):**
- Does the exposure index behave sensibly?
- Are major countries classified plausibly (US, India, China, Brazil, Nigeria, Egypt, etc.)?
- Do missing-data countries cluster anywhere consequential?
- Does panel d tell a useful story?
- Do all 6 kill criteria pass?

If yes → Phase 1. If no → fallback or redesign.

### Phase 1 — buffer aggregation refinement + classification refit + choropleths

After Phase 0 sign-off (lock-in 2026-04-25 with modified max-rule):

- **Aggregation method (Phase 1 substitution):** Country **median** pixel value
  for buffer proxy, instead of mean. This is robust to peat/forest outliers
  (Phase 0 saw mean-based ranking pulled by mineral-soil regions). A proper
  cropland-weighted aggregation requires a global cropland-fraction raster;
  none was openly downloadable without registration during Phase 0 (probed
  EarthStat, GAEZ, OpenLandMap, NASA SEDAC, FAO GLC-SHARE, etc. — most behind
  Earthdata login or 403/404). Median is documented as a Phase 1 prototype
  simplification; Phase 2 should replace with cropland-weighted mean once a
  cropland mask is acquired (Earth Engine, Earthdata login, or paid GAEZ
  bulk).
- **Re-export filter:** apply tightened criteria (§2.2).
- **Classification:** apply modified two-pathway max-rule (§2.2).
- **Sensitivity:** stake-floor + rule comparison tables (always produced).
- **Choropleths:** geopandas-based panels b, c; stacked-bar panel d.
- **Panel a (cropland raster):** deferred to Phase 2 (requires SoilGrids 250 m
  tile mosaic + cropland mask).

### Phase 2 — sensitivity + manuscript integration

- SOC-only vs SOC/CN buffer
- Exposure weight sensitivity {0.7/0.3, 0.3/0.7}
- N intensity cap at 200 / 300 / 400
- ±20 % threshold shifts
- One SI sensitivity table; only escalate to extra maps if patterns diverge meaningfully
- Move current NUE figure to SI
- New main-text Figure 4 + short Results paragraph + SI methods + scope caveat (§0)

---

## 10. Sign-off (Wallenstein, 2026-04-25)

| # | Question | Resolution |
|---|---|---|
| 1 | Buffer proxy formula | `SOC / C:N`, named **soil organic N buffer proxy**; not mineralizable N |
| 2 | Panel c rule | combined exposure index, threshold ≥ 0.66 |
| 3 | Year window | 2018–2020 mean ("pre-shock structural exposure") |
| 4 | Cropland mask | EarthStat for aggregation; ESA WorldCover only for raster visual in Phase 1 |
| 5 | Re-export hubs | flag in data + caption footnote; no hatching unless materially misleading |
| 6 | Panel d | stacked by region; plus `top_countries_by_class.csv` |

**Build order:** Phase 0 country-level prototype → checkpoint → Phase 1 raster + choropleths → Phase 2 sensitivity + manuscript integration.
