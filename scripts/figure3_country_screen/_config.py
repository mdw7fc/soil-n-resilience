"""Shared configuration for the Figure 3 country-level soil-buffer / exposure
screen pipeline.

All paths, constants, thresholds, and FAOSTAT/SoilGrids URLs live here so every
script imports from a single source of truth.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout: scripts at scripts/figure3_country_screen/, data at
# data/figure3_country_screen/{raw,processed,outputs}, figures at
# figures/figure3_country_screen/.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]

DIR_RAW       = REPO_ROOT / "data" / "figure3_country_screen" / "raw"
DIR_PROCESSED = REPO_ROOT / "data" / "figure3_country_screen" / "processed"
DIR_OUTPUTS   = REPO_ROOT / "data" / "figure3_country_screen" / "outputs"
DIR_FIGURES   = REPO_ROOT / "figures" / "figure3_country_screen"
DIR_DOCS      = SCRIPT_DIR / "docs"

for d in (DIR_RAW, DIR_PROCESSED, DIR_OUTPUTS, DIR_FIGURES, DIR_DOCS):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Year window — locked per SPEC §1
# ---------------------------------------------------------------------------
YEAR_START = 2018
YEAR_END   = 2020
YEARS      = list(range(YEAR_START, YEAR_END + 1))
YEAR_LABEL = f"{YEAR_START}-{YEAR_END} mean"

# ---------------------------------------------------------------------------
# Thresholds — locked per SPEC §2
# ---------------------------------------------------------------------------
N_INTENSITY_CAP       = 300.0  # kg N / ha; values above this are scaled to 1.0

# --- Combined-index parameters (kept for SI sensitivity cross-check only) ---
EXPOSURE_W_INTENSITY  = 0.5
EXPOSURE_W_RELIANCE   = 0.5
EXPOSURE_LOW          = 0.33   # baseline combined-index thresholds; used
EXPOSURE_HIGH         = 0.66   # only by sensitivity scripts now

# --- Phase-1-locked classification: Modified Two-Pathway Max-Rule (per SPEC §2.2) ---
HIGH_INTENSITY_KG_HA      = 150.0   # pathway A floor
HIGH_RELIANCE             = 0.70    # pathway B reliance floor
MATERIAL_STAKE_KG_HA      = 25.0    # pathway B intensity floor (Phase 1 lock)
LOW_INTENSITY_KG_HA       = 50.0    # AND-rule for low exposure
LOW_RELIANCE              = 0.30
STAKE_FLOOR_SENSITIVITY   = (10.0, 25.0, 50.0)  # SI sensitivity sweep

# --- Re-export filter (tightened per Phase 0 finding) ---
REEXPORT_RATIO_FLAG       = 3.0
REEXPORT_TRADE_FLOOR_T    = 50_000.0   # Imports + Exports > 50 kt N
REEXPORT_AC_FLOOR_T       = 10_000.0   # Apparent_consumption > 10 kt N

# Buffer thresholds: pre-specified terciles, computed within cropland-bearing
# countries at runtime. Tercile cut points written into map_methods_summary.txt.

# ---------------------------------------------------------------------------
# FAOSTAT bulk-download URLs (normalized format → long table with Year column)
# ---------------------------------------------------------------------------
FAOSTAT_FERT_NUTRIENT_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Inputs_FertilizersNutrient_E_All_Data_(Normalized).zip"
)
FAOSTAT_LANDUSE_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Inputs_LandUse_E_All_Data_(Normalized).zip"
)

FAOSTAT_FERT_ZIP = DIR_RAW / "faostat_fertilizers_nutrient.zip"
FAOSTAT_LANDUSE_ZIP = DIR_RAW / "faostat_landuse.zip"

# ---------------------------------------------------------------------------
# FAOSTAT element / item codes we care about
#   (Element/Item names taken from FAOSTAT "Fertilizers by Nutrient" + "Land Use".)
# ---------------------------------------------------------------------------
FERT_ITEM_N            = "Nutrient nitrogen N (total)"
FERT_ELEMENT_USE       = "Agricultural Use"
FERT_ELEMENT_PRODUCTION= "Production"
FERT_ELEMENT_IMPORT    = "Import quantity"
FERT_ELEMENT_EXPORT    = "Export quantity"
FERT_ELEMENTS_TONNES   = (FERT_ELEMENT_USE, FERT_ELEMENT_PRODUCTION,
                          FERT_ELEMENT_IMPORT, FERT_ELEMENT_EXPORT)

LANDUSE_ITEM_CROPLAND  = "Cropland"
LANDUSE_ELEMENT_AREA   = "Area"
LANDUSE_CROPLAND_UNIT_FACTOR = 1000.0  # FAO unit "1000 ha" → multiply to get hectares

# ---------------------------------------------------------------------------
# Region groupings (FAO-style, used for panel d)
# ---------------------------------------------------------------------------
REGION_GROUPS = {
    "Sub-Saharan Africa": [
        "AGO","BDI","BEN","BFA","BWA","CAF","CIV","CMR","COD","COG","COM","CPV",
        "DJI","ERI","ETH","GAB","GHA","GIN","GMB","GNB","GNQ","IOT","KEN","LBR",
        "LSO","MDG","MLI","MOZ","MRT","MUS","MWI","MYT","NAM","NER","NGA","REU",
        "RWA","SDN","SEN","SHN","SLE","SOM","SSD","STP","SWZ","SYC","TCD","TGO",
        "TZA","UGA","ZAF","ZMB","ZWE",
    ],
    "North Africa & West Asia": [
        "ARE","ARM","AZE","BHR","CYP","DZA","EGY","ESH","GEO","IRN","IRQ","ISR",
        "JOR","KWT","LBN","LBY","MAR","OMN","PSE","QAT","SAU","SYR","TUN","TUR",
        "YEM",
    ],
    "Central Asia": [
        "KAZ","KGZ","TJK","TKM","UZB",
    ],
    "South Asia": [
        "AFG","BGD","BTN","IND","LKA","MDV","NPL","PAK",
    ],
    "East & Southeast Asia": [
        "BRN","CHN","HKG","IDN","JPN","KHM","KOR","LAO","MAC","MMR","MNG","MYS",
        "PHL","PRK","SGP","THA","TLS","TWN","VNM",
    ],
    "Europe": [
        "ALB","AND","AUT","BEL","BGR","BIH","BLR","CHE","CZE","DEU","DNK","ESP",
        "EST","FIN","FRA","FRO","GBR","GIB","GRC","HRV","HUN","IRL","ISL","ITA",
        "LIE","LTU","LUX","LVA","MCO","MDA","MKD","MLT","MNE","NLD","NOR","POL",
        "PRT","ROU","RUS","SJM","SMR","SRB","SVK","SVN","SWE","UKR","VAT","XKX",
    ],
    "Northern America": [
        "BMU","CAN","GRL","SPM","USA",
    ],
    "Latin America & Caribbean": [
        "ABW","AIA","ARG","ATG","BES","BHS","BLZ","BOL","BRA","BRB","CHL","COL",
        "CRI","CUB","CUW","CYM","DMA","DOM","ECU","FLK","GLP","GRD","GTM","GUF",
        "GUY","HND","HTI","JAM","KNA","LCA","MEX","MSR","MTQ","NIC","PAN","PER",
        "PRI","PRY","SLV","SUR","SXM","TCA","TTO","URY","VCT","VEN","VGB","VIR",
    ],
    "Oceania": [
        "ASM","AUS","COK","FJI","FSM","GUM","KIR","MHL","MNP","NCL","NFK","NIU",
        "NRU","NZL","PCN","PLW","PNG","PYF","SLB","TKL","TON","TUV","VUT","WLF",
        "WSM",
    ],
}

ISO3_TO_REGION = {iso: region for region, isos in REGION_GROUPS.items() for iso in isos}


def iso3_to_region(iso3: str) -> str:
    return ISO3_TO_REGION.get(iso3, "Unassigned")
