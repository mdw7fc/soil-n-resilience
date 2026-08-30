"""Build countries_master.csv — the FAO ↔ M49 ↔ ISO3 crosswalk + region tag.

Output:
  data_processed/countries_master.csv
    iso3, fao_area_code, m49_code, fao_name, region

Strategy: pull FAOSTAT's own area-codes table (which gives FAO Area Code + M49),
then map M49 → ISO3 via a curated mapping. Aggregates ("World", "Africa", ...) and
non-country areas are tagged with iso3 = NA and excluded from downstream joins.

Notes / gotchas (per SPEC §6):
  - FAOSTAT has historical entities (USSR, Yugoslav SFR, Sudan (former)). We keep
    their FAO codes so historical data can be retrieved, but tag them
    `historical=True` and exclude from current-period joins.
  - China: FAO splits into "China, mainland", "China, Taiwan Province of",
    "China, Hong Kong SAR", "China, Macao SAR", and aggregate "China". We keep
    the four sub-entities and drop the aggregate from the country-level table.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import (  # noqa: E402
    DIR_PROCESSED,
    DIR_RAW,
    FAOSTAT_FERT_ZIP,
    iso3_to_region,
)


# Curated M49 → ISO3 overrides for FAO's idiosyncratic areas.
# (FAO sometimes uses M49=156 for China-aggregate, M49=158 for Taiwan, etc.;
#  we resolve to the sub-entity ISO3 codes the analysis uses downstream.)
M49_OVERRIDE_ISO3 = {
    "156": "CHN",   # China, mainland (FAO uses 156 for "China, mainland")
    "158": "TWN",   # China, Taiwan Province of
    "344": "HKG",   # China, Hong Kong SAR
    "446": "MAC",   # China, Macao SAR
    "830": None,    # Channel Islands aggregate → drop
    "736": "SDN",   # Sudan (former) — historical, will be tagged
    "729": "SDN",   # Sudan (post-2011)
    "728": "SSD",   # South Sudan
    "891": None,    # Serbia and Montenegro (former) → drop, use SRB / MNE
    "688": "SRB",
    "499": "MNE",
}

# FAO Area Codes ≥ 5000 are continent / regional / economic-grouping aggregates
# (e.g. 5000 World, 51000 Africa, 57060 EU-28, 58020 Land-Locked Developing).
# These are excluded from the country master.
AGGREGATE_FAO_AREA_MIN = 5000


def m49_to_iso3_fallback(m49: str) -> str | None:
    """Lightweight pycountry-free M49 → ISO3 lookup using a packaged table.

    Reads a small static mapping shipped in this script (below). Avoids adding
    a pycountry dependency to the venv for Phase 0.
    """
    return M49_TO_ISO3.get(str(m49).zfill(3))


# Static M49 → ISO3 table (UNSD M49, abbreviated to current sovereign states).
# Only ~250 rows; pasted inline so the pipeline has zero external dependencies
# beyond pandas/numpy/requests/matplotlib.
M49_TO_ISO3: dict[str, str] = {
    "004":"AFG","008":"ALB","010":"ATA","012":"DZA","016":"ASM","020":"AND",
    "024":"AGO","028":"ATG","031":"AZE","032":"ARG","036":"AUS","040":"AUT",
    "044":"BHS","048":"BHR","050":"BGD","051":"ARM","052":"BRB","056":"BEL",
    "060":"BMU","064":"BTN","068":"BOL","070":"BIH","072":"BWA","074":"BVT",
    "076":"BRA","084":"BLZ","086":"IOT","090":"SLB","092":"VGB","096":"BRN",
    "100":"BGR","104":"MMR","108":"BDI","112":"BLR","116":"KHM","120":"CMR",
    "124":"CAN","132":"CPV","136":"CYM","140":"CAF","144":"LKA","148":"TCD",
    "152":"CHL","156":"CHN","158":"TWN","162":"CXR","166":"CCK","170":"COL",
    "174":"COM","175":"MYT","178":"COG","180":"COD","184":"COK","188":"CRI",
    "191":"HRV","192":"CUB","196":"CYP","203":"CZE","204":"BEN","208":"DNK",
    "212":"DMA","214":"DOM","218":"ECU","222":"SLV","226":"GNQ","231":"ETH",
    "232":"ERI","233":"EST","234":"FRO","238":"FLK","239":"SGS","242":"FJI",
    "246":"FIN","248":"ALA","250":"FRA","254":"GUF","258":"PYF","260":"ATF",
    "262":"DJI","266":"GAB","268":"GEO","270":"GMB","275":"PSE","276":"DEU",
    "288":"GHA","292":"GIB","296":"KIR","300":"GRC","304":"GRL","308":"GRD",
    "312":"GLP","316":"GUM","320":"GTM","324":"GIN","328":"GUY","332":"HTI",
    "334":"HMD","336":"VAT","340":"HND","344":"HKG","348":"HUN","352":"ISL",
    "356":"IND","360":"IDN","364":"IRN","368":"IRQ","372":"IRL","376":"ISR",
    "380":"ITA","384":"CIV","388":"JAM","392":"JPN","398":"KAZ","400":"JOR",
    "404":"KEN","408":"PRK","410":"KOR","414":"KWT","417":"KGZ","418":"LAO",
    "422":"LBN","426":"LSO","428":"LVA","430":"LBR","434":"LBY","438":"LIE",
    "440":"LTU","442":"LUX","446":"MAC","450":"MDG","454":"MWI","458":"MYS",
    "462":"MDV","466":"MLI","470":"MLT","474":"MTQ","478":"MRT","480":"MUS",
    "484":"MEX","492":"MCO","496":"MNG","498":"MDA","499":"MNE","500":"MSR",
    "504":"MAR","508":"MOZ","512":"OMN","516":"NAM","520":"NRU","524":"NPL",
    "528":"NLD","531":"CUW","533":"ABW","534":"SXM","535":"BES","540":"NCL",
    "548":"VUT","554":"NZL","558":"NIC","562":"NER","566":"NGA","570":"NIU",
    "574":"NFK","578":"NOR","580":"MNP","581":"UMI","583":"FSM","584":"MHL",
    "585":"PLW","586":"PAK","591":"PAN","598":"PNG","600":"PRY","604":"PER",
    "608":"PHL","612":"PCN","616":"POL","620":"PRT","624":"GNB","626":"TLS",
    "630":"PRI","634":"QAT","638":"REU","642":"ROU","643":"RUS","646":"RWA",
    "652":"BLM","654":"SHN","659":"KNA","660":"AIA","662":"LCA","663":"MAF",
    "666":"SPM","670":"VCT","674":"SMR","678":"STP","682":"SAU","686":"SEN",
    "688":"SRB","690":"SYC","694":"SLE","702":"SGP","703":"SVK","704":"VNM",
    "705":"SVN","706":"SOM","710":"ZAF","716":"ZWE","724":"ESP","728":"SSD",
    "729":"SDN","732":"ESH","740":"SUR","744":"SJM","748":"SWZ","752":"SWE",
    "756":"CHE","760":"SYR","762":"TJK","764":"THA","768":"TGO","772":"TKL",
    "776":"TON","780":"TTO","784":"ARE","788":"TUN","792":"TUR","795":"TKM",
    "796":"TCA","798":"TUV","800":"UGA","804":"UKR","807":"MKD","818":"EGY",
    "826":"GBR","834":"TZA","840":"USA","850":"VIR","854":"BFA","858":"URY",
    "860":"UZB","862":"VEN","876":"WLF","882":"WSM","887":"YEM","894":"ZMB",
}


def main() -> int:
    print("Building countries_master.csv ...")

    # Pull FAOSTAT's own AreaCodes table (smaller than re-deriving from data file)
    with zipfile.ZipFile(FAOSTAT_FERT_ZIP) as zf:
        with zf.open("Inputs_FertilizersNutrient_E_AreaCodes.csv") as fh:
            raw = pd.read_csv(fh, encoding="latin-1")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.rename(columns={"Area Code": "fao_area_code",
                              "M49 Code": "m49_raw",
                              "Area": "fao_name"})

    # FAO's M49 column ships as "'004" or similar — strip leading apostrophe.
    raw["m49_code"] = raw["m49_raw"].astype(str).str.replace("'", "", regex=False).str.zfill(3)
    raw["m49_code"] = raw["m49_code"].where(raw["m49_code"].str.match(r"^\d{3}$"), other=None)

    # Drop FAO aggregate areas (continents, World, regional groupings).
    raw["is_aggregate"] = raw["fao_area_code"].astype(int) >= AGGREGATE_FAO_AREA_MIN
    aggregates = raw[raw["is_aggregate"]].copy()
    countries = raw[~raw["is_aggregate"]].copy()

    # Map M49 → ISO3, with overrides
    def to_iso3(m49: str | None) -> str | None:
        if m49 is None or pd.isna(m49):
            return None
        if m49 in M49_OVERRIDE_ISO3:
            return M49_OVERRIDE_ISO3[m49]
        return m49_to_iso3_fallback(m49)

    countries["iso3"] = countries["m49_code"].apply(to_iso3)
    countries["region"] = countries["iso3"].apply(
        lambda x: iso3_to_region(x) if isinstance(x, str) else "Unassigned"
    )

    # Tag historical / unmapped
    HISTORICAL_NAMES = (
        "Belgium-Luxembourg", "Czechoslovakia", "Ethiopia PDR",
        "Netherlands Antilles (former)", "Pacific Islands Trust Territory",
        "Serbia and Montenegro", "USSR", "Yugoslav SFR", "Sudan (former)",
    )
    countries["historical"] = countries["fao_name"].isin(HISTORICAL_NAMES)
    countries["unmapped"] = countries["iso3"].isna()

    out = countries[
        ["iso3", "fao_area_code", "m49_code", "fao_name", "region",
         "historical", "unmapped"]
    ].sort_values(["region", "fao_name"]).reset_index(drop=True)

    out_path = DIR_PROCESSED / "countries_master.csv"
    out.to_csv(out_path, index=False)

    n = len(out)
    n_hist = int(out["historical"].sum())
    n_unmap = int(out["unmapped"].sum())
    n_unassigned = int((out["region"] == "Unassigned").sum())
    print(f"  rows: {n}")
    print(f"  historical entities: {n_hist}")
    print(f"  unmapped (no ISO3): {n_unmap}")
    print(f"  region=Unassigned: {n_unassigned}")
    print(f"  written: {out_path}")

    if n_unassigned - n_unmap - n_hist > 0:
        print("\n  WARN: countries with ISO3 but no region — review:")
        bad = out[(out["region"] == "Unassigned") & ~out["unmapped"] & ~out["historical"]]
        print(bad[["iso3", "fao_name"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
