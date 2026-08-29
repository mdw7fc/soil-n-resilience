# Parameter ledger (completed SOL audit)

The authoritative CSV contains **582 semantic entries**. The companion numeric-literal audit contains **2216 code-line entries**.

## Scope and completeness rule

A live parameter is any primitive that can change a state, calibration, forcing, scenario, weight, statistic or acceptance decision without changing program structure. The ledger therefore includes climate vectors, crop calendars, calibration targets, solver settings, priors, seeds and test tolerances as well as named model coefficients.

Plot colors, fonts, line widths, dimensions and label positions are excluded as non-scientific formatting. Every numeric literal in model and reproduction code is separately listed in `NUMERIC_LITERAL_AUDIT_sol.csv` with that disposition visible.

## Duplicate-definition decisions

- SOC stock-to-percentage conversion, residue carbon fraction and water-stress smoothing constants now have one source in `parameter_registry.py`.
- Baseline BNF is derived once from legume fraction, net N credit and free-living fixation. `RegionParams.bnf_potential` is populated from that derived registry value; it is not a second primitive.
- Regional `yield_max_regional` is a disabled zero sentinel. The only reported regional ceiling is the ERA5 runtime Brent calibration.
- ERA5 JSON is the sole canonical climate forcing; repeated loader implementations delegate to one validated function.
- Fertilizer cost share and N price in yield units are derived from audited nitrogen price, crop price, N rate and modeled yield.
- The OFRA benchmark reads the generated Table S4 values; it no longer hardcodes the SSA ceiling or no-synthetic-N yield.

## Audit summary

- Derived/calibrated entries explicitly marked: 30
- Entries explicitly fixed and not varied: 212
- Fixed-but-unvaried entries are limitations, not silently implied sources of certainty.

## Entry counts

- BNF derived: 8
- BNF primitive: 32
- SOM default: 12
- acceptance threshold: 6
- algorithm/solver: 24
- analysis design: 11
- benchmark design: 6
- climate forcing: 48
- crop calendar: 16
- crop default: 9
- economic default: 13
- economic scenario: 65
- feedback default: 7
- monthly N default: 14
- regional biophysical: 144
- regional economic: 64
- regional price: 12
- shared physical/economic: 7
- spatial-screen design: 20
- uncertainty prior: 48
- yield calibration: 16
