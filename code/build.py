#!/usr/bin/env python3
"""build.py -- the deposit's build graph (F-009, F-014).

WHAT THIS IS

Every generated artifact in the deposit is declared here together with the
script that makes it and the files that script reads. The graph is then the
only place that knows how a number got onto disk. Three questions become
answerable mechanically:

  1. Is this artifact current, or was it written before the inputs it claims
     to derive from?            ->  node status OK / STALE
  2. Does anything produce this file?                ->  orphan scan
  3. Does anything produce the files this node reads? ->  unsourced scan

Question 2 and question 3 are not the same question, and F-009 was found by
asking both. `data/climate_swap_comparison.csv` was a file nobody produced and
nobody declared -- an ORPHAN -- and it held pre-recalibration numbers that
disagreed with the live copy on every row. A file that is *declared as an
input* by some node but produced by no node is invisible to an orphan scan,
because somebody declares it; it needs its own rule, and it is reported here as
UNSOURCED. A figure drawn from an unsourced input is a figure of unknown
provenance.

THE PARAMS FINGERPRINT (F-012)

Nodes that run the model depend on `code/model/params.yaml`. Hashing its bytes
made every such node STALE the moment anyone added a `note:` or an
`affects_claims:` entry -- an edit that cannot change any output. A staleness
signal that fires when nothing is wrong stops being a staleness signal, and the
only way to read it was to learn to ignore it. `params_fingerprint()` therefore
hashes the *document with DOCUMENTARY_KEYS removed*. The list is a denylist, so
a key nobody has thought about is fingerprinted by default and has to be
exempted deliberately.

THE UNSTAMPED BASELINE

Standing this graph up over an existing deposit finds a tree full of artifacts
that predate it and so carry no provenance sidecar. That set is not an error,
it is the regeneration to-do list; it is recorded once in
`.build/unstamped_baseline.json` and pruned on every successful `verify`, so a
node that has been stamped even once loses its exemption permanently and cannot
silently regress to unstamped.

USAGE

    python code/build.py status              # one line per node, exit 0
    python code/build.py verify              # gate: exit 1 on any defect
    python code/build.py run <node> [...]    # run generators, then stamp
    python code/build.py run --all           # run every runnable node
    python code/build.py run --stale         # run only nodes that are not OK
    python code/build.py stamp [<node> ...]  # record current hashes
    python code/build.py graph               # topological order and costs
    python code/build.py fingerprint         # print the params fingerprint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - the deposit requires PyYAML
    yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BUILD_DIR = os.path.join(ROOT, ".build")
PARAMS = "code/model/params.yaml"

# ---------------------------------------------------------------------------
# The params fingerprint (F-012)
# ---------------------------------------------------------------------------

# Keys whose value cannot reach any number the model computes. A denylist, not
# an allowlist: a new key is fingerprinted until somebody argues it out.
#
# `provenance`, `source`, `citation`, `note`, `notes`, `comment`, `rationale`,
# `basis`, `authority` and `reference` are prose. `used_by` and `used_in` name
# call sites. `affects_claims` is the forward half of the claim/parameter index
# (F-012) and is read by the claim gate, never by the model. `benchmark` names
# an observed comparator and is read by the benchmark suite (F-008).
# `units` and `label` are read by table generators, which are themselves nodes
# and so depend on the file's bytes through their generator hash, not through
# this fingerprint.
DOCUMENTARY_KEYS = frozenset({
    "affects_claims",
    "authority",
    "basis",
    "benchmark",
    "citation",
    "comment",
    "comments",
    "description",
    "doi",
    "label",
    "note",
    "notes",
    "provenance",
    "rationale",
    "reference",
    "references",
    "source",
    "sources",
    "units",
    "used_by",
    "used_in",
})


def strip_documentary(obj):
    """Return `obj` with every DOCUMENTARY_KEYS mapping key removed, at depth."""
    if isinstance(obj, dict):
        return {k: strip_documentary(v) for k, v in obj.items()
                if k not in DOCUMENTARY_KEYS}
    if isinstance(obj, list):
        return [strip_documentary(v) for v in obj]
    return obj


def params_fingerprint(path: Optional[str] = None) -> str:
    """sha256 of params.yaml with the documentary keys removed.

    A documentation edit must not restale artifacts (F-012). A value edit must.
    """
    path = path or os.path.join(ROOT, PARAMS)
    if yaml is None:
        raise RuntimeError("PyYAML is required to fingerprint params.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    payload = json.dumps(strip_documentary(doc), sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Node declarations
# ---------------------------------------------------------------------------

# Model sources every model-running node depends on. params.yaml is deliberately
# absent: it enters through params_fingerprint() instead of through its bytes.
MODEL_SOURCES = (
    "code/model/registry.py",
    "code/model/soil_n_model.py",
    "code/model/monthly_model_v3.py",
    "code/model/coupled_monthly.py",
    "code/model/coupled_econ_biophysical.py",
    "code/model/parameter_registry.py",
    "code/model/prices.py",
    "code/model/seams.py",
)

ERA5 = "data/era5_regional_climates.json"


@dataclass
class Node:
    name: str
    generator: str
    outputs: Tuple[str, ...]
    inputs: Tuple[str, ...] = ()
    libs: Tuple[str, ...] = ()
    uses_model: bool = True
    minutes: float = 0.5
    note: str = ""
    argv: Tuple[str, ...] = ()
    blocked: str = ""   # non-empty: running this node would destroy evidence

    def all_inputs(self) -> Tuple[str, ...]:
        extra = MODEL_SOURCES if self.uses_model else ()
        return tuple(sorted(set(self.inputs) | set(extra)))


NODES: List[Node] = [
    # -- the canonical run and everything read off it ----------------------
    Node("canonical", "code/repro/run_canonical.py",
         outputs=("data/canonical_ERA5_y30.csv", "data/canonical_ERA5_y30.json",
                  "outputs/global_S3_losses.txt"),
         inputs=(ERA5,), minutes=0.2,
         note="root of the graph; S3 to year 30 on the ERA5 climate"),
    Node("table_s3", "code/repro/make_table_s3.py",
         outputs=("outputs/table_S3_correlations.csv",),
         inputs=("data/canonical_ERA5_y30.json",), uses_model=False, minutes=0.1),
    Node("figure_s6", "code/repro/make_figure_s6.py",
         outputs=("figures/Figure_S6_pairwise_diagnostics.png",),
         inputs=("data/canonical_ERA5_y30.json",), uses_model=False, minutes=0.1),

    # -- trajectories -------------------------------------------------------
    Node("scenario_trajectories", "code/repro/make_scenario_trajectories.py",
         outputs=("data/scenario_trajectories.csv",),
         inputs=(ERA5,), minutes=0.2,
         note="S3, SC1, SC2 and PULSE1 global loss by year. Was BLOCKED on two "
              "counts and both are now paid. (a) The PULSE1 capability died "
              "with the v15 tree; it is rebuilt on the coupled_econ_biophysical"
              ".supply_state seam, which also removed the four copies of the "
              "disruption timeline the two coupled models each carried. The "
              "rebuild reproduces the lost column's years 1 and 2 to three "
              "decimals (2.316, 0.492) and diverges from year 3, which is the "
              "eps_F_N signature and is the evidence that it is the same "
              "scenario rather than a plausible new one. (b) The deposited CSV "
              "was the last place in the deposit still reading the superseded "
              "eps_F_N = -0.5 family; regenerating moved S3 year-10 from 3.032 "
              "to 3.198, in agreement with the canonical. C-060 and C-061 are "
              "now scored against the same family as everything else (F-021). "
              "Frozen pre-regeneration copy: "
              "baseline/surviving_v15/scenario_trajectories.csv"),
    Node("s3_shock_calibration", "code/repro/make_s3_shock_calibration.py",
         outputs=("results/s3_shock_calibration.csv",), inputs=(ERA5,),
         minutes=8.0,
         note="per-region shock-calibration diagnostics C-050 reads; was the "
              "last unsourced input (a deposit from the lost v15 tree, stale "
              "at the zero eps_F_N central) until F-027 gave it a generator"),
    Node("soc_trajectories", "code/repro/make_soc_trajectories.py",
         outputs=("data/soc_trajectories.csv", "data/soc_trajectories.json"),
         inputs=(ERA5,), minutes=2.0,
         note="the 30-year carbon series the SOC-decline sentence is read "
              "from; recovered from the session transcript after the v15 tree "
              "was lost and regenerated under the current eps_F_N (F-018)"),
    Node("sc_trajectories", "code/repro/make_sc_trajectories.py",
         outputs=("data/SC1_regional_trajectory.csv",
                  "data/SC2_regional_trajectory.csv"),
         inputs=(ERA5,), minutes=0.3),

    # -- climate robustness -------------------------------------------------
    Node("climate_swap", "code/repro/climate_comparison.py",
         outputs=("outputs/climate_swap_comparison.csv",
                  "results/climate_swap_stats.txt"),
         inputs=(ERA5,), minutes=0.3,
         note="expert vs ERA5 climate; reported in the response letter. The "
              "stats file is declared so the two headline numbers are a "
              "tracked artifact and not a console line (F-009)"),

    # -- farm gradient, main figures ---------------------------------------
    Node("price_shock", "code/repro/run_price_shock_analysis.py",
         outputs=("data/figure1_farm_gradient.json",
                  "data/figure2_soc_gradient.json"),
         inputs=(ERA5,), minutes=2.0),
    Node("figure_1", "code/repro/make_figure_1.py",
         outputs=("data/figure1_soc_gradient.csv",
                  "figures/Figure_1_farm_buffering.png",
                  "figures/Figure_1_farm_buffering.pdf"),
         inputs=("data/figure1_farm_gradient.json",), uses_model=False,
         minutes=0.2),
    Node("figure_2", "code/repro/make_figure_2.py",
         outputs=("data/figure2_panels.json",
                  "figures/Figure_2_regional_vulnerability.png",
                  "figures/Figure_2_regional_vulnerability.pdf"),
         inputs=("data/figure2_soc_gradient.json",
                 "data/canonical_ERA5_y30.json"),
         minutes=0.5),

    # -- elasticity sensitivity --------------------------------------------
    Node("figS8_curves", "code/repro/compute_figS8_curves.py",
         outputs=("data/figS8_curves.json",), inputs=(ERA5,), minutes=1.0),
    Node("figure_s8", "code/repro/make_figure_s8.py",
         outputs=("figures/Figure_S8_elasticity_sensitivity.png",),
         inputs=("data/figS8_curves.json",), uses_model=False, minutes=0.1),
    Node("figure_s7", "code/repro/make_figure_s7.py",
         outputs=("data/figS7_farm_elasticity_gradient.json",
                  "figures/Figure_S7_farm_elasticity_gradient.png",
                  "figures/Figure_S7_farm_elasticity_gradient.pdf"),
         inputs=(ERA5,), minutes=2.5),

    # -- calibration table and the crop-response curves ---------------------
    Node("table_s4", "code/repro/make_table_s4_sol.py",
         outputs=("outputs/Table_S4_calibration_sol.csv",
                  "data/figS12_curves.json",
                  "data/crop_response_calibration_table.csv"),
         inputs=(ERA5,), minutes=3.0,
         note="writes THREE outputs; the second is Figure S12's input (F-009) "
              "and the third is Figure S13's, added by D3"),
    Node("figure_s12", "code/repro/make_figure_s12.py",
         outputs=("figures/Figure_S12_crop_response_calibration.png",),
         inputs=("data/figS12_curves.json",), uses_model=False, minutes=0.1),

    # -- Monte Carlo ensemble ----------------------------------------------
    Node("mc_ensemble", "code/repro/run_mc_ensemble.py",
         outputs=("data/mc_ensemble/mc_posterior.csv.gz",
                  "data/mc_ensemble/mc_summary.csv",
                  "data/mc_ensemble/mc_probabilities.csv",
                  "data/mc_ensemble/mc_summary.txt",
                  "data/mc_ensemble/mc_priors.json"),
         inputs=(ERA5,), minutes=90.0,
         note="1,000 joint-prior draws; the expensive node. Regenerated 2026-08-29 under F-026: the eps_F_N central is restored to -0.50 and the clearing is realized-yield (F-025), so the deposited ensemble was superseded on both counts. The pre-rerun deposit is snapshotted outside the tree; F-013 claim strength reproduces against the regenerated ensemble from this run onward. ~20 min at 2 workers."),
    Node("figure_s9", "code/repro/make_figure_s9.py",
         outputs=("figures/Figure_S9_mc_ensemble.png",
                  "figures/Figure_S9_mc_ensemble.pdf"),
         inputs=("data/mc_ensemble/mc_posterior.csv.gz",), uses_model=False,
         minutes=0.3),

    # -- SI sweeps ----------------------------------------------------------
    Node("figure_s10", "code/repro/make_figure_s10.py",
         outputs=("data/figS10_nue_sensitivity.json",
                  "figures/Figure_S10_nue_sensitivity.png",
                  "figures/Figure_S10_nue_sensitivity.pdf"),
         inputs=(ERA5,), minutes=1.5),
    Node("figure_s11", "code/repro/make_figure_s11.py",
         outputs=("data/figS11_severity_sweep.json",
                  "figures/Figure_S11_severity_gradient.png",
                  "figures/Figure_S11_severity_gradient.pdf"),
         inputs=(ERA5,), minutes=2.0),

    # -- prices and food-price response ------------------------------------
    Node("food_price_table", "code/repro/make_food_price_table.py",
         outputs=("data/food_price_response.csv",), inputs=(ERA5,), minutes=0.5),

    # -- benchmarks ---------------------------------------------------------
    Node("broadbalk_benchmark", "code/repro/make_broadbalk_benchmark.py",
         outputs=("data/benchmarks/broadbalk_yield_benchmark_sol.csv",
                  "figures/Figure_S2_broadbalk_benchmark.png",
                  "figures/Figure_S2_broadbalk_benchmark.pdf"),
         inputs=("code/model/data/benchmark_broadbalk/soc_trajectories_broadbalk.csv",),
         minutes=1.0),
    Node("hindcast_benchmark", "code/repro/make_hindcast_benchmark.py",
         outputs=("data/benchmarks/hindcast_benchmark_sol.csv",
                  "figures/Figure_S4_hindcast_sensitivity.png",
                  "figures/Figure_S4_hindcast_sensitivity.pdf"),
         inputs=("data/benchmarks/hindcast_observed_2022.csv",), minutes=0.5),
    Node("ofra_validation", "code/repro/make_ofra_validation.py",
         outputs=("figures/Figure_S13_OFRA_SSA_validation.png",),
         inputs=("data/ofra_maize_N_responsefunctions.csv",
                 "outputs/Table_S4_calibration_sol.csv"),
         uses_model=False, minutes=0.1,
         note="declared crop_response_calibration_table.csv until D3; the "
              "script reads Table_S4_calibration_sol.csv and always has "
              "(make_ofra_validation.py line 15)"),
    Node("benchmarks", "code/repro/run_benchmarks.py",
         outputs=("outputs/benchmarks.csv", "outputs/benchmarks.json"),
         inputs=("data/benchmarks/observed_values.yaml",
                 "data/canonical_ERA5_y30.json",
                 "data/benchmarks/broadbalk_yield_benchmark_sol.csv"),
         minutes=1.0,
         note="verdicts frozen in data/benchmarks/baseline_verdicts.json (F-008)"),

    # -- acceptance sweeps that deposit tables ------------------------------
    Node("zero_shock_invariance", "code/repro/test_zero_shock_invariance.py",
         outputs=("outputs/zero_shock_invariance.csv",), inputs=(ERA5,),
         minutes=0.5),
    Node("structural_sensitivity", "code/repro/run_structural_sensitivity_sol.py",
         outputs=("outputs/structural_sensitivity_sol.csv",
                  "outputs/price_convention_sensitivity_sol.csv"),
         inputs=(ERA5,), minutes=3.0),
    Node("parameter_extremes", "code/repro/test_parameter_extremes_sol.py",
         outputs=("outputs/parameter_extreme_acceptance_sol.csv",),
         inputs=(ERA5,), minutes=3.0),
    Node("calibration_production_path",
         "code/repro/make_calibration_production_path.py",
         outputs=("results/calibration_production_path.csv",), inputs=(ERA5,),
         minutes=0.5),

    # -- ledgers ------------------------------------------------------------
    Node("parameter_ledger", "code/repro/make_parameter_ledger_sol.py",
         outputs=("PARAMETER_LEDGER_sol.csv", "PARAMETER_LEDGER_sol.md",
                  "NUMERIC_LITERAL_AUDIT_sol.csv"),
         inputs=(ERA5,), minutes=1.0),

    # -- the claim register (F-012, F-013) ---------------------------------
    Node("claims_report", "code/repro/make_claim_report.py",
         outputs=("results/claims_report.md", "outputs/claims_status.csv"),
         libs=("code/repro/claim_resolvers.py",),
         inputs=("docs/claims.yaml",
                 "data/scenario_trajectories.csv", "data/soc_trajectories.json",
                 "data/figure2_panels.json", "data/figS8_curves.json",
                 "data/food_price_response.csv",
                 "data/figure1_farm_gradient.json",
                 "data/figS11_severity_sweep.json",
                 "data/mc_ensemble/mc_probabilities.csv",
                 "results/s3_shock_calibration.csv",
                 "outputs/price_convention_sensitivity_sol.csv",
                 "results/calibration_production_path.csv"),
         minutes=0.2,
         note="baselines are written with --write-baseline, not on every run"),
    Node("claim_strength", "code/repro/make_claim_strength.py",
         outputs=("results/claim_strength.csv", "results/claim_strength.md"),
         libs=("code/repro/claim_resolvers.py",),
         inputs=("docs/claims.yaml",
                 "data/mc_ensemble/mc_posterior.csv.gz"),
         minutes=0.3),
]

NODE_BY_NAME = {n.name: n for n in NODES}

# ---------------------------------------------------------------------------
# Files that are not node outputs and are not defects
# ---------------------------------------------------------------------------

# Observations, retrievals and hand-transcribed inputs. The note is the point:
# an external input with no note is an artifact whose provenance nobody wrote
# down, which is the defect this file exists to catch (F-014, the `prices`
# node).
EXTERNAL_INPUTS: Dict[str, str] = {
    "data/era5_regional_climates.json":
        "ERA5 monthly normals, retrieved by code/era5/fetch_era5_climate.py "
        "from the Open-Meteo archive; needs network, so it is retrieved once "
        "and deposited",
    "data/ofra_maize_N_responsefunctions.csv":
        "OFRA maize N-response functions, published dataset, transcribed",
    "data/benchmarks/hindcast_observed_2022.csv":
        "official FAOSTAT 2021-2022 changes, transcribed (F-008)",
    "data/benchmarks/observed_values.yaml":
        "the benchmark suite's observed-value compilation, hand-assembled with "
        "a source per row (F-008)",
    "data/benchmarks/validation_data_extraction.csv":
        "extracted long-term-trial observations (F-008)",
    "data/benchmarks/validation_targets_and_caveats.csv":
        "the caveat table for the above (F-008)",
    "data/benchmarks/baseline_verdicts.json":
        "frozen benchmark verdicts; written by run_benchmarks.py "
        "--write-baseline, deliberately not on every run (F-008, F-009)",
    "code/model/data/benchmark_broadbalk/soc_trajectories_broadbalk.csv":
        "Broadbalk observed SOC, transcribed from the Rothamsted archive",
    "code/model/params.yaml":
        "the parameter registry itself; every value carries its own provenance "
        "field (F-001)",
    "docs/claims.yaml":
        "the claim register; text and location are transcribed from the "
        "manuscript and are not model outputs (F-012)",
    "docs/claims_baseline.json":
        "frozen claim verdicts, written with --write-baseline (F-012)",
    "docs/claims_index_baseline.json":
        "frozen claim/parameter index, written with --write-baseline (F-012)",
    "docs/claim_strength_baseline.json":
        "frozen claim-strength verdicts, written with --write-baseline (F-013)",
}

# Deposited pipelines that live outside code/ and are documented in MANIFEST.md.
# Declared so the orphan scan does not report them, and named so that "who made
# this" has an answer.
EXTERNAL_OUTPUTS: Dict[str, str] = {
    "figures/Figure_3_mechanism_screen.png":
        "spatial_screen/scripts/19_fig4_mechanism_screen_v10.py",
    "figures/Figure_3_mechanism_screen.pdf":
        "spatial_screen/scripts/19_fig4_mechanism_screen_v10.py",
}

# Artifacts written by test scripts rather than by generators. `make verify`
# runs them; they are declared here so that the orphan scan does not mistake a
# test's deposit for a file nobody made. A test artifact is evidence about the
# code, not a result the manuscript cites.
TEST_ARTIFACTS: Dict[str, str] = {
    "results/cap_market_clearing.txt": "code/repro/test_cap_market_clearing.py",
    "results/seam_contract_checks.yaml": "code/tests/test_seam_contracts.py",
    "results/calibration_fingerprint_checks.yaml":
        "code/tests/test_calibration_fingerprint.py",
    "results/spinup_partition_characterisation.yaml":
        "code/tests/test_spinup_partition_independence.py",
    "results/spinup_partition_characterisation.json":
        "code/tests/test_spinup_partition_independence.py",
    "results/wp1_equality_log.txt": "code/tests/test_wp1_registry_wiring.py",
    "results/mutation_coverage.csv": "code/tests/run_mutation_coverage.py",
    "results/mutation_coverage_summary.txt": "code/tests/run_mutation_coverage.py",
    "outputs/zero_shock_invariance.csv": "code/repro/test_zero_shock_invariance.py",
    "baseline/canonical_before.json": "pinned WP1 refactor baseline",
    "baseline/canonical_expected_delta.json": "pinned WP1 refactor baseline",
    "baseline/econ_targets_before.json": "pinned WP1 refactor baseline",
    "baseline/regional_fields_before.json": "pinned WP1 refactor baseline",
    "baseline/spinup_characterisation_expected.json":
        "code/tests/test_spinup_partition_independence.py",
}

# Caches: written by a generator to make a rerun cheaper, read by nothing that
# the manuscript cites. Declared so they are not orphans and labelled so nobody
# mistakes one for a result.
CACHES: Dict[str, str] = {
    "data/mc_ensemble/ym_cache.json":
        "calibrated y_max cache, written by run_mc_ensemble.py on first run",
}

# Snapshots kept deliberately. `baseline/surviving_v15/` holds the artifacts
# that survived the crashed v15 session and whose generators did not, frozen
# before this package regenerated the chain over them (see
# results/build_reconciliation.md).
# `baseline/f022_f025_evidence/` holds the pre-F-025 diagnostics that justified
# adopting realized market clearing; rerunning their generators against the
# current model would measure a gap the model no longer has, so they are
# frozen evidence, not regenerable artifacts (see the README there).
PRESERVED_PREFIXES = ("baseline/surviving_v15/", "baseline/f022_f025_evidence/",
                      # F-028: archived pre-F-025 claim register, kept for the record
                      "docs/archive/")

# Hand-written prose. Reconciliation notes are arguments about artifacts, not
# artifacts. They are listed rather than pattern-matched so that a generated
# .md cannot hide among them.
NARRATIVE: Set[str] = {
    "results/benchmark_reconciliation.md",
    "results/claims_reconciliation.md",
    "results/claim_strength_surviving_v15.md",
    "results/mutation_coverage_reconciliation.md",
    "results/build_reconciliation.md",
    "outputs/table_S1_parameters.md",
}

# Directories scanned for orphans.
ARTIFACT_DIRS = ("data", "outputs", "figures", "results", "docs", "baseline")

# Never scanned. `excluded_legacy_sol/` is outside the evidentiary chain by
# construction (MANIFEST.md); `.build/` is this tool's own bookkeeping;
# `era5_raw/` is the provenance dump behind the single ERA5 input.
SCAN_EXCLUDE_DIRS = ("data/era5_raw", "data/mc_ensemble/chunks", ".build")
SCAN_EXCLUDE_SUFFIXES = (".DS_Store", ".gitkeep")


# ---------------------------------------------------------------------------
# Hashing and sidecars
# ---------------------------------------------------------------------------

def abspath(rel: str) -> str:
    return os.path.join(ROOT, rel)


def file_hash(rel: str) -> Optional[str]:
    path = abspath(rel)
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_payload_hash(rel: str) -> Optional[str]:
    """Hash the *contents* of a gzip file, not its container.

    F-014: regenerating the ensemble left `mc_posterior.csv.gz` byte-different
    while its decompressed payload was identical, because gzip records an mtime.
    Hashing the container would report a change that is not one.
    """
    import gzip
    path = abspath(rel)
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(rel: str) -> Optional[str]:
    if rel.endswith(".gz"):
        return gzip_payload_hash(rel)
    return file_hash(rel)


# ---------------------------------------------------------------------------
# Output canonicalization (F-028)
# ---------------------------------------------------------------------------
# The third external audit ran the graph on another machine and found a node
# whose regenerated output differed from the stamped one at the 1e-14 scale:
# libm/BLAS last-ulp differences, amplified through the monthly loop, land in
# the full-precision float text that json.dump and csv writers emit. A
# staleness gate that hashes bytes then reports a defect where there is none.
#
# The repair is at the write side, not the compare side: every node's textual
# outputs are rewritten to a canonical form -- float literals carrying more
# than six significant digits are re-rendered at six -- immediately after the
# generator runs and before the sidecar is stamped. Six significant digits is
# two orders finer than any value the documents quote (three to four), and
# coarse enough that noise at 1e-13 relative cannot move a rounding boundary
# (flip probability ~1e-7 per value). Tolerance-based comparison was rejected
# because it would have to live in every consumer; canonical bytes fix every
# consumer at once.

_FLOAT_CELL = re.compile(r'-?(?:\d+\.\d+|\.\d+)(?:[eE][+-]?\d+)?')


def _sig_digits(lit: str) -> int:
    mant = lit.split('e')[0].split('E')[0].lstrip('-+').replace('.', '')
    return len(mant.lstrip('0'))


def _canon_float(x) -> float:
    return float('%.6g' % float(x))


def _canon_json_text(text: str) -> str:
    obj = json.loads(text, parse_float=_canon_float)
    return json.dumps(obj, indent=2, ensure_ascii=False) + '\n'


def _canon_csv_text(text: str) -> str:
    import csv as _csv
    import io as _io
    rows = list(_csv.reader(_io.StringIO(text)))
    for row in rows:
        for i, cell in enumerate(row):
            if _FLOAT_CELL.fullmatch(cell) and _sig_digits(cell) > 6:
                row[i] = '%.6g' % float(cell)
    out = _io.StringIO()
    _csv.writer(out, lineterminator='\n').writerows(rows)
    return out.getvalue()


def canonicalize_file(rel: str) -> bool:
    """Rewrite one output file in canonical form. True if bytes changed."""
    path = abspath(rel)
    if not os.path.exists(path):
        return False
    if rel.endswith('.csv.gz'):
        import gzip as _gzip
        with _gzip.open(path, 'rt', encoding='utf-8') as fh:
            old = fh.read()
        new = _canon_csv_text(old)
        # the container is rewritten unconditionally (mtime=0, no name) so the
        # gzip bytes are as deterministic as the payload; content_hash() hashes
        # the payload, so the return value tracks payload change only
        with open(path, 'wb') as raw:
            with _gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                                mtime=0) as gz:
                gz.write(new.encode('utf-8'))
        return new != old
    if rel.endswith('.json'):
        old = open(path, encoding='utf-8').read()
        new = _canon_json_text(old)
    elif rel.endswith('.csv'):
        old = open(path, encoding='utf-8').read()
        new = _canon_csv_text(old)
    else:
        return False
    if new == old:
        return False
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(new)
    return True


def canonicalize_outputs(node: "Node") -> List[str]:
    return [p for p in node.outputs if canonicalize_file(p)]


def generator_hash(node: "Node") -> Optional[str]:
    """Hash the generator together with the libraries it imports.

    A resolver library is source, not an artifact: it belongs in the generator
    fingerprint, not in the input list, or the graph reports a source file as a
    result nobody produces.
    """
    parts = [file_hash(node.generator)]
    if parts[0] is None:
        return None
    for lib in sorted(node.libs):
        parts.append("%s=%s" % (lib, file_hash(lib)))
    return hashlib.sha256("|".join(str(x) for x in parts).encode()).hexdigest()


def sidecar_path(name: str) -> str:
    return os.path.join(BUILD_DIR, name + ".json")


def read_sidecar(name: str) -> Optional[dict]:
    path = sidecar_path(name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=20)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def node_state(node: Node) -> dict:
    """Current hashes for one node, independent of any sidecar."""
    return {
        "node": node.name,
        "generator": node.generator,
        "generator_sha": generator_hash(node),
        "params_fingerprint": params_fingerprint() if node.uses_model else None,
        "inputs": {p: content_hash(p) for p in node.all_inputs()},
        "outputs": {p: content_hash(p) for p in node.outputs},
    }


def write_sidecar(node: Node) -> dict:
    os.makedirs(BUILD_DIR, exist_ok=True)
    state = node_state(node)
    state["stamped_at_commit"] = git_commit()
    state["stamped_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(sidecar_path(node.name), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return state


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

OK = "OK"
STALE = "STALE"
MISSING_GENERATOR = "MISSING_GENERATOR"
MISSING_OUTPUT = "MISSING_OUTPUT"
MISSING_INPUT = "MISSING_INPUT"
UNSTAMPED = "UNSTAMPED"
UNSTAMPED_BASELINE = "UNSTAMPED_BASELINE"
# A node whose generator is BEHIND its deposited artifact: running it would
# destroy something the tree cannot rebuild. Reported on every status and
# verify, with the reason, so it cannot become invisible.
BLOCKED = "BLOCKED"

DEFECT_STATES = {STALE, MISSING_GENERATOR, MISSING_OUTPUT, MISSING_INPUT,
                 UNSTAMPED}


def unstamped_baseline_path() -> str:
    return os.path.join(BUILD_DIR, "unstamped_baseline.json")


def read_unstamped_baseline() -> Set[str]:
    path = unstamped_baseline_path()
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        return set(json.load(fh).get("nodes", []))


def write_unstamped_baseline(names: Iterable[str], reason: str) -> None:
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(unstamped_baseline_path(), "w", encoding="utf-8") as fh:
        json.dump({"reason": reason, "nodes": sorted(names)}, fh, indent=2)
        fh.write("\n")


def _plain_status(node: Node, baseline: Set[str]) -> Tuple[str, List[str]]:
    """The verdict a node would get if it were not blocked."""
    detail: List[str] = []
    now = node_state(node)
    missing_out = [p for p, h in now["outputs"].items() if h is None]
    missing_in = [p for p, h in now["inputs"].items() if h is None]
    if missing_in:
        return MISSING_INPUT, ["input absent: " + ", ".join(sorted(missing_in))]
    if missing_out:
        return MISSING_OUTPUT, ["output absent: " + ", ".join(sorted(missing_out))]
    old = read_sidecar(node.name)
    if old is None:
        if node.name in baseline:
            return UNSTAMPED_BASELINE, ["never regenerated under this graph"]
        return UNSTAMPED, ["no provenance sidecar and not in the baseline"]
    if old.get("generator_sha") != now["generator_sha"]:
        detail.append("generator changed")
    if old.get("params_fingerprint") != now["params_fingerprint"]:
        detail.append("params fingerprint changed")
    for p_, h in sorted(now["inputs"].items()):
        if old.get("inputs", {}).get(p_) != h:
            detail.append("input changed: %s" % p_)
    for p_, h in sorted(now["outputs"].items()):
        if old.get("outputs", {}).get(p_) != h:
            detail.append("output changed since stamp: %s" % p_)
    return (STALE, detail) if detail else (OK, [])


def status_of(node: Node, baseline: Set[str]) -> Tuple[str, List[str]]:
    detail: List[str] = []
    if file_hash(node.generator) is None:
        return MISSING_GENERATOR, ["%s does not exist" % node.generator]
    missing_lib = [p for p in node.libs if file_hash(p) is None]
    if missing_lib:
        return MISSING_GENERATOR, ["library absent: " + ", ".join(missing_lib)]

    if node.blocked:
        # A blocked node still gets its ordinary verdict computed and reported.
        # A state that hides another state is the defect this file exists to
        # catch: BLOCKED alone would have said nothing about whether the frozen
        # artifact is also out of date.
        under, under_detail = _plain_status(node, baseline)
        detail = [node.blocked, "underlying status: %s%s" % (
            under, (" -- " + "; ".join(under_detail)) if under_detail else "")]
        return BLOCKED, detail

    now = node_state(node)
    missing_out = [p for p, h in now["outputs"].items() if h is None]
    missing_in = [p for p, h in now["inputs"].items() if h is None]
    if missing_in:
        return MISSING_INPUT, ["input absent: " + ", ".join(sorted(missing_in))]
    if missing_out:
        return MISSING_OUTPUT, ["output absent: " + ", ".join(sorted(missing_out))]

    old = read_sidecar(node.name)
    if old is None:
        if node.name in baseline:
            return UNSTAMPED_BASELINE, ["predates the graph; on the "
                                        "regeneration to-do list"]
        return UNSTAMPED, ["no provenance sidecar and not in the baseline"]

    if old.get("generator_sha") != now["generator_sha"]:
        detail.append("generator changed")
    if old.get("params_fingerprint") != now["params_fingerprint"]:
        detail.append("params fingerprint changed")
    for p, h in sorted(now["inputs"].items()):
        if old.get("inputs", {}).get(p) != h:
            detail.append("input changed: %s" % p)
    for p, h in sorted(now["outputs"].items()):
        if old.get("outputs", {}).get(p) != h:
            detail.append("output changed since stamp: %s" % p)
    if detail:
        return STALE, detail
    return OK, []


# ---------------------------------------------------------------------------
# Orphan and unsourced scans
# ---------------------------------------------------------------------------

def declared_outputs() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in NODES:
        for p in n.outputs:
            out[p] = n.name
    return out


def scan_files() -> List[str]:
    found: List[str] = []
    for d in ARTIFACT_DIRS:
        base = abspath(d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = os.path.relpath(dirpath, ROOT)
            if any(rel_dir == x or rel_dir.startswith(x + os.sep)
                   for x in SCAN_EXCLUDE_DIRS):
                dirnames[:] = []
                continue
            for fn in filenames:
                if fn.endswith(SCAN_EXCLUDE_SUFFIXES) or fn.startswith("."):
                    continue
                found.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    return sorted(found)


def declared_inputs() -> Set[str]:
    out: Set[str] = set()
    for n in NODES:
        out |= set(n.inputs)
    return out


def orphans() -> List[str]:
    """Files in the artifact tree that no node makes AND nobody declares.

    F-009's rule, both halves. A file some node declares as an input is not an
    orphan even when nothing writes it -- it is UNSOURCED, which is a different
    verdict reported by `unsourced()`. Collapsing the two loses the distinction
    that found `data/figS12_curves.json`.
    """
    known = set(declared_outputs())
    known |= declared_inputs()
    known |= set(EXTERNAL_INPUTS)
    known |= set(EXTERNAL_OUTPUTS)
    known |= set(TEST_ARTIFACTS)
    known |= NARRATIVE
    known |= set(CACHES)
    return [p for p in scan_files()
            if p not in known
            and not p.startswith(PRESERVED_PREFIXES)]


def unsourced() -> List[Tuple[str, List[str]]]:
    """Declared inputs that no node writes and no external declaration covers.

    This is the second direction of the orphan check and needs its own rule:
    such a file is declared -- by the node that reads it -- so an orphan scan
    never sees it (F-009).
    """
    produced = set(declared_outputs()) | set(EXTERNAL_INPUTS) | set(EXTERNAL_OUTPUTS)
    produced |= set(TEST_ARTIFACTS) | set(CACHES)
    hits: Dict[str, List[str]] = {}
    for n in NODES:
        for p in n.inputs:
            if p not in produced:
                hits.setdefault(p, []).append(n.name)
    return sorted((p, sorted(v)) for p, v in hits.items())


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def topo_order(names: Sequence[str]) -> List[str]:
    producer = declared_outputs()
    wanted = list(names)
    order: List[str] = []
    seen: Set[str] = set()
    marks: Set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name in marks:
            raise RuntimeError("cycle in the build graph at %s" % name)
        marks.add(name)
        node = NODE_BY_NAME[name]
        for p in node.inputs:
            upstream = producer.get(p)
            if upstream and upstream != name and upstream in wanted:
                visit(upstream)
        marks.discard(name)
        seen.add(name)
        order.append(name)

    for n in wanted:
        visit(n)
    return order


def run_node(node: Node, log_dir: Optional[str] = None) -> int:
    cmd = [sys.executable, abspath(node.generator), *node.argv]
    print("--> %-28s %s" % (node.name, node.generator), flush=True)
    started = time.time()
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "%s.log" % node.name)
        with open(log_path, "w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=fh,
                                  stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - started
    if proc.returncode == 0:
        canon = canonicalize_outputs(node)
        write_sidecar(node)
        note = "  (canonicalized %d)" % len(canon) if canon else ""
        print("    ok   %.1fs%s" % (dt, note), flush=True)
    else:
        print("    FAIL exit %d after %.1fs" % (proc.returncode, dt), flush=True)
    return proc.returncode


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cmd_status(args) -> int:
    baseline = read_unstamped_baseline()
    rows = []
    for node in NODES:
        st, detail = status_of(node, baseline)
        rows.append((node, st, detail))

    width = max(len(n.name) for n in NODES)
    for node, st, detail in rows:
        line = "%-*s  %-19s %s" % (width, node.name, st,
                                   "; ".join(detail) if detail else "")
        print(line.rstrip())

    tally: Dict[str, int] = {}
    for _, st, _ in rows:
        tally[st] = tally.get(st, 0) + 1
    print("")
    print("%d nodes: %s" % (len(rows), ", ".join(
        "%s %d" % (k, v) for k, v in sorted(tally.items()))))

    orph = orphans()
    print("orphans (%d):" % len(orph))
    for p in orph:
        print("    %s" % p)
    uns = unsourced()
    print("unsourced inputs (%d):" % len(uns))
    for p, readers in uns:
        print("    %s  <- read by %s" % (p, ", ".join(readers)))
    if args.json:
        payload = {
            "nodes": [{"name": n.name, "status": s, "detail": d}
                      for n, s, d in rows],
            "tally": tally,
            "orphans": orph,
            "unsourced": [{"path": p, "readers": r} for p, r in uns],
            "params_fingerprint": params_fingerprint(),
            "commit": git_commit(),
        }
        with open(abspath(args.json), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print("\nwrote %s" % args.json)
    return 0


def cmd_verify(args) -> int:
    """The gate. Exit 1 on any defect the graph can see."""
    baseline = read_unstamped_baseline()
    failures: List[str] = []
    stamped_now: List[str] = []
    tally: Dict[str, int] = {}
    for node in NODES:
        st, detail = status_of(node, baseline)
        tally[st] = tally.get(st, 0) + 1
        if st == BLOCKED:
            print("BLOCKED  %s: %s" % (node.name, "; ".join(detail)))
        if st in DEFECT_STATES:
            failures.append("%s %s: %s" % (st, node.name, "; ".join(detail)))
        if st == OK and node.name in baseline:
            stamped_now.append(node.name)

    orph = orphans()
    uns = unsourced()
    known_orphans = set(args.allow_orphan or [])
    known_unsourced = set(args.allow_unsourced or [])
    for p in orph:
        if p not in known_orphans:
            failures.append("ORPHAN %s" % p)
    for p, readers in uns:
        if p not in known_unsourced:
            failures.append("UNSOURCED %s (read by %s)" % (p, ", ".join(readers)))

    for f in failures:
        print("FAIL  " + f)
    if failures:
        print("\nbuild graph: %d defects" % len(failures))
        return 1

    # A node stamped even once loses its exemption permanently.
    if stamped_now and baseline:
        remaining = baseline - set(stamped_now)
        if remaining != baseline:
            write_unstamped_baseline(
                remaining,
                "pruned on a successful verify; a stamped node cannot regress")
            print("pruned %d node(s) from the unstamped baseline"
                  % (len(baseline) - len(remaining)))
    # Report the tally, never "all OK". A gate that says all OK while a node
    # is exempt is the defect this file exists to catch.
    print("build graph: %d nodes -- %s" % (
        len(NODES), ", ".join("%s %d" % (k, v) for k, v in sorted(tally.items()))))
    if tally.get(UNSTAMPED_BASELINE):
        print("  (%d node(s) still carry the pre-graph exemption; they have "
              "never been regenerated under this graph)"
              % tally[UNSTAMPED_BASELINE])
    print("  %d orphan(s) and %d unsourced input(s) allowed by name on the "
          "command line" % (len(orph), len(uns)))
    return 0


def cmd_run(args) -> int:
    baseline = read_unstamped_baseline()
    if args.all:
        names = [n.name for n in NODES
                 if status_of(n, baseline)[0] not in (MISSING_GENERATOR, BLOCKED)]
    elif args.stale:
        names = [n.name for n in NODES
                 if status_of(n, baseline)[0] not in (OK, UNSTAMPED_BASELINE,
                                                      MISSING_GENERATOR, BLOCKED)]
    else:
        names = list(args.nodes)
    refused = [n for n in names
               if NODE_BY_NAME.get(n) is not None and NODE_BY_NAME[n].blocked]
    if refused and not args.force:
        for n in refused:
            print("REFUSING to run %s: %s" % (n, NODE_BY_NAME[n].blocked))
        print("(--force overrides; the frozen copy is in baseline/surviving_v15/)")
        names = [n for n in names if n not in set(refused)]
    unknown = [n for n in names if n not in NODE_BY_NAME]
    if unknown:
        print("unknown node(s): %s" % ", ".join(unknown))
        return 2
    if args.skip:
        names = [n for n in names if n not in set(args.skip)]
    order = topo_order(names)
    budget = sum(NODE_BY_NAME[n].minutes for n in order)
    print("running %d node(s), about %.0f min:\n  %s\n"
          % (len(order), budget, " ".join(order)))
    if args.dry_run:
        return 0
    rc = 0
    for name in order:
        code = run_node(NODE_BY_NAME[name], args.log_dir)
        if code != 0:
            rc = code
            if not args.keep_going:
                return rc
    return rc


def cmd_stamp(args) -> int:
    names = args.nodes or [n.name for n in NODES]
    n_ok = 0
    for name in names:
        node = NODE_BY_NAME[name]
        st, _ = status_of(node, read_unstamped_baseline())
        if st in (MISSING_GENERATOR, MISSING_OUTPUT, MISSING_INPUT):
            print("skip  %-28s %s" % (name, st))
            continue
        write_sidecar(node)
        n_ok += 1
        print("stamp %s" % name)
    print("stamped %d node(s)" % n_ok)
    return 0


def cmd_baseline(args) -> int:
    """Record the nodes that predate the graph. Run once, at stand-up."""
    names = [n.name for n in NODES if read_sidecar(n.name) is None]
    write_unstamped_baseline(
        names, "nodes present in the deposit before code/build.py existed")
    print("unstamped baseline: %d node(s)" % len(names))
    for n in names:
        print("    %s" % n)
    return 0


def cmd_graph(args) -> int:
    order = topo_order([n.name for n in NODES])
    producer = declared_outputs()
    total = 0.0
    for name in order:
        node = NODE_BY_NAME[name]
        total += node.minutes
        deps = sorted({producer[p] for p in node.inputs
                       if p in producer and producer[p] != name})
        print("%-28s %5.1f min  <- %s" % (name, node.minutes,
                                          ", ".join(deps) or "(external only)"))
    print("\n%d nodes, about %.0f min end to end" % (len(NODES), total))
    return 0


def cmd_fingerprint(args) -> int:
    print(params_fingerprint())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("status", help="one line per node")
    p.add_argument("--json", help="also write the report to this path")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("verify", help="gate; exit 1 on any defect")
    p.add_argument("--allow-orphan", action="append", default=[])
    p.add_argument("--allow-unsourced", action="append", default=[])
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("run", help="run generators and stamp them")
    p.add_argument("nodes", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--stale", action="store_true")
    p.add_argument("--skip", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="run nodes marked blocked; destroys evidence")
    p.add_argument("--log-dir", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("stamp", help="record current hashes")
    p.add_argument("nodes", nargs="*")
    p.set_defaults(func=cmd_stamp)

    p = sub.add_parser("baseline", help="record the pre-graph nodes, once")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("graph", help="topological order and cost")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("fingerprint", help="print the params fingerprint")
    p.set_defaults(func=cmd_fingerprint)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
