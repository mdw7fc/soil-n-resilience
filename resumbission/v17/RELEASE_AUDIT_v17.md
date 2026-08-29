# v17 release audit

Status: **PASS**, pending Dale Manning's review of the net-revenue reframing.

v17 is the v14 document base carried forward through the F-018 to F-025 model
work and Dale Manning's August review, released 2026-08-29 at repository
commit `9efffb6` (see FINDINGS.md entries F-018 through F-025 for the full
ledger). It supersedes both the v14 line and the parallel v16_sol line; where
v16_sol and v17 differ, v17's central configuration is eps_F_N = 0 with the
realized-yield market clearing, and v16_sol's expert-elicited eps_F_N = -0.50
remains a structural sensitivity.

What this release changes:

- Realized-yield market clearing (F-025). The food price is root-found at
  every timestep so that demand equals the production response of the
  biophysical model itself; the Mitscherlich elasticities are diagnostics and
  the physical supply ceiling is a quantity constraint inside the same solve.
  Clearing residual below 1e-8 at every step of every published run, verified
  externally from reported outputs. Figure 1's price recovery clears the same
  way. Yield trajectories moved by at most 0.05 percentage points; year-10
  food-price responses rose 1-2 percentage points in most regions.
- Partial net-revenue reframing. The financial outcome is named "crop revenue
  net of nitrogen-fertilizer expenditure" throughout, with exclusions stated.
  The audited price pairs put the nitrogen-cost share at ~15% in South Asia
  and ~4% in Sub-Saharan Africa, so the submitted claim that SSA has the
  largest financial loss is explicitly retracted; the SI's ensemble-based
  SSA-worst-margin sentence is withdrawn pending an ensemble rerun under
  audited prices. The reversal is confirmed independently by the v16_sol
  line (same cost shares and ordering under the other elasticity central and
  the old clearing), so it is driven by the price audit.
- The one-year pulse (PULSE1) is rebuilt on the single-definition disruption
  timeline (F-021); the global carbon-retention fallback is deleted (F-019);
  the SOC trajectories run in the current eps_F_N family (F-018).
- Dale's August comments are addressed: FAOSTAT anchor stated as the
  2019-2021 mean with the vintage rationale, the dangling optional-analyses
  cross-reference repaired, the AI-use statement moved to back matter, the
  mean-SOC price-application procedure stated operationally in Methods, and
  the response letter carries a fifth analysis item disclosing the clearing
  change.

Final verification (logs/run_238_verify.log at commit `56a4246`, unchanged by
the subsequent document-only commits):

- 15 analytical test suites passed; build graph 31 nodes, 30 OK, 1 BLOCKED
  by design (mc_ensemble, owed a CHECK rerun under audited prices).
- Claim register: 19 claims, 70 checks; baseline refrozen under F-025 with
  every refreeze authorised by a FINDINGS entry.
- Manuscript, SI and author response validate against their v14/base
  originals with every text change tracked as "Matthew Wallenstein".

Central results:

- Production-weighted global S3 loss: 2.32%, 3.18%, 3.30% at years 1/10/30.
- Regional year-10 range: 1.21% (East Asia) to 5.51% (FSU/Central Asia);
  South Asia 5.10%, Sub-Saharan Africa 4.92%.
- Year-10 food-price responses: 2.70% (East Asia) to 12.96% (FSU); global
  7.12%.
- Net revenue at regional mean SOC under a 100% spike: SSA +0.0%, Latin
  America -1.5%, North America -1.0%, South Asia -6.8%; half-to-mean SOC
  retains 0.3-1.0 percentage points.
- 30-year SOC decline: South Asia 2.23% and SSA 2.24%, not separable.
- PULSE1 year-5 residual 0.038%.

Reproducibility: `reproducibility/ERFS-100341-soil-resilience_v17.bundle`
(git bundle, d979a0c..9efffb6 on the v15_base history) and
`v17_release_code_docs.zip`. Restore procedure in the repository's
HANDOFF.md. GitHub push (branch v15-rebuild) awaits Matthew's credentials.

Open before submission: Dale's sign-off on the reframing and the clearing
disclosure; the mc_ensemble CHECK rerun; the clean standalone model
description document.
