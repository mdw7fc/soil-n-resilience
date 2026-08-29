# v17 release audit

Status: **PASS** — second pass, after the external audit of 2026-08-29.

v17 is the v14 document base carried forward through F-018 to F-026. The
external audit's central finding was a code-document fork: the SI declared
the soil-N depletion elasticity "default -0.50; active in S3, SC1 and SC2"
while the code central sat at 0.0. F-026 restored the documented -0.50,
regenerated the entire graph and the 1,000-draw Monte Carlo ensemble, rebased
the claim register onto this document set, and re-included the two previously
excluded release tests. Where v16_sol and v17 differ, v17 supersedes it: same
central elasticity, plus the realized-yield market clearing (F-025) that
v16_sol listed as a limitation.

Final verification (logs/run_250_verify.log, and reproduced from a clean
checkout):

- 17 analytical test suites passed, including the re-included
  parameter-consistency and cross-document-consistency tests; build graph
  32 nodes, 0 blocked, 0 orphans.
- Claim register: 19 claims, 70 checks, 70 AGREES, 0 DRIFTED,
  document_basis v17.
- Monte Carlo regenerated under the corrected clearing and audited prices:
  buffering universal (P = 1.0 in every region, 1,000 draws), median
  cross-region buffer 0.88 pp, global median year-1 loss 2.51%,
  P(SSA worst year-1 net revenue among priced regions) = 0.001.
- Manuscript, SI and response letter validate with every change tracked as
  "Matthew Wallenstein"; Table S4, Figure S6's annotation, and Figures 1, 2,
  S4, S6, S7, S8, S10, S11, S12, S13 are re-embedded from the regenerated
  artifacts.

Central results (eps_F_N = -0.50, realized clearing):

- Production-weighted global S3 loss: 2.32%, 3.02%, 3.07% at years 1/10/30.
- Regional year-10 range: 1.2% (East Asia) to 5.1% (FSU/Central Asia);
  South Asia 4.8%, Sub-Saharan Africa 4.7%.
- Year-10 food-price responses: 2.6% (East Asia) to 11.9% (FSU); global 6.7%.
- Net revenue at regional mean SOC under a 100% spike: SSA +0.0%, Latin
  America -1.5%, North America -1.0%, South Asia -6.8%; half-to-mean SOC
  retains 0.3-1.0 pp. Nitrogen-cost shares: South Asia 14.7%, NA 6.1%,
  LATAM 4.9%, SSA 3.6%.
- 30-year SOC decline: SSA 2.14%, South Asia 2.09%.
- One-year-pulse year-5 residual: 0.01%. S3 realized fertilizer reduction:
  approximately 19%.
- Year-10 SOC gradients are non-monotone at low SOC in LATAM, SSA and FSU
  under the global -0.50 elasticity; the documents state year-1 buffering as
  universal and year-10 gradients as conditional on that scenario assumption.

Reproducibility: reproducibility/ERFS-100341-soil-resilience_v17.bundle and
v17_release_code_docs.zip (refresh both from the current commit). Restore
procedure in the repository's HANDOFF.md.

Open before submission: Dale Manning's sign-off on the net-revenue reframing,
the restored elasticity disclosure in the response letter, and the AI-use
statement's scope (Claude is named; a second AI system, Codex, assisted the
independent audit and the v16_sol reconstruction, and whether the journal's
policy requires naming it is Matthew's decision). The four-pool Figure S5
retains its indicative-only caveat pending the 4-pool re-port.
