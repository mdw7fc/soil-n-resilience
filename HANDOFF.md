# HANDOFF — ERFS-100341 "Soil organic matter buffers fertilizer supply disruptions"

Wallenstein & Manning, *Environmental Research: Food Systems*. v15 rebuild line,
written 2026-08-29 at commit `56a4246`. This file is the entry point for anyone
(or any session) picking the project up cold. The older
`HANDOFF_v15_model_assurance.md` describes the original rebuild plan and is
superseded by this file wherever they disagree; FINDINGS.md is the full ledger
and wins over both.

## Where the truth lives

The durable copy is the git bundles in the Mac folder
`.../paper2-soil-resilience/v15/`. Working containers are ephemeral and have
been reclaimed mid-session at least four times; nothing that matters may exist
only in a container. To restore a working tree:

    git clone <path>/v15_base.bundle repo        # full history through d979a0c
    cd repo
    git fetch <path>/v15_f025.bundle HEAD:work   # the 9 commits on top
    git checkout work                            # HEAD 56a4246
    make verify                                  # expect: 15 suites, exit 0

`v15_release_code_docs.zip` in the same folder is a snapshot of the tree for
humans; the bundles are authoritative. The Mac v15 folder's own git checkout is
stuck at `d979a0c` because the device bridge cannot clear `.git/index.lock`;
apply the bundles from a machine with a normal shell rather than fighting it.

Rule of the whole rebuild: when a document and the model disagree, the default
repair is to the document. `docs/claims.yaml` is the claim register (19 claims,
70 checks); `make verify` runs it and everything else.

## State of the model

The model is green: 17 suites and the 32-node build graph, exit 0, nothing
blocked (the MC ensemble regenerated under F-026). The registry central
eps_F_N is -0.50 (F-026, closing the code-document fork an external audit
found); the claim register reads 70/70 AGREES on document_basis v17. The
defining structural change:

**F-025 — realized-yield market clearing.** The food price is root-found at
every timestep so that demand equals the production response of the monthly
biophysical model itself (`CoupledMonthlyModel._clear_market_realized`);
each candidate price implies a fertilizer rate, the nitrogen balance runs at
that rate from a soil-state snapshot, and the price is accepted when
eta*PY = ln(yield_frac) + alpha*lambda_L*PY. The Mitscherlich elasticities
beta and gamma are recorded diagnostics and enter nothing. The supply ceiling
is a quantity constraint inside the residual, so no separate capped solver
exists. `run_price_shock_analysis.py` (Figure 1) clears the same way.
`code/repro/test_cap_market_clearing.py` (third form) verifies the structural
equations from reported columns at every step and requires the old linear
supply relation to disagree somewhere, so it cannot pass both clearings.

How this came about, in order: F-022 (Dale Manning asked which of two yields
the price was clearing; measurement found the equilibrium exact but clearing a
log-linear production change, gap 1.54 pp worst), F-023 (decomposition: a land
accounting term plus a within-step linearization that runs 0.44 pp too
optimistic in the chronic phase; price bias about -1 pp at year 10), F-024
(prototype; its linear mode reproduced the old model to 0.0e+00 before its
realized mode was believed), F-025 (adoption; both refreeze gates fired and
were answered with documented refreezes). Pre-change diagnostics are frozen in
`baseline/f022_f025_evidence/` and are not regenerable — rerunning them would
measure a gap the model no longer has.

Earlier landmarks still load-bearing: F-018 (v15 tree lost and recovered from
the Mac; eps_F_N family signature: year 1 identical, later years diverge),
F-019 (cre_base deleted; `region_cre()` raises instead of substituting 0.11),
F-020 (widened mutation fingerprint; three price parameters scored UNTESTED),
F-021 (PULSE1 rebuilt on the `supply_state()` seam — the disruption timeline
exists exactly once; square pulse, inclusive boundary, and the reason).

## Current headline numbers (post-F-025)

Global S3 loss 2.32 / 3.02 / 3.07 % at years 1/10/30 (F-026: eps_F_N central
restored to -0.50). Regional year-10: EA 1.18, NA 1.72, LATAM 2.41, SEA 3.67,
EU 3.41, SSA 4.74, SA 4.80, FSU 5.09. Year-30: SSA 5.08, SA 4.79, FSU 5.19.
SOC 30-year decline: SSA 2.14 %, SA 2.09 % (they separate under this family;
F-018 not-separable was a zero-central property). Food price indices, year
10: NA 5.71, EU 10.24, EA 2.64, SA 8.32, SEA 6.83, LATAM 4.74, SSA 6.92, FSU
11.89 %, global 6.72 (year 1 global 5.20, range 2.44–10.29).
Figure-1 margin gaps mean-vs-half SOC: NA 0.98, SSA 0.93, LATAM 0.69, SA 0.27
pp; SSA at the mean is essentially unaffected (+0.0 %) and South Asia is the
margin-vulnerable region (nitrogen cost share ~15 % vs SSA ~4 % at audited
prices — the manuscript's framing was inverted and has been corrected).
PULSE1: year 1 2.317 (equals S3 by construction), year 5 residual 0.010 %.
SC1 year-10 3.70, SC2 1.88 falling to 0.08 by year 20. Year-10 SOC gradients
are non-monotone at low SOC in LATAM, SSA and FSU (conditional on the global
-0.50 elasticity; year-1 buffering is universal, MC P = 1.0 everywhere). S3 realized fertilizer
reduction: 19 % (state the averaging basis). WHC sensitivity: 3.5 mm per pp
SOC central (Minasny & McBratney 2018), 8.4 is the upper bound only. SSA
nitrogen price: $2.30/kg N.

## Documents

Three document generations exist. Do not mix them:

- **7_22 set** — what Matthew sent Dale in August. Carries a pre-rebuild
  number family roughly 2x the current model's. Reference only; never
  propagate its values.
- **v14 set** (`resumbission/v14/`) — the basis the claim register tracks
  (`document_basis: v14` in claims.yaml).
- **v17 set** (`resumbission/v17/`, also committed in-repo; released 2026-08-29, RELEASE_AUDIT_v17.md there) — the
  live set: v14 base plus tracked changes, every insertion and deletion
  authored "Matthew Wallenstein". Contains all 29 drifted-number restatements,
  the realized-clearing Methods description, the mean-SOC operational
  sentence, the AI statement moved to back matter, WHC 3.5, and re-embedded
  Figures 1, 2, S4, S6, S8, S10. The author response is built on Dale's own
  commented file (his comments preserved, all four answered, a fifth
  "analysis tasks" item disclosing the clearing change).

Editing procedure that works: unzip the docx, run the docx skill's
`merge_runs.py`, do anchored string replacement wrapping edits in
`<w:ins>`/`<w:del>` (author "Matthew Wallenstein"), rezip, and validate with
`validate.py --original <base> --author "Matthew Wallenstein"`. Anchors must
sit inside a single run; when a needle fails, dump the run texts and shorten
the anchor. All file metadata (creator, lastModifiedBy) is set to Matthew
Wallenstein; keep it that way.

## GitHub

`CITATION.cff` names https://github.com/mdw7fc/soil-n-resilience. It is
reachable read-only from the container; its HEAD (`7026193`) predates the v15
lineage. Neither the container nor the device VM holds credentials, so the
push is Matthew's, from any machine with his auth:

    git clone <path>/v15_base.bundle soil-n-resilience && cd soil-n-resilience
    git fetch <path>/v15_f025.bundle HEAD:v15-rebuild
    git remote set-url origin https://github.com/mdw7fc/soil-n-resilience
    git push origin v15-rebuild

Push as a branch; do not force over the old main without deciding that
deliberately.

## Open work, in rough priority

1. Dale's sign-off on two things: adopting the realized clearing (done on
   Matthew's instruction; Dale called it "possibly preferred") and the
   inverted SSA/South-Asia margin framing in the discussion.
2. `mc_ensemble` is BLOCKED: the deposited ensemble is the only surviving v15
   one and F-013's claim strength reproduces against it. It is owed a rerun as
   a CHECK (snapshot first). Under realized clearing a rerun costs ~8x the
   biophysical work; it is year-1-scoped so the multiplier bites less, but
   budget hours, and expect P3/P4 strength numbers to need a FINDINGS entry.
3. The mutation-coverage sweep (F-020 harness) has not rerun under F-025.
   The three UNTESTED price parameters (`crop_price_usd_t`, `n_price_wedge`,
   `n_benchmark_usd_kg`) move 164 published fields each and still have no
   test; the 22 UNTESTED biophysical leaves from F-011 stand.
4. C-033 (South Asia farmer-paid net revenue, 0.95 → -0.59 %) drifted and its
   sentence lives in a document variant ("SI-sol [81]") not located in the
   v14 SI; find it before the next freeze or retire the claim to the register.
5. The 4-pool re-port (coupled_4pool, som_4pool_monthly, run_4pool_comparison,
   run_cue_decomp_matched) and the F-004 planting-month registration remain
   from the original plan; README:156-158's "known gap" paragraph waits on the
   4-pool port.
6. Zenodo/journal deposit refresh once Dale approves the v17 set.

## Environment gotchas (all learned the hard way)

Containers reset cwd between Bash calls and are killed at ~2 min per call
(poll with short-sleep loops; `nohup ... &` for long jobs — and note that
`cd X && cmd &` backgrounds the `cd` too). pytest is not installed; tests run
standalone via `make`. The device bridge cannot delete files or clear
`.git/index.lock` (no git on the Mac from here; use `device_commit_files`,
20 MB/file cap). SendUserFile caps at 30 MiB. `Edit` on repo files is less
reliable than `python3` heredocs with an assert-count before writing; anchor
indented needles with a leading newline. Output discipline: long output goes
to `logs/run_NN.log`, read back tails and counts, findings get written to
FINDINGS.md before they get reported.
