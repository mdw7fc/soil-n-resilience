# Makefile -- reproduce and gate the ERFS-100341 deposit (F-009, F-014).
#
#   make status     what the build graph thinks of the tree right now
#   make verify     THE GATE. Test suites plus the build graph. Exit 1 on any
#                   defect. This is what CI and a reviewer should run.
#   make all        regenerate every artifact except the Monte Carlo ensemble
#   make everything regenerate every artifact including the ensemble (~2 h)
#   make stale      regenerate only the nodes the graph reports as not OK
#   make figures    the figure nodes only
#   make tests      the test suites, without the build graph
#   make graph      topological order and cost per node
#   make baseline   record the pre-graph nodes once, at stand-up
#
# Nothing here writes a frozen baseline. `data/benchmarks/baseline_verdicts.json`,
# `docs/claims_baseline.json`, `docs/claims_index_baseline.json` and
# `docs/claim_strength_baseline.json` are rewritten only by their generator's
# explicit `--write-baseline`, because a baseline that a build target can
# refresh is not a baseline (F-008, F-009, F-012).

PY      := python3
BUILD   := $(PY) code/build.py
LOGDIR  := logs/build

# ---------------------------------------------------------------------------
# Test suites run by `make verify`.
# ---------------------------------------------------------------------------
# Ordered cheapest first so a broken model is reported in seconds rather than
# minutes.
TESTS := \
	code/tests/test_seam_contracts.py \
	code/tests/test_calibration_fingerprint.py \
	code/tests/test_claims.py \
	code/repro/test_parameter_boundaries_sol.py \
	code/repro/test_dimensional_consistency_sol.py \
	code/repro/test_zero_shock_invariance.py \
	code/repro/test_full_zero_shock_sol.py \
	code/repro/test_cap_market_clearing.py \
	code/repro/test_mc_robustness_sol.py \
	code/tests/test_benchmark_baseline.py \
	code/tests/test_spinup_partition_independence.py \
	code/tests/test_wp1_registry_wiring.py \
	code/tests/test_soc_trajectories.py \
	code/tests/test_supply_state.py \
	code/repro/test_parameter_extremes_sol.py

# Excluded from the gate, by name and with the reason. A test excluded without
# a reason written down is a test somebody turned off.
#
#   test_parameter_consistency_sol.py
#       Red on purpose. Three derived nitrogen cost shares moved when F-002
#       recalibrated the production path (SSA 0.0358 against a hardcoded
#       0.037); the repair is a document edit owed to the claim register, not a
#       code change. Re-include it the moment that edit lands.
#
#   test_cross_document_consistency_sol.py
#       Reads the manuscript .docx, which is not in the deposit. It belongs to
#       the document packages (D1/D2), where the file is present.
#
#   run_mutation_coverage.py
#       A 20-minute sweep, not a gate. `make mutation` runs it.
EXCLUDED_TESTS := \
	code/repro/test_parameter_consistency_sol.py \
	code/repro/test_cross_document_consistency_sol.py

.PHONY: all everything stale figures tests verify status graph baseline \
        stamp mutation clean-build help

help:
	@sed -n '2,20p' Makefile

status:
	@$(BUILD) status

graph:
	@$(BUILD) graph

baseline:
	@$(BUILD) baseline

fingerprint:
	@$(BUILD) fingerprint

# --- regeneration ----------------------------------------------------------

all:
	@mkdir -p $(LOGDIR)
	$(BUILD) run --all --skip mc_ensemble --log-dir $(LOGDIR)

everything:
	@mkdir -p $(LOGDIR)
	$(BUILD) run --all --log-dir $(LOGDIR)

stale:
	@mkdir -p $(LOGDIR)
	$(BUILD) run --stale --log-dir $(LOGDIR)

figures:
	@mkdir -p $(LOGDIR)
	$(BUILD) run figure_1 figure_2 figure_s6 figure_s7 figure_s8 figure_s9 \
		figure_s10 figure_s11 figure_s12 broadbalk_benchmark \
		hindcast_benchmark ofra_validation --log-dir $(LOGDIR)

stamp:
	@$(BUILD) stamp

# --- the gate --------------------------------------------------------------

tests:
	@rc=0; \
	for t in $(TESTS); do \
		printf '%-58s' "$$t"; \
		if $(PY) $$t > /tmp/$$(basename $$t .py).log 2>&1; then \
			echo "ok"; \
		else \
			echo "FAIL  (see /tmp/$$(basename $$t .py).log)"; rc=1; \
		fi; \
	done; \
	exit $$rc

verify: tests
	@echo ""
	@$(BUILD) verify \
		--allow-unsourced results/s3_shock_calibration.csv
	@echo ""
	@echo "verify: $(words $(TESTS)) suites and the build graph, exit 0"

# The two allowances above are debts, not exemptions. Each is an artifact the
# manuscript cites whose generator does not exist in this tree:
#
#   results/s3_shock_calibration.csv     make_s3_shock_calibration.py (F-015)
#
# data/soc_trajectories.csv|.json was the second line. The generator was written
# in the lost v15 tree, recovered verbatim from the session transcript and
# replayed here (F-018), so the debt is paid and the line is gone rather than
# exempted; soc_trajectories is now a build node.
#
# data/crop_response_calibration_table.csv was the third line. D3 made
# make_table_s4_sol.py write it, which is what MANIFEST.md had claimed since the
# v14 deposit, so the line is gone rather than exempted.
#
# Writing either remaining generator removes its line from this file. Adding a
# third line requires an entry in FINDINGS.md saying why.

mutation:
	$(PY) code/tests/run_mutation_coverage.py

clean-build:
	rm -rf .build
