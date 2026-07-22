
## v12b — y_max reconciled to the simulation-calibrated ceiling
The reported calibrated ceiling y_max is now the value the coupled model actually
uses: `get_calibrated_ym`, a **Brent's-method root-finder** that adjusts y_max until
the model's **year-2 simulated yield** (under current N inputs and
the ERA5 climate) equals the FAOSTAT target — not a closed-form Mitscherlich inversion.
Canonical y_max (t/ha): NA 6.28, EU 6.12, EA 6.09, SA 3.64, SEA 4.87, LATAM 5.60,
SSA 3.88, FSU 4.29.

- **Table S4**: y_max → canonical; the no-synthetic-N column is now the **simulated**
  year-2 yield (`y_no_synth_sim`), e.g. NA 3.71, EU 2.29, SSA 1.26 (was closed-form
  5.08/4.42/1.28). Column relabeled pred → sim.
- **SI Methods**: closed-form `y_max = y_obs/(1−e^(−cN))` replaced with the numerical
  year-2 procedure; `yield_max_regional` noted as a legacy fallback parameter.
- **SI emergent-yield text**: "≈5.1 (NA), 4.4 (EU)" → "≈3.7 (NA), 2.3 (EU)" (SSA ≈1.3
  unchanged); still well above the empirical floors.
- **Figure S12**: regenerated as simulated N-response curves (FAOSTAT points now lie on
  the curves; canonical y_max asymptotes).
- **Figure S14 (OFRA)**: regenerated with SSA y_max=3.88 (was 3.47); conclusion revised
  from "low edge of the OFRA envelope" to "below the observed median but within the
  interquartile range." Observed OFRA envelope (n=364, B_gain>0) unchanged.
- **Response letter**: SSA ceiling 3.47→3.88; OFRA conclusion reworded to match.
- **Manuscript**: SSA SOC decline 6.2%→6.0% (canonical total-SOC value).
- Losses are reported as fractions of baseline; they vary slightly with y_max through
  residue return and SOM feedbacks and were NOT claimed to be y_max-independent — the
  reported losses use the y_max the canonical run actually applied.
