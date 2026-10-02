# POST-HOC EXPLORATORY AUDIT

> **These analyses were specified AFTER observing the primary results. They are exploratory diagnostics.**

All tables and the calibration figure carry the same post-hoc status. The script makes no JEV calls and leaves the frozen `results/` artifacts unchanged.
Human vote shares are a trained-rater reference distribution, not ontological truth. JEV `noul` is the provider-documented probability of the bounded Yes/True judgment; it is not defined as a human-vote probability. These analyses empirically examine whether it maps to that reference distribution.

## What the frozen primary analyses establish

- The prespecified hard-minus-probability squared-loss difference was 0.03452 (95% comment-bootstrap interval 0.03357 to 0.03550); positive values favor the probability over its own hard threshold under squared loss.
- The frozen hard-minus-probability absolute-loss difference was -0.02570 (95% interval -0.02685 to -0.02460); negative values favor the hard label under absolute loss.
- On observed held-out MFRC source cells, soft-probability MAE was 0.10278 (cell-size weighted 0.10531) and hard-label MAE was 0.06709 (weighted 0.06804). These are selected-sample cell means, not Reddit population prevalence.
- The frozen entropy-validity macro Spearman correlation between `H(p)` and `H(h)` was 0.2925 (95% comment-bootstrap interval 0.2871 to 0.2974). It does not by itself show that model entropy specifically identifies mixed human judgments.

## What these post-hoc analyses can and cannot establish

They can describe marginal mismatch and held-out loss changes for this pinned sample, compare ranking signals for a defined mixed-case outcome, and evaluate a calibration map fit on development items. They cannot establish transport to other samples or models, validate a deployable human-review policy, or turn trained-rater shares into truth.

### A. Does p contain useful information?

The frozen soft-versus-hard squared-loss comparison says that retaining probability values helped under that loss. It is a within-instrument representation comparison, not an independent validation claim. The new held-out method table compares raw, hard, development-prevalence, and development-calibrated predictions under squared loss, absolute loss, bias, and source-cell MAE.
For the macro held-out rows, raw p had squared loss 0.05482 and absolute loss 0.15049; dev-fitted calibration had squared loss 0.02288 and absolute loss 0.08903. Its calibrated-minus-raw squared-loss difference was -0.03194 (95% paired interval -0.03269 to -0.03115), and the absolute-loss difference was -0.06145 (95% interval -0.06242 to -0.06047).
A ranking score can order comments usefully while its numeric values are miscalibrated. Calibration changes numeric scale and offset; discrimination concerns order. The affine mapping is fitted by weighted least squares on the 1,000 canonical development items only (item weight = retained annotator count), then clipped and applied unchanged to all held-out rows. The constant baseline is the unweighted mean development human vote share by foundation.

### B. Is raw p calibrated to the human vote distribution?

Across foundations, held-out mean signed raw bias was 0.10476 (positive means p exceeds the human vote share on average). See `marginal_calibration.csv` for DEV and TEST means, losses, and foundation-specific biases; `reliability_bins.csv` and the figure show binned means.
The reliability bins are equal-frequency. Items tied on probability are ordered deterministically by item ID, so a tie can span adjacent bins. Agreement with the diagonal describes marginal calibration at those bins; it does not measure ranking quality. `raw_probability_sd_sharpness` and mean forecast entropy describe prediction concentration (sharpness): more concentrated probabilities can still be miscalibrated or poorly ranked.

### C. Does H(p) specifically recover disagreement?

On held-out data, mean ROC-AUC was 0.7335 for entropy and 0.7827 for raw p when predicting whether human share was mixed; entropy had the higher AUC in 0 of 6 foundations. Foundation-level prevalence and mean p for all-zero, mixed, and all-one reference cases are in `entropy_disagreement.csv`.
If raw p predicts mixed cases as well as or better than entropy, entropy is not uniquely detecting human uncertainty: p may carry a positive-score-strength signal. AUC measures discrimination for this binary target, not marginal calibration or probability sharpness. The three questions remain separate.

### Selective-review diagnostics

At 20%, held-out matrix MSE was 0.03323 for entropy targeting, 0.02929 for mean-probability targeting, 0.04387 for random review, and 0.02471 for oracle-error targeting. The table includes every configured budget. Random review resamples comments; all six predictions for a selected comment are replaced by its full human vote-share vector. Oracle-error ranking uses actual held-out JEV MSE and is an upper-bound diagnostic only, never a deployable strategy. Entropy versus mean-probability targeting tests whether entropy adds triage value beyond positive-score strength.

### Strict wording sensitivity

On the fixed 1,000-case sensitivity subset, macro canonical-to-strict probability Spearman correlation was 0.9670. Strict-minus-canonical squared-loss change was -0.01885 (95% paired interval -0.02043 to -0.01734); absolute-loss change was -0.03158 (95% interval -0.03359 to -0.02969); change in absolute aggregate signed-bias magnitude was -0.04051 (95% interval -0.04404 to -0.03894).
A high canonical/strict rank correlation alongside a change in signed-bias magnitude is consistent with wording affecting calibration more than ordering in this subset. It does not identify a general wording effect; canonical remains the frozen primary instrument.

## Files

- `calibration_by_foundation.png` — equal-frequency reliability diagnostics, including held-out raw and development-calibrated probabilities.
- `marginal_calibration.csv` and `reliability_bins.csv` — raw DEV/TEST summaries and binned calibration diagnostics.
- `dev_calibration_parameters.csv`, `heldout_method_comparison.csv`, and `calibration_paired_bootstrap.csv` — development fit and held-out comparisons.
- `entropy_disagreement.csv`, `selective_review.csv`, `strict_sensitivity.csv`, and `strict_sensitivity_bootstrap.csv` — disagreement, review, and wording diagnostics.

No calibration parameter was fitted from test outcomes. No original results, estimands, prompts, approval records, or predictions were changed.
