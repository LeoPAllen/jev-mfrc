# Prespecified methods

## Representation

Literal exact duplicate rows are counted and removed separately. Repeated judgments by one annotator are collapsed only when the parsed unordered label sets and confidence agree. The current official snapshot has 42 items with irreconcilable same-annotator repeats; all rows for those items are excluded rather than adjudicated. Disagreement across annotators is retained in the trained-rater reference distribution.

For each eligible comment `i` and foundation `f`:

- `h_if`: fraction of retained trained annotators selecting the foundation;
- `p_if`: the provider-documented JEV `noul` probability that the canonical bounded presence question is answered Yes/True. This is a probability of that judgment, not a moral-intensity score;
- `q_if = I(p_if >= .5)`.

Use natural-log binary entropy with `H(0)=H(1)=0`. Entropy is a symmetric uncertainty index, not construct intensity.

## 1. Uncertainty validity

For each foundation separately, compute Spearman correlation across held-out comments between `H(p_if)` and `H(h_if)`. The primary summary is the unweighted macro mean of the six correlations.

Bootstrap by resampling comments with replacement; in every draw keep all six foundations for a selected comment together. Report percentile 95% intervals for the foundation correlations and their unweighted macro mean. The macro mean is primary; foundation-specific estimates are diagnostics. This avoids treating six codes from the same comment as independent and avoids a pooled correlation being driven by cross-foundation base rates.

Secondary: average JEV entropy across foundations for each comment and correlate it with reverse-coded mean annotator confidence. Public confidence levels are mapped monotonically to 0/.5/1 and reversed as `1 - mean_confidence`; Spearman is invariant to monotone affine recoding.

## 2. Information retained by probabilities

Compare the same JEV prediction before and after thresholding, so model differences cannot explain the contrast.

Primary: squared loss to human vote share, `(prediction - h_if)^2`. Secondary: absolute loss `|prediction - h_if|`.

Report mean hard-minus-soft loss, overall and by foundation, with comment-cluster bootstrap intervals. Positive values mean the probability preserves information relative to its own hard version.
This is a representation comparison, not an independent validation test of JEV: it quantifies what is lost by thresholding the same instrument under the chosen human-reference loss. Its interpretation therefore belongs alongside the uncertainty-validity and downstream-recovery analyses.

Do **not** call squared loss to a fractional vote share itself a Brier score. However, for binary coder judgments, the *difference* in average Brier loss between two predictions is exactly the same as their difference in squared loss to the coder vote share: the within-item coder-variance term `h_if(1-h_if)` is common to both predictions and cancels. This gives the hard-vs-soft squared-loss comparison a direct individual-coder interpretation without pretending the vote share is a realized binary outcome.

## 3. Selective review simulation

Review is at the comment level. Aggregate JEV uncertainty is mean entropy across the six probabilities.

For budgets `{0, .05, .10, .20, .30}`:

- targeted oracle: replace all six JEV probabilities with the human vote shares for the highest-uncertainty `b` fraction of comments;
- random oracle: replace the same number of random comments, repeated 200 times.

Outcome: matrix MSE to the full human vote-share reference. Report targeted MSE, random mean/interval, and targeted-minus-random difference.

This is an **oracle / idealized reference-replacement simulation**. It asks whether model uncertainty ranks valuable review cases. It does not estimate actual minutes saved or the outcome of adding one more coder.

## 4. Downstream descriptive recovery

For each observed held-out MFRC `(bucket, subreddit) × foundation` source cell, calculate the human-share mean and estimates from soft JEV, hard JEV, targeted-review hybrids, and random-review hybrids.

Primary recovery loss is unweighted mean absolute deviation across observed cells; cell-size-weighted mean absolute deviation is a sensitivity diagnostic.

These are MFRC **sample-cell means**, not natural subreddit prevalence, because corpus sampling enriched moral content. They describe recovery in this selected sample and do not support population-prevalence or causal claims.

## Prespecified secondary robustness checks (frozen before instrument approval)

These checks do not change the primary entropy, loss, oracle-review, or source-cell estimands above.

### Coder-count-adjusted disagreement

Empirical binary entropy of `human_share` remains the primary coder-disagreement quantity. As a secondary robustness operationalization, for `n` retained annotators and positive share `h = k/n`, compute the proportion of unordered coder pairs that disagree:

`pairwise_disagreement = 2 * n * h * (1 - h) / (n - 1)`.

For each foundation, correlate JEV binary entropy with this pairwise disagreement using Spearman, and report the six foundation estimates and their unweighted macro mean. These are point estimates. Pairwise disagreement is not a new target or a truth measure: annotations are a trained-rater reference distribution, and coder disagreement can reflect ambiguity, perspective, coder noise, missing context, or task limitations.

### Exact-text dependence and bootstrap units

The primary bootstrap unit is the item/comment row, with all six foundations on a selected row kept together. The pinned prepared test set has 16,709 rows and 16,709 unique `content_id` values (see `docs/DATA_AUDIT.md`). Thus each exact-text cluster contains one row, the row and content-cluster units coincide, and no separate content-cluster bootstrap is applicable. Content clustering addresses dependence from exact duplicate text only.

### Unequal source-cell sizes

The pinned held-out sample has 12 observed `(bucket, subreddit)` cells. Their item counts are min 98, Q1 1,258.5, median 1,415.5, Q3 1,657.75, and max 2,478 (linear-interpolation quartiles; `docs/DATA_AUDIT.md`). Keep unweighted source-cell MAE as primary. Report cell-size-weighted MAE as the unequal-cell-size robustness diagnostic; do not filter cells or add a minimum-cell-size cutoff. This weighting does not turn the estimates into Reddit population prevalence.

## Wording sensitivity

Run the fixed strict bundle only on the preselected held-out sensitivity subset. Report mean absolute probability change, hard-label flip rate, canonical/strict correlation, and direction of the main loss/uncertainty results on that subset.

Canonical remains primary regardless of comparative performance. Strict wording is never run as a competing development specification.

## Statistical emphasis

This is a diagnostic methods study, not a large null-hypothesis-testing exercise. Prefer effect sizes, paired losses, correlations, bootstrap intervals, and plots. Foundation-specific estimates are diagnostics, not six independent confirmatory hypotheses.
