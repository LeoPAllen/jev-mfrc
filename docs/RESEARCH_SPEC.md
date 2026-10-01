# Frozen research specification

## Research question

When a decision model returns probabilities for theory-derived text codes, does that uncertainty correspond to disagreement among trained human coders, and can researchers use it to preserve measurement information and allocate scarce human review?

## Data

Use the Moral Foundations Reddit Corpus (MFRC), public Hugging Face dataset `USC-MOLA-Lab/MFRC`, split `train_dedup`.

On first use, resolve the requested dataset revision and save one local raw snapshot with its SHA-256 and resolved revision. Reuse that snapshot only when its checksum and source metadata verify; stop on a missing manifest, partial snapshot, checksum drift, or config mismatch. Do not silently refresh or replace raw data.

Primary constructs are six nonexclusive binary domains: Care, Equality, Proportionality, Loyalty, Authority, and Purity. Thin Morality and Implicit/Explicit Morality are outside this MVP.

MFRC deliberately enriched morally loaded content. Preserve both `bucket` and `subreddit`; downstream source cells are `(bucket, subreddit)`. Source-cell estimates are therefore recovery of this selected MFRC sample, not natural Reddit prevalence.

## Unit and inclusion rule

The public release exposes no canonical comment ID. Reconstruct an item from the exact `(bucket, subreddit, text)` tuple and hash that tuple. This keeps source-cell identity explicit.
Because the release has no lower-level comment identifier, two physically distinct comments with identical text in the same `(bucket, subreddit)` cell cannot be distinguished from repeated annotation rows. Treating the tuple as one item is therefore an explicit reconstruction assumption to report, not a hidden claim about original comment identity.

Exact text can legitimately appear in multiple source cells. Preserve those source-specific items, but assign every item sharing the exact same text to the same dev/test side so literal text leakage cannot cross the split.

After exact-row deduplication, require at least 3 unique annotators per reconstructed item. Preserve excluded items in an audit table. If one reconstructed item has conflicting rows for the same annotator, stop rather than guess.

## Human reference

For comment `i` and foundation `f`:

`human_share_if = number of retained annotators selecting f / number of retained annotators`.

This is the observed distribution of trained judgments, not truth or a precisely estimated latent probability. When a human-majority category is needed for diagnostics, a share above 0.5 is a positive majority, a share below 0.5 is a negative majority, and exactly 0.5 is a tie. Primary analyses retain the fractional shares directly.

Coder confidence is secondary because it applies to the whole multilabel annotation rather than one foundation.

## Splits fixed before JEV inference

- Seed: 42.
- Development target: 1,000 eligible items selected by deterministic exact-text-group rank. Grouping can make the realized count slightly exceed the target.
- Held-out test: all remaining eligible items.
- Sensitivity target: 1,000 deterministic held-out items, also selected by exact-text groups.

No rebalancing, stratification, or split change after seeing JEV output. A different split/config produces a different experiment fingerprint; cached calls from the old experiment are ignored rather than reused.

## Instruments

Primary: one canonical bundle of six independent Noul questions sent together in one JEV request per comment. Development inference uses **canonical only**.

Each `noul` value is interpreted as the provider-documented probability of Yes/True for that bounded foundation-presence question; it is not a measure of moral intensity.

Sensitivity: one stricter wording bundle on the preselected held-out sensitivity subset only. It is specified before held-out inference and cannot replace canonical because it performs better.

Before paid calls, record the configured request model's advertised metadata. Always request that configured model name, record the response model, and stop if observable model identity drifts within the experiment. Do not assume the response-model string is itself a requestable pin.

The instrument bundle is human-approved after dev-only review. Any later wording edit invalidates that approval.

## Primary analyses

1. **Uncertainty validity:** within each foundation, relate JEV binary entropy to human vote entropy; macro-average six foundation-specific associations with comment-cluster bootstrap interval. Secondary: JEV uncertainty versus reverse coder confidence.
2. **Information retention:** compare native JEV probabilities with their own `p >= .5` hard labels against human vote shares using paired squared loss (primary) and absolute loss (secondary).
3. **Selective review simulation:** at fixed comment-level budgets, compare uncertainty-targeted versus random **oracle/reference replacement**.
4. **Downstream descriptive recovery:** compare soft, hard, and review-hybrid recovery of `(bucket, subreddit) × foundation` MFRC sample-cell means.

## Claims outside scope

- JEV probability is not moral-foundation intensity.
- Human vote share is not ontological truth or a definitive latent construct score.
- Human disagreement can reflect ambiguity, perspective, coder noise, missing context, or task limitations; the design does not identify which mechanism generated each disagreement.
- Source-cell comparisons are descriptive and not causal or population prevalence estimates.
- One dataset and one model family do not establish universal generalizability.
- Null or adverse results are valid and may not be specification-shopped away.
