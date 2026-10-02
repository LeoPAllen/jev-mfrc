# JEV × MFRC reproducibility bundle

This archive contains the completed prespecified held-out analysis and the non-text data needed to recalculate it. It includes 16,709 canonical held-out rows and 1,000 strict-wording sensitivity rows. The canonical file also flags the 1,000 rows selected for wording sensitivity.

## Contents and use

- `analysis.json`, `summary.md`, `tables/`, and `figures/` are the archived analysis outputs.
- `data/canonical_heldout.csv.gz` contains the six JEV probabilities and six trained-rater vote shares for every held-out item.
- `data/strict_sensitivity.csv.gz` contains strict-wording predictions for the prespecified sensitivity subset, alongside the canonical comparison fields.
- `manifest.json` records data and instrument provenance, model identity, analysis settings, the generating Git commit, and SHA-256 hashes for every other release file.

No Reddit comment text, annotator identifiers, raw MFRC files, JEV cache, or API credentials are included. `item_id` and `content_id` are one-way identifiers; the bucket and subreddit fields preserve the specified source-cell analyses.

The data use `human_<foundation>` for trained-rater shares and `p_<foundation>` for the probability of the bounded JEV yes/no judgment. These are not moral-intensity scores or ontological truth labels. Source-cell estimates describe this selected MFRC sample, not Reddit population prevalence.

The pinned analysis settings and seed are in `manifest.json`. The methods are specified in [`../docs/METHODS.md`](../docs/METHODS.md), with implementation in [`../src/jev_mfrc/metrics.py`](../src/jev_mfrc/metrics.py). Recalculation from this bundle requires no JEV request, local call cache, raw MFRC file, or comment text.
