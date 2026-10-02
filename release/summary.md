# Analysis summary

Generated from prespecified held-out analyses. Interpret human shares as trained-rater reference distributions, not truth.

- Held-out comments: 16709
- Sensitivity comments: 1000
- Uncertainty-validity macro Spearman: 0.2925 [0.2871, 0.2974]
- Hard minus probabilistic squared loss to human vote share: 0.034519 [0.033568, 0.035504]
- Soft source-cell MAE: 0.102782
- Hard source-cell MAE: 0.067086

**Unequal-cell-size robustness diagnostic (cell-size-weighted source-cell MAE; unweighted MAE remains primary):**
- Soft: 0.105312
- Hard: 0.068041

The squared loss is to the human vote share, not itself a Brier score. The hard-minus-probability squared-loss difference equals the same difference averaged over retained individual binary coder judgments because the within-comment coder-variance term cancels.

Selective-review results are an oracle / idealized reference-replacement simulation: reviewed comments receive their human vote shares. They do not estimate labor time or minutes saved.

Source-cell results describe recovery of observed MFRC sample-cell means in this selected sample. They do not estimate population prevalence or causal effects.

Canonical wording remains primary; strict wording is reported only as the prespecified held-out sensitivity check.

See results/tables/ for foundation-specific diagnostics, absolute-loss secondary results, and selective-review results.

## Secondary robustness

Pairwise coder disagreement is the coder-count-adjusted robustness operationalization `2*n*h*(1-h)/(n-1)`. It is not a new target; coder disagreement may reflect ambiguity, perspective, coder noise, missing context, or task limits.

| Foundation | Spearman rho |
|---|---:|
| Care | 0.3893 |
| Equality | 0.3008 |
| Proportionality | 0.2272 |
| Loyalty | 0.1922 |
| Authority | 0.3411 |
| Purity | 0.3041 |
| Unweighted macro mean | 0.2925 |

Primary bootstrap unit: item/comment row. Exact-text content-cluster unit: exact-text content_id cluster.
Every held-out row has a unique content_id, so row and exact-text content-cluster units coincide and no separate content-cluster bootstrap was run.
