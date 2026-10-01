# Task 04 — analysis and review simulation

Goal: audit whether the code answers the methods-paper questions without adding unnecessary estimands.

Focus on `src/jev_mfrc/metrics.py`, `src/jev_mfrc/report.py`, `docs/METHODS.md`, `tests/test_metrics.py`.

Keep four core analyses:
1. foundation-wise association between JEV entropy and human-vote entropy, with comment-cluster bootstrap and macro summary;
2. native probability versus its own 0.5-thresholded label against human vote share, with squared loss primary and absolute loss secondary;
3. comment-level uncertainty-targeted oracle review versus equal-budget random review;
4. `(bucket, subreddit) × foundation` recovery of MFRC sample-cell means.

Scientific checks:
- Explain that the hard-vs-soft squared-loss difference against human vote share is exactly the same difference one obtains by averaging Brier loss against the retained individual binary coder judgments; the coder-variance term cancels. Do not call the fractional-share loss itself a Brier score.
- Keep all six foundations together when bootstrapping comments.
- Call review simulation "oracle" or "idealized reference replacement" and make no labor-time claim.
- Strict wording stays a sensitivity check and cannot replace canonical based on results.
- Source-cell results are sample recovery, not population prevalence or causal inference.

Run the task checks, append a short note to `BUILD_LOG.md`, and stop.
