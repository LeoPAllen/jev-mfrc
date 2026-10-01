# Risks and edge cases

| Risk | Research-first response |
|---|---|
| Human labels are subjective | Preserve vote shares/disagreement; call them trained-human reference judgments, not truth. |
| Few coders per item | Require at least 3 for the primary sample; interpret shares as empirical references, not precise latent probabilities. |
| Human-disagreement entropy has coarse resolution with 3 coders | Treat entropy correlation as a diagnostic association; report coder-count distribution and avoid claims of precise latent ambiguity. |
| Disagreement has multiple causes | Do not equate disagreement with ambiguity; discuss perspective, noise, missing context, and construct boundaries. |
| Publication/live-data counts differ | Audit the exact public snapshot and report differences rather than forcing publication counts. |
| No canonical public comment ID | Reconstruct source-specific items from `(bucket, subreddit, text)`. |
| Same-cell identical text may represent distinct physical comments | The public release cannot distinguish them; reconstruction treats the tuple as one item and reports this as a limitation rather than inventing identity. |
| Exact text appears in multiple source cells | Preserve each source-specific item but group identical text onto the same dev/test side to prevent literal leakage. |
| Duplicate annotation rows | Drop/count exact duplicates; conflicting item+annotator records stop for inspection. |
| Multi-label parsing | Require exact focal-token matches after trimming; audit observed tokens and reject unexpected focal spellings/case. |
| Confidence vocabulary changes | Use observed known levels; unknown values stop rather than being guessed. |
| Class imbalance | Report foundation-specific diagnostics; do not make overall accuracy the main metric. |
| Soft-vs-hard loss can favor soft values for representational reasons | Treat this as the measured cost of thresholding the *same* instrument, not independent evidence that JEV is valid; interpret it alongside uncertainty validity and downstream recovery. |
| Cross-foundation base rates | Estimate uncertainty association within foundation, then macro-average. |
| MFRC morally enriched sampling | Never describe source-cell means as natural subreddit prevalence. |
| Prompt degrees of freedom | Canonical dev only; one strict wording frozen for held-out sensitivity; no post-test prompt selection. |
| Data/split iteration during development | New fingerprint means new scoped cache keys; old experiment rows may remain safely in SQLite but are ignored. |
| Stale derived data | Rebuild deterministically from immutable raw data instead of maintaining a complex derived-cache protocol. |
| Moving JEV alias | Freeze advertised identifying metadata at experiment start, keep submitting the documented request model name, record returned model, and stop on observed drift. |
| Alias changes that preserve all exposed identifiers are unobservable | Record the exact dates/metadata/returned model and acknowledge this provider-provenance limit; do not add speculative pinning machinery. |
| Response model is not documented as requestable | Never resubmit it merely because the response returned it. Treat it as recorded provenance/drift evidence. |
| Timeout/connection/5xx after possible processing | Stop; idempotency is undocumented, so blind retry could duplicate cost. |
| Rate limiting | Retry only explicit 429 with bounded backoff/`Retry-After`. |
| Partial paid run | Commit each valid response immediately; exact completed calls resume from SQLite. |
| Prompt/data/method/model change after approval | Held-out stage refuses the mismatched approval; regenerate canonical dev evidence where relevant and re-approve. |
| Harmless code/report refactor after approval | No automatic re-approval merely because implementation bytes or Markdown formatting changed. |
| Review simulation overstates real review | Call it oracle/idealized reference replacement; make no labor-time claim. |
| Luna task becomes too broad | Five bounded tasks, focused tests, fresh invocation; split a task only when an actual dependency warrants it. |
| Agent invents results/citations | Numerical claims come from generated results; paper citations come from vetted sources or are marked for verification. |
| Crash after response but before local commit | A narrow duplicate-billing window remains because the provider documents no idempotency key; use provider usage logs if exact reconciliation matters. |

## What is intentionally *not* a blocker

- build-task completion state;
- unrelated files in the SQLite cache from a different experiment fingerprint;
- report formatting changes;
- a description-only edit in `/v1/models` metadata;
- identical text in different source cells, provided split grouping prevents leakage;
- live dataset counts differing from paper prose when the audited snapshot is internally valid.
