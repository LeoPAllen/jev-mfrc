# Validation record

Validated: 2026-09-30

This revision intentionally removes engineering ceremony from the original scaffold and keeps only controls that materially protect the study.

## External contracts rechecked

- OpenAI's current GPT-6 Luna model page lists model ID `gpt-6-luna` and supports reasoning effort through `max`.
- Current Codex guidance supports scripted `codex exec` workflows with bounded write permissions; the repository therefore uses one fresh, precisely scoped task per invocation instead of a custom session-log verification system.
- TypeSafe's current OpenAPI says `/v1/models` returns model names/aliases accepted by the request, Noul is the probability of yes/true, and the response model may differ from the requested alias. It does not promise that a response-model string is requestable; the client therefore records/detects drift rather than trying to pin by resubmitting an undocumented identifier.
- The MFRC publication describes 16,123 comments from 12 subreddits, annotated by at least three trained annotators for the six focal foundations plus Thin and Implicit/Explicit Morality. The live public `train_dedup` viewer currently exposes 53,827 annotation rows, 11 subreddit values, 3 bucket values, 6 annotator values, and the expected text/subreddit/bucket/annotator/annotation/confidence fields. Those live viewer counts differ from the paper's 16,123-comment/12-subreddit description because they are not the same unit and the public release itself currently exposes 11 subreddit values. The pipeline therefore audits and fingerprints the exact downloaded snapshot rather than hard-coding publication counts.

## What was simplified

Removed from the research workflow: change-scope policing, model-log parsing, wrapper receipts, immutable invariant harnesses, build-state gating of scientific execution, implementation-source hashes in approvals, report-format hashes in approvals, forced fresh SQLite caches for every experiment, and the assumption that JEV's returned model must be a requestable concrete pin.

Development inference now runs the canonical instrument only. The strict wording is frozen for held-out sensitivity. Derived MFRC tables rebuild deterministically from immutable raw data. Exact text repeated across source cells is grouped to the same split rather than treated as a fatal dataset error.

## What remains a hard scientific guardrail

Raw checksum/source provenance; deterministic split/inclusion choices; exact-text leakage prevention; >=3-coder primary inclusion; schema/range validation for JEV responses; experiment-scoped cache keys; observable model-drift detection; complete canonical dev evidence; a human approval gate before held-out inference; and prespecified methods/configuration.

## Local validation

The final source tree passes 52/52 repository tests, Python compilation, shell syntax checks, and the documentation consistency gate. The build harness was smoke-tested with a fake successful Codex command (exactly one task advanced) and a fake failing Codex command (exit code propagated and no task advanced). A synthetic no-network analysis path was also exercised. The release ZIP is re-extracted and checked again before delivery.

No live GPT-6 Luna Codex session or paid JEV request was run here. Those depend on the user's authenticated account/API key and are intentionally runtime checks rather than fabricated validation claims.
