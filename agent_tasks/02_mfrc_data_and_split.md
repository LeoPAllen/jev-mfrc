# Task 02 — MFRC data and split

Goal: audit the MFRC ingestion and split logic for scientific correctness and simplicity.

Focus on `src/jev_mfrc/data.py`, `tests/test_data.py`, `docs/RESEARCH_SPEC.md`, and `docs/RISKS.md`.

Requirements:
- Raw snapshot is downloaded once, checksum-verified, and never silently replaced.
- Derived item tables are cheap enough to rebuild deterministically from raw data; prefer rebuilding over complicated stale-cache logic.
- Item identity preserves exact `(bucket, subreddit, text)` because the public release has no canonical comment ID.
- If identical text appears in more than one source cell, keep the source-specific items but assign identical text to the same dev/test side so exact-text leakage cannot cross the split. Do not fail merely because repeated text exists.
- Exact duplicate rows may be removed; conflicting rows for the same reconstructed item + annotator must remain visible as a data problem.
- Require at least 3 unique annotators in the primary sample and report exclusions.
- Dev and sensitivity selection must be deterministic and selected before JEV output.
- Experiment provenance should bind the immutable raw snapshot, relevant split/inclusion settings, and the actual processed-item content — not arbitrary source-code hashes.

Use local fixtures only. Run the task checks, append a short note to `BUILD_LOG.md`, and stop.
