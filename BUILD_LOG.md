# Build log

2026-09-30 — Initial research-first review

Simplified the original scaffold so development can proceed in five bounded GPT-6 Luna Max chunks while preserving the scientific guardrails that matter. Build tasks intentionally remain pending; `./continue_build.sh` advances one task per successful invocation. See `VALIDATION.md` for the review record.

2026-10-01 — Task 01 baseline and runner

Deferred analysis imports until analysis stages so routine runner startup does not load Matplotlib. Focused checks passed (11 tests); shell syntax and read-only `--print` behavior also passed. No unresolved issues.

2026-10-01 — Task 02 MFRC data and split

Preserved literal raw strings, strengthened exact tuple IDs and text-group split assignment, enforced the three-annotator minimum, and bound provenance to the raw snapshot plus processed-file bytes and scientific settings. Documented snapshot immutability; `pytest -q tests/test_data.py` passed (19 tests). No unresolved issues.

2026-10-01 — Task 03 instrument and JEV client

Aligned canonical domains to MFRC Coding Guide-2, made the approval manifest reviewable and bound to the exact split, both prompt hashes, frozen model metadata, complete canonical dev evidence, and methods/config, and preserved valid paid responses before drift checks for safe resumption. 26 focused prompt/client tests passed; no live provider calls were made. Human semantic approval remains required.

2026-10-01 — Task 04 analysis and review simulation

Kept outputs aligned with the four core analyses, added foundation and macro entropy intervals from shared comment-cluster resamples, and removed unprespecified review MAE, cell RMSE, majority-agreement, and standalone strict-variant outputs. Clarified the coder-level Brier identity, oracle review limits, and sample-cell interpretation in methods and summary reporting. `pytest -q tests/test_metrics.py` passed (13 tests); no unresolved issues.
