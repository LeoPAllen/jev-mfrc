# Build log

2026-09-30 — Initial research-first review

Simplified the original scaffold so development could proceed in five bounded GPT-6 Luna Max chunks while preserving the scientific guardrails that matter. Those implementation tasks are complete; build state is separate from scientific inference.

2026-10-01 — Task 01 baseline and runner

Deferred analysis imports until analysis stages so routine runner startup does not load Matplotlib. Focused checks passed (11 tests); shell syntax and read-only `--print` behavior also passed. No unresolved issues.

2026-10-01 — Task 02 MFRC data and split

Preserved literal raw strings, strengthened exact tuple IDs and text-group split assignment, enforced the three-annotator minimum, and bound provenance to the raw snapshot plus processed-file bytes and scientific settings. Documented snapshot immutability; `pytest -q tests/test_data.py` passed (19 tests). No unresolved issues.

2026-10-01 — Task 03 instrument and JEV client

Aligned canonical domains to MFRC Coding Guide-2, made the approval manifest reviewable and bound to the exact split, both prompt hashes, frozen model metadata, complete canonical dev evidence, and methods/config, and preserved valid paid responses before drift checks for safe resumption. 26 focused prompt/client tests passed; no live provider calls were made. Human semantic approval remains required.

2026-10-01 — Task 04 analysis and review simulation

Kept outputs aligned with the four core analyses, added foundation and macro entropy intervals from shared comment-cluster resamples, and removed unprespecified review MAE, cell RMSE, majority-agreement, and standalone strict-variant outputs. Clarified the coder-level Brier identity, oracle review limits, and sample-cell interpretation in methods and summary reporting. `pytest -q tests/test_metrics.py` passed (13 tests); no unresolved issues.

2026-10-01 — Task 05 end-to-end package

Clarified fresh setup and resumable study phases; kept `./continue_build.sh` as the sole build loop; removed the unused auto-loop, shell alias, and stale validation record; sourced the study literature and bounded the paper outline. Full suite passed (69 tests), as did shell syntax, documentation gate, Python compilation, and a synthetic no-network analysis smoke test. Human instrument approval remains required before held-out inference; no other package issue remains.

2026-10-01 — Localized review and provenance fixes

Ignored the raw metadata and generated audit JSON, namespaced JEV model snapshots by experiment/provider/request, and expanded deterministic canonical-only development review with explicit human tie handling and annotator counts. Clarified the optional auto-commit wrapper in the README and updated the majority-category definition. Focused suite passed (57 JEV, metrics, and data tests); no paid calls or held-out inference were run, and no unresolved issue remains.

2026-10-01 — MFRC semantic duplicate ingestion fix

Canonicalized annotation label sets for duplicate detection, added semantic-removal audit counts, and retained strict diagnostics for genuine item/annotator conflicts. The raw snapshot audit found 250 semantically duplicate groups, 42 different-label-set conflict groups, and no confidence-only groups (0 literal exact duplicates). `pytest -q tests/test_data.py` passed (21 tests). The local data stage correctly remains blocked by the 42 genuine conflicts; raw data were not modified.

2026-10-01 — MFRC contradictory duplicate item exclusion

Kept literal and semantic duplicate counts separate, collapsed only same-label-set/same-confidence repeats, and excluded all rows for each of the 42 affected items. Added retained item/annotator uniqueness enforcement, audit counts, pipeline version 9, and concise methods/data/risks documentation with the resolved snapshot revision and SHA. Real `./run.sh --stage data` passed on 53,827 raw rows: 250 semantic duplicates removed, 42 conflict groups/items and 170 rows excluded, 17,844 items remaining before the minimum-annotator filter, and 17,709 eligible items (1,000 dev; 16,709 test). Full suite passed (75 tests) and documentation gate passed. Raw data were unchanged; no JEV/API calls; no unresolved issue.

2026-10-01 — Final structural data audit before JEV inference

Added `docs/DATA_AUDIT.md` with verified snapshot provenance, normalization, split/content duplication, annotator, held-out source-cell, category, and environment summaries; updated `RESEARCH_SPEC.md` to state that config explicitly requests the resolved SHA. Independent raw-to-processed checks passed; 38 blank raw confidence fields are reported descriptively, with no pipeline change. Focused data tests (21), documentation gate, and full suite (75) passed. No JEV/API calls or unresolved audit issue.

2026-10-01 — Task 04 final statistical robustness pass

Added coder-count-adjusted pairwise disagreement as a secondary robustness result and explicitly reported weighted source-cell MAE as the unequal-cell-size diagnostic. The pinned test data have unique `content_id` values, so row and exact-text cluster bootstrap units coincide and no cluster bootstrap was added. Updated the pre-approval methods note and hand-calculated/reporting tests. Focused metrics suite passed (19 tests); full suite passed (78 tests). No JEV/API calls or held-out predictions were inspected; human instrument approval remains required before live inference.

2026-10-01 — Final pre-inference audit

Required raw requested/resolved revisions to match the configured immutable pin and added the direct analysis approval guard. Clarified primary/secondary analysis roles and the data → canonical dev → human review sequence. Data stage passed without `TYPESAFE_API_KEY`; full suite passed (80 tests), as did compilation, shell syntax, documentation gate, snapshot ignore check, and diff check. No live JEV calls were made. Still pending: authenticated request, actual dev predictions, human semantic approval, and held-out/sensitivity inference.

2026-10-01 — Post-results reproducibility bundle

Added `scripts/build_release_bundle.py` and a non-ignored `release/` archive with hashed, text-free canonical held-out and strict sensitivity data, archived analysis outputs, provenance, and schema/integrity validation tests. Updated `VALIDATION.md` to preserve the pre-inference checkpoint while recording completed inference and post-run checks. Bundle validation passed; one isolated analysis regeneration matched all eight numerical CSV/JSON/summary artifacts byte-for-byte, and both PNGs were byte-identical in this environment. Full suite passed (96 tests), including deterministic compressed-CSV output. Documentation, compilation, and diff checks passed. No JEV calls were made and original results were not overwritten. Repository index tracking remains for the wrapper/user because `.git` is read-only in this workspace.

2026-10-02 — Post-hoc diagnostics

Added a clearly labeled exploratory diagnostic script, tables, calibration figure, audit, and synthetic tests under `posthoc/` and `tests/`. Calibration is fit only on canonical dev outcomes; held-out comparisons and comment bootstraps are separate from frozen results. The script ran successfully and the complete suite passed (101 tests; `PYTHONPATH=.:src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -vv`). No JEV calls or frozen-output changes; no unresolved issue.

2026-10-02 — Literature positioning and confirmatory next-study plan

Added literature positioning centered on information, calibration, and disagreement, with Liu (2026) identified as the closest prior and current MFRC calibration findings labeled post hoc. Specified a GoEmotions raw-rater replication, six fixed mapped emotion questions, data-quality and split rules, one development calibration set, primary coder-level loss, secondary diagnostics, and an optional pinned generative-LLM extension. No results or raw data changed; no JEV calls or tests were run. Unresolved pre-run item: exact GoEmotions file and repository hashes must be recorded in the corpus lock before any inference.
