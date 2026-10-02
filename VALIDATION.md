# Post-run validation record

Updated 2026-10-01 after completion of the JEV-MFRC experiment and creation of the text-free release bundle.

## Pinned data and split

MFRC `USC-MOLA-Lab/MFRC`, split `train_dedup`, revision `ddc21d2f03e156732fd1ba95a51c4c29f07975be`. The raw SHA-256 is `ad5c1e695a7c52e8d92ee452c3bca66a3400c76880a2b424627902046bb6511d`; requested and resolved revisions match.

Observed from 53,827 raw rows: 0 literal exact duplicates; 250 semantic duplicate rows collapsed; 42 irreconcilable same-annotator items (170 rows) excluded in full; 135 items below the three-annotator minimum excluded; 17,709 eligible items. The fixed split has 1,000 development / 16,709 held-out / 1,000 sensitivity items. Retained item/annotator votes are unique, cross-annotator disagreement remains, and no exact text or `content_id` crosses development/held-out. Reconstructed eligible vote shares match the prepared item table. These counts match `docs/DATA_AUDIT.md`.

## Historical pre-inference checkpoint

The following records preserve the validation status before live inference. The pending items were subsequently resolved as recorded below.

- `env -u TYPESAFE_API_KEY ./run.sh --stage data` completed from the verified local raw snapshot without a JEV request.
- The full suite passed with **80 tests** at that checkpoint.
- Python compilation, shell syntax, documentation gate, `git diff --check`, and the generated model-snapshot ignore rule passed.
- No full-analysis regeneration had yet been run; the suite then contained offline synthetic metric tests.
- At that point, authenticated requests, complete development predictions, human instrument approval, held-out inference, and sensitivity inference were still pending.

## Completed post-run checks

- The approved frozen instrument was used for 1,000 canonical development predictions, 16,709 canonical held-out predictions, and 1,000 strict sensitivity predictions. Requested model identity is `jev-latest`; every released prediction reports returned model `jev-1.13.0`.
- The release records the MFRC revision and raw/processed-data hashes, experiment fingerprint, approval-file and instrument hashes, canonical/strict prompt hashes, model identity, counts, analysis configuration, source Git commit, and hashes for every file other than the manifest.
- `scripts/build_release_bundle.py` ran successfully and validated the bundle. Its data files contain only the specified IDs, source-cell fields, human shares/confidence/counts, JEV probabilities, variant and provenance fields. No comment text, annotator identifiers, provider credentials, raw MFRC files, or JEV cache is included.
- `release/` is outside the ignore rules and appears in the worktree. The Git index was not changed because `.git` is read-only in this workspace.
- The full test suite passed: **96 passed**, including a deterministic compressed-CSV check and release validation regressions.
- The documentation gate, Python compilation of the release builder/tests, and `git diff --check` passed.
- One `run.py --stage analyze` regeneration was run in an isolated temporary copy, leaving the archived `results/` outputs untouched. It made no JEV request. The regenerated `analysis.json`, `summary.md`, and all six numerical CSV tables were byte-identical to their archived originals.
- The two regenerated PNGs were also byte-identical to the archived files in this environment. PNG bytes can vary across renderers or metadata settings; no such variation appeared in this comparison, and it would not indicate a numerical-result difference.

The release can be checked and rebuilt with `python scripts/build_release_bundle.py`. Its bundled data and analysis outputs support verification without the JEV API, local call cache, Reddit comment text, or raw MFRC files.
