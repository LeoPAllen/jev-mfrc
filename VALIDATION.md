# Final pre-inference validation

## Pinned data

MFRC `USC-MOLA-Lab/MFRC`, split `train_dedup`, revision `ddc21d2f03e156732fd1ba95a51c4c29f07975be`. The raw SHA-256 is `ad5c1e695a7c52e8d92ee452c3bca66a3400c76880a2b424627902046bb6511d`; requested and resolved revisions match.

Observed from 53,827 raw rows: 0 literal exact duplicates; 250 semantic duplicate rows collapsed; 42 irreconcilable same-annotator items (170 rows) excluded in full; 135 items below the three-annotator minimum excluded; 17,709 eligible items. The split is 1,000 development / 16,709 test / 1,000 sensitivity. Retained item/annotator votes are unique, cross-annotator disagreement remains, and no exact text or `content_id` crosses development/test. Reconstructed eligible vote shares match the prepared item table. These counts match `docs/DATA_AUDIT.md`.

## Validated

- `env -u TYPESAFE_API_KEY ./run.sh --stage data` completed from the verified local raw snapshot without a JEV request.
- Full pytest suite: **80 passed**.
- Python compilation, shell syntax, documentation gate, `git diff --check`, and the generated model-snapshot ignore rule passed.
- No separate full-analysis smoke test is present; the suite includes offline synthetic metric tests.

## Not yet validated

- Authenticated live TypeSafe request.
- Actual JEV development predictions.
- Human semantic approval of the instrument.
- Held-out or sensitivity inference and outputs.

The repository is ready for `./run.sh --stage dev --max-items 10` after setting `TYPESAFE_API_KEY`.
