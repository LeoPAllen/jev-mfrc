# Structural data audit

This descriptive audit records the local MFRC snapshot and the eligible item table before JEV inference. It contains aggregate metadata only; no Reddit comment text is included.

## Snapshot

| Field | Value |
|---|---|
| Dataset repo | `USC-MOLA-Lab/MFRC` |
| Split | `train_dedup` |
| Requested revision (pinned in `config.json`) | `ddc21d2f03e156732fd1ba95a51c4c29f07975be` |
| Resolved revision (raw metadata) | `ddc21d2f03e156732fd1ba95a51c4c29f07975be` |
| Raw SHA-256 | `ad5c1e695a7c52e8d92ee452c3bca66a3400c76880a2b424627902046bb6511d` |
| Data-pipeline version | 9 |

## Release normalization

| Measure | Count |
|---|---:|
| Raw rows | 53,827 |
| Literal exact duplicate rows removed | 0 |
| Semantic annotation duplicate groups | 250 |
| Semantic annotation duplicate rows removed | 250 |
| Genuine conflicting same-item/same-annotator groups | 42 |
| Items excluded for those conflicts | 42 |
| Rows removed with those conflicted items | 170 |
| Items remaining after conflict exclusion | 17,844 |
| Items excluded for fewer than 3 unique annotators | 135 |
| Final eligible items | 17,709 |

Annotation labels are compared as unordered sets, so the same label set in a different comma order is the same multilabel judgment (when confidence also matches). Each of the 42 items with genuinely conflicting repeated judgments by one annotator is excluded in full rather than adjudicated. Disagreement across different annotators is preserved.

## Split and content duplication

| Subset | Items |
|---|---:|
| Development | 1,000 |
| Held-out test | 16,709 |
| Sensitivity subset of test | 1,000 |

The exact-text leak check found **0 text groups crossing dev/test**. There are **17,709 unique `content_id` values** among eligible items and **0 `content_id` groups represented by more than one item** (0 items in such groups).

| Scope | Unique `content_id` values | Repeated `content_id` groups (>1 item) | Items in repeated groups |
|---|---:|---:|---:|
| All eligible | 17,709 | 0 | 0 |
| Development | 1,000 | 0 | 0 |
| Held-out test | 16,709 | 0 | 0 |

No `content_id` crosses dev/test. The eligible table therefore contains no repeated-content groups for a secondary content-cluster resample.

## Annotators

| Unique annotators per item | All eligible items | Held-out test items |
|---:|---:|---:|
| 3 | 17,701 | 16,702 |
| 4 | 6 | 5 |
| 5 | 2 | 2 |

Exact-half human-share ties by foundation:

| Foundation | All eligible | Held-out test |
|---|---:|---:|
| Care | 1 | 1 |
| Equality | 1 | 1 |
| Proportionality | 0 | 0 |
| Loyalty | 0 | 0 |
| Authority | 0 | 0 |
| Purity | 0 | 0 |

## Held-out source cells

For the 12 observed `(bucket, subreddit)` cells in test, item counts are: **min 98; Q1 1,258.5; median 1,415.5; Q3 1,657.75; max 2,478**. Quartiles use linear interpolation.

## Observed categories

- **Subreddits (11):** `AmItheAsshole`, `Conservative`, `antiwork`, `confession`, `europe`, `geopolitics`, `neoliberal`, `nostalgia`, `politics`, `relationship_advice`, `worldnews`.
- **Buckets (3):** `Everyday Morality`, `French politics`, `US Politics`.
- **Annotation tokens:** `Authority`, `Care`, `Equality`, `Loyalty`, `Non-Moral`, `Proportionality`, `Purity`, `Thin Morality`.
- **Confidence values:** `Confident`, `Somewhat Confident`, `Not Confident`; 38 raw rows have a blank confidence field.

## Reproducibility and structural checks

Environment used to prepare this audit: Python 3.13.5, pandas 2.3.3, numpy 2.5.3, scipy 1.18.1, datasets 4.8.5, pyarrow 24.0.0.

Independent checks confirmed that `item_id` is unique, every eligible item has at least 3 annotators, sensitivity is a subset of test, exact text and `content_id` never cross dev/test, and realized counts match `audit.json`. Reconstructing normalized retained records from the raw snapshot found unique `(item_id, annotator)` pairs, so no duplicated vote survives into the eligible aggregate; the processed item summaries match those records. Processed tables contain one aggregate row per item and no raw annotator-vote rows.
