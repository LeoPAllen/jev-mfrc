# Next study: a confirmatory probability-calibration replication

## Objective and scope

Test prospectively whether a small, fixed human development set can calibrate JEV Noul probabilities to the observed distribution of individual GoEmotions annotator judgments on untouched comments. The study asks when model-native probabilities can serve as distributional annotations and whether calibration is a necessary empirical step. It does not test whether a model “reproduces humans.” The reference is the vote distribution of the GoEmotions annotator panel, not a latent truth or a population-wide human distribution.

The confirmatory replication is the JEV-only study below. A conventional generative-LLM baseline is an optional, separately reported model-comparison extension. Keep both to one corpus, one frozen question bundle, one dev/test split, and one affine calibration rule.

GoEmotions is preferred because it has about 58,000 Reddit comments with raw individual annotations, three or five raters per comment, and six binary targets that map to the six Ekman emotions used in prior disagreement work. Its published annotation definitions and group mapping make the six Noul questions explicit. It also extends the external validation beyond the especially close 2026 HateXplain study rather than repeating that corpus. The corpus is still curated, subreddit-balanced, and annotated by a specific panel; it is not representative of Reddit users or all human judgments.

## A. Corpus and pre-run data lock

Use the **original raw GoEmotions release** from Demszky et al. (2020), not the agreement-filtered `train.tsv`/`dev.tsv`/`test.tsv` files. The official release describes 58,009 comments, raw one-row-per-rater annotations, 82 annotators, and three or five raters per comment; it also documents the source-defined `example_very_unclear` flag. See the [ACL paper and Appendix A](https://aclanthology.org/2020.acl-main.372/) and the [official Google Research repository](https://github.com/google-research/google-research/tree/master/goemotions).

The raw files are the three official full-data objects:

- `https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv`
- `https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv`
- `https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv`

Before opening the corpus for analysis and **before any model inference**, save a read-only snapshot and commit a lock record containing (1) the exact Google Research `google-research` commit SHA used to identify the release documentation/code, (2) the retrieval date and canonical source URLs, (3) each raw CSV's byte size and SHA-256, (4) the SHA-256 of the source `emotions.txt`, any upstream mapping file used, the GoEmotions paper PDF, and the committed derived six-group mapping table, and (5) a combined manifest hash. Content hashes, rather than a mutable `master` URL or an unfiltered mirror, define the data revision used for this study. Verify the hashes before every rebuild and stop on a missing lock or mismatch. Never replace a snapshot silently.

**Lock status:** the exact GoEmotions SHA-256 values are not yet recorded in this documentation-only task because the corpus has not been downloaded. No inference may start until the values and repository commit are filled into a versioned `corpus_lock.json`. The official source and hash procedure are fixed here; the byte-level lock is a pre-run gate.

## B. Duplicate and annotator-quality rules

Use the raw comment `id` as the item key and retain source `rater_id` and the per-rater emotion vector during preprocessing.

1. Count exact duplicate raw rows and remove duplicates only when `id`, `rater_id`, text/metadata, the full emotion vector, and the unclear flag all match.
2. If one `id` has irreconcilable text or item metadata, exclude that item and record the count. If an `(id, rater_id)` pair has nonidentical emotion vectors or conflicting unclear flags after exact duplicates are removed, exclude the entire item; do not choose the response that agrees with other raters.
3. Treat a source-flagged `example_very_unclear` response as missing for all six outcomes, not as six negative labels. It is an original response-quality flag, and the paper reports removing examples with no emotion label. Keep all other complete responses.
4. Require at least **three unique, non-unclear annotators per item** after these rules. Exclude items below this fixed minimum and report the flow counts. Do not add a majority-agreement filter.
5. Do not remove annotators because their rates, agreement, demographics, or model disagreement look unusual. The source has no independent gold-based rater-quality measure for this purpose; outcome-based filtering would change the reference distribution. Do not exclude labels by JEV agreement.
6. Keep distinct comment IDs as distinct items even when exact text repeats, but assign byte-identical stored text to the same split and bootstrap cluster. This prevents text leakage without discarding separate rating records.

The primary human reference for item `i` and emotion group `f` is `h_if = k_if / n_i`, where `k_if` is the number of retained annotators whose raw GoEmotions response maps to that group and `n_i` is the number of valid annotators. Multiple labels are allowed. An `example_very_unclear` response is missing, never an all-zero vote.

## C. Deterministic development/test split

After applying only the structural rules in section B, group items by exact stored text (UTF-8 bytes; do not lowercase, trim, or otherwise normalize). Assign the whole text group to development if the first 16 hexadecimal characters of

```text
SHA256("goemotions-calibration-v1\0" || exact_text_utf8)
```

interpreted as an unsigned integer are below `0.10 × 16^16`; assign all other groups to test. The salt and threshold are fixed. The expected allocation is approximately 10% development (about 5,800 comments) and 90% held-out test (about 52,000 comments), with the actual item and text-group counts recorded after the corpus lock. Do not stratify, rebalance, or alter assignment after seeing any JEV output or human label rates. The single development set is the only set used to estimate prevalence and calibration.

For the fixed wording sensitivity, select approximately 10% of test text groups by the same rule with the distinct salt `goemotions-strict-v1\0`; hash the exact text bytes and apply the same threshold. This subset is frozen before inference. Canonical JEV predictions are run on all eligible development and test items; strict wording is run on the frozen sensitivity subset only.

## D. Constructs and fixed question wording

The original GoEmotions task asked annotators to identify emotions expressed by the writer using published definitions and allowed multiple labels. Use its published **Ekman-level grouping** for the six binary targets. For every individual annotator, the group is positive if they selected at least one constituent raw label; otherwise it is negative. This aggregation is fixed before model calls and preserves the source definitions and raw rater data. It is not a post-hoc fit to the observed GoEmotions label frequencies.

Submit the following six independent Noul questions together for each comment. In each question, “True” means that the writer expresses the defined GoEmotions group in the comment; “False” means none of its listed constituent categories is expressed. The definitions below paraphrase GoEmotions Appendix A; membership follows the published six-way mapping in the main paper.

| Target | Fixed canonical question and positive criterion |
|---|---|
| Anger | **Does the writer express the GoEmotions anger group in this comment?** True covers anger or antagonism, irritation, or expressed disapproval (raw labels: `anger`, `annoyance`, `disapproval`). |
| Disgust | **Does the writer express disgust in this comment?** True covers revulsion or strong disapproval prompted by something unpleasant or offensive (raw label: `disgust`). |
| Fear | **Does the writer express the GoEmotions fear group in this comment?** True covers being afraid, worried, or apprehensive (raw labels: `fear`, `nervousness`). |
| Joy | **Does the writer express the GoEmotions joy group in this comment?** True covers any category in the published positive-emotion group: `admiration`, `amusement`, `approval`, `caring`, `desire`, `excitement`, `gratitude`, `joy`, `love`, `optimism`, `pride`, or `relief`. |
| Sadness | **Does the writer express the GoEmotions sadness group in this comment?** True covers emotional pain or sorrow and the mapped categories for disappointment, embarrassment, grief, and remorse (raw labels: `sadness`, `disappointment`, `embarrassment`, `grief`, `remorse`). |
| Surprise | **Does the writer express the GoEmotions surprise group in this comment?** True covers any category in the published ambiguous-emotion group: `realization`, `surprise`, `curiosity`, or `confusion`. |

For each Noul question, set the machine-readable criteria to `true = this emotion group is expressed by the writer in the comment` and `false = no constituent emotion in this group is expressed`. Preserve original comment text exactly as released. Do not add chain-of-thought requests, examples, or label explanations after the instrument is frozen. The six outputs are separate yes/no judgments, not a single-choice emotion classification; multiple groups may be True.

The GoEmotions paper reports the mapping used above: anger = anger/annoyance/disapproval; disgust = disgust; fear = fear/nervousness; joy = all positive emotions; sadness = sadness/disappointment/embarrassment/grief/remorse; surprise = all ambiguous emotions. It also reports that its annotators were native English speakers from India and that the corpus was deliberately curated and balanced. Results therefore describe this corpus and its annotator panel, not Reddit prevalence or a universal human emotion distribution.

### One fixed stricter wording sensitivity

On the preselected test subset only, use the same six definitions and group membership with this fixed modifier added to the stem and criteria:

> “Answer True only when the comment clearly supports at least one listed category from the writer's text itself. Do not infer an emotion that the text does not express.”

Apply the same modifier to all six questions, keep the canonical wording primary, and report probability change, hard-label flips at `.5`, loss direction, and canonical/strict rank correlation. This is one prespecified sensitivity check; it cannot replace the canonical bundle based on performance.

## E. Human calibration set and instrument gate

The hash-selected 10% development portion is the **one human calibration set**, fixed before predictions. Use its individual annotator judgments to compute each emotion's development prevalence and to fit the six calibration functions. Do not split it again or search among alternate calibration rules.

Before opening test predictions, a human author reviews the six definitions, exact serialized questions, and a development-only sample of model/human disagreements to verify that the operationalization is legible and faithful to the published coding scheme. No test labels, predictions, or error summaries may be used to edit questions. If any instrument wording or target mapping changes, record a dated protocol amendment and re-freeze the instrument before test inference; do not treat that revised run as the original confirmatory specification. Freeze the model identity, response identity, prompt hashes, request parameters, split manifest, and analysis code with the approval record.

Freeze the JEV release as well. To match the completed MFRC run as closely as possible, use `jev-1.13.0` if it remains requestable; otherwise select one available versioned release and record its full advertised model metadata and response identity before any GoEmotions labels or summaries are analyzed. Do not use a moving `latest` alias or change the selected release after test inference.

## F. Pre-registered JEV method comparisons

For each emotion, derive all four methods from the same canonical JEV output and fixed human development set:

1. **Raw JEV probability:** `p_if`, the provider-documented Noul probability of True.
2. **Hard JEV label:** `I(p_if >= 0.5)`. The threshold is fixed at `.5` for all emotions and is never tuned.
3. **Development prevalence baseline:** one constant per emotion, equal to the development-set positive coder rate `sum_i k_if / sum_i n_i`.
4. **Development-fitted weighted affine calibration:** fit `a_f + b_f p_if` by weighted least squares on development items, targeting `h_if` with item weight `n_i`; then apply `clip(a_f + b_f p_if, 0, 1)` unchanged to test items. Fit one intercept and slope per emotion. No isotonic, logistic, nonlinear, or test-fitted recalibration is allowed.

This weighted least-squares fit minimizes squared loss over the development annotator judgments, up to the within-item human-variance term that is common to all predictions. The constant prevalence baseline uses the same coder-weighted development distribution. Save and publish all six `(a_f, b_f)` pairs before test evaluation.

## G. Outcomes and inference

### Primary confirmatory outcome

The primary outcome is the **held-out coder-level-equivalent squared loss**, macro-averaged equally over the six emotion groups. For method score `s_if` and the `n_i` retained binary coder labels `y_ijf`:

```text
L_if(s) = (1 / n_i) * sum_j (s_if - y_ijf)^2
        = (s_if - h_if)^2 + h_if * (1 - h_if)
```

For each emotion, average `L_if` over test coder judgments, weighting each item by `n_i`; then average the six emotion-specific losses with equal weight. Also report the equivalent weighted squared loss to the vote share. The difference between methods is identical on these two forms because `h_if(1-h_if)` is common to all methods.

The single primary contrast is **development-fitted affine calibration minus raw JEV probability**. The preregistered directional hypothesis is that this difference is below zero. Report its paired estimate and two-sided 95% interval; support requires the interval to exclude zero in the predicted direction. Report the other planned pairwise comparisons among raw, hard, prevalence, and affine methods as secondary contrasts. Keep emotion-specific estimates visible; do not treat the six correlated outcomes as six independent replications.

Use 2,000 paired bootstrap resamples of exact-text clusters, retaining all comment IDs in a selected text cluster, all coder votes, and all six emotions together. Recompute the macro and per-emotion losses in each draw. No held-out outcome may alter the questions, `.5` threshold, exclusions, split, mapping, calibration function, or claims.

### Secondary outcomes

- **Absolute loss:** mean individual-coder absolute loss, calculated over the retained test votes using each method's probability or binary score.
- **Calibration bias and reliability:** coder-weighted mean signed bias (`prediction - vote share`), and ten equal-frequency reliability bins per emotion with deterministic item-ID tie-breaking. Show mean prediction and mean vote share in each bin with paired cluster-bootstrap intervals. These bins are diagnostics only and never refit test probabilities.
- **Sharpness:** show each method's probability distribution, standard deviation, and binary entropy separately from calibration. Include the squared/Brier score and reliability displays together; a concentrated forecast is not automatically a calibrated forecast.
- **Disagreement association:** for each item/emotion, calculate the observed coder-pair disagreement fraction `2 n_i h_if (1-h_if)/(n_i-1)`. Report Spearman association of raw `H(p_if)` with this measure and with human vote entropy. Also report association of raw `p_if` with disagreement so a positive-score-strength signal can be distinguished from entropy-specific information.
- **Disagreement detection:** label an item mixed when its retained coder votes include both 0 and 1. Report ROC-AUC for raw JEV entropy and for raw `p`, per emotion and macro-averaged. AUC concerns ranking for this mixed-vote criterion, not calibration or recovery of annotator identities.
- **Source/group recovery, if substantively justified:** as a descriptive secondary analysis only, compare predictions and human vote shares within observed `subreddit × emotion` cells having at least 200 held-out exact-text clusters. Preserve the six labels and report selected-corpus cell recovery; do not interpret those estimates as subreddit or Reddit population prevalence or causal differences. If the authors cannot justify a source-group question before test inference, omit this optional analysis rather than inventing one after results.

## H. Power and sample-size logic

Run the full fixed corpus after the quality rules. The published release contains about 58,000 comments, giving approximately 5,800 development comments for the six two-parameter affine maps and approximately 52,000 held-out comments for primary estimates, before structural exclusions. Each comment has three or five original annotations, so the test should retain on the order of 150,000–260,000 coder decisions per emotion if the release counts are close to the published totals.

This is a census of the eligible released corpus, not an arbitrarily chosen `N`. With roughly 52,000 independent test text clusters, a conservative normal-approximation planning half-width for a paired per-cluster loss difference bounded in `[-1, 1]` is about `1.96 / sqrt(52,000) = 0.009`, using the maximum possible variance of one. Exact-text clustering can reduce the effective cluster count, which will be reported and used in the bootstrap. The same sample supplies tens of thousands of held-out predictions per emotion for reliability curves. The 10% development allocation leaves thousands of calibration items for only two fitted parameters per emotion. These calculations use corpus size and the bounded loss range, not held-out outcomes or an assumed effect size. If fewer than 40,000 held-out text clusters remain after the fixed data rules, report the smaller precision and amend the power statement before any inference; do not change the split to chase a favorable result.

## Optional model-comparison extension

If included, select one conventional generative LLM and an immutable model/version **before inspecting or analyzing GoEmotions data**. Record provider, exact dated model ID, endpoint/API version, decoding parameters, call date, and response model identity. If an immutable model version is unavailable, do not run this extension with a moving alias. Use the same six construct definitions, text, development/test split, and human calibration set. Request one direct probability of True per emotion and, if the model exposes a hard output, retain it under a separate label. Compare its direct probabilities, the same predeclared development-fitted weighted affine map, and its hard output where available. Report the model comparison as an extension; it does not replace the JEV-only primary hypothesis. Exact provider/model selection remains an explicit pre-analysis decision.

## Proposed paper framing

**Question:** When can model-native probabilities function as distributional annotations, and is calibration a necessary empirical step?

The paper should not claim “JEV reproduces humans.” It should distinguish probability information, probability-level calibration, and human disagreement; evaluate all three against raw individual annotations; and test a development-fitted correction on a large untouched set. Human votes remain the observed reference distribution for this annotation process. Generalization beyond GoEmotions' Reddit sample, annotator panel, wording, and pinned model requires further studies.

Do not write manuscript claims from the calibration hypothesis until the confirmatory results exist and a human author has reviewed the interpretation.

## Key sources

- Demszky, D., et al. (2020). GoEmotions: A dataset of fine-grained emotions. [ACL paper](https://aclanthology.org/2020.acl-main.372/); [official raw data README](https://github.com/google-research/google-research/tree/master/goemotions).
- Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. [DOI](https://doi.org/10.1111/j.1467-9868.2007.00587.x).
- Liu, J. (2026). Rubric-conditioned large language model labeling: Agreement, uncertainty, and label consistency in subjective text annotation. [Computers in Human Behavior](https://www.sciencedirect.com/science/article/abs/pii/S0747563226000853).
- See [`LITERATURE_POSITIONING.md`](LITERATURE_POSITIONING.md) for the broader methods and disagreement literature, including the closest-prior boundary.
