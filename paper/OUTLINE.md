# Conference paper outline

## Working title
Beyond Hard Labels: Uncertainty-Aware AI Measurement of Subjective Text Constructs

## 1. Introduction

- Automated text labels can become generated variables in later analyses, which makes construct definition and measurement error consequential (MacKenzie et al., 2011; Qiao & Huang, 2021).
- Prompt and inference choices can affect LLM annotations and downstream analyses, motivating a bounded, prespecified sensitivity check (Carlson & Burbano, 2026; Son & Lee, 2026).
- Trained raters can disagree, and a single aggregate label can hide variation in their judgments (Mostafazadeh Davani et al., 2022; Fornaciari et al., 2021; Weerasooriya et al., 2023).
- JEV returns probability-valued responses to bounded yes/no questions; the provider contract defines these as probabilities of Yes/True ([System One API documentation](https://api.typesafe.ai/docs)).
- Research question: on held-out MFRC items, how does JEV probability entropy associate with trained-rater vote entropy, and what differences are observed when probabilities are thresholded or used in prespecified descriptive analyses?
- Contributions are empirical measurements: uncertainty correspondence, paired loss differences from thresholding, an oracle review-ranking simulation, and recovery of observed sample-cell means.

## 2. Background

### 2.1 Text coding as construct measurement
MacKenzie et al. (2011); Qiao and Huang (2021).

### 2.2 AI as rater and specification sensitivity
Carlson and Burbano (2026); Son and Lee (2026).

### 2.3 Human disagreement and label distributions
Mostafazadeh Davani et al. (2022); Fornaciari et al. (2021); Weerasooriya et al. (2023).

### 2.4 MFRC and bounded judgment
Trager et al. (2026), including the Coding Guide-2 appendix. Distinguish probability of category presence from moral intensity.

## 3. Data and methods

- MFRC public snapshot, source-cell reconstruction, and selected/enriched sampling design (Trager et al., 2026; audit the exact [public release](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)).
- Six Moral Foundations, trained-rater rows, inclusion rules, and the exact-text-grouped split.
- Canonical JEV instrument, human approval gate, and one strict held-out wording sensitivity.
- Four prespecified analyses in `docs/METHODS.md`; report the wording sensitivity separately.

## 4. Results

1. Foundation-level and macro uncertainty-validity associations.
2. Paired soft-versus-hard loss differences.
3. Uncertainty-targeted versus random oracle/reference-replacement simulation.
4. Soft, hard, and review-hybrid recovery of observed MFRC `(bucket, subreddit) × foundation` sample-cell means.
5. Prespecified wording sensitivity on its fixed held-out subset.

## 5. Discussion

- Interpret estimates with their uncertainty intervals and the trained-rater reference distribution.
- Discuss probability retention only as a comparison under the prespecified losses; pair it with the uncertainty-validity and sample-cell analyses.
- Describe review results as an idealized ranking exercise. Do not infer actual review time or labor savings.
- State the limits: one dataset, few raters per item, a selected Reddit sample, one decision-model family, and no observed incremental human adjudication.
- Keep disagreement mechanisms open: ambiguity, perspective, coder noise, missing context, and task limitations are not separated by this design.
- Future work can replicate in another construct/domain and measure real additional-coder decisions and review costs.
