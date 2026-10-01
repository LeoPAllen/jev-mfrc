# Conference paper outline

## Working title
Beyond Hard Labels: Uncertainty-Aware AI Measurement of Subjective Text Constructs

## 1. Introduction
- Automated text coding increasingly produces variables treated as observed measures.
- Management/IS methods work warns that construct validity, prompt specification, and generated-variable error matter for downstream conclusions.
- Subjective coding adds another issue: trained raters can disagree, and aggregation erases that information.
- New decision models expose probabilities directly.
- Research question: does model uncertainty correspond to trained-rater disagreement, and does retaining it improve measurement and allocation of human review?
- Contributions: uncertainty-validity evidence; direct estimate of information lost through thresholding; selective-review workflow; downstream sample-cell recovery demonstration.

## 2. Background
### 2.1 Text coding as construct measurement
MacKenzie et al.; related organizational text-measurement logic.

### 2.2 AI as rater and specification sensitivity
Carlson & Burbano; Son & Lee.

### 2.3 Human disagreement as information
Mostafazadeh Davani et al.; Fornaciari et al.; Weerasooriya et al.

### 2.4 Generated-variable consequences
Qiao & Huang.

End by distinguishing probability of category membership from construct intensity.

## 3. Data and methods
- MFRC and its selected/enriched sampling design.
- Six Moral Foundations and trained human coder rows.
- Public-snapshot audit and inclusion rules.
- Frozen dev/test/sensitivity split.
- Canonical JEV instrument + one strict sensitivity wording.
- Four prespecified analyses in `docs/METHODS.md`.

## 4. Results
1. Within-foundation uncertainty correspondence.
2. Soft versus hard loss.
3. Targeted versus random review simulation.
4. Recovery of MFRC bucket × subreddit × foundation sample-cell means.
5. Bounded wording sensitivity.

## 5. Discussion
- Preserve probabilities when they demonstrably contain measurement information.
- Validate model uncertainty rather than assuming confidence equals human ambiguity.
- Use uncertainty to prioritize costly review when validated.
- Boundary conditions: one dataset, few raters/item, selected Reddit sample, one decision-model family, idealized review simulation.
- Future extension: replicate on another construct/domain and directly observe incremental human adjudication.
