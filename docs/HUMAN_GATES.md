# Human-only decisions

## Gate 1 — instrument meaning

After canonical development inference completes:

```bash
python3 scripts/approve_instrument.py
```

Read `results/dev_review.md` and the printed canonical + strict question bundles. The development artifact contains **canonical results only**; the strict wording is prespecified for held-out sensitivity and is not a development competitor.

The human author decides:

> Are the canonical questions a defensible operationalization of the MFRC coding guide, and is the strict bundle a reasonable prespecified wording sensitivity check?

If wording changes, do it before approval, rerun canonical development under the new prompt hash, and review again. Do not inspect held-out outcomes while revising wording.

When satisfied:

```bash
python3 scripts/approve_instrument.py --approve
```

Approval is tied to the data/split experiment, both prompt bundles, the JEV request/model snapshot, complete canonical dev evidence, and the prespecified methods/config. Git commit is recorded only as audit context.

Ordinary source refactors or report-format edits do not automatically invalidate the approval. A change to an actual scientific decision should be reflected in the methods/config or protocol version and then re-approved.

## Gate 2 — manuscript claims

Before submission, a human author verifies:

- construct interpretation;
- whether disagreement is accidentally described as pure ambiguity or model error;
- whether probability is accidentally described as moral intensity;
- whether MFRC sample-cell results are accidentally called population prevalence;
- whether descriptive comparisons are accidentally causal;
- whether one-dataset/one-model limitations are clear;
- every citation and numerical claim.

Agents may draft and critique; final interpretive claims remain a human responsibility.
