# JEV × MFRC — research-first study package

A compact, resumable methods study of probabilistic AI text measurement. It asks whether JEV's native probabilities preserve information that thresholded labels discard and whether uncertainty ranks cases for an idealized human-review exercise.

## Study scope

One public dataset (MFRC), six Moral Foundations, one primary JEV instrument, one prespecified stricter wording sensitivity check, and four analyses:

1. Does JEV entropy associate with disagreement among trained human coders?
2. What information is lost when the same JEV probabilities are thresholded to 0/1?
3. At fixed budgets, does uncertainty-targeted oracle reference replacement reduce error faster than random replacement?
4. How closely do soft and hard JEV outputs recover observed MFRC `(bucket, subreddit) × foundation` sample-cell means?

The study does not add annotations, fine-tune models, make causal claims, compare an LLM leaderboard, or shop among prompts.

## Fresh setup

Requires Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

The scientific runner uses the local `src/` package through `./run.sh`; no editable install is needed. Keep provider secrets outside the repository. Set `TYPESAFE_API_KEY` in the shell before paid JEV inference.

## Build the repository

Use `continue_build.sh` as the one-task primitive:

```bash
./continue_build.sh
```

Each invocation gives Codex one pending task, runs that task's focused checks, and advances the task state only when those checks pass. The command also accepts `--status` and `--print` for inspection. `continue_all.sh` is an optional sequential wrapper that runs one-task invocations and auto-commits each successful task. Both build commands need the Codex CLI installed and signed in; see the [official Codex CLI quickstart](https://developers.openai.com/codex/cli). The scientific runner does not depend on Codex or build-task state.

## Run the study

Set the provider key in your shell before paid inference. To limit inference to at most 100 new calls per invocation and resume from cached responses:

```bash
export TYPESAFE_API_KEY="<your key>"
./run.sh --stage all --max-items 100
# Repeat until the development review is ready.
```

When development inference is complete, `all` writes `results/dev_review.md` and stops for the human instrument decision. Inspect the review and the canonical and strict question bundles:

```bash
python3 scripts/approve_instrument.py
```

After reviewing the instrument and manifest, approve it explicitly:

```bash
python3 scripts/approve_instrument.py --approve
./run.sh --stage all --max-items 100
```

Repeat `all` to resume held-out canonical inference, run the fixed sensitivity subset, and analyze once each preceding phase is complete. Each invocation advances at most one scientific phase after preparing and auditing the data. `--max-items` limits new paid calls in that invocation.

Available explicit stages are `data`, `dev`, `test`, `sensitivity`, `analyze`, and `all`. Use `data` to prepare the dataset without JEV calls. Stages are resumable from the call cache. The approval check is enforced for held-out inference and analysis.

## Scientific guardrails

- Raw MFRC data are checksum-verified and immutable after download; derived tables rebuild from that snapshot.
- Exact text is grouped onto one dev/test side to prevent literal leakage; primary items require at least three unique annotators.
- Human vote shares are trained-rater reference distributions, not ontological truth.
- JEV probability is the probability of the bounded yes/no judgment, not moral intensity.
- The review exercise is oracle/reference replacement, not an estimate of labor time saved.
- MFRC source-cell results describe this selected sample, not Reddit population prevalence; comparisons are descriptive, not causal.
- Held-out outcomes do not select prompts, exclusions, thresholds, or claims.

See `docs/RESEARCH_SPEC.md`, `docs/METHODS.md`, `docs/HUMAN_GATES.md`, and `docs/LITERATURE.md` for the frozen design, approval gate, and sources.
