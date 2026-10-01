# JEV × MFRC — research-first Luna Max scaffold

A compact, resumable repository for a methods study of uncertainty-aware AI text measurement. JEV is the measurement instrument; the study asks whether its native probabilities preserve information that thresholded labels discard and whether its uncertainty can help prioritize human review.

## Study scope

One public dataset (MFRC), six Moral Foundations, one primary JEV instrument, one prespecified stricter wording sensitivity check, and four analyses:

1. Does JEV uncertainty track disagreement among trained human coders?
2. How much information is lost when JEV probabilities are thresholded to 0/1?
3. At fixed review budgets, does uncertainty-targeted idealized review reduce error faster than random review?
4. Do soft JEV probabilities recover MFRC `(bucket, subreddit) × foundation` sample-cell means better than hard JEV labels?

Not in scope: new annotation, fine-tuning, causal claims, an LLM leaderboard, prompt shopping, or a large workflow framework.

## Setup

Requirements: Python 3.11+; Codex CLI only if you want the Luna build loop.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Keep provider secrets outside the repository. Before paid JEV inference:

```bash
export TYPESAFE_API_KEY="..."
```

## Develop it in bounded Luna 6 Max chunks

The main build command is:

```bash
./continue_build.sh
```

Each invocation gives Codex exactly one task from `agent_tasks/`, requests `gpt-6-luna` with max reasoning, runs that task's focused checks, marks it complete only if those checks pass, and stops. There are five deliberately broad-enough tasks rather than a long micro-task queue.

Useful inspection commands:

```bash
./continue_build.sh --status
./continue_build.sh --print
```

If Codex is unavailable, the fallback prints the next task and leaves state unchanged. `work.sh` is only a compatibility alias for `continue_build.sh`.

The build harness is intentionally lightweight. It does **not** parse model logs, police file-write scopes, require receipts, or block the scientific runner based on task state. The repository tests and the frozen research documents are the useful constraints.

## Run the study

```bash
./run.sh --stage data
./run.sh --stage dev --max-items 100
# repeat until development inference is complete
python3 scripts/approve_instrument.py          # inspect only
# after personally reviewing results/dev_review.md and both question bundles:
python3 scripts/approve_instrument.py --approve
./run.sh --stage test --max-items 1000
./run.sh --stage sensitivity --max-items 1000
./run.sh --stage analyze
```

Or use `./run.sh --stage all --max-items N`. It advances one scientific phase per invocation, stops at the human instrument gate, and resumes from cached paid responses.

## What is deliberately strict

Only safeguards that protect the study remain hard gates:

- raw MFRC snapshot checksum/provenance;
- deterministic dev/test split, with identical exact text grouped onto the same side to avoid leakage;
- at least three unique annotators per primary item;
- prompt/data/methods/model identity at the pre-held-out human approval gate;
- complete, schema-valid paid responses cached one at a time;
- no mixing of observably different returned JEV models within one experiment;
- no use of held-out outcomes to choose prompts, exclusions, thresholds, or claims.

Derived MFRC tables simply rebuild from raw data. A SQLite cache can safely contain multiple experiment fingerprints. Ordinary code refactors, report formatting changes, and build-task status do not force scientific re-approval.

## JEV model provenance

The runner always submits the configured requestable model name (default `jev-latest`). Before new paid calls it records that name's advertised `/v1/models` metadata. It also records the `model` returned by each System One response and stops on observed within-experiment drift.

It intentionally does **not** assume the response `model` string is a requestable pinned identifier; that guarantee is not part of the provider contract.

## Interpretation guardrails

- Human vote shares are trained-rater reference distributions, not ontological truth.
- JEV probability is probability of the bounded yes/no coding judgment, not moral intensity.
- MFRC deliberately enriches moral content; source-cell results recover this selected sample, not Reddit population prevalence.
- The selective-review exercise is an oracle/idealized reference-replacement simulation, not a claim about actual review labor.

See `docs/RESEARCH_SPEC.md`, `docs/METHODS.md`, and `docs/RISKS.md` for the study specification.
