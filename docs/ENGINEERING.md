# Engineering design

## Principle

Keep the repository small enough that a coding agent can understand the relevant slice in one pass. Optimize for scientific correctness, legibility, and resumability rather than software-process ceremony.

The scientific path is linear:

`download/audit -> canonical dev inference -> human approval -> canonical held-out inference -> strict sensitivity -> analysis`

Use ordinary Python, compressed CSV/JSON/Markdown, pytest, and one SQLite cache. No workflow framework, cloud database, Docker requirement, web UI, model-log parser, write-scope police, or separate invariant framework.

## Luna build loop

`continue_build.sh` launches one bounded Codex task at a time using `gpt-6-luna` and max reasoning. The task file supplies the goal, relevant files, scientific constraints, and checks. After Codex exits, `taskctl.py` runs focused tests and advances `state/tasks.json` only if they pass.

This build state is a convenience, not a scientific gate. The study runner never reads it.

`AGENTS.md` stays short so each task carries only the context it needs. Agents should prefer deletion/simplification to adding infrastructure.

## Data reproducibility

- First download records the requested and resolved Hugging Face revision plus raw SHA-256.
- Raw data are immutable thereafter; checksum or source mismatch stops.
- Derived item tables rebuild deterministically each time the data stage runs. They are cheap and do not need a stale-cache subsystem.
- The experiment fingerprint binds the raw snapshot, relevant dataset/split/inclusion settings, and exact processed-item content.
- Exact text repeated across source cells is preserved as separate source-specific items but grouped onto the same dev/test side.
- Exact duplicate annotation rows are removed and counted; conflicting reconstructed item+annotator rows stop for inspection.

## Paid-call resumability

The SQLite key includes experiment fingerprint, provider endpoint, item ID, prompt variant/hash, requested model name, and state-text hash. Valid responses are committed immediately.

Different experiments may coexist safely in the same database because every lookup is scoped by the experiment fingerprint. This is simpler than forcing a fresh cache for every split/config iteration.

Automatic retry is limited to explicit 429 responses. Timeout, connection, malformed-success, and 5xx outcomes stop because provider-side completion/cost may be ambiguous.

## JEV model identity

Before new paid calls, authenticated `GET /v1/models` must advertise the configured request model. The experiment records the complete advertised metadata row, including name, release date, and description. Name/release-date drift blocks additional calls; description-only edits do not.

Every POST continues to request the configured model name. The returned response `model` field is recorded, and a different returned value within the same experiment stops. The code does not assume that returned string can itself be submitted as a requestable pinned version.

## Human approval

Held-out inference requires one human approval after complete canonical development review. Approval binds scientifically relevant decisions:

- immutable data/split experiment and processed content;
- canonical + strict prompt hashes;
- requested provider/model identity and observed model snapshot;
- complete canonical development evidence;
- request protocol version;
- prespecified analysis configuration and `docs/METHODS.md`.

It does **not** bind arbitrary implementation-file bytes, report formatting, Git cleanliness, or build-task receipts. Refactors that preserve the frozen scientific decisions therefore do not create spurious re-approval work.

## Fail loudly on what can invalidate the study

Examples: raw checksum drift, missing/ambiguous item identity, conflicting annotation records, unknown focal-label spelling, exact-text split leakage, malformed JEV probabilities, incomplete required inference, model drift, or approval mismatch.

Do not fail simply because publication prose and the live public dataset differ in counts; audit and report the observed snapshot.
