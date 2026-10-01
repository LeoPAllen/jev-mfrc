# Task 03 — instrument and JEV client

Goal: make the measurement instrument faithful to MFRC and make paid inference resumable without assuming undocumented provider guarantees.

Focus on `src/jev_mfrc/prompts.py`, `src/jev_mfrc/jev.py`, `scripts/approve_instrument.py`, `src/jev_mfrc/report.py`, tests, and human-gate docs.

Requirements:
- Canonical questions should track the MFRC Coding Guide-2: whether the text expresses a concern/belief/attitude/emotion grounded in each of the six domains; generic thin morality is not a positive foundation label.
- Keep one canonical bundle and one prespecified stricter wording sensitivity bundle. Development inference uses canonical only; strict is held-out sensitivity, not a prompt competitor.
- Treat JEV `noul` as the provider-documented probability of yes/true.
- Before a paid run, call authenticated `GET /v1/models`; freeze and record the configured request model's advertised metadata for this experiment.
- Do not assume the `model` string returned by `POST /v1/systemone` is a requestable pinned version. Always request the configured model name; record the returned model and stop if identifying model metadata (name/release date) or returned-model identity observably drifts within the experiment; description-only copy edits need not invalidate a run.
- Cache each valid response immediately and resume exact completed calls.
- 429 may retry with bounded backoff. Ambiguous timeout/connection/5xx outcomes should stop rather than be blindly retried.
- Held-out inference requires human approval of the exact data split, canonical + strict instrument hashes, model snapshot, complete canonical dev evidence, and prespecified methods/config. Implementation-file hashes and report-format hashes must not create spurious re-approval requirements.
- Do not make live provider calls in this task.

Run the task checks, append a short note to `BUILD_LOG.md`, and stop.
