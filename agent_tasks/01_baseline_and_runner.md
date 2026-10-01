# Task 01 — baseline and runner

Goal: make the repository easy to build and run without adding process machinery.

Focus on: `README.md`, `run.py`, `run.sh`, `continue_build.sh`, `scripts/taskctl.py`, config/imports, and focused tests.

Requirements:
- `./continue_build.sh` must run exactly one bounded Luna task, then run that task's focused checks and stop.
- If Codex is unavailable, `./continue_build.sh --print` or the fallback path must print the exact next task without changing state.
- The research runner must not depend on build-task completion state.
- Keep ordinary Python + shell. No CI framework, container requirement, orchestration service, change-scope verifier, model-log parser, or receipt gate.
- Do not make network or paid calls.

Run the stated task checks, append a short note to `BUILD_LOG.md`, and stop.
