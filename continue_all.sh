#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

if [ ! -x ./continue_build.sh ]; then
  echo "./continue_build.sh is missing or not executable." >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This wrapper must be run from a Git repository." >&2
  exit 2
fi

# For a brand-new repository, snapshot the scaffold once before Luna starts.
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "No Git history found; creating the initial scaffold commit."
  git add -A
  if git diff --cached --quiet; then
    echo "Nothing to commit for the initial scaffold." >&2
    exit 2
  fi
  git commit -m "Initialize JEV-MFRC scaffold"
elif [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is not clean. Commit or discard existing changes before running the automated build." >&2
  git status --short >&2
  exit 2
fi

while true; do
  TASK_ID="$("$PY" scripts/taskctl.py next-id)"

  if [ "$TASK_ID" = "DONE" ]; then
    echo "============================================================"
    echo "All build tasks are complete. Local commits are ready to push."
    echo "============================================================"
    exit 0
  fi

  PROMPT_FILE="$("$PY" scripts/taskctl.py prompt-file "$TASK_ID")"
  TASK_SLUG="$(basename "$PROMPT_FILE" .md)"
  TASK_SLUG="${TASK_SLUG#${TASK_ID}_}"

  echo
  echo "============================================================"
  echo "TASK $TASK_ID — $TASK_SLUG"
  echo "============================================================"

  set +e
  ./continue_build.sh
  CODE=$?
  set -e

  if [ "$CODE" -ne 0 ]; then
    echo "Task $TASK_ID failed with exit code $CODE. Stopping." >&2
    exit "$CODE"
  fi

  NEXT_ID="$("$PY" scripts/taskctl.py next-id)"
  if [ "$NEXT_ID" = "$TASK_ID" ]; then
    echo "Task $TASK_ID returned successfully but did not advance task state. Stopping." >&2
    exit 3
  fi

  git add -A

  if git diff --cached --quiet; then
    echo "Task $TASK_ID advanced but produced no staged changes. Stopping." >&2
    exit 3
  fi

  git commit -m "build: complete task $TASK_ID ($TASK_SLUG)"

  if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is dirty after committing task $TASK_ID. Stopping." >&2
    git status --short >&2
    exit 3
  fi
done
