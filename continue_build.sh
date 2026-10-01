#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

case "${1:-}" in
  --status)
    exec "$PY" scripts/taskctl.py status
    ;;
  --print)
    exec "$PY" scripts/taskctl.py prompt
    ;;
  "") ;;
  *)
    echo "usage: ./continue_build.sh [--status|--print]" >&2
    exit 2
    ;;
esac

TASK_ID="$("$PY" scripts/taskctl.py next-id)"
if [ "$TASK_ID" = "DONE" ]; then
  echo "Build plan complete. Run ./run.sh --stage data when you are ready to start the study."
  exit 0
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "Codex CLI not found. Here is the next bounded Luna task:" >&2
  "$PY" scripts/taskctl.py prompt
  exit 2
fi

PROMPT_FILE="$("$PY" scripts/taskctl.py prompt-file "$TASK_ID")"
echo "Running one bounded build task with GPT-6 Luna, max reasoning: $TASK_ID"
set +e
{
  cat AGENTS.md
  printf '\n--- ASSIGNED TASK ---\n\n'
  cat "$PROMPT_FILE"
} | codex exec \
      --model gpt-6-luna \
      --config 'model_reasoning_effort="max"' \
      --config 'approval_policy="never"' \
      --sandbox workspace-write \
      --skip-git-repo-check -
CODE=$?
set -e
if [ "$CODE" -ne 0 ]; then
  echo "Codex exited with code $CODE. The task remains pending." >&2
  exit "$CODE"
fi

"$PY" scripts/taskctl.py check "$TASK_ID"
echo "Task $TASK_ID passed its focused checks and was marked complete. Run ./continue_build.sh again for the next chunk."
