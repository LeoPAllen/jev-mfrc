#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "tasks.json"

CHECKS = {
    "01": [[sys.executable, "-m", "pytest", "-q", "tests/test_config.py", "tests/test_runner.py"]],
    "02": [[sys.executable, "-m", "pytest", "-q", "tests/test_data.py"]],
    "03": [[sys.executable, "-m", "pytest", "-q", "tests/test_prompts.py", "tests/test_jev.py"]],
    "04": [[sys.executable, "-m", "pytest", "-q", "tests/test_metrics.py"]],
    "05": [
        [sys.executable, "-m", "pytest", "-q", "tests"],
        ["bash", "-n", "continue_build.sh"],
        ["bash", "-n", "run.sh"],
        [sys.executable, "scripts/check_docs.py"],
    ],
}


def load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def next_task(state: dict | None = None) -> dict | None:
    state = load() if state is None else state
    return next((task for task in state["tasks"] if task["status"] != "complete"), None)


def prompt_file(task_id: str) -> Path:
    matches = sorted((ROOT / "agent_tasks").glob(f"{task_id}_*.md"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one prompt for task {task_id}; found {len(matches)}")
    return matches[0]


def show_prompt(task: dict) -> None:
    print((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    print("\n--- ASSIGNED TASK ---\n")
    print(prompt_file(task["id"]).read_text(encoding="utf-8"))


def run_checks(task_id: str) -> None:
    for cmd in CHECKS[task_id]:
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    state = load()
    if cmd == "status":
        for task in state["tasks"]:
            print(f"{task['id']}  {task['status']:<8}  {task['name']}")
        return
    task = next_task(state)
    if cmd == "next-id":
        print(task["id"] if task else "DONE")
        return
    if cmd == "prompt":
        if task is None:
            print("All build tasks are complete.")
        else:
            show_prompt(task)
        return
    if cmd == "prompt-file":
        if len(sys.argv) != 3:
            raise SystemExit("usage: taskctl.py prompt-file TASK_ID")
        print(prompt_file(sys.argv[2]))
        return
    if cmd == "check":
        if len(sys.argv) != 3:
            raise SystemExit("usage: taskctl.py check TASK_ID")
        task_id = sys.argv[2]
        if task is None or task["id"] != task_id:
            raise SystemExit(f"{task_id} is not the next pending task")
        run_checks(task_id)
        task["status"] = "complete"
        task["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(state)
        return
    raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
