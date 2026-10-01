from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _load_run():
    p = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("jev_mfrc_runner", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _base(monkeypatch, tmp_path):
    run = _load_run()
    monkeypatch.setattr(run, "load_config", lambda: {"jev": {"model": "jev-latest"}})
    monkeypatch.setattr(run, "ensure_data", lambda cfg: None)
    monkeypatch.setattr(run, "path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(sys, "argv", ["run.py", "--stage", "all", "--max-items", "7"])
    return run


def test_runner_import_defers_analysis_dependencies():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import run, sys; assert 'matplotlib' not in sys.modules",
        ],
        cwd=root,
        env=env,
        check=True,
    )


def test_all_advances_only_dev_when_dev_incomplete(monkeypatch, tmp_path):
    run = _base(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(run, "completed_calls", lambda cfg, stage: {"dev": 0, "test": 0, "sensitivity": 0}[stage])
    monkeypatch.setattr(run, "expected_calls", lambda cfg, stage: {"dev": 2, "test": 2, "sensitivity": 1}[stage])
    monkeypatch.setattr(run, "run_inference", lambda cfg, stage, n: calls.append((stage, n)) or True)
    monkeypatch.setattr(run, "export_dev", lambda cfg: calls.append(("export_dev", None)))
    assert run.main() == 0
    assert calls == [("dev", 7), ("export_dev", None)]


def test_all_stops_at_human_gate(monkeypatch, tmp_path, capsys):
    run = _base(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(run, "completed_calls", lambda cfg, stage: {"dev": 2, "test": 0, "sensitivity": 0}[stage])
    monkeypatch.setattr(run, "expected_calls", lambda cfg, stage: {"dev": 2, "test": 2, "sensitivity": 1}[stage])
    monkeypatch.setattr(run, "run_inference", lambda cfg, stage, n: calls.append((stage, n)) or True)
    monkeypatch.setattr(run, "export_dev", lambda cfg: calls.append(("export_dev", None)))
    assert run.main() == 0
    assert calls == [("export_dev", None)]
    assert "approve the instrument" in capsys.readouterr().out


def test_all_advances_only_test_after_approval(monkeypatch, tmp_path):
    run = _base(monkeypatch, tmp_path)
    (tmp_path / "approvals").mkdir(parents=True)
    (tmp_path / "approvals/instrument.json").write_text("{}")
    calls = []
    monkeypatch.setattr(run, "completed_calls", lambda cfg, stage: {"dev": 2, "test": 0, "sensitivity": 0}[stage])
    monkeypatch.setattr(run, "expected_calls", lambda cfg, stage: {"dev": 2, "test": 2, "sensitivity": 1}[stage])
    monkeypatch.setattr(run, "run_inference", lambda cfg, stage, n: calls.append((stage, n)) or True)
    monkeypatch.setattr(run, "export_dev", lambda cfg: calls.append(("export_dev", None)))
    assert run.main() == 0
    assert calls == [("export_dev", None), ("test", 7)]


def test_scientific_runner_does_not_depend_on_build_task_state(monkeypatch):
    run = _load_run()
    monkeypatch.setattr(run, "load_config", lambda: {})
    calls = []
    monkeypatch.setattr(run, "ensure_data", lambda cfg: calls.append("data"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--stage", "data"])
    assert run.main() == 0
    assert calls == ["data"]
    assert not hasattr(run, "require_implementation_ready")


def test_taskctl_check_persists_completion_on_the_loaded_state(tmp_path, monkeypatch):
    import importlib.util
    taskctl_path = Path(__file__).resolve().parents[1] / "scripts" / "taskctl.py"
    spec = importlib.util.spec_from_file_location("taskctl_under_test", taskctl_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    state_path = tmp_path / "tasks.json"
    state_path.write_text('{"tasks":[{"id":"01","name":"x","status":"pending"},{"id":"02","name":"y","status":"pending"}]}')
    monkeypatch.setattr(mod, "STATE", state_path)
    monkeypatch.setattr(mod, "run_checks", lambda task_id: None)
    monkeypatch.setattr(sys, "argv", ["taskctl.py", "check", "01"])
    mod.main()
    import json
    state = json.loads(state_path.read_text())
    assert state["tasks"][0]["status"] == "complete"
    assert state["tasks"][1]["status"] == "pending"
