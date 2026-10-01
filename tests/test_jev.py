from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from jev_mfrc import FOUNDATIONS
from jev_mfrc import jev
from jev_mfrc.prompts import instrument_bundle_hash, prompt_hash

EXP = "e" * 64
PROV = {
    "repo_id": "x", "requested_revision": "main", "resolved_revision": "sha", "split": "train_dedup",
    "raw_sha256": "r", "data_config_fingerprint": "c", "processed_items_sha256": "i",
}
CFG = {
    "seed": 42,
    "jev": {"base_url": "https://api.typesafe.ai", "model": "jev-latest"},
    "analysis": {"review_budgets": [0.0, 0.1], "random_review_reps": 5, "bootstrap_reps": 10},
}


def _payload(model="jev-2026-09-15"):
    return {
        "model": model,
        "answers": {f: {"type": "noul", "noul": 0.25} for f in FOUNDATIONS},
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }


def _patch_provenance(monkeypatch):
    monkeypatch.setattr(jev, "experiment_fingerprint", lambda cfg: EXP)
    monkeypatch.setattr(jev, "prepared_data_provenance", lambda cfg: dict(PROV))


def _items(root):
    p = root / "data/processed"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"item_id": "d1", "text": "dev1", "split": "dev", "sensitivity": False},
        {"item_id": "d2", "text": "dev2", "split": "dev", "sensitivity": False},
        {"item_id": "t1", "text": "test1", "split": "test", "sensitivity": True},
        {"item_id": "t2", "text": "test2", "split": "test", "sensitivity": False},
    ]).to_csv(p / "items.csv.gz", index=False, compression="gzip")


def _snapshot(root, returned="jev-2026-09-15", release_date="2026-09-15"):
    p = root / "cache"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"jev_model_{EXP[:16]}.json").write_text(json.dumps({
        "request_protocol_version": jev.REQUEST_PROTOCOL_VERSION,
        "experiment_fingerprint": EXP,
        "provider_base_url": CFG["jev"]["base_url"],
        "requested_model": CFG["jev"]["model"],
        "advertised_model": {"name": CFG["jev"]["model"], "description": "alias", "release_date": release_date},
        "observed_response_model": returned,
        "created_at": "now",
    }))


def _seed_dev(root, returned="jev-2026-09-15"):
    _snapshot(root, returned)
    con = jev.connect(root / "cache/jev.sqlite")
    for item, text in [("d1", "dev1"), ("d2", "dev2")]:
        payload = _payload(returned)
        con.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (EXP, CFG["jev"]["base_url"], item, "canonical", prompt_hash("canonical"), CFG["jev"]["model"],
             returned, jev._state_hash(text), json.dumps(payload, sort_keys=True), 1, 0, "now"),
        )
    con.commit(); con.close()


def _approve(root):
    _seed_dev(root)
    (root / "approvals").mkdir(parents=True, exist_ok=True)
    payload = {
        "instrument_bundle_hash": instrument_bundle_hash(),
        "prompt_hashes": {v: prompt_hash(v) for v in ("canonical", "strict")},
        "experiment_fingerprint": EXP,
        "data_provenance": dict(PROV),
        "request_protocol_version": jev.REQUEST_PROTOCOL_VERSION,
        "requested_jev_model": CFG["jev"]["model"],
        "jev_provider_base_url": CFG["jev"]["base_url"],
        "model_snapshot": jev.model_snapshot_scientific(CFG),
        "prespecified_methods": jev.methods_plan_details(CFG),
        "methods_plan_fingerprint": jev.methods_plan_fingerprint(CFG),
        "split_summary": jev.split_summary(),
        "development_evidence": {
            "completed_calls": jev.completed_calls(CFG, "dev"),
            "expected_calls": jev.expected_calls(CFG, "dev"),
            "sha256": jev.dev_evidence_fingerprint(CFG),
        },
    }
    p = root / "approvals/instrument.json"
    p.write_text(json.dumps(payload))
    return p


def test_job_plan_uses_canonical_dev_only_and_blocks_heldout_without_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path)
    dev = jev._jobs(CFG, "dev")
    assert len(dev) == 2
    assert set(dev["variant"]) == {"canonical"}
    sensitivity = jev._jobs_unchecked("sensitivity")
    assert list(sensitivity["item_id"]) == ["t1"]
    assert set(sensitivity["variant"]) == {"strict"}
    with pytest.raises(RuntimeError, match="Held-out inference is blocked"):
        jev._jobs(CFG, "test")


def test_completed_calls_resume_without_another_provider_request(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path); _seed_dev(tmp_path)
    monkeypatch.setattr(jev, "_preflight_model_available", lambda cfg: pytest.fail("cached calls should not trigger preflight"))
    monkeypatch.setattr(jev, "_post", lambda *args, **kwargs: pytest.fail("cached calls must not be submitted again"))

    result = jev.infer(CFG, stage="dev")

    assert result == {"new_calls": 0, "missing_before": 0, "remaining": 0}


def test_validate_response():
    out = jev.validate_response(_payload())
    assert out["care"] == 0.25
    bad = _payload(); bad["answers"]["care"]["noul"] = 1.5
    with pytest.raises(ValueError, match="outside"):
        jev.validate_response(bad)


def test_cache_primary_key_scopes_experiment_provider_prompt_model_and_state(tmp_path):
    con = jev.connect(tmp_path / "cache.sqlite")
    base = ("e1", "https://provider", "i", "canonical", "h1", "m1", "returned", "state1", "{}", 1, 0, "now")
    con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", base); con.commit()
    assert jev._cached(con, "e1", "https://provider", "i", "canonical", "h1", "m1", "state1")
    assert not jev._cached(con, "e2", "https://provider", "i", "canonical", "h1", "m1", "state1")
    assert not jev._cached(con, "e1", "https://other", "i", "canonical", "h1", "m1", "state1")
    assert not jev._cached(con, "e1", "https://provider", "i", "canonical", "h2", "m1", "state1")
    assert not jev._cached(con, "e1", "https://provider", "i", "canonical", "h1", "m2", "state1")
    assert not jev._cached(con, "e1", "https://provider", "i", "canonical", "h1", "m1", "state2")
    con.close()


def test_cache_can_safely_hold_multiple_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path)
    con = jev.connect(tmp_path / "cache/jev.sqlite")
    con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("old-exp", CFG["jev"]["base_url"], "d1", "canonical", "x", "jev-latest", "old", "x", "{}", 1, 0, "now"))
    con.commit(); con.close()
    # Foreign rows are harmless because experiment fingerprint is part of every lookup key.
    assert len(jev._jobs(CFG, "dev")) == 2


def test_legacy_cache_with_rows_fails_loudly(tmp_path):
    p = tmp_path / "legacy.sqlite"
    con = sqlite3.connect(p); con.execute("CREATE TABLE predictions (item_id TEXT)"); con.execute("INSERT INTO predictions VALUES ('x')"); con.commit(); con.close()
    with pytest.raises(RuntimeError, match="Legacy JEV cache schema"):
        jev.connect(p)


def test_preflight_records_advertised_model_and_ignores_description_only_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    description = {"value": "first"}
    seen = {}
    class Response:
        status_code = 200; text = ""
        def json(self):
            return {"models": [{"name": "jev-latest", "description": description["value"], "release_date": "2026-09-15", "account_tier": "research"}]}
    def fake_get(url, headers, timeout):
        seen.update(url=url, headers=headers, timeout=timeout)
        return Response()
    monkeypatch.setattr(jev.requests, "get", fake_get)
    snap = jev._preflight_model_available(CFG)
    assert snap["advertised_model"] == {"name": "jev-latest", "description": "first", "release_date": "2026-09-15", "account_tier": "research"}
    assert seen == {
        "url": "https://api.typesafe.ai/v1/models",
        "headers": {"Authorization": "Bearer test-key"},
        "timeout": 60.0,
    }
    description["value"] = "copy edit"
    again = jev._preflight_model_available(CFG)  # documentation-only drift is not a blocker
    assert again["advertised_model"]["description"] == "first"  # the initial metadata remains frozen


def test_preflight_stops_if_alias_release_date_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    release = {"value": "2026-09-15"}
    class Response:
        status_code = 200; text = ""
        def json(self):
            return {"models": [{"name": "jev-latest", "description": "x", "release_date": release["value"]}]}
    monkeypatch.setattr(jev.requests, "get", lambda *a, **k: Response())
    jev._preflight_model_available(CFG)
    release["value"] = "2026-09-30"
    with pytest.raises(RuntimeError, match="different identifying metadata"):
        jev._preflight_model_available(CFG)


def test_post_always_sends_requested_alias_not_response_model(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _snapshot(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen = {}
    class Response:
        status_code = 200; headers = {}; text = ""
        def json(self): return _payload("jev-2026-09-15")
    def fake_post(url, headers, json, timeout):
        seen["body"] = json; return Response()
    monkeypatch.setattr(jev.requests, "post", fake_post)
    jev._post(CFG, "hello", "canonical")
    assert seen["body"]["model"] == "jev-latest"


def test_response_model_drift_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _snapshot(tmp_path, "jev-A")
    with pytest.raises(RuntimeError, match="changed within the experiment"):
        jev._record_returned_model(CFG, "jev-B")


def test_infer_caches_valid_response_before_stopping_on_returned_model_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path); _seed_dev(tmp_path, "jev-A")
    con = jev.connect(tmp_path / "cache/jev.sqlite")
    con.execute("DELETE FROM predictions WHERE item_id='d2'")
    con.commit(); con.close()
    monkeypatch.setattr(jev, "_preflight_model_available", lambda cfg: jev.model_snapshot(cfg))
    monkeypatch.setattr(jev, "_post", lambda cfg, text, variant: _payload("jev-B"))

    with pytest.raises(RuntimeError, match="changed within the experiment"):
        jev.infer(CFG, stage="dev")

    con = jev.connect(tmp_path / "cache/jev.sqlite")
    row = con.execute("SELECT returned_model, response_json FROM predictions WHERE item_id='d2'").fetchone()
    assert row is not None
    assert row[0] == "jev-B"
    assert json.loads(row[1])["model"] == "jev-B"
    with pytest.raises(RuntimeError, match="mix returned JEV models"):
        jev._assert_model_cache_consistency(con, CFG)
    con.close()


def test_model_identity_recovers_if_process_stops_after_response_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path); _snapshot(tmp_path, returned=None)
    con = jev.connect(tmp_path / "cache/jev.sqlite")
    con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (EXP, CFG["jev"]["base_url"], "d1", "canonical", prompt_hash("canonical"), CFG["jev"]["model"],
                 "jev-A", jev._state_hash("dev1"), json.dumps(_payload("jev-A")), 1, 0, "now"))
    con.commit()

    jev._assert_model_cache_consistency(con, CFG)
    con.close()

    assert jev.model_snapshot(CFG)["observed_response_model"] == "jev-A"


def test_post_retries_only_429_with_bounded_backoff(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    cfg = json.loads(json.dumps(CFG))
    cfg["jev"].update(rate_limit_retries=2, rate_limit_backoff_seconds=5)
    calls = []
    sleeps = []
    class Response:
        def __init__(self, status, retry_after=None):
            self.status_code = status
            self.headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
            self.text = "limited" if status == 429 else ""
        def json(self): return _payload()
    responses = [Response(429, 120), Response(429), Response(200)]
    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"]["model"])
        return responses[len(calls) - 1]
    monkeypatch.setattr(jev.requests, "post", fake_post)
    monkeypatch.setattr(jev.time, "sleep", sleeps.append)

    jev._post(cfg, "hello", "canonical")

    assert calls == ["jev-latest"] * 3
    assert sleeps == [60.0, 10.0]


@pytest.mark.parametrize("failure", ["timeout", "server"])
def test_post_does_not_blindly_retry_ambiguous_outcomes(monkeypatch, failure):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    calls = []
    def fake_post(*args, **kwargs):
        calls.append(1)
        if failure == "timeout":
            raise jev.requests.Timeout("ambiguous")
        class Response:
            status_code = 503; headers = {}; text = "temporarily unavailable"
        return Response()
    monkeypatch.setattr(jev.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="ambiguously|not auto-retrying"):
        jev._post(CFG, "hello", "canonical")

    assert calls == [1]


def test_current_cached_predictions_require_matching_model_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path)
    con = jev.connect(tmp_path / "cache/jev.sqlite")
    con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (EXP, CFG["jev"]["base_url"], "d1", "canonical", prompt_hash("canonical"), "jev-latest", "jev-A", jev._state_hash("dev1"), json.dumps(_payload("jev-A")), 1, 0, "now"))
    con.commit(); con.close()
    with pytest.raises(RuntimeError, match="model snapshot is missing"):
        jev.completed_calls(CFG, "dev")


def test_export_rejects_cached_payload_model_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path); _seed_dev(tmp_path, "jev-A")
    con = jev.connect(tmp_path / "cache/jev.sqlite")
    con.execute("UPDATE predictions SET response_json=? WHERE item_id='d1'", (json.dumps(_payload("jev-B"), sort_keys=True),)); con.commit(); con.close()
    with pytest.raises(RuntimeError, match="response/model mismatch"):
        jev.export_dev_predictions(CFG)


def test_approval_allows_heldout_then_methods_change_invalidates_it(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path); _approve(tmp_path)
    assert len(jev._jobs(CFG, "test")) == 2
    sensitivity = jev._jobs(CFG, "sensitivity")
    assert len(sensitivity) == 1
    assert set(sensitivity["variant"]) == {"strict"}
    changed = json.loads(json.dumps(CFG)); changed["analysis"]["review_budgets"] = [0.0, 0.2]
    with pytest.raises(RuntimeError, match="Prespecified analysis settings"):
        jev._jobs(changed, "test")


def test_approval_invalidated_by_model_snapshot_or_dev_evidence_change(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path); p = _approve(tmp_path)
    payload = json.loads(p.read_text())
    payload["model_snapshot"]["observed_response_model"] = "other"
    p.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="model identity changed"):
        jev._jobs(CFG, "test")


def test_final_export_requires_human_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "path", lambda *p: tmp_path.joinpath(*p)); _patch_provenance(monkeypatch); _items(tmp_path)
    with pytest.raises(RuntimeError, match="Held-out inference is blocked"):
        jev.export_predictions(CFG)


def test_methods_plan_fingerprint_tracks_methods_and_config_not_metrics_source(monkeypatch):
    real_hash = jev._sha256_file
    from pathlib import Path
    methods_path = Path(jev.__file__).resolve().parents[2] / "docs" / "METHODS.md"
    monkeypatch.setattr(jev, "_sha256_file", lambda p: "methods-v1" if Path(p) == methods_path else real_hash(Path(p)))
    a = jev.methods_plan_fingerprint(CFG)
    changed = json.loads(json.dumps(CFG)); changed["analysis"]["bootstrap_reps"] = 11
    b = jev.methods_plan_fingerprint(changed)
    assert a != b
