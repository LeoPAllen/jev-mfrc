from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import FOUNDATIONS
from .config import path
from .data import experiment_fingerprint, prepared_data_provenance, _sha256_file
from .prompts import instrument_bundle_hash, prompt_hash, question_set

REQUEST_PROTOCOL_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    experiment_fingerprint TEXT NOT NULL,
    provider_base_url TEXT NOT NULL,
    item_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    returned_model TEXT NOT NULL,
    state_sha256 TEXT NOT NULL,
    response_json TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (experiment_fingerprint, provider_base_url, item_id, variant, prompt_hash, requested_model, state_sha256)
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or path("cache", "jev.sqlite")
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone()
    if exists:
        cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()]
        if "experiment_fingerprint" not in cols or "provider_base_url" not in cols:
            n = int(con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            if n:
                con.close()
                raise RuntimeError(
                    "Legacy JEV cache schema detected with existing predictions. Archive/delete cache/jev.sqlite, then rerun."
                )
            con.execute("DROP TABLE predictions")
    con.execute(SCHEMA)
    con.commit()
    return con


def validate_response(payload: dict) -> dict[str, float]:
    """Validate a response; each ``noul`` value is the provider's P(yes/true)."""
    if not isinstance(payload, dict):
        raise ValueError("JEV response is not an object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("JEV response is missing returned model")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("JEV response is missing answers")
    probs: dict[str, float] = {}
    for f in FOUNDATIONS:
        ans = answers.get(f)
        if not isinstance(ans, dict) or ans.get("type") != "noul":
            raise ValueError(f"Missing/invalid noul answer for {f}: {ans}")
        try:
            p = float(ans["noul"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid probability for {f}: {ans}") from exc
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability outside [0,1] for {f}: {p}")
        probs[f] = p
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("JEV response is missing usage")
    for key in ("input_tokens", "output_tokens"):
        if key not in usage or int(usage[key]) < 0:
            raise ValueError(f"JEV response has invalid usage field {key}")
    return probs


def _state_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _current_experiment(cfg: dict) -> str:
    return experiment_fingerprint(cfg)


def _provider_base_url(cfg: dict) -> str:
    return cfg["jev"]["base_url"].rstrip("/")


def methods_plan_details(cfg: dict) -> dict:
    """Return the prespecified analysis inputs that a human reviews at the gate."""
    methods_path = Path(__file__).resolve().parents[2] / "docs" / "METHODS.md"
    if not methods_path.exists():
        raise RuntimeError("docs/METHODS.md is missing; cannot freeze the analysis plan")
    return {
        "seed": int(cfg["seed"]),
        "analysis": cfg["analysis"],
        "methods_sha256": _sha256_file(methods_path),
    }


def methods_plan_fingerprint(cfg: dict) -> str:
    """Freeze research decisions, not implementation-file bytes."""
    payload = methods_plan_details(cfg)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# Backward-compatible name for any external notebook using the prior scaffold.
def analysis_plan_fingerprint(cfg: dict) -> str:
    return methods_plan_fingerprint(cfg)


def _model_snapshot_path(cfg: dict) -> Path:
    namespace = json.dumps(
        [_current_experiment(cfg), _provider_base_url(cfg), cfg["jev"]["model"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
    return path("cache", f"jev_model_{key}.json")


def model_snapshot(cfg: dict) -> dict:
    p = _model_snapshot_path(cfg)
    if not p.exists():
        raise RuntimeError("JEV model snapshot is missing; run development inference first.")
    snap = json.loads(p.read_text(encoding="utf-8"))
    if snap.get("experiment_fingerprint") != _current_experiment(cfg):
        raise RuntimeError("JEV model snapshot belongs to a different data/split experiment.")
    if snap.get("requested_model") != cfg["jev"]["model"]:
        raise RuntimeError("JEV model snapshot was created under a different requested model name.")
    if snap.get("provider_base_url") != _provider_base_url(cfg):
        raise RuntimeError("JEV model snapshot was created against a different provider endpoint.")
    return snap


def model_snapshot_scientific(cfg: dict) -> dict:
    snap = model_snapshot(cfg)
    adv = snap.get("advertised_model") or {}
    return {
        "request_protocol_version": int(snap.get("request_protocol_version", 0)),
        "experiment_fingerprint": snap.get("experiment_fingerprint"),
        "provider_base_url": snap.get("provider_base_url"),
        "requested_model": snap.get("requested_model"),
        "advertised_model": adv,
        "observed_response_model": snap.get("observed_response_model"),
        "created_at": snap.get("created_at"),
    }


def _write_snapshot(p: Path, payload: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def _preflight_model_available(cfg: dict) -> dict:
    """Record the advertised requestable model and detect meaningful alias drift."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set in the environment")
    requested = cfg["jev"]["model"]
    timeout = float(cfg["jev"].get("timeout_seconds", 60))
    url = _provider_base_url(cfg) + "/v1/models"
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError("Could not verify the configured JEV model before inference; no paid call was made.") from exc
    if response.status_code != 200:
        raise RuntimeError(f"Could not verify JEV model availability ({response.status_code}): {response.text[:500]}")
    try:
        response_payload = response.json()
        models = response_payload["models"]
        row = next(m for m in models if str(m.get("name")) == requested)
        if not isinstance(row, dict):
            raise TypeError("model metadata row is not an object")
    except (AttributeError, ValueError, KeyError, TypeError, StopIteration) as exc:
        names = []
        try:
            names = sorted(str(m.get("name")) for m in response_payload.get("models", []))
        except Exception:
            pass
        raise RuntimeError(
            f"Configured JEV model/alias {requested!r} is not advertised for this account. Available names: {names}."
        ) from exc

    # Preserve the complete advertised row for this experiment. Drift checks
    # below intentionally use only identifying name/release-date metadata.
    advertised = dict(row)
    p = _model_snapshot_path(cfg)
    if p.exists():
        snap = model_snapshot(cfg)
        old_adv = snap.get("advertised_model") or {}
        # Description-only edits are documentation, not a scientific reason to halt.
        if old_adv.get("name") != advertised["name"] or old_adv.get("release_date") != advertised["release_date"]:
            raise RuntimeError(
                "The configured JEV alias now advertises different identifying metadata than at the start of this experiment. "
                "Review the model change before making more calls."
            )
        return snap

    snap = {
        "request_protocol_version": REQUEST_PROTOCOL_VERSION,
        "experiment_fingerprint": _current_experiment(cfg),
        "provider_base_url": _provider_base_url(cfg),
        "requested_model": requested,
        "advertised_model": advertised,
        "observed_response_model": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_snapshot(p, snap)
    return snap


def _record_returned_model(cfg: dict, returned_model: str) -> None:
    snap = model_snapshot(cfg)
    observed = snap.get("observed_response_model")
    if observed is None:
        snap["observed_response_model"] = returned_model
        _write_snapshot(_model_snapshot_path(cfg), snap)
    elif observed != returned_model:
        raise RuntimeError(
            f"JEV response model changed within the experiment: first={observed!r}, now={returned_model!r}. "
            "Do not mix model versions in one study run."
        )


def _assert_model_cache_consistency(con: sqlite3.Connection, cfg: dict) -> None:
    exp = _current_experiment(cfg)
    provider = _provider_base_url(cfg)
    rows = con.execute(
        "SELECT DISTINCT returned_model FROM predictions WHERE experiment_fingerprint=? AND provider_base_url=? AND requested_model=?",
        (exp, provider, cfg["jev"]["model"]),
    ).fetchall()
    if not rows:
        return
    observed = {str(r[0]) for r in rows}
    if len(observed) != 1:
        raise RuntimeError(f"Cached predictions mix returned JEV models: {sorted(observed)}")
    snap = model_snapshot(cfg)
    snap_model = snap.get("observed_response_model")
    if snap_model is None:
        # A process can stop just after committing a valid response and before
        # updating the JSON snapshot. Recover its identity from that exact cache.
        snap["observed_response_model"] = next(iter(observed))
        _write_snapshot(_model_snapshot_path(cfg), snap)
    elif snap_model not in observed:
        raise RuntimeError("Cached JEV predictions do not match the experiment's model snapshot.")


def _jobs_unchecked(stage: str) -> pd.DataFrame:
    items_path = path("data", "processed", "items.csv.gz")
    if not items_path.exists():
        raise RuntimeError("Prepared items missing; run --stage data first")
    items = pd.read_csv(items_path)
    if stage == "dev":
        out = items.loc[items["split"] == "dev", ["item_id", "text"]].copy()
        out["variant"] = "canonical"
    elif stage == "test":
        out = items.loc[items["split"] == "test", ["item_id", "text"]].copy()
        out["variant"] = "canonical"
    elif stage == "sensitivity":
        out = items.loc[(items["split"] == "test") & (items["sensitivity"]), ["item_id", "text"]].copy()
        out["variant"] = "strict"
    else:
        raise ValueError(f"Unknown inference stage: {stage}")
    if out.empty:
        raise RuntimeError(f"No jobs found for stage {stage}")
    return out.reset_index(drop=True)


def _wanted_keys(stage: str) -> set[tuple[str, str, str, str]]:
    jobs = _jobs_unchecked(stage)
    return {(r.item_id, r.variant, prompt_hash(r.variant), _state_hash(r.text)) for r in jobs.itertuples(index=False)}


def split_summary() -> dict[str, int]:
    """Small human-readable summary of the exact, separately hashed split table."""
    items_path = path("data", "processed", "items.csv.gz")
    if not items_path.exists():
        raise RuntimeError("Prepared items missing; run --stage data first")
    items = pd.read_csv(items_path)
    return {
        "dev_items": int((items["split"] == "dev").sum()),
        "test_items": int((items["split"] == "test").sum()),
        "sensitivity_items": int(items["sensitivity"].sum()),
    }


def _current_prediction_rows(cfg: dict, stage: str) -> pd.DataFrame:
    wanted = _wanted_keys(stage)
    exp = _current_experiment(cfg)
    provider = _provider_base_url(cfg)
    con = connect()
    _assert_model_cache_consistency(con, cfg)
    rows = pd.read_sql_query(
        "SELECT * FROM predictions WHERE experiment_fingerprint=? AND provider_base_url=? AND requested_model=?",
        con,
        params=(exp, provider, cfg["jev"]["model"]),
    )
    con.close()
    if rows.empty:
        raise RuntimeError(f"{stage} inference is incomplete: 0 of {len(wanted)} current calls")
    mask = rows.apply(
        lambda r: (r["item_id"], r["variant"], r["prompt_hash"], r["state_sha256"]) in wanted,
        axis=1,
    )
    rows = rows.loc[mask].copy()
    if len(rows) != len(wanted):
        raise RuntimeError(f"{stage} inference is incomplete: {len(rows)} of {len(wanted)} current calls")
    if rows.duplicated(["item_id", "variant", "prompt_hash", "state_sha256"]).any():
        raise RuntimeError(f"Duplicate current prediction keys detected for {stage}")
    return rows.sort_values(["item_id", "variant"]).reset_index(drop=True)


def dev_evidence_fingerprint(cfg: dict) -> str:
    rows = _current_prediction_rows(cfg, "dev")
    records = []
    for r in rows.itertuples(index=False):
        records.append(
            {
                "item_id": r.item_id,
                "variant": r.variant,
                "prompt_hash": r.prompt_hash,
                "state_sha256": r.state_sha256,
                "requested_model": r.requested_model,
                "returned_model": r.returned_model,
                "response_sha256": hashlib.sha256(str(r.response_json).encode("utf-8")).hexdigest(),
            }
        )
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_approval(cfg: dict) -> dict:
    p = path("approvals", "instrument.json")
    if not p.exists():
        raise RuntimeError(
            "Held-out inference is blocked. Review results/dev_review.md and run "
            "`PYTHONPATH=src python scripts/approve_instrument.py --approve`."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    expected_hashes = {v: prompt_hash(v) for v in ("canonical", "strict")}
    if payload.get("prompt_hashes") != expected_hashes or payload.get("instrument_bundle_hash") != instrument_bundle_hash():
        raise RuntimeError("Instrument approval does not match the current questions; re-review and re-approve.")
    if payload.get("request_protocol_version") != REQUEST_PROTOCOL_VERSION:
        raise RuntimeError("JEV request protocol changed after approval; re-review before held-out inference.")
    if payload.get("requested_jev_model") != cfg["jev"]["model"] or payload.get("jev_provider_base_url") != _provider_base_url(cfg):
        raise RuntimeError("Instrument approval was made under a different JEV model request or provider endpoint.")
    if payload.get("methods_plan_fingerprint") != methods_plan_fingerprint(cfg):
        raise RuntimeError("Prespecified analysis settings or methods changed after approval; review and re-approve.")
    if payload.get("prespecified_methods") != methods_plan_details(cfg):
        raise RuntimeError("Approved prespecified methods/configuration do not match the current plan.")

    current_prov = prepared_data_provenance(cfg)
    current_exp = _current_experiment(cfg)
    if payload.get("experiment_fingerprint") != current_exp or payload.get("data_provenance") != current_prov:
        raise RuntimeError("Instrument approval belongs to a different MFRC snapshot/split experiment; re-run dev review and approve again.")
    if payload.get("split_summary") != split_summary():
        raise RuntimeError("Approved MFRC split counts do not match the current prepared experiment.")
    if payload.get("model_snapshot") != model_snapshot_scientific(cfg):
        raise RuntimeError("JEV model identity changed relative to the approved development run.")
    evidence = {
        "completed_calls": completed_calls(cfg, "dev"),
        "expected_calls": expected_calls(cfg, "dev"),
        "sha256": dev_evidence_fingerprint(cfg),
    }
    if evidence["completed_calls"] != evidence["expected_calls"] or payload.get("development_evidence") != evidence:
        raise RuntimeError("Instrument approval does not match the current complete development predictions.")
    return payload


def _jobs(cfg: dict, stage: str) -> pd.DataFrame:
    prepared_data_provenance(cfg)
    if stage in {"test", "sensitivity"}:
        _validate_approval(cfg)
    return _jobs_unchecked(stage)


def _cached(
    con: sqlite3.Connection,
    exp: str,
    provider: str,
    item_id: str,
    variant: str,
    p_hash: str,
    model: str,
    state_hash: str,
) -> bool:
    row = con.execute(
        "SELECT 1 FROM predictions WHERE experiment_fingerprint=? AND provider_base_url=? AND item_id=? AND variant=? AND prompt_hash=? AND requested_model=? AND state_sha256=?",
        (exp, provider, item_id, variant, p_hash, model, state_hash),
    ).fetchone()
    return row is not None


def _post(cfg: dict, text: str, variant: str) -> dict:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set in the environment")
    jcfg = cfg["jev"]
    url = _provider_base_url(cfg) + "/v1/systemone"
    # Always submit the documented requestable name/alias. The response's model
    # field is recorded for drift detection; it is not assumed to be requestable.
    body = {"state": text, "model": jcfg["model"], "questions": question_set(variant)}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    retries = int(jcfg.get("rate_limit_retries", 5))
    backoff = float(jcfg.get("rate_limit_backoff_seconds", 1.0))
    timeout = float(jcfg.get("timeout_seconds", 60))

    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                "JEV request ended ambiguously (timeout/connection/response error). It was not retried automatically; rerun deliberately."
            ) from exc
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("JEV returned non-JSON success payload") from exc
            validate_response(payload)
            return payload
        if response.status_code == 429:
            if attempt >= retries:
                raise RuntimeError("JEV rate-limit retries exhausted")
            wait = backoff * (2**attempt)
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            time.sleep(min(wait, 60.0))
            continue
        if response.status_code >= 500:
            raise RuntimeError(
                f"JEV server error {response.status_code}; not auto-retrying because request completion/cost is ambiguous"
            )
        raise RuntimeError(f"JEV API error {response.status_code}: {response.text[:500]}")
    raise RuntimeError("JEV rate-limit retries exhausted")


def infer(cfg: dict, *, stage: str, max_items: int | None = None) -> dict:
    jobs = _jobs(cfg, stage)
    model = cfg["jev"]["model"]
    exp = _current_experiment(cfg)
    provider = _provider_base_url(cfg)
    con = connect()
    try:
        _assert_model_cache_consistency(con, cfg)
        pending: list[tuple[str, str, str, str, str]] = []
        for row in jobs.itertuples(index=False):
            p_hash = prompt_hash(row.variant)
            s_hash = _state_hash(row.text)
            if not _cached(con, exp, provider, row.item_id, row.variant, p_hash, model, s_hash):
                pending.append((row.item_id, row.text, row.variant, p_hash, s_hash))

        if pending and (max_items is None or max_items > 0):
            _preflight_model_available(cfg)

        new_calls = 0
        for item_id, text, variant, p_hash, s_hash in pending:
            if max_items is not None and new_calls >= max_items:
                break
            payload = _post(cfg, text, variant)
            usage = payload["usage"]
            con.execute(
                """INSERT INTO predictions
                   (experiment_fingerprint, provider_base_url, item_id, variant, prompt_hash, requested_model, returned_model, state_sha256,
                    response_json, input_tokens, output_tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp, provider, item_id, variant, p_hash, model, payload["model"], s_hash,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True), int(usage["input_tokens"]),
                    int(usage["output_tokens"]), datetime.now(timezone.utc).isoformat(),
                ),
            )
            # Persist every valid paid response before checking returned-model
            # identity, so observable drift cannot cause that call to be paid again.
            con.commit()
            new_calls += 1
            _record_returned_model(cfg, payload["model"])
            if new_calls % 100 == 0:
                print(f"JEV: {new_calls} new calls completed this run")
    finally:
        con.close()
    return {"new_calls": new_calls, "missing_before": len(pending), "remaining": max(0, len(pending) - new_calls)}


def expected_calls(cfg: dict, stage: str) -> int:
    return int(len(_jobs(cfg, stage)))


def completed_calls(cfg: dict, stage: str) -> int:
    jobs = _jobs(cfg, stage)
    wanted = {(r.item_id, r.variant, prompt_hash(r.variant), _state_hash(r.text)) for r in jobs.itertuples(index=False)}
    exp = _current_experiment(cfg)
    provider = _provider_base_url(cfg)
    con = connect()
    _assert_model_cache_consistency(con, cfg)
    rows = con.execute(
        "SELECT item_id, variant, prompt_hash, state_sha256 FROM predictions WHERE experiment_fingerprint=? AND provider_base_url=? AND requested_model=?",
        (exp, provider, cfg["jev"]["model"]),
    ).fetchall()
    con.close()
    return sum((i, v, ph, sh) in wanted for i, v, ph, sh in rows)


def _records_from_rows(rows: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in rows.itertuples(index=False):
        payload = json.loads(row.response_json)
        probs = validate_response(payload)
        if payload.get("model") != row.returned_model:
            raise RuntimeError(
                f"Cached response/model mismatch for item {row.item_id}: row={row.returned_model!r}, payload={payload.get('model')!r}"
            )
        rec = {
            "item_id": row.item_id,
            "variant": row.variant,
            "prompt_hash": row.prompt_hash,
            "requested_model": row.requested_model,
            "returned_model": row.returned_model,
            "state_sha256": row.state_sha256,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
        }
        rec.update({f"p_{f}": probs[f] for f in FOUNDATIONS})
        records.append(rec)
    return records


def export_dev_predictions(cfg: dict) -> Path:
    rows = _current_prediction_rows(cfg, "dev")
    out_df = pd.DataFrame(_records_from_rows(rows))
    if out_df.empty:
        raise RuntimeError("No current development predictions available")
    out = path("data", "processed", "dev_predictions.csv.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    return out


def export_predictions(cfg: dict) -> Path:
    _validate_approval(cfg)
    test_rows = _current_prediction_rows(cfg, "test")
    sensitivity_rows = _current_prediction_rows(cfg, "sensitivity")
    rows = pd.concat([test_rows, sensitivity_rows], ignore_index=True)
    out_df = pd.DataFrame(_records_from_rows(rows))
    if out_df.empty:
        raise RuntimeError("No current held-out predictions available")
    out = path("data", "processed", "jev_predictions.csv.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})

    exp = _current_experiment(cfg)
    con = connect()
    all_rows = pd.read_sql_query(
        "SELECT variant, returned_model, input_tokens, output_tokens FROM predictions WHERE experiment_fingerprint=? AND provider_base_url=? AND requested_model=?",
        con,
        params=(exp, _provider_base_url(cfg), cfg["jev"]["model"]),
    )
    con.close()
    usage = all_rows.groupby(["variant", "returned_model"], as_index=False)[["input_tokens", "output_tokens"]].sum()
    usage_out = path("results", "tables", "jev_usage.csv")
    usage_out.parent.mkdir(parents=True, exist_ok=True)
    usage.to_csv(usage_out, index=False)
    return out
