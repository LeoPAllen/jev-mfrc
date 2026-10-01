#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_mfrc.config import load_config, path
from jev_mfrc.data import experiment_fingerprint, prepared_data_provenance, prepare_items
from jev_mfrc.jev import (
    REQUEST_PROTOCOL_VERSION,
    completed_calls,
    dev_evidence_fingerprint,
    expected_calls,
    export_dev_predictions,
    methods_plan_details,
    methods_plan_fingerprint,
    model_snapshot_scientific,
    split_summary,
)
from jev_mfrc.prompts import instrument_bundle_hash, prompt_hash, question_set
from jev_mfrc.report import write_dev_review


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and approve the frozen research instrument after development inference.")
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    # Keep the approval tied to a freshly rebuilt deterministic derived table.
    prepare_items(cfg)
    provenance = prepared_data_provenance(cfg)
    if completed_calls(cfg, "dev") != expected_calls(cfg, "dev"):
        raise SystemExit("Development inference is incomplete.")

    export_dev_predictions(cfg)
    write_dev_review()
    review_path = path("results", "dev_review.md")
    if not review_path.exists():
        raise SystemExit("Development review artifact could not be generated.")
    snapshot = model_snapshot_scientific(cfg)
    if not snapshot.get("observed_response_model"):
        raise SystemExit("JEV model snapshot has no observed response model; development provenance is incomplete.")

    print(json.dumps({v: question_set(v) for v in ("canonical", "strict")}, indent=2, ensure_ascii=False))
    print(f"\nReview artifact: {review_path}")
    evidence = {
        "completed_calls": completed_calls(cfg, "dev"),
        "expected_calls": expected_calls(cfg, "dev"),
        "sha256": dev_evidence_fingerprint(cfg),
    }
    manifest = {
        "instrument_bundle_hash": instrument_bundle_hash(),
        "prompt_hashes": {v: prompt_hash(v) for v in ("canonical", "strict")},
        "experiment_fingerprint": experiment_fingerprint(cfg),
        "data_provenance": provenance,
        "split_summary": split_summary(),
        "request_protocol_version": REQUEST_PROTOCOL_VERSION,
        "requested_jev_model": cfg["jev"]["model"],
        "jev_provider_base_url": cfg["jev"]["base_url"].rstrip("/"),
        "model_snapshot": snapshot,
        "prespecified_methods": methods_plan_details(cfg),
        "methods_plan_fingerprint": methods_plan_fingerprint(cfg),
        "development_evidence": evidence,
    }
    print("\nApproval manifest to inspect:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not args.approve:
        raise SystemExit("No approval written. Inspect the review artifact, question bundles, and manifest; re-run with --approve after human review.")

    out = path("approvals", "instrument.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **manifest,
        "approved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "statement": "Human author reviewed the canonical development-only evidence, both prespecified question sets, and this exact approval manifest before held-out inference.",
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
