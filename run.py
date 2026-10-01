#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from jev_mfrc.config import load_config, path
from jev_mfrc.data import download_raw, prepare_items
from jev_mfrc.jev import completed_calls, expected_calls, export_dev_predictions, export_predictions, infer
from jev_mfrc.report import write_dev_review, write_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run/resume the JEV × MFRC experiment.")
    p.add_argument("--stage", choices=["data", "dev", "test", "sensitivity", "analyze", "all"], default="all")
    p.add_argument("--max-items", type=int, default=None, help="Maximum NEW paid JEV calls in this invocation")
    return p.parse_args()


def ensure_data(cfg: dict) -> None:
    raw, meta = download_raw(cfg)
    # Derived MFRC tables are small: rebuild deterministically instead of maintaining
    # a complicated stale-cache protocol.
    items = prepare_items(cfg)
    print(f"Raw data: {raw}\nDataset metadata: {meta}\nPrepared items: {items}")


def run_inference(cfg: dict, stage: str, max_items: int | None) -> bool:
    progress = infer(cfg, stage=stage, max_items=max_items)
    done = completed_calls(cfg, stage)
    total = expected_calls(cfg, stage)
    print(f"{stage}: {done}/{total} complete; {progress['new_calls']} new calls this run")
    return done == total


def export_dev(cfg: dict) -> None:
    export_dev_predictions(cfg)
    write_dev_review()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")

    # Build-task state is deliberately independent from scientific execution. The
    # task harness helps development; it is not a research gate.
    ensure_data(cfg)
    if args.stage == "data":
        return 0
    if args.stage == "dev":
        if run_inference(cfg, "dev", args.max_items):
            export_dev(cfg)
        return 0
    if args.stage in {"test", "sensitivity"}:
        run_inference(cfg, args.stage, args.max_items)
        return 0
    if args.stage == "analyze":
        from jev_mfrc.metrics import analyze

        export_predictions(cfg)
        summary = analyze(cfg)
        write_summary()
        print(json.dumps(summary, indent=2))
        return 0

    # `all` advances one scientific phase per call and stops at the human semantic gate.
    if completed_calls(cfg, "dev") < expected_calls(cfg, "dev"):
        if run_inference(cfg, "dev", args.max_items):
            export_dev(cfg)
        return 0
    export_dev(cfg)
    if not path("approvals", "instrument.json").exists():
        print("Development inference complete. Review results/dev_review.md and approve the instrument before held-out inference.")
        return 0
    if completed_calls(cfg, "test") < expected_calls(cfg, "test"):
        run_inference(cfg, "test", args.max_items)
        return 0
    if completed_calls(cfg, "sensitivity") < expected_calls(cfg, "sensitivity"):
        run_inference(cfg, "sensitivity", args.max_items)
        return 0
    from jev_mfrc.metrics import analyze

    export_predictions(cfg)
    summary = analyze(cfg)
    write_summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
