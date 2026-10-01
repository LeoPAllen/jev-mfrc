from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate_config(cfg: dict) -> dict:
    try:
        seed = int(cfg["seed"])
        dataset = cfg["dataset"]
        sampling = cfg["sampling"]
        jev = cfg["jev"]
        analysis = cfg["analysis"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"config.json is missing or has an invalid required field: {exc}") from exc

    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("repo_id", "revision", "split"):
        if not isinstance(dataset.get(key), str) or not dataset[key].strip():
            raise ValueError(f"dataset.{key} must be a nonempty string")
    if int(dataset.get("min_annotators", 0)) < 3:
        raise ValueError("dataset.min_annotators must be at least 3 for the prespecified MFRC study")
    if int(sampling.get("dev_items", 0)) < 1 or int(sampling.get("sensitivity_items", 0)) < 1:
        raise ValueError("sampling.dev_items and sampling.sensitivity_items must be positive integers")

    if not isinstance(jev.get("base_url"), str) or not jev["base_url"].startswith("https://"):
        raise ValueError("jev.base_url must be an https URL")
    if not isinstance(jev.get("model"), str) or not jev["model"].strip():
        raise ValueError("jev.model must be a nonempty model identifier")
    if float(jev.get("timeout_seconds", 0)) <= 0:
        raise ValueError("jev.timeout_seconds must be positive")
    if int(jev.get("rate_limit_retries", -1)) < 0 or float(jev.get("rate_limit_backoff_seconds", -1)) < 0:
        raise ValueError("JEV retry settings must be nonnegative")

    budgets = analysis.get("review_budgets")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("analysis.review_budgets must be a nonempty list")
    vals = [float(x) for x in budgets]
    if vals != sorted(set(vals)) or any(x < 0 or x > 1 for x in vals) or 0.0 not in vals:
        raise ValueError("analysis.review_budgets must be unique, sorted, in [0,1], and include 0")
    if int(analysis.get("random_review_reps", 0)) < 1 or int(analysis.get("bootstrap_reps", 0)) < 1:
        raise ValueError("analysis random_review_reps and bootstrap_reps must be positive integers")
    return cfg


def load_config() -> dict:
    return validate_config(json.loads((ROOT / "config.json").read_text(encoding="utf-8")))


def path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
