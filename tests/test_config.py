from __future__ import annotations

import copy

import pytest

from jev_mfrc.config import validate_config


BASE = {
    "seed": 42,
    "dataset": {"repo_id": "x/y", "revision": "main", "split": "train_dedup", "min_annotators": 3},
    "sampling": {"dev_items": 10, "sensitivity_items": 10},
    "jev": {"base_url": "https://api.example", "model": "jev-x", "timeout_seconds": 60, "rate_limit_retries": 2, "rate_limit_backoff_seconds": 1},
    "analysis": {"review_budgets": [0.0, 0.1], "random_review_reps": 10, "bootstrap_reps": 10},
}


def test_config_accepts_prespecified_shape():
    assert validate_config(copy.deepcopy(BASE))["seed"] == 42


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c["dataset"].update(min_annotators=2), "min_annotators"),
        (lambda c: c["analysis"].update(review_budgets=[0.1, 0.0]), "review_budgets"),
        (lambda c: c["analysis"].update(bootstrap_reps=0), "bootstrap_reps"),
        (lambda c: c["jev"].update(base_url="http://unsafe"), "base_url"),
    ],
)
def test_config_rejects_invalid_edge_cases(mutate, match):
    cfg = copy.deepcopy(BASE)
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)
