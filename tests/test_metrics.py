import numpy as np
import pandas as pd
import pytest

from jev_mfrc import FOUNDATIONS
from jev_mfrc.metrics import (
    _bootstrap_loss_deltas,
    _bootstrap_macro_delta,
    _cell_recovery,
    _selective_review,
    _sensitivity,
    binary_entropy,
    squared_loss_to_share,
)


def test_entropy_boundaries_and_midpoint():
    x = binary_entropy([0, 0.5, 1])
    assert x[0] == 0 and x[2] == 0
    assert x[1] == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        binary_entropy([1.2])


def test_squared_loss_to_human_share():
    p, h = 0.7, 2 / 3
    assert squared_loss_to_share([p], [h])[0] == pytest.approx((p - h) ** 2)


def test_hard_vs_soft_squared_loss_delta_equals_coder_level_brier_delta():
    coders = np.array([1.0, 1.0, 0.0])
    h = coders.mean()
    p = 0.72
    hard = float(p >= 0.5)
    share_delta = (hard - h) ** 2 - (p - h) ** 2
    coder_delta = np.mean((hard - coders) ** 2 - (p - coders) ** 2)
    assert share_delta == pytest.approx(coder_delta)


def _analysis_df(n=20):
    rows = []
    for i in range(n):
        r = {"item_id": str(i), "subreddit": f"s{i % 3}", "bucket": f"b{i % 2}"}
        for j, f in enumerate(FOUNDATIONS):
            h = 1.0 if (i + j) % 3 else 0.0
            p = 0.9 if h else 0.1
            if i < 4:
                p = 0.51
            r[f"human_{f}"] = h
            r[f"p_{f}"] = p
        rows.append(r)
    return pd.DataFrame(rows)


def test_selective_review_is_reproducible_and_budgeted_by_comment():
    df = _analysis_df()
    a = _selective_review(df, [0.0, 0.2], 20, 42)
    b = _selective_review(df, [0.0, 0.2], 20, 42)
    pd.testing.assert_frame_equal(a, b)
    row0 = a.loc[a["budget"] == 0.0].iloc[0]
    row20 = a.loc[a["budget"] == 0.2].iloc[0]
    assert row20["comments_reviewed"] == 4
    assert row20["targeted_mse"] <= row0["targeted_mse"]
    assert "random_mse_ci95_low" in a and "targeted_minus_random_mse" in a


def test_bootstrap_loss_deltas_reproducible_and_has_both_losses():
    df = _analysis_df()
    out1 = _bootstrap_loss_deltas(df, 50, 42)
    out2 = _bootstrap_loss_deltas(df, 50, 42)
    assert out1 == out2
    assert set(out1) == {"squared", "absolute", "reps"}
    assert set(out1["squared"]["foundation"]) == set(FOUNDATIONS)
    legacy = _bootstrap_macro_delta(df, 50, 42)
    assert legacy["ci95_low"] <= legacy["estimate"] <= legacy["ci95_high"]


def test_cell_recovery_contains_review_hybrids_and_weighted_losses():
    df = _analysis_df(30)
    cells, summary = _cell_recovery(df, [0.0, 0.2], 10, 42)
    assert "targeted_20pct" in cells
    assert "random_mean_20pct" in cells
    assert {"soft", "hard", "targeted", "random_mean"}.issubset(set(summary["method"]))
    assert "mae_cell_size_weighted" in summary


def test_sensitivity_reports_prespecified_comparisons():
    c = _analysis_df(20)
    s = c[["item_id"] + [f"p_{f}" for f in FOUNDATIONS]].copy()
    for f in FOUNDATIONS:
        s[f"p_{f}"] = np.clip(s[f"p_{f}"] + 0.05, 0, 1)
    table, summary = _sensitivity(c, s)
    assert len(table) == len(FOUNDATIONS)
    assert "mean_absolute_probability_change" in table
    assert "hard_label_flip_rate" in table
    assert "canonical_strict_spearman" in table
    assert "strict_minus_canonical_squared_loss_macro" in summary


def test_json_safe_converts_nonfinite_values_to_null_ready_values():
    from jev_mfrc.metrics import _json_safe
    out = _json_safe({"x": float("nan"), "y": [float("inf"), 1.0]})
    assert out == {"x": None, "y": [None, 1.0]}


def test_spearman_ignores_missing_secondary_confidence_values():
    from jev_mfrc.metrics import _spearman
    assert _spearman([1, 2, 3, 4], [1, float("nan"), 3, 4]) == pytest.approx(1.0)


def test_cell_recovery_keeps_same_subreddit_in_different_buckets_separate():
    df = _analysis_df(12)
    df["subreddit"] = "same"
    # There are still two distinct MFRC sampling source cells because bucket differs.
    cells, _ = _cell_recovery(df, [0.0], 2, 42)
    observed = cells[["bucket", "subreddit"]].drop_duplicates()
    assert len(observed) == 2
    assert set(observed["bucket"]) == {"b0", "b1"}


def test_write_summary_reads_current_source_cell_key(tmp_path, monkeypatch):
    import json
    import jev_mfrc.report as report
    monkeypatch.setattr(report, "path", lambda *p: tmp_path.joinpath(*p))
    (tmp_path / "results").mkdir(parents=True)
    payload = {
        "test_items": 10,
        "sensitivity_items": 3,
        "uncertainty_validity": {"macro_mean": 0.2, "ci95_low": 0.1, "ci95_high": 0.3},
        "loss_information_retention": {"squared": {"macro": {"estimate": 0.01, "ci95": [0.0, 0.02]}}},
        "source_sample_cell_recovery": {"soft_mae": 0.1, "hard_mae": 0.2},
    }
    (tmp_path / "results/analysis.json").write_text(json.dumps(payload))
    report.write_summary()
    text = (tmp_path / "results/summary.md").read_text()
    assert "Soft source-cell MAE: 0.100000" in text
    assert "Hard source-cell MAE: 0.200000" in text


def test_dev_review_does_not_truncate_selected_comment_text(tmp_path, monkeypatch):
    import jev_mfrc.report as report
    monkeypatch.setattr(report, "path", lambda *p: tmp_path.joinpath(*p))
    (tmp_path / "data/processed").mkdir(parents=True)
    long_text = "x" * 700 + " END_MARKER"
    item = {"item_id": "d1", "text": long_text, "split": "dev"}
    for f in FOUNDATIONS:
        item[f"human_{f}"] = 1.0
    pd.DataFrame([item]).to_csv(tmp_path / "data/processed/items.csv.gz", index=False, compression="gzip")
    row = {"item_id": "d1", "variant": "canonical"}
    for f in FOUNDATIONS:
        row[f"p_{f}"] = 0.5
    pd.DataFrame([row]).to_csv(tmp_path / "data/processed/dev_predictions.csv.gz", index=False, compression="gzip")
    report.write_dev_review()
    assert "END_MARKER" in (tmp_path / "results/dev_review.md").read_text()
