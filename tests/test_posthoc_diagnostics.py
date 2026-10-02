import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import posthoc_diagnostics as posthoc


def test_weighted_affine_fit_matches_weighted_least_squares():
    intercept, slope = posthoc.fit_weighted_affine(
        probability=[0.0, 0.5, 1.0],
        human_share=[0.0, 0.0, 1.0],
        weights=[1.0, 1.0, 10.0],
    )
    assert intercept == pytest.approx(-0.196078431372549)
    assert slope == pytest.approx(1.1764705882352942)
    assert (intercept, slope) != pytest.approx((-1 / 6, 1.0))  # unweighted fit


def test_auc_uses_mann_whitney_ranks_and_averages_ties():
    assert posthoc.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.9]) == pytest.approx(0.875)
    assert posthoc.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert np.isnan(posthoc.roc_auc([1, 1], [0.2, 0.8]))


def test_test_outcomes_cannot_change_development_calibration():
    development = pd.DataFrame({
        "n_annotators": [3, 5, 4],
        "p_care": [0.1, 0.55, 0.9],
        "human_care": [0.0, 0.4, 1.0],
    })
    evaluation = pd.DataFrame({
        "p_care": [0.2, 0.7],
        "human_care": [0.0, 1.0],
    })
    fitted_a, predictions_a = posthoc.fit_and_apply_dev_calibration(
        development, evaluation, foundations=["care"]
    )
    changed_evaluation = evaluation.copy()
    changed_evaluation["human_care"] = [1.0, 0.0]
    fitted_b, predictions_b = posthoc.fit_and_apply_dev_calibration(
        development, changed_evaluation, foundations=["care"]
    )
    assert fitted_a == fitted_b
    np.testing.assert_array_equal(predictions_a, predictions_b)


def test_selective_review_ranks_entropy_probability_and_oracle_scores_separately():
    frame = pd.DataFrame({
        "item_id": ["a", "b", "c"],
        "p_one": [0.5, 0.9, 0.1],
        "p_two": [0.5, 0.9, 0.1],
        "human_one": [0.0, 0.0, 0.8],
        "human_two": [1.0, 0.0, 0.8],
    })
    entropy_order = posthoc.select_review_indices(frame, "entropy", foundations=["one", "two"])
    probability_order = posthoc.select_review_indices(frame, "mean_probability", foundations=["one", "two"])
    oracle_order = posthoc.select_review_indices(frame, "oracle_error", foundations=["one", "two"])
    assert frame.iloc[entropy_order[0]]["item_id"] == "a"
    assert frame.iloc[probability_order[0]]["item_id"] == "b"
    assert frame.iloc[oracle_order[0]]["item_id"] == "b"


def test_selective_review_table_is_deterministic():
    frame = pd.DataFrame({
        "item_id": [f"item-{i:02d}" for i in range(12)],
        "p_one": np.linspace(0.05, 0.95, 12),
        "p_two": np.linspace(0.9, 0.1, 12),
        "human_one": [0.0, 0.0, 0.5, 1.0] * 3,
        "human_two": [1.0, 0.5, 0.0, 0.0] * 3,
    })
    first = posthoc.selective_review_table(
        frame, [0.0, 0.25, 0.5, 1.0], random_reps=25, seed=117, foundations=["one", "two"]
    )
    second = posthoc.selective_review_table(
        frame, [0.0, 0.25, 0.5, 1.0], random_reps=25, seed=117, foundations=["one", "two"]
    )
    pd.testing.assert_frame_equal(first, second)

