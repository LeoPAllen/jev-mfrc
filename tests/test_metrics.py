import numpy as np
import pandas as pd
import pytest

from jev_mfrc import FOUNDATIONS
from jev_mfrc.metrics import (
    _bootstrap_loss_deltas,
    _bootstrap_macro_delta,
    _cell_recovery,
    _pairwise_coder_disagreement,
    _selective_review,
    _sensitivity,
    _spearman,
    _uncertainty_bootstrap,
    binary_entropy,
    pairwise_disagreement,
    squared_loss_to_share,
)


def test_entropy_boundaries_and_midpoint():
    x = binary_entropy([0, 0.5, 1])
    assert x[0] == 0 and x[2] == 0
    assert x[1] == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        binary_entropy([1.2])


def test_pairwise_disagreement_hand_calculations_and_symmetry():
    got = pairwise_disagreement([0.0, 1.0, 1 / 3, 2 / 3, 0.5], [3, 3, 3, 3, 4])
    np.testing.assert_allclose(got, [0.0, 0.0, 2 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(
        pairwise_disagreement([0.0, 0.2, 1 / 3, 0.5], [3, 5, 3, 4]),
        pairwise_disagreement([1.0, 0.8, 2 / 3, 0.5], [3, 5, 3, 4]),
    )


def test_pairwise_disagreement_rejects_invalid_annotator_counts():
    with pytest.raises(ValueError, match="greater than 1"):
        pairwise_disagreement(0.5, 1)
    with pytest.raises(ValueError, match="greater than 1"):
        pairwise_disagreement([0.0, 0.5], [3, 0])


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


def test_pairwise_coder_disagreement_reports_each_foundation_and_macro_mean():
    rows = []
    for i in range(60):
        row = {"n_annotators": 3 + i % 2}
        n = row["n_annotators"]
        for j, f in enumerate(FOUNDATIONS):
            row[f"human_{f}"] = ((i * (j + 1) + j) % (n + 1)) / n
            row[f"p_{f}"] = ((i * (j + 3) + 7 * j) % 101) / 100
        rows.append(row)
    df = pd.DataFrame(rows)

    got = _pairwise_coder_disagreement(df)
    expected = {
        f: _spearman(
            binary_entropy(df[f"p_{f}"]),
            pairwise_disagreement(df[f"human_{f}"], df["n_annotators"]),
        )
        for f in FOUNDATIONS
    }
    assert got["foundation_spearman"] == expected
    assert got["macro_mean"] == pytest.approx(np.mean(list(expected.values())))


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
    assert "targeted_mae" not in a and "random_mae_mean" not in a


def test_bootstrap_loss_deltas_reproducible_and_has_both_losses():
    df = _analysis_df()
    out1 = _bootstrap_loss_deltas(df, 50, 42)
    out2 = _bootstrap_loss_deltas(df, 50, 42)
    assert out1 == out2
    assert set(out1) == {"squared", "absolute", "reps"}
    assert set(out1["squared"]["foundation"]) == set(FOUNDATIONS)
    legacy = _bootstrap_macro_delta(df, 50, 42)
    assert legacy["ci95_low"] <= legacy["estimate"] <= legacy["ci95_high"]


def test_uncertainty_bootstrap_resamples_comments_together_across_foundations():
    rows = []
    for i in range(40):
        row = {"item_id": str(i)}
        for j, f in enumerate(FOUNDATIONS):
            row[f"human_{f}"] = ((i + j) % 5) / 4
            row[f"p_{f}"] = ((i * 2 + 3 * j + 1) % 5) / 4
        rows.append(row)
    df = pd.DataFrame(rows)
    reps, seed = 40, 7
    actual = _uncertainty_bootstrap(df, reps, seed)

    # Independently reconstruct the bootstrap with one shared row sample for all six codes.
    rng = np.random.default_rng(seed + 1)
    by_foundation = {f: [] for f in FOUNDATIONS}
    macro = []
    for _ in range(reps):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        cors = {
            f: _spearman(binary_entropy(sample[f"human_{f}"]), binary_entropy(sample[f"p_{f}"]))
            for f in FOUNDATIONS
        }
        for f, value in cors.items():
            by_foundation[f].append(value)
        macro.append(float(np.nanmean(list(cors.values()))))

    for f in FOUNDATIONS:
        values = np.asarray(by_foundation[f])
        np.testing.assert_allclose(actual["foundation_ci95"][f], np.quantile(values[np.isfinite(values)], [0.025, 0.975]))
    np.testing.assert_allclose(
        [actual["ci95_low"], actual["ci95_high"]],
        np.quantile(np.asarray(macro)[np.isfinite(macro)], [0.025, 0.975]),
    )


def test_cell_recovery_contains_review_hybrids_and_weighted_losses():
    df = _analysis_df(30)
    cells, summary = _cell_recovery(df, [0.0, 0.2], 10, 42)
    assert "targeted_20pct" in cells
    assert "random_mean_20pct" in cells
    assert {"soft", "hard", "targeted", "random_mean"}.issubset(set(summary["method"]))
    assert "mae_cell_size_weighted" in summary
    assert "rmse_unweighted" not in summary


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
        "source_sample_cell_recovery": {
            "soft_mae": 0.1,
            "hard_mae": 0.2,
            "soft_weighted_mae": 0.11,
            "hard_weighted_mae": 0.21,
        },
        "foundations": list(FOUNDATIONS),
        "robustness": {
            "pairwise_coder_disagreement": {
                "foundation_spearman": {foundation: 0.2 for foundation in FOUNDATIONS},
                "macro_mean": 0.2,
            },
        },
        "bootstrap_units": {
            "primary": "item/comment row",
            "content_cluster_robustness": "exact-text content_id cluster",
            "heldout_rows": 10,
            "heldout_unique_content_ids": 10,
            "row_and_content_cluster_units_coincide": True,
            "content_cluster_bootstrap_applicable": False,
        },
    }
    (tmp_path / "results/analysis.json").write_text(json.dumps(payload))
    report.write_summary()
    text = (tmp_path / "results/summary.md").read_text()
    assert "Soft source-cell MAE: 0.100000" in text
    assert "Hard source-cell MAE: 0.200000" in text
    assert "Unequal-cell-size robustness diagnostic" in text
    assert "Soft: 0.110000" in text
    assert "Hard: 0.210000" in text
    assert "Pairwise coder disagreement" in text
    assert "Unweighted macro mean" in text
    assert "Every held-out row has a unique content_id" in text
    assert "not itself a Brier score" in text
    assert "within-comment coder-variance term cancels" in text
    assert "oracle / idealized reference-replacement simulation" in text
    assert "do not estimate population prevalence or causal effects" in text


def test_dev_review_does_not_truncate_selected_comment_text(tmp_path, monkeypatch):
    import jev_mfrc.report as report
    monkeypatch.setattr(report, "path", lambda *p: tmp_path.joinpath(*p))
    (tmp_path / "data/processed").mkdir(parents=True)
    long_text = "x" * 700 + " END_MARKER"
    item = {"item_id": "d1", "text": long_text, "split": "dev"}
    item["n_annotators"] = 4
    for f in FOUNDATIONS:
        item[f"human_{f}"] = 1.0
    pd.DataFrame([item]).to_csv(tmp_path / "data/processed/items.csv.gz", index=False, compression="gzip")
    row = {"item_id": "d1", "variant": "canonical"}
    for f in FOUNDATIONS:
        row[f"p_{f}"] = 0.5
    pd.DataFrame([row]).to_csv(tmp_path / "data/processed/dev_predictions.csv.gz", index=False, compression="gzip")
    report.write_dev_review()
    review = (tmp_path / "results/dev_review.md").read_text()
    assert "END_MARKER" in review
    assert "item_id: `d1`" in review
    assert "JEV p=0.500; human share=1.000; retained annotators=4" in review
    assert "exactly 0.5 is a tie excluded from both" in review


def test_dev_review_selection_is_deterministic_and_disjoint(tmp_path):
    from jev_mfrc.report import _select_dev_review_cases

    rows = []
    for i, (p, h) in enumerate([
        (0.50, 0.50), (0.49, 0.00), (0.51, 0.00), (0.99, 0.00),
        (0.01, 1.00), (0.98, 0.25), (0.02, 0.75), (0.50, 1.00),
        (0.50, 0.00), (0.50, 0.50), (0.50, 0.25), (0.50, 0.75),
    ]):
        rows.append({"item_id": f"id-{i:02}", "p_care": p, "human_care": h})
    frame = pd.DataFrame(rows)

    selected = _select_dev_review_cases(frame, "care")
    shuffled = _select_dev_review_cases(frame.sample(frac=1, random_state=12), "care")
    assert {key: list(value.item_id) for key, value in selected.items()} == {
        key: list(value.item_id) for key, value in shuffled.items()
    }
    selected_ids = [item_id for section in selected.values() for item_id in section.item_id]
    assert len(selected_ids) == len(set(selected_ids))
    assert list(selected["Closest to JEV p = 0.5"].item_id) == ["id-00", "id-07", "id-08"]


def test_dev_review_ties_are_excluded_from_majority_disagreements():
    from jev_mfrc.report import _select_dev_review_cases

    frame = pd.DataFrame([
        {"item_id": "tie-positive-p", "p_care": 0.99, "human_care": 0.5},
        {"item_id": "tie-negative-p", "p_care": 0.01, "human_care": 0.5},
        {"item_id": "negative-majority", "p_care": 0.99, "human_care": 0.0},
        {"item_id": "positive-majority", "p_care": 0.01, "human_care": 1.0},
        {"item_id": "uncertain-1", "p_care": 0.5, "human_care": 0.0},
        {"item_id": "uncertain-2", "p_care": 0.5, "human_care": 1.0},
        {"item_id": "uncertain-3", "p_care": 0.5, "human_care": 0.5},
    ])

    sections = _select_dev_review_cases(frame, "care")
    positive = set(sections["Confident JEV-positive / human-negative-majority disagreements"].item_id)
    negative = set(sections["Confident JEV-negative / human-positive-majority disagreements"].item_id)
    assert positive == {"negative-majority"}
    assert negative == {"positive-majority"}


def test_dev_review_rejects_noncanonical_development_predictions(tmp_path, monkeypatch):
    import jev_mfrc.report as report

    monkeypatch.setattr(report, "path", lambda *p: tmp_path.joinpath(*p))
    (tmp_path / "data/processed").mkdir(parents=True)
    item = {"item_id": "d1", "text": "development example", "split": "dev", "n_annotators": 3}
    prediction = {"item_id": "d1", "variant": "strict"}
    for foundation in FOUNDATIONS:
        item[f"human_{foundation}"] = 0.5
        prediction[f"p_{foundation}"] = 0.5
    pd.DataFrame([item]).to_csv(tmp_path / "data/processed/items.csv.gz", index=False, compression="gzip")
    pd.DataFrame([prediction]).to_csv(tmp_path / "data/processed/dev_predictions.csv.gz", index=False, compression="gzip")

    with pytest.raises(RuntimeError, match="non-canonical prompt variant"):
        report.write_dev_review()
