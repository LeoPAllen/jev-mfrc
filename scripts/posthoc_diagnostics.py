#!/usr/bin/env python3
"""Generate exploratory, post-results diagnostics without changing frozen outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_mfrc import FOUNDATIONS  # noqa: E402

POSTHOC_STATUS = "POST-HOC EXPLORATORY — specified after observing the primary results"
EXPECTED_DEV_ITEMS = 1_000
EXPECTED_TEST_ITEMS = 16_709
EXPECTED_SENSITIVITY_ITEMS = 1_000
N_RELIABILITY_BINS = 10
EPS = 1e-12


def binary_entropy(values: np.ndarray | list[float]) -> np.ndarray:
    """Natural-log binary entropy with exact zero entropy at probabilities 0 and 1."""
    x = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("Entropy inputs must be finite probabilities in [0, 1]")
    clipped = np.clip(x, EPS, 1 - EPS)
    out = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    return np.where((x == 0) | (x == 1), 0.0, out)


def fit_weighted_affine(
    probability: np.ndarray | list[float],
    human_share: np.ndarray | list[float],
    weights: np.ndarray | list[float],
) -> tuple[float, float]:
    """Fit h = a + b*p by weighted least squares, returning (a, b)."""
    p = np.asarray(probability, dtype=float)
    h = np.asarray(human_share, dtype=float)
    w = np.asarray(weights, dtype=float)
    if p.ndim != 1 or h.ndim != 1 or w.ndim != 1 or not (len(p) == len(h) == len(w)):
        raise ValueError("Probability, human share, and weights must be equal-length vectors")
    if len(p) == 0:
        raise ValueError("At least one development item is required")
    if (
        np.any(~np.isfinite(p))
        or np.any(~np.isfinite(h))
        or np.any(~np.isfinite(w))
        or np.any((p < 0) | (p > 1))
        or np.any((h < 0) | (h > 1))
        or np.any(w <= 0)
    ):
        raise ValueError("Inputs must be finite, probabilities/shares in [0, 1], and weights positive")

    weight_sum = float(w.sum())
    mean_p = float(np.average(p, weights=w))
    mean_h = float(np.average(h, weights=w))
    centered_p = p - mean_p
    variance = float(np.dot(w, centered_p**2))
    if variance <= EPS * max(weight_sum, 1.0):
        slope = 0.0
    else:
        covariance = float(np.dot(w, centered_p * (h - mean_h)))
        slope = covariance / variance
    intercept = mean_h - slope * mean_p
    return float(intercept), float(slope)


def fit_calibration_models(
    development: pd.DataFrame,
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
) -> dict[str, dict[str, float]]:
    """Fit calibration parameters from a development frame only."""
    models: dict[str, dict[str, float]] = {}
    weights = development["n_annotators"].to_numpy(float)
    for foundation in foundations:
        intercept, slope = fit_weighted_affine(
            development[f"p_{foundation}"].to_numpy(float),
            development[f"human_{foundation}"].to_numpy(float),
            weights,
        )
        models[foundation] = {"intercept": intercept, "slope": slope}
    return models


def fit_and_apply_dev_calibration(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Fit from development outcomes and apply to evaluation probabilities only."""
    models = fit_calibration_models(development, foundations)
    calibrated = np.column_stack([
        apply_calibration(evaluation[f"p_{foundation}"].to_numpy(float), models[foundation])
        for foundation in foundations
    ])
    return models, calibrated


def apply_calibration(probability: np.ndarray, model: dict[str, float]) -> np.ndarray:
    """Apply a fitted affine mapping and clip its output to the probability range."""
    p = np.asarray(probability, dtype=float)
    return np.clip(model["intercept"] + model["slope"] * p, 0.0, 1.0)


def roc_auc(labels: np.ndarray | list[int], scores: np.ndarray | list[float]) -> float:
    """ROC-AUC from average ranks (the Mann–Whitney U equivalence)."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("Labels and scores must be equal-length vectors")
    if np.any(~np.isfinite(s)) or not set(np.unique(y)).issubset({0, 1, False, True}):
        raise ValueError("Scores must be finite and labels must be binary")
    y = y.astype(bool)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    positive_rank_sum = float(ranks[y].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def select_review_indices(
    frame: pd.DataFrame,
    method: str,
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
) -> np.ndarray:
    """Rank comments for review with stable item-ID tie breaking."""
    p = frame[[f"p_{f}" for f in foundations]].to_numpy(float)
    if method == "entropy":
        score = binary_entropy(p).mean(axis=1)
    elif method == "mean_probability":
        score = p.mean(axis=1)
    elif method == "oracle_error":
        h = frame[[f"human_{f}" for f in foundations]].to_numpy(float)
        score = ((p - h) ** 2).mean(axis=1)
    else:
        raise ValueError(f"Unknown review ranking method: {method}")
    order = pd.DataFrame(
        {"score": score, "item_id": frame["item_id"].astype(str).to_numpy(), "row": np.arange(len(frame))}
    ).sort_values(["score", "item_id"], ascending=[False, True], kind="mergesort")
    return order["row"].to_numpy(int)


def _prediction_matrix(frame: pd.DataFrame, foundations: tuple[str, ...] | list[str]) -> np.ndarray:
    return frame[[f"p_{foundation}" for foundation in foundations]].to_numpy(float)


def _human_matrix(frame: pd.DataFrame, foundations: tuple[str, ...] | list[str]) -> np.ndarray:
    return frame[[f"human_{foundation}" for foundation in foundations]].to_numpy(float)


def _macro(values: np.ndarray) -> float:
    return float(np.mean(values))


def evaluate_methods(
    test: pd.DataFrame,
    methods: dict[str, np.ndarray],
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
) -> pd.DataFrame:
    """Evaluate predictions at comment and source-cell levels on held-out rows."""
    h = _human_matrix(test, foundations)
    result_rows: list[dict] = []
    for method, predictions in methods.items():
        pred = np.asarray(predictions, dtype=float)
        if pred.shape != h.shape:
            raise ValueError(f"{method} predictions must have shape {h.shape}")
        sq = (pred - h) ** 2
        absolute = np.abs(pred - h)
        bias = pred - h
        cell_metrics: list[dict[str, float]] = []
        for column, foundation in enumerate(foundations):
            cell_frame = test[["bucket", "subreddit"]].copy()
            cell_frame["human"] = h[:, column]
            cell_frame["estimate"] = pred[:, column]
            cells = cell_frame.groupby(["bucket", "subreddit"], sort=True).agg(
                n=("human", "size"), human_share=("human", "mean"), estimate=("estimate", "mean")
            )
            cell_error = np.abs(cells["estimate"].to_numpy() - cells["human_share"].to_numpy())
            cell_n = cells["n"].to_numpy(float)
            cell_metrics.append({
                "source_cell_mae_unweighted": float(cell_error.mean()),
                "source_cell_mae_cell_size_weighted": float(np.average(cell_error, weights=cell_n)),
            })
            result_rows.append({
                "split": "TEST",
                "method": method,
                "foundation": foundation,
                "n_comments": len(test),
                "squared_loss_to_human_share": float(sq[:, column].mean()),
                "absolute_loss_to_human_share": float(absolute[:, column].mean()),
                "signed_bias_prediction_minus_human": float(bias[:, column].mean()),
                **cell_metrics[-1],
            })
        result_rows.append({
            "split": "TEST",
            "method": method,
            "foundation": "macro",
            "n_comments": len(test),
            "squared_loss_to_human_share": _macro(sq.mean(axis=0)),
            "absolute_loss_to_human_share": _macro(absolute.mean(axis=0)),
            "signed_bias_prediction_minus_human": _macro(bias.mean(axis=0)),
            "source_cell_mae_unweighted": _macro(np.array([x["source_cell_mae_unweighted"] for x in cell_metrics])),
            "source_cell_mae_cell_size_weighted": _macro(
                np.array([x["source_cell_mae_cell_size_weighted"] for x in cell_metrics])
            ),
        })
    return pd.DataFrame(result_rows)


def bootstrap_paired_differences(
    test: pd.DataFrame,
    calibrated: np.ndarray,
    comparators: dict[str, np.ndarray],
    *,
    reps: int,
    seed: int,
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Bootstrap paired loss/bias differences by comments, retaining all codes per draw."""
    if reps < 1 or batch_size < 1:
        raise ValueError("Bootstrap repetitions and batch size must be positive")
    h = _human_matrix(test, foundations)
    candidate = np.asarray(calibrated, dtype=float)
    if candidate.shape != h.shape:
        raise ValueError("Calibrated predictions do not align with test data")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    metric_deltas: dict[str, dict[str, np.ndarray]] = {}
    for name, comparison in comparators.items():
        other = np.asarray(comparison, dtype=float)
        if other.shape != h.shape:
            raise ValueError(f"Comparator {name} does not align with test data")
        metric_deltas[name] = {
            "squared_loss": (candidate - h) ** 2 - (other - h) ** 2,
            "absolute_loss": np.abs(candidate - h) - np.abs(other - h),
            "signed_bias": (candidate - h) - (other - h),
        }

    for comparison_name, metrics in metric_deltas.items():
        for metric_name, delta in metrics.items():
            point = delta.mean(axis=0)
            boot_foundations = np.empty((reps, len(foundations)), dtype=float)
            for start in range(0, reps, batch_size):
                count = min(batch_size, reps - start)
                sampled_rows = rng.integers(0, len(test), size=(count, len(test)))
                boot_foundations[start:start + count] = delta[sampled_rows].mean(axis=1)
            boot_macro = boot_foundations.mean(axis=1)
            for index, foundation in enumerate(foundations):
                limits = np.quantile(boot_foundations[:, index], [0.025, 0.975])
                rows.append({
                    "comparison": f"calibrated_minus_{comparison_name}",
                    "metric": metric_name,
                    "foundation": foundation,
                    "estimate": float(point[index]),
                    "ci95_low": float(limits[0]),
                    "ci95_high": float(limits[1]),
                    "bootstrap_reps": reps,
                })
            limits = np.quantile(boot_macro, [0.025, 0.975])
            rows.append({
                "comparison": f"calibrated_minus_{comparison_name}",
                "metric": metric_name,
                "foundation": "macro",
                "estimate": _macro(point),
                "ci95_low": float(limits[0]),
                "ci95_high": float(limits[1]),
                "bootstrap_reps": reps,
            })
    return pd.DataFrame(rows)


def strict_sensitivity_tables(
    canonical: pd.DataFrame,
    strict: pd.DataFrame,
    *,
    reps: int,
    seed: int,
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the fixed wording subset and paired comment-bootstrap changes."""
    canonical = canonical.sort_values("item_id", kind="mergesort").reset_index(drop=True)
    strict = strict.sort_values("item_id", kind="mergesort").reset_index(drop=True)
    if canonical["item_id"].tolist() != strict["item_id"].tolist():
        raise ValueError("Canonical and strict sensitivity items must match one-to-one")
    h = _human_matrix(canonical, foundations)
    p = _prediction_matrix(canonical, foundations)
    q = _prediction_matrix(strict, foundations)
    rows: list[dict] = []
    for index, foundation in enumerate(foundations):
        rho = float(spearmanr(p[:, index], q[:, index]).statistic)
        rows.append({
            "foundation": foundation,
            "n_comments": len(canonical),
            "mean_human_share": float(h[:, index].mean()),
            "mean_canonical_probability": float(p[:, index].mean()),
            "mean_strict_probability": float(q[:, index].mean()),
            "canonical_signed_bias": float((p[:, index] - h[:, index]).mean()),
            "strict_signed_bias": float((q[:, index] - h[:, index]).mean()),
            "canonical_strict_probability_spearman": rho,
        })
    foundation_rows = rows.copy()
    rows.append({
        "foundation": "macro",
        "n_comments": len(canonical),
        "mean_human_share": _macro(h.mean(axis=0)),
        "mean_canonical_probability": _macro(p.mean(axis=0)),
        "mean_strict_probability": _macro(q.mean(axis=0)),
        "canonical_signed_bias": _macro((p - h).mean(axis=0)),
        "strict_signed_bias": _macro((q - h).mean(axis=0)),
        "canonical_strict_probability_spearman": _macro(
            np.array([row["canonical_strict_probability_spearman"] for row in foundation_rows])
        ),
    })
    summary = pd.DataFrame(rows)

    deltas = {
        "squared_loss": (q - h) ** 2 - (p - h) ** 2,
        "absolute_loss": np.abs(q - h) - np.abs(p - h),
    }
    point_values: dict[str, np.ndarray] = {
        name: values.mean(axis=0) for name, values in deltas.items()
    }
    point_values["absolute_signed_bias_magnitude"] = np.abs((q - h).mean(axis=0)) - np.abs((p - h).mean(axis=0))
    rng = np.random.default_rng(seed)
    boot: dict[str, np.ndarray] = {
        name: np.empty((reps, len(foundations)), dtype=float) for name in point_values
    }
    for draw in range(reps):
        sampled = rng.integers(0, len(canonical), size=len(canonical))
        for name, values in deltas.items():
            boot[name][draw] = values[sampled].mean(axis=0)
        boot["absolute_signed_bias_magnitude"][draw] = (
            np.abs((q[sampled] - h[sampled]).mean(axis=0))
            - np.abs((p[sampled] - h[sampled]).mean(axis=0))
        )

    bootstrap_rows: list[dict] = []
    for metric, estimate_by_foundation in point_values.items():
        macro_boot = boot[metric].mean(axis=1)
        for index, foundation in enumerate(foundations):
            limits = np.quantile(boot[metric][:, index], [0.025, 0.975])
            bootstrap_rows.append({
                "metric": metric,
                "foundation": foundation,
                "estimate": float(estimate_by_foundation[index]),
                "ci95_low": float(limits[0]),
                "ci95_high": float(limits[1]),
                "bootstrap_reps": reps,
            })
        limits = np.quantile(macro_boot, [0.025, 0.975])
        bootstrap_rows.append({
            "metric": metric,
            "foundation": "macro",
            "estimate": _macro(estimate_by_foundation),
            "ci95_low": float(limits[0]),
            "ci95_high": float(limits[1]),
            "bootstrap_reps": reps,
        })
    return summary, pd.DataFrame(bootstrap_rows)


def selective_review_table(
    test: pd.DataFrame,
    budgets: list[float],
    random_reps: int,
    seed: int,
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
) -> pd.DataFrame:
    """Compare exploratory comment-level reference-replacement review rankings."""
    if random_reps < 1:
        raise ValueError("At least one random review replicate is required")
    p = _prediction_matrix(test, foundations)
    h = _human_matrix(test, foundations)
    raw_mse = float(((p - h) ** 2).mean())
    orders = {
        method: select_review_indices(test, method, foundations)
        for method in ("entropy", "mean_probability", "oracle_error")
    }
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for budget in budgets:
        k = int(round(len(test) * float(budget)))
        metrics: dict[str, float] = {}
        for method, order in orders.items():
            reviewed = order[:k]
            hybrid = p.copy()
            if k:
                hybrid[reviewed] = h[reviewed]
            metrics[method] = float(((hybrid - h) ** 2).mean())
        random_mse = np.empty(random_reps, dtype=float)
        for repetition in range(random_reps):
            reviewed = rng.choice(len(test), size=k, replace=False) if k else np.array([], dtype=int)
            hybrid = p.copy()
            if k:
                hybrid[reviewed] = h[reviewed]
            random_mse[repetition] = float(((hybrid - h) ** 2).mean())
        limits = np.quantile(random_mse, [0.025, 0.975])
        rows.append({
            "budget": float(budget),
            "comments_reviewed": k,
            "baseline_raw_mse": raw_mse,
            "entropy_targeted_mse": metrics["entropy"],
            "mean_probability_targeted_mse": metrics["mean_probability"],
            "random_review_mse_mean": float(random_mse.mean()),
            "random_review_mse_ci95_low": float(limits[0]),
            "random_review_mse_ci95_high": float(limits[1]),
            "oracle_error_targeted_mse": metrics["oracle_error"],
        })
    return pd.DataFrame(rows)


def _marginal_table(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    foundations: tuple[str, ...] | list[str] = FOUNDATIONS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for split, frame in (("DEV", dev), ("TEST", test)):
        for foundation in foundations:
            h = frame[f"human_{foundation}"].to_numpy(float)
            p = frame[f"p_{foundation}"].to_numpy(float)
            rows.append({
                "split": split,
                "foundation": foundation,
                "n_comments": len(frame),
                "mean_human_vote_share": float(h.mean()),
                "mean_raw_jev_probability": float(p.mean()),
                "mean_signed_bias_p_minus_h": float((p - h).mean()),
                "squared_loss_to_human_share": float(((p - h) ** 2).mean()),
                "absolute_loss_to_human_share": float(np.abs(p - h).mean()),
                "raw_probability_sd_sharpness": float(p.std(ddof=0)),
                "mean_raw_probability_entropy": float(binary_entropy(p).mean()),
            })
    return pd.DataFrame(rows)


def _reliability_bins(
    frame: pd.DataFrame,
    split: str,
    foundation: str,
    prediction_method: str,
    prediction: np.ndarray,
    *,
    bins: int = N_RELIABILITY_BINS,
) -> list[dict]:
    """Build equal-frequency bins; stable item IDs break equal-probability ties."""
    probability = np.asarray(prediction, dtype=float)
    count = min(int(bins), len(frame))
    if count < 1:
        return []
    work = pd.DataFrame({
        "probability": probability,
        "human": frame[f"human_{foundation}"].to_numpy(float),
        "item_id": frame["item_id"].astype(str).to_numpy(),
    }).sort_values(["probability", "item_id"], kind="mergesort").reset_index(drop=True)
    work["bin"] = np.arange(len(work), dtype=int) * count // len(work)
    rows: list[dict] = []
    for bin_id, group in work.groupby("bin", sort=True):
        rows.append({
            "split": split,
            "foundation": foundation,
            "prediction_method": prediction_method,
            "bin": int(bin_id) + 1,
            "mean_predicted_probability": float(group["probability"].mean()),
            "mean_human_vote_share": float(group["human"].mean()),
            "n": int(len(group)),
        })
    return rows


def _disagreement_table(test: pd.DataFrame, foundations: tuple[str, ...] | list[str] = FOUNDATIONS) -> pd.DataFrame:
    rows = []
    for foundation in foundations:
        h = test[f"human_{foundation}"].to_numpy(float)
        p = test[f"p_{foundation}"].to_numpy(float)
        mixed = (h > 0) & (h < 1)
        rows.append({
            "foundation": foundation,
            "n_comments": len(test),
            "mixed_case_prevalence": float(mixed.mean()),
            "auc_entropy_predicting_mixed": roc_auc(mixed.astype(int), binary_entropy(p)),
            "auc_raw_probability_predicting_mixed": roc_auc(mixed.astype(int), p),
            "mean_p_human_share_zero": float(p[h == 0].mean()) if np.any(h == 0) else float("nan"),
            "mean_p_human_share_mixed": float(p[mixed].mean()) if np.any(mixed) else float("nan"),
            "mean_p_human_share_one": float(p[h == 1].mean()) if np.any(h == 1) else float("nan"),
        })
    return pd.DataFrame(rows)


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    items_path = ROOT / "data/processed/items.csv.gz"
    dev_prediction_path = ROOT / "data/processed/dev_predictions.csv.gz"
    prediction_path = ROOT / "data/processed/jev_predictions.csv.gz"
    for source in (items_path, dev_prediction_path, prediction_path):
        if not source.exists():
            raise FileNotFoundError(f"Missing post-hoc input {source}; complete the frozen inference run first")
    items = pd.read_csv(items_path)
    dev_predictions = pd.read_csv(dev_prediction_path)
    predictions = pd.read_csv(prediction_path)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    dev_items = items.loc[items["split"] == "dev"].copy()
    test_items = items.loc[items["split"] == "test"].copy()
    dev = dev_predictions.loc[dev_predictions["variant"] == "canonical"].merge(
        dev_items, on="item_id", how="inner", validate="one_to_one"
    )
    canonical = predictions.loc[predictions["variant"] == "canonical"].merge(
        test_items, on="item_id", how="inner", validate="one_to_one"
    )
    strict_items = test_items.loc[test_items["sensitivity"].astype(bool)].copy()
    strict = predictions.loc[predictions["variant"] == "strict"].merge(
        strict_items, on="item_id", how="inner", validate="one_to_one"
    )
    if len(dev) != EXPECTED_DEV_ITEMS or len(dev_items) != EXPECTED_DEV_ITEMS:
        raise RuntimeError(f"Expected {EXPECTED_DEV_ITEMS} canonical dev rows; found {len(dev)}")
    if len(canonical) != EXPECTED_TEST_ITEMS or len(test_items) != EXPECTED_TEST_ITEMS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_ITEMS} canonical test rows; found {len(canonical)}")
    if len(strict) != EXPECTED_SENSITIVITY_ITEMS or len(strict_items) != EXPECTED_SENSITIVITY_ITEMS:
        raise RuntimeError(f"Expected {EXPECTED_SENSITIVITY_ITEMS} strict sensitivity rows; found {len(strict)}")
    if set(dev["item_id"]) & set(canonical["item_id"]):
        raise RuntimeError("Development and held-out item IDs overlap")
    required = [*(f"p_{f}" for f in FOUNDATIONS), *(f"human_{f}" for f in FOUNDATIONS), "n_annotators"]
    for label, frame in (("development", dev), ("held-out", canonical), ("strict sensitivity", strict)):
        if frame[required].isna().any().any():
            raise RuntimeError(f"Missing probability, vote-share, or annotator-count data in {label} frame")
        probabilities = frame[[f"p_{f}" for f in FOUNDATIONS]].to_numpy(float)
        shares = frame[[f"human_{f}" for f in FOUNDATIONS]].to_numpy(float)
        if np.any((probabilities < 0) | (probabilities > 1)) or np.any((shares < 0) | (shares > 1)):
            raise RuntimeError(f"Invalid probability or vote share in {label} frame")
    return dev, canonical, strict_items, strict, config


def _write_table(frame: pd.DataFrame, output: Path) -> None:
    table = frame.copy()
    table.insert(0, "analysis_status", POSTHOC_STATUS)
    table.to_csv(output, index=False, float_format="%.10g")


def _write_calibration_figure(reliability: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    styles = {
        "dev_raw": ("#4C78A8", "Development raw"),
        "test_raw": ("#F58518", "Held-out raw"),
        "test_dev_calibrated": ("#54A24B", "Held-out, dev calibrated"),
    }
    for axis, foundation in zip(axes.flat, FOUNDATIONS):
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        for method, (color, label) in styles.items():
            points = reliability.loc[
                (reliability["foundation"] == foundation)
                & (reliability["prediction_method"] == method)
            ]
            axis.plot(
                points["mean_predicted_probability"],
                points["mean_human_vote_share"],
                marker="o",
                color=color,
                label=label,
            )
        axis.set_title(foundation.title())
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes[1, 1].set_xlabel("Mean predicted probability")
    axes[1, 0].set_xlabel("Mean predicted probability")
    axes[1, 2].set_xlabel("Mean predicted probability")
    axes[0, 0].set_ylabel("Mean human vote share")
    axes[1, 0].set_ylabel("Mean human vote share")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.suptitle(
        "POST-HOC EXPLORATORY — specified after observing the primary results\n"
        "Reliability by foundation; Noul probability is empirically compared with human vote share",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.91))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _fmt(value: float, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def _row(frame: pd.DataFrame, **filters) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _write_audit(
    output: Path,
    marginal: pd.DataFrame,
    methods: pd.DataFrame,
    calibration_bootstrap: pd.DataFrame,
    disagreement: pd.DataFrame,
    review: pd.DataFrame,
    strict_summary: pd.DataFrame,
    strict_bootstrap: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    primary = json.loads((ROOT / "results/analysis.json").read_text(encoding="utf-8"))
    squared = primary["loss_information_retention"]["squared"]["macro"]
    absolute = primary["loss_information_retention"]["absolute"]["macro"]
    uncertainty = primary["uncertainty_validity"]
    source_cells = primary["source_sample_cell_recovery"]
    raw_test = _row(methods, method="raw", foundation="macro")
    calibrated_test = _row(methods, method="dev_fitted_calibrated_probability", foundation="macro")
    calibrated_vs_raw_sq = _row(
        calibration_bootstrap, comparison="calibrated_minus_raw", metric="squared_loss", foundation="macro"
    )
    calibrated_vs_raw_abs = _row(
        calibration_bootstrap, comparison="calibrated_minus_raw", metric="absolute_loss", foundation="macro"
    )
    auc_h = float(disagreement["auc_entropy_predicting_mixed"].mean())
    auc_p = float(disagreement["auc_raw_probability_predicting_mixed"].mean())
    entropy_wins = int(
        (disagreement["auc_entropy_predicting_mixed"] > disagreement["auc_raw_probability_predicting_mixed"]).sum()
    )
    review_20 = review.loc[np.isclose(review["budget"], 0.2)]
    review_20_text = "The configured review budgets do not include 20%."
    if len(review_20) == 1:
        r = review_20.iloc[0]
        review_20_text = (
            f"At 20%, held-out matrix MSE was {_fmt(r['entropy_targeted_mse'], 5)} for entropy targeting, "
            f"{_fmt(r['mean_probability_targeted_mse'], 5)} for mean-probability targeting, "
            f"{_fmt(r['random_review_mse_mean'], 5)} for random review, and "
            f"{_fmt(r['oracle_error_targeted_mse'], 5)} for oracle-error targeting."
        )
    strict_macro = _row(strict_summary, foundation="macro")
    strict_sq = _row(strict_bootstrap, metric="squared_loss", foundation="macro")
    strict_abs = _row(strict_bootstrap, metric="absolute_loss", foundation="macro")
    strict_bias = _row(strict_bootstrap, metric="absolute_signed_bias_magnitude", foundation="macro")
    dev_test_marginal = marginal.pivot(index="foundation", columns="split", values="mean_signed_bias_p_minus_h")
    raw_bias_macro = float(dev_test_marginal["TEST"].mean())
    lines = [
        "# POST-HOC EXPLORATORY AUDIT",
        "",
        "> **These analyses were specified AFTER observing the primary results. They are exploratory diagnostics.**",
        "",
        "All tables and the calibration figure carry the same post-hoc status. The script makes no JEV calls and leaves the frozen `results/` artifacts unchanged.",
        "Human vote shares are a trained-rater reference distribution, not ontological truth. JEV `noul` is the provider-documented probability of the bounded Yes/True judgment; it is not defined as a human-vote probability. These analyses empirically examine whether it maps to that reference distribution.",
        "",
        "## What the frozen primary analyses establish",
        "",
        f"- The prespecified hard-minus-probability squared-loss difference was {_fmt(squared['estimate'], 5)} "
        f"(95% comment-bootstrap interval {_fmt(squared['ci95'][0], 5)} to {_fmt(squared['ci95'][1], 5)}); positive values favor the probability over its own hard threshold under squared loss.",
        f"- The frozen hard-minus-probability absolute-loss difference was {_fmt(absolute['estimate'], 5)} "
        f"(95% interval {_fmt(absolute['ci95'][0], 5)} to {_fmt(absolute['ci95'][1], 5)}); negative values favor the hard label under absolute loss.",
        f"- On observed held-out MFRC source cells, soft-probability MAE was {_fmt(source_cells['soft_mae'], 5)} "
        f"(cell-size weighted {_fmt(source_cells['soft_weighted_mae'], 5)}) and hard-label MAE was {_fmt(source_cells['hard_mae'], 5)} "
        f"(weighted {_fmt(source_cells['hard_weighted_mae'], 5)}). These are selected-sample cell means, not Reddit population prevalence.",
        f"- The frozen entropy-validity macro Spearman correlation between `H(p)` and `H(h)` was {_fmt(uncertainty['macro_mean'], 4)} "
        f"(95% comment-bootstrap interval {_fmt(uncertainty['ci95_low'], 4)} to {_fmt(uncertainty['ci95_high'], 4)}). It does not by itself show that model entropy specifically identifies mixed human judgments.",
        "",
        "## What these post-hoc analyses can and cannot establish",
        "",
        "They can describe marginal mismatch and held-out loss changes for this pinned sample, compare ranking signals for a defined mixed-case outcome, and evaluate a calibration map fit on development items. They cannot establish transport to other samples or models, validate a deployable human-review policy, or turn trained-rater shares into truth.",
        "",
        "### A. Does p contain useful information?",
        "",
        "The frozen soft-versus-hard squared-loss comparison says that retaining probability values helped under that loss. It is a within-instrument representation comparison, not an independent validation claim. The new held-out method table compares raw, hard, development-prevalence, and development-calibrated predictions under squared loss, absolute loss, bias, and source-cell MAE.",
        f"For the macro held-out rows, raw p had squared loss {_fmt(raw_test['squared_loss_to_human_share'], 5)} and absolute loss {_fmt(raw_test['absolute_loss_to_human_share'], 5)}; dev-fitted calibration had squared loss {_fmt(calibrated_test['squared_loss_to_human_share'], 5)} and absolute loss {_fmt(calibrated_test['absolute_loss_to_human_share'], 5)}. Its calibrated-minus-raw squared-loss difference was {_fmt(calibrated_vs_raw_sq['estimate'], 5)} (95% paired interval {_fmt(calibrated_vs_raw_sq['ci95_low'], 5)} to {_fmt(calibrated_vs_raw_sq['ci95_high'], 5)}), and the absolute-loss difference was {_fmt(calibrated_vs_raw_abs['estimate'], 5)} (95% interval {_fmt(calibrated_vs_raw_abs['ci95_low'], 5)} to {_fmt(calibrated_vs_raw_abs['ci95_high'], 5)}).",
        "A ranking score can order comments usefully while its numeric values are miscalibrated. Calibration changes numeric scale and offset; discrimination concerns order. The affine mapping is fitted by weighted least squares on the 1,000 canonical development items only (item weight = retained annotator count), then clipped and applied unchanged to all held-out rows. The constant baseline is the unweighted mean development human vote share by foundation.",
        "",
        "### B. Is raw p calibrated to the human vote distribution?",
        "",
        f"Across foundations, held-out mean signed raw bias was {_fmt(raw_bias_macro, 5)} (positive means p exceeds the human vote share on average). See `marginal_calibration.csv` for DEV and TEST means, losses, and foundation-specific biases; `reliability_bins.csv` and the figure show binned means.",
        "The reliability bins are equal-frequency. Items tied on probability are ordered deterministically by item ID, so a tie can span adjacent bins. Agreement with the diagonal describes marginal calibration at those bins; it does not measure ranking quality. `raw_probability_sd_sharpness` and mean forecast entropy describe prediction concentration (sharpness): more concentrated probabilities can still be miscalibrated or poorly ranked.",
        "",
        "### C. Does H(p) specifically recover disagreement?",
        "",
        f"On held-out data, mean ROC-AUC was {_fmt(auc_h, 4)} for entropy and {_fmt(auc_p, 4)} for raw p when predicting whether human share was mixed; entropy had the higher AUC in {entropy_wins} of {len(disagreement)} foundations. Foundation-level prevalence and mean p for all-zero, mixed, and all-one reference cases are in `entropy_disagreement.csv`.",
        "If raw p predicts mixed cases as well as or better than entropy, entropy is not uniquely detecting human uncertainty: p may carry a positive-score-strength signal. AUC measures discrimination for this binary target, not marginal calibration or probability sharpness. The three questions remain separate.",
        "",
        "### Selective-review diagnostics",
        "",
        f"{review_20_text} The table includes every configured budget. Random review resamples comments; all six predictions for a selected comment are replaced by its full human vote-share vector. Oracle-error ranking uses actual held-out JEV MSE and is an upper-bound diagnostic only, never a deployable strategy. Entropy versus mean-probability targeting tests whether entropy adds triage value beyond positive-score strength.",
        "",
        "### Strict wording sensitivity",
        "",
        f"On the fixed 1,000-case sensitivity subset, macro canonical-to-strict probability Spearman correlation was {_fmt(strict_macro['canonical_strict_probability_spearman'], 4)}. Strict-minus-canonical squared-loss change was {_fmt(strict_sq['estimate'], 5)} (95% paired interval {_fmt(strict_sq['ci95_low'], 5)} to {_fmt(strict_sq['ci95_high'], 5)}); absolute-loss change was {_fmt(strict_abs['estimate'], 5)} (95% interval {_fmt(strict_abs['ci95_low'], 5)} to {_fmt(strict_abs['ci95_high'], 5)}); change in absolute aggregate signed-bias magnitude was {_fmt(strict_bias['estimate'], 5)} (95% interval {_fmt(strict_bias['ci95_low'], 5)} to {_fmt(strict_bias['ci95_high'], 5)}).",
        "A high canonical/strict rank correlation alongside a change in signed-bias magnitude is consistent with wording affecting calibration more than ordering in this subset. It does not identify a general wording effect; canonical remains the frozen primary instrument.",
        "",
        "## Files",
        "",
        "- `calibration_by_foundation.png` — equal-frequency reliability diagnostics, including held-out raw and development-calibrated probabilities.",
        "- `marginal_calibration.csv` and `reliability_bins.csv` — raw DEV/TEST summaries and binned calibration diagnostics.",
        "- `dev_calibration_parameters.csv`, `heldout_method_comparison.csv`, and `calibration_paired_bootstrap.csv` — development fit and held-out comparisons.",
        "- `entropy_disagreement.csv`, `selective_review.csv`, `strict_sensitivity.csv`, and `strict_sensitivity_bootstrap.csv` — disagreement, review, and wording diagnostics.",
        "",
        "No calibration parameter was fitted from test outcomes. No original results, estimands, prompts, approval records, or predictions were changed.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    out_dir = ROOT / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev, test, strict_items, strict, config = _load_frames()
    models, calibrated_test = fit_and_apply_dev_calibration(dev, test)
    test_raw = _prediction_matrix(test, FOUNDATIONS)
    dev_prevalence = np.array([dev[f"human_{f}"].mean() for f in FOUNDATIONS], dtype=float)
    methods = {
        "raw": test_raw,
        "hard": (test_raw >= 0.5).astype(float),
        "constant_dev_prevalence": np.tile(dev_prevalence, (len(test), 1)),
        "dev_fitted_calibrated_probability": calibrated_test,
    }

    marginal = _marginal_table(dev, test)
    _write_table(marginal, out_dir / "marginal_calibration.csv")

    parameter_rows = []
    reliability_rows: list[dict] = []
    for column, foundation in enumerate(FOUNDATIONS):
        model = models[foundation]
        parameter_rows.append({
            "foundation": foundation,
            "development_items": len(dev),
            "development_weight_sum_n_annotators": int(dev["n_annotators"].sum()),
            "intercept_a": model["intercept"],
            "slope_b": model["slope"],
            "constant_dev_human_prevalence": dev_prevalence[column],
        })
        reliability_rows += _reliability_bins(
            dev, "DEV", foundation, "dev_raw", dev[f"p_{foundation}"].to_numpy(float)
        )
        reliability_rows += _reliability_bins(
            test, "TEST", foundation, "test_raw", test[f"p_{foundation}"].to_numpy(float)
        )
        reliability_rows += _reliability_bins(
            test, "TEST", foundation, "test_dev_calibrated", calibrated_test[:, column]
        )
    parameters = pd.DataFrame(parameter_rows)
    reliability = pd.DataFrame(reliability_rows)
    _write_table(parameters, out_dir / "dev_calibration_parameters.csv")
    _write_table(reliability, out_dir / "reliability_bins.csv")
    _write_calibration_figure(reliability, out_dir / "calibration_by_foundation.png")

    method_table = evaluate_methods(test, methods)
    _write_table(method_table, out_dir / "heldout_method_comparison.csv")
    bootstrap = bootstrap_paired_differences(
        test,
        calibrated_test,
        {
            "raw": methods["raw"],
            "hard": methods["hard"],
            "constant_dev_prevalence": methods["constant_dev_prevalence"],
        },
        reps=int(config["analysis"]["bootstrap_reps"]),
        seed=int(config["seed"]),
    )
    _write_table(bootstrap, out_dir / "calibration_paired_bootstrap.csv")

    disagreement = _disagreement_table(test)
    _write_table(disagreement, out_dir / "entropy_disagreement.csv")

    budgets = [float(value) for value in config["analysis"]["review_budgets"]]
    review = selective_review_table(
        test,
        budgets,
        int(config["analysis"]["random_review_reps"]),
        int(config["seed"]),
    )
    _write_table(review, out_dir / "selective_review.csv")

    canonical_sensitivity = test.loc[test["sensitivity"].astype(bool)].copy()
    strict = strict.copy()
    # The source rows are already joined to the fixed sensitivity subset; preserve a stable paired order.
    strict_summary, strict_bootstrap = strict_sensitivity_tables(
        canonical_sensitivity,
        strict,
        reps=int(config["analysis"]["bootstrap_reps"]),
        seed=int(config["seed"]),
    )
    _write_table(strict_summary, out_dir / "strict_sensitivity.csv")
    _write_table(strict_bootstrap, out_dir / "strict_sensitivity_bootstrap.csv")

    _write_audit(
        out_dir / "POSTHOC_AUDIT.md",
        marginal,
        method_table,
        bootstrap,
        disagreement,
        review,
        strict_summary,
        strict_bootstrap,
        parameters,
    )
    print(f"Wrote exploratory post-hoc diagnostics to {out_dir}")


if __name__ == "__main__":
    run()
