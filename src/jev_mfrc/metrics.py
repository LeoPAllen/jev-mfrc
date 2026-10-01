from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import FOUNDATIONS
from .config import path
from .data import _sha256_file, experiment_fingerprint, prepared_data_provenance
from .prompts import instrument_bundle_hash, prompt_hash

EPS = 1e-12


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def binary_entropy(values):
    x = np.asarray(values, dtype=float)
    if np.any((x < 0) | (x > 1) | ~np.isfinite(x)):
        raise ValueError("Entropy input must be finite probabilities in [0,1]")
    z = np.clip(x, EPS, 1 - EPS)
    out = -(z * np.log(z) + (1 - z) * np.log(1 - z))
    return np.where((x == 0) | (x == 1), 0.0, out)


def squared_loss_to_share(pred, human_share):
    p = np.asarray(pred, dtype=float)
    h = np.asarray(human_share, dtype=float)
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)) or np.any((h < 0) | (h > 1) | ~np.isfinite(h)):
        raise ValueError("Squared-loss inputs must be finite probabilities/shares in [0,1]")
    return (p - h) ** 2


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _foundation_metrics(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for f in FOUNDATIONS:
        h = df[f"human_{f}"].to_numpy(float)
        p = df[f"p_{f}"].to_numpy(float)
        hard = (p >= 0.5).astype(float)
        soft_sq = squared_loss_to_share(p, h)
        hard_sq = squared_loss_to_share(hard, h)
        soft_abs = np.abs(p - h)
        hard_abs = np.abs(hard - h)
        rows.append({
            "variant": variant,
            "foundation": f,
            "n_comments": len(df),
            "entropy_spearman": _spearman(binary_entropy(h), binary_entropy(p)),
            "squared_loss_probability": float(soft_sq.mean()),
            "squared_loss_hard": float(hard_sq.mean()),
            "hard_minus_probability_squared_loss": float((hard_sq - soft_sq).mean()),
            "mae_probability_to_human_share": float(soft_abs.mean()),
            "mae_hard_to_human_share": float(hard_abs.mean()),
            "hard_minus_probability_absolute_loss": float((hard_abs - soft_abs).mean()),
        })
    return pd.DataFrame(rows)


def _loss_deltas(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    sq: dict[str, float] = {}
    ab: dict[str, float] = {}
    for f in FOUNDATIONS:
        h = df[f"human_{f}"].to_numpy(float)
        p = df[f"p_{f}"].to_numpy(float)
        hard = (p >= 0.5).astype(float)
        sq[f] = float((squared_loss_to_share(hard, h) - squared_loss_to_share(p, h)).mean())
        ab[f] = float((np.abs(hard - h) - np.abs(p - h)).mean())
    return sq, ab


def _bootstrap_loss_deltas(df: pd.DataFrame, reps: int, seed: int) -> dict:
    point_sq, point_abs = _loss_deltas(df)
    n = len(df)
    rng = np.random.default_rng(seed)
    boot_sq = {f: [] for f in FOUNDATIONS}
    boot_abs = {f: [] for f in FOUNDATIONS}
    macro_sq, macro_abs = [], []
    for _ in range(reps):
        sample = df.iloc[rng.integers(0, n, n)]
        sq, ab = _loss_deltas(sample)
        for f in FOUNDATIONS:
            boot_sq[f].append(sq[f])
            boot_abs[f].append(ab[f])
        macro_sq.append(float(np.mean(list(sq.values()))))
        macro_abs.append(float(np.mean(list(ab.values()))))

    def ci(values):
        a = np.asarray(values, dtype=float)
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]

    return {
        "squared": {
            "macro": {"estimate": float(np.mean(list(point_sq.values()))), "ci95": ci(macro_sq)},
            "foundation": {f: {"estimate": point_sq[f], "ci95": ci(boot_sq[f])} for f in FOUNDATIONS},
        },
        "absolute": {
            "macro": {"estimate": float(np.mean(list(point_abs.values()))), "ci95": ci(macro_abs)},
            "foundation": {f: {"estimate": point_abs[f], "ci95": ci(boot_abs[f])} for f in FOUNDATIONS},
        },
        "reps": int(reps),
    }


def _bootstrap_macro_delta(df: pd.DataFrame, reps: int, seed: int) -> dict:
    """Backward-compatible helper used by tests; returns squared-loss macro delta."""
    out = _bootstrap_loss_deltas(df, reps, seed)["squared"]["macro"]
    return {"estimate": out["estimate"], "ci95_low": out["ci95"][0], "ci95_high": out["ci95"][1], "reps": int(reps)}


def _uncertainty_bootstrap(df: pd.DataFrame, reps: int, seed: int) -> dict:
    by_f = {
        f: _spearman(binary_entropy(df[f"human_{f}"]), binary_entropy(df[f"p_{f}"]))
        for f in FOUNDATIONS
    }
    estimate = float(np.nanmean(list(by_f.values())))
    rng = np.random.default_rng(seed + 1)
    boot = {f: [] for f in FOUNDATIONS}
    n = len(df)
    for _ in range(reps):
        sample = df.iloc[rng.integers(0, n, n)]
        for f in FOUNDATIONS:
            boot[f].append(_spearman(binary_entropy(sample[f"human_{f}"]), binary_entropy(sample[f"p_{f}"])))

    def ci(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return [float("nan"), float("nan")]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    macro = [float(np.nanmean([boot[f][i] for f in FOUNDATIONS])) for i in range(reps)]
    macro_finite = np.asarray(macro, dtype=float)
    macro_finite = macro_finite[np.isfinite(macro_finite)]
    macro_ci = ci(macro_finite)
    return {
        "foundation_spearman": by_f,
        "foundation_ci95": {f: ci(boot[f]) for f in FOUNDATIONS},
        "macro_mean": estimate,
        "ci95_low": macro_ci[0],
        "ci95_high": macro_ci[1],
        "reps": int(reps),
    }


def _review_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = df[[f"p_{f}" for f in FOUNDATIONS]].to_numpy(float)
    uncertainty = binary_entropy(p).mean(axis=1)
    return p, np.argsort(-uncertainty)


def _selective_review(df: pd.DataFrame, budgets: list[float], random_reps: int, seed: int) -> pd.DataFrame:
    pcols = [f"p_{f}" for f in FOUNDATIONS]
    hcols = [f"human_{f}" for f in FOUNDATIONS]
    p = df[pcols].to_numpy(float)
    h = df[hcols].to_numpy(float)
    _, order = _review_indices(df)
    rng = np.random.default_rng(seed + 2)
    rows = []
    for budget in budgets:
        k = int(round(len(df) * float(budget)))
        target_p = p.copy()
        if k:
            target_p[order[:k], :] = h[order[:k], :]
        target_mse = float(np.mean((target_p - h) ** 2))
        random_mse = []
        for _ in range(random_reps):
            idx = rng.choice(len(df), size=k, replace=False) if k else np.array([], dtype=int)
            rp = p.copy()
            if k:
                rp[idx, :] = h[idx, :]
            random_mse.append(float(np.mean((rp - h) ** 2)))
        mse_arr = np.asarray(random_mse)
        rows.append({
            "budget": float(budget),
            "comments_reviewed": k,
            "targeted_mse": target_mse,
            "random_mse_mean": float(mse_arr.mean()),
            "random_mse_ci95_low": float(np.quantile(mse_arr, .025)),
            "random_mse_ci95_high": float(np.quantile(mse_arr, .975)),
            "targeted_minus_random_mse": float(target_mse - mse_arr.mean()),
        })
    return pd.DataFrame(rows)


def _cell_table(df: pd.DataFrame, estimates: np.ndarray, label: str) -> pd.DataFrame:
    work = df[["bucket", "subreddit"]].copy()
    for j, f in enumerate(FOUNDATIONS):
        work[f"est_{f}"] = estimates[:, j]
        work[f"human_{f}"] = df[f"human_{f}"].to_numpy(float)
    rows = []
    for (bucket, subreddit), g in work.groupby(["bucket", "subreddit"], sort=True):
        for f in FOUNDATIONS:
            rows.append({
                "bucket": bucket,
                "subreddit": subreddit,
                "foundation": f,
                "n": len(g),
                "human_share": float(g[f"human_{f}"].mean()),
                label: float(g[f"est_{f}"].mean()),
            })
    return pd.DataFrame(rows)


def _cell_loss(cells: pd.DataFrame, estimate_col: str) -> dict:
    err = np.abs(cells[estimate_col] - cells["human_share"])
    weights = cells["n"].to_numpy(float)
    return {
        "mae_unweighted": float(err.mean()),
        "mae_cell_size_weighted": float(np.average(err, weights=weights)),
    }


def _cell_recovery(df: pd.DataFrame, budgets: list[float], random_reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = df[[f"p_{f}" for f in FOUNDATIONS]].to_numpy(float)
    h = df[[f"human_{f}" for f in FOUNDATIONS]].to_numpy(float)
    hard = (p >= 0.5).astype(float)
    _, order = _review_indices(df)

    keys = ["bucket", "subreddit", "foundation"]
    base = _cell_table(df, h, "reference")[keys + ["n", "human_share"]]
    soft = _cell_table(df, p, "soft")[keys + ["soft"]]
    hard_cells = _cell_table(df, hard, "hard")[keys + ["hard"]]
    cells = base.merge(soft, on=keys).merge(hard_cells, on=keys)
    summaries = [
        {"method": "soft", "budget": 0.0, **_cell_loss(cells, "soft")},
        {"method": "hard", "budget": 0.0, **_cell_loss(cells, "hard")},
    ]

    rng = np.random.default_rng(seed + 3)
    for budget in budgets:
        if float(budget) == 0.0:
            continue
        k = int(round(len(df) * float(budget)))
        targeted = p.copy(); targeted[order[:k], :] = h[order[:k], :]
        label = f"targeted_{int(round(100*budget))}pct"
        tc = _cell_table(df, targeted, label)[keys + [label]]
        cells = cells.merge(tc, on=keys)
        summaries.append({"method": "targeted", "budget": float(budget), **_cell_loss(cells, label)})

        random_cell_estimates = []
        random_losses = []
        for _ in range(random_reps):
            idx = rng.choice(len(df), size=k, replace=False)
            rp = p.copy(); rp[idx, :] = h[idx, :]
            rc = _cell_table(df, rp, "random")
            random_cell_estimates.append(rc["random"].to_numpy(float))
            temp = base.copy(); temp["random"] = rc["random"].to_numpy(float)
            random_losses.append(_cell_loss(temp, "random"))
        arr = np.vstack(random_cell_estimates)
        rlabel = f"random_mean_{int(round(100*budget))}pct"
        cells[rlabel] = arr.mean(axis=0)
        summaries.append({
            "method": "random_mean",
            "budget": float(budget),
            "mae_unweighted": float(np.mean([x["mae_unweighted"] for x in random_losses])),
            "mae_cell_size_weighted": float(np.mean([x["mae_cell_size_weighted"] for x in random_losses])),
        })
    return cells, pd.DataFrame(summaries)


def _sensitivity(canonical: pd.DataFrame, strict: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = ["item_id"] + [f"human_{f}" for f in FOUNDATIONS]
    c = canonical[keys + [f"p_{f}" for f in FOUNDATIONS]].copy()
    s = strict[["item_id"] + [f"p_{f}" for f in FOUNDATIONS]].copy()
    s = s.rename(columns={f"p_{f}": f"strict_{f}" for f in FOUNDATIONS})
    x = c.merge(s, on="item_id", how="inner", validate="one_to_one")
    rows = []
    for f in FOUNDATIONS:
        h = x[f"human_{f}"].to_numpy(float)
        p = x[f"p_{f}"].to_numpy(float)
        q = x[f"strict_{f}"].to_numpy(float)
        can_loss = float(squared_loss_to_share(p, h).mean())
        strict_loss = float(squared_loss_to_share(q, h).mean())
        can_unc = _spearman(binary_entropy(h), binary_entropy(p))
        strict_unc = _spearman(binary_entropy(h), binary_entropy(q))
        rows.append({
            "foundation": f,
            "n": len(x),
            "mean_absolute_probability_change": float(np.abs(q - p).mean()),
            "hard_label_flip_rate": float(np.mean((p >= .5) != (q >= .5))),
            "canonical_strict_spearman": _spearman(p, q),
            "canonical_squared_loss": can_loss,
            "strict_squared_loss": strict_loss,
            "strict_minus_canonical_squared_loss": strict_loss - can_loss,
            "canonical_uncertainty_spearman": can_unc,
            "strict_uncertainty_spearman": strict_unc,
            "strict_minus_canonical_uncertainty_spearman": strict_unc - can_unc,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "mean_absolute_probability_change_macro": float(frame["mean_absolute_probability_change"].mean()),
        "hard_label_flip_rate_macro": float(frame["hard_label_flip_rate"].mean()),
        "canonical_strict_spearman_macro": float(frame["canonical_strict_spearman"].mean()),
        "strict_minus_canonical_squared_loss_macro": float(frame["strict_minus_canonical_squared_loss"].mean()),
        "strict_minus_canonical_uncertainty_spearman_macro": float(frame["strict_minus_canonical_uncertainty_spearman"].mean()),
    }
    return frame, summary


def analyze(cfg: dict) -> dict:
    items = pd.read_csv(path("data", "processed", "items.csv.gz"))
    pred = pd.read_csv(path("data", "processed", "jev_predictions.csv.gz"))
    test = items.loc[items["split"] == "test"].copy()

    canonical = pred.loc[pred["variant"] == "canonical"].merge(test, on="item_id", how="inner", validate="one_to_one")
    if len(canonical) != len(test):
        raise RuntimeError(f"Canonical held-out predictions incomplete: {len(canonical)} of {len(test)}")
    strict_expected = test.loc[test["sensitivity"]].copy()
    strict = pred.loc[pred["variant"] == "strict"].merge(strict_expected, on="item_id", how="inner", validate="one_to_one")
    if len(strict) != len(strict_expected):
        raise RuntimeError(f"Strict sensitivity predictions incomplete: {len(strict)} of {len(strict_expected)}")

    for frame in (canonical, strict):
        pcols = [f"p_{f}" for f in FOUNDATIONS]
        if frame[pcols].isna().any().any():
            raise RuntimeError("Missing prediction probabilities in analysis frame")

    tables = path("results", "tables"); figs = path("results", "figures")
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    # Strict wording is summarized only as the prespecified sensitivity check below.
    fm = _foundation_metrics(canonical, "canonical")
    fm.to_csv(tables / "foundation_metrics.csv", index=False)

    review = _selective_review(canonical, cfg["analysis"]["review_budgets"], int(cfg["analysis"]["random_review_reps"]), int(cfg["seed"]))
    review.to_csv(tables / "selective_review.csv", index=False)
    cells, cell_summary = _cell_recovery(canonical, cfg["analysis"]["review_budgets"], int(cfg["analysis"]["random_review_reps"]), int(cfg["seed"]))
    cells.to_csv(tables / "source_cell_recovery.csv", index=False)
    cell_summary.to_csv(tables / "source_cell_recovery_summary.csv", index=False)

    strict_canonical = canonical.loc[canonical["sensitivity"]].copy()
    sens_table, sens_summary = _sensitivity(strict_canonical, strict)
    sens_table.to_csv(tables / "wording_sensitivity.csv", index=False)

    jev_item_uncertainty = np.column_stack([binary_entropy(canonical[f"p_{f}"].to_numpy(float)) for f in FOUNDATIONS]).mean(axis=1)
    reverse_confidence = 1.0 - canonical["human_confidence_mean"].to_numpy(float)
    confidence_corr = _spearman(jev_item_uncertainty, reverse_confidence)
    loss_deltas = _bootstrap_loss_deltas(canonical, int(cfg["analysis"]["bootstrap_reps"]), int(cfg["seed"]))
    uncertainty = _uncertainty_bootstrap(canonical, int(cfg["analysis"]["bootstrap_reps"]), int(cfg["seed"]))

    plt.figure(figsize=(6, 4))
    plt.plot(review["budget"] * 100, review["targeted_mse"], marker="o", label="Uncertainty-targeted")
    plt.plot(review["budget"] * 100, review["random_mse_mean"], marker="o", label="Random")
    plt.xlabel("Comments reviewed (%)"); plt.ylabel("Mean squared loss to human vote share"); plt.legend(); plt.tight_layout()
    plt.savefig(figs / "selective_review.png", dpi=160); plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(cells["human_share"], cells["soft"], alpha=.7, label="Probability")
    plt.scatter(cells["human_share"], cells["hard"], alpha=.7, label="Hard label")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Human-reference sample cell mean"); plt.ylabel("JEV-estimated sample cell mean"); plt.legend(); plt.tight_layout()
    plt.savefig(figs / "cell_recovery.png", dpi=160); plt.close()

    soft_row = cell_summary.loc[cell_summary["method"] == "soft"].iloc[0]
    hard_row = cell_summary.loc[cell_summary["method"] == "hard"].iloc[0]
    returned_models = sorted(pred["returned_model"].dropna().astype(str).unique().tolist()) if "returned_model" in pred else []
    requested_models = sorted(pred["requested_model"].dropna().astype(str).unique().tolist()) if "requested_model" in pred else []
    summary = {
        "provenance": {
            "experiment_fingerprint": experiment_fingerprint(cfg),
            "data": prepared_data_provenance(cfg),
            "instrument_bundle_hash": instrument_bundle_hash(),
            "prompt_hashes": {"canonical": prompt_hash("canonical"), "strict": prompt_hash("strict")},
            "provider_base_url": cfg["jev"]["base_url"].rstrip("/"),
            "configured_jev_model": cfg["jev"]["model"],
            "requested_models_in_export": requested_models,
            "returned_models_in_export": returned_models,
            "seed": int(cfg["seed"]),
            "analysis_config": cfg["analysis"],
            "analysis_code_sha256": _sha256_file(Path(__file__)),
            "heldout_export_sha256": _sha256_file(path("data", "processed", "jev_predictions.csv.gz")),
        },
        "test_items": int(len(canonical)),
        "sensitivity_items": int(len(strict)),
        "foundations": FOUNDATIONS,
        "uncertainty_validity": uncertainty,
        "loss_information_retention": loss_deltas,
        "jev_uncertainty_vs_reverse_coder_confidence_spearman": confidence_corr,
        "source_sample_cell_recovery": {
            "soft_mae": float(soft_row["mae_unweighted"]),
            "hard_mae": float(hard_row["mae_unweighted"]),
            "soft_weighted_mae": float(soft_row["mae_cell_size_weighted"]),
            "hard_weighted_mae": float(hard_row["mae_cell_size_weighted"]),
        },
        "wording_sensitivity": sens_summary,
    }
    safe_summary = _json_safe(summary)
    path("results", "analysis.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return safe_summary
