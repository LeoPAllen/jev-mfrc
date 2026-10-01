from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import FOUNDATIONS
from .config import path


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _select_dev_review_cases(frame: pd.DataFrame, foundation: str) -> dict[str, pd.DataFrame]:
    """Select canonical development examples in stable, non-overlapping sections."""
    p_col = f"p_{foundation}"
    h_col = f"human_{foundation}"
    work = frame.copy()
    work["_item_sort"] = work["item_id"].astype(str)
    work["_confidence"] = np.abs(work[p_col].astype(float) - 0.5)
    work["_human_mixed_distance"] = np.abs(work[h_col].astype(float) - 0.5)
    used: set[str] = set()

    def take(ordered: pd.DataFrame, count: int) -> pd.DataFrame:
        available = ordered.loc[~ordered["item_id"].astype(str).isin(used)].head(count).copy()
        used.update(available["item_id"].astype(str))
        return available

    uncertainty = work.sort_values(
        ["_confidence", "_item_sort"], ascending=[True, True], kind="mergesort"
    )
    positive_disagreement = work.loc[(work[p_col] > 0.5) & (work[h_col] < 0.5)].sort_values(
        ["_confidence", "_item_sort"], ascending=[False, True], kind="mergesort"
    )
    negative_disagreement = work.loc[(work[p_col] < 0.5) & (work[h_col] > 0.5)].sort_values(
        ["_confidence", "_item_sort"], ascending=[False, True], kind="mergesort"
    )

    low_reference = work.loc[work[h_col] < 0.5].sort_values(
        [h_col, "_item_sort"], ascending=[True, True], kind="mergesort"
    )
    mixed_reference = work.loc[(work[h_col] > 0.0) & (work[h_col] < 1.0)].sort_values(
        ["_human_mixed_distance", "_item_sort"], ascending=[True, True], kind="mergesort"
    )
    high_reference = work.loc[work[h_col] > 0.5].sort_values(
        [h_col, "_item_sort"], ascending=[False, True], kind="mergesort"
    )

    return {
        "Closest to JEV p = 0.5": take(uncertainty, 3),
        "Confident JEV-positive / human-negative-majority disagreements": take(positive_disagreement, 3),
        "Confident JEV-negative / human-positive-majority disagreements": take(negative_disagreement, 3),
        "Deterministic reference cases — low human share": take(low_reference, 1),
        "Deterministic reference cases — mixed human share": take(mixed_reference, 1),
        "Deterministic reference cases — high human share": take(high_reference, 1),
    }


def write_dev_review() -> None:
    items = pd.read_csv(path("data", "processed", "items.csv.gz"))
    pred_path = path("data", "processed", "dev_predictions.csv.gz")
    if not pred_path.exists():
        return
    pred = pd.read_csv(pred_path)
    dev = items.loc[items["split"] == "dev"].copy()
    lines = [
        "# Development-only instrument review",
        "",
        "Purpose: semantic validation of the canonical instrument only. The strict wording is prespecified for held-out sensitivity and is not compared on development performance.",
        "JEV values are provider-documented `noul` probabilities of Yes/True to each bounded foundation-presence question, not moral-intensity scores. Human shares are the observed trained-rater reference distribution, not truth.",
        "Majority-disagreement sections use strict human-share cutoffs: above 0.5 is a positive majority, below 0.5 is a negative majority, and exactly 0.5 is a tie excluded from both.",
        "Review whether each question operationalizes the MFRC construct defensibly and inspect obvious failure modes.",
        "",
    ]
    frame = pred.merge(dev, on="item_id", how="inner")
    if set(frame["variant"].unique()) - {"canonical"}:
        raise RuntimeError("Development review unexpectedly contains a non-canonical prompt variant")
    for f in FOUNDATIONS:
        sections = _select_dev_review_cases(frame, f)
        lines += [f"## {f.title()}", ""]
        for label, ex in sections.items():
            lines.append(f"**{label}**")
            if ex.empty:
                lines.append("- None available in the development sample.")
            else:
                for r in ex.itertuples(index=False):
                    item_text = " ".join(str(r.text).splitlines())
                    lines.append(
                        f"- item_id: `{r.item_id}`; JEV p={getattr(r, 'p_'+f):.3f}; "
                        f"human share={getattr(r, 'human_'+f):.3f}; "
                        f"retained annotators={int(r.n_annotators)}; text: {item_text}"
                    )
            lines.append("")
    out = path("results", "dev_review.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary() -> None:
    payload = json.loads(path("results", "analysis.json").read_text(encoding="utf-8"))
    lines = [
        "# Analysis summary",
        "",
        "Generated from prespecified held-out analyses. Interpret human shares as trained-rater reference distributions, not truth.",
        "",
        f"- Held-out comments: {payload['test_items']}",
        f"- Sensitivity comments: {payload['sensitivity_items']}",
        f"- Uncertainty-validity macro Spearman: {_fmt(payload['uncertainty_validity']['macro_mean'])} "
        f"[{_fmt(payload['uncertainty_validity']['ci95_low'])}, {_fmt(payload['uncertainty_validity']['ci95_high'])}]",
        f"- Hard minus probabilistic squared loss to human vote share: {_fmt(payload['loss_information_retention']['squared']['macro']['estimate'], 6)} "
        f"[{_fmt(payload['loss_information_retention']['squared']['macro']['ci95'][0], 6)}, {_fmt(payload['loss_information_retention']['squared']['macro']['ci95'][1], 6)}]",
        f"- Soft source-cell MAE: {_fmt(payload['source_sample_cell_recovery']['soft_mae'], 6)}",
        f"- Hard source-cell MAE: {_fmt(payload['source_sample_cell_recovery']['hard_mae'], 6)}",
        "",
        "**Unequal-cell-size robustness diagnostic (cell-size-weighted source-cell MAE; unweighted MAE remains primary):**",
        f"- Soft: {_fmt(payload['source_sample_cell_recovery']['soft_weighted_mae'], 6)}",
        f"- Hard: {_fmt(payload['source_sample_cell_recovery']['hard_weighted_mae'], 6)}",
        "",
        "The squared loss is to the human vote share, not itself a Brier score. The hard-minus-probability squared-loss difference equals the same difference averaged over retained individual binary coder judgments because the within-comment coder-variance term cancels.",
        "",
        "Selective-review results are an oracle / idealized reference-replacement simulation: reviewed comments receive their human vote shares. They do not estimate labor time or minutes saved.",
        "",
        "Source-cell results describe recovery of observed MFRC sample-cell means in this selected sample. They do not estimate population prevalence or causal effects.",
        "",
        "Canonical wording remains primary; strict wording is reported only as the prespecified held-out sensitivity check.",
        "",
        "See results/tables/ for foundation-specific diagnostics, absolute-loss secondary results, and selective-review results.",
    ]
    pairwise = payload.get("robustness", {}).get("pairwise_coder_disagreement")
    if pairwise:
        lines += [
            "",
            "## Secondary robustness",
            "",
            "Pairwise coder disagreement is the coder-count-adjusted robustness operationalization `2*n*h*(1-h)/(n-1)`. It is not a new target; coder disagreement may reflect ambiguity, perspective, coder noise, missing context, or task limits.",
            "",
            "| Foundation | Spearman rho |",
            "|---|---:|",
        ]
        for foundation in payload["foundations"]:
            lines.append(f"| {foundation.title()} | {_fmt(pairwise['foundation_spearman'][foundation])} |")
        lines += [
            f"| Unweighted macro mean | {_fmt(pairwise['macro_mean'])} |",
        ]
    bootstrap_units = payload.get("bootstrap_units")
    if bootstrap_units:
        lines += [
            "",
            f"Primary bootstrap unit: {bootstrap_units['primary']}. Exact-text content-cluster unit: {bootstrap_units['content_cluster_robustness']}.",
        ]
        if bootstrap_units["row_and_content_cluster_units_coincide"]:
            lines.append("Every held-out row has a unique content_id, so row and exact-text content-cluster units coincide and no separate content-cluster bootstrap was run.")
    out = path("results", "summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
