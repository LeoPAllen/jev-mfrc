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
        "Review whether each question operationalizes the MFRC construct defensibly and inspect obvious failure modes.",
        "",
    ]
    frame = pred.merge(dev, on="item_id", how="inner")
    if set(frame["variant"].unique()) - {"canonical"}:
        raise RuntimeError("Development review unexpectedly contains a non-canonical prompt variant")
    for f in FOUNDATIONS:
        frame["uncertainty_distance"] = np.abs(frame[f"p_{f}"] - 0.5)
        uncertain = frame.sort_values(["uncertainty_distance", "item_id"], kind="mergesort").head(2)
        disagreement = frame.loc[(frame[f"p_{f}"] >= .5) != (frame[f"human_{f}"] >= .5)].copy()
        if not disagreement.empty:
            disagreement["certainty"] = np.abs(disagreement[f"p_{f}"] - .5)
            disagreement = disagreement.sort_values(["certainty", "item_id"], ascending=[False, True], kind="mergesort").head(2)
        lines += [f"## {f.title()}", ""]
        for label, ex in (("Most uncertain", uncertain), ("Confident human-majority disagreements", disagreement)):
            lines.append(f"**{label}**")
            if ex.empty:
                lines.append("- None in development sample.")
            else:
                for r in ex.itertuples(index=False):
                    text = str(r.text).replace("\n", " ")
                    lines.append(f"- JEV={getattr(r, 'p_'+f):.3f}; human share={getattr(r, 'human_'+f):.3f}; text: {text}")
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
        "See results/tables/ for foundation-specific diagnostics and selective-review results.",
    ]
    out = path("results", "summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
