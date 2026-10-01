from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from . import FOUNDATIONS
from .config import path

EXPECTED_COLUMNS = {"text", "subreddit", "bucket", "annotator", "annotation", "confidence"}
CONFIDENCE = {"Not Confident": 0.0, "Somewhat Confident": 0.5, "Confident": 1.0}
ITEM_IDENTITY_COLUMNS = ["bucket", "subreddit", "text"]
ANNOTATION_RECORD_COLUMNS = [*ITEM_IDENTITY_COLUMNS, "annotator"]
PRIMARY_MIN_ANNOTATORS = 3
DATA_PIPELINE_VERSION = 7


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_config_fingerprint(cfg: dict) -> str:
    """Fingerprint scientific data/split choices, not implementation source code."""
    payload = {
        "foundations": list(FOUNDATIONS),
        "confidence_mapping": CONFIDENCE,
        "item_identity_columns": ITEM_IDENTITY_COLUMNS,
        "annotation_record_columns": ANNOTATION_RECORD_COLUMNS,
        "dataset": {
            "repo_id": cfg["dataset"]["repo_id"],
            "revision": cfg["dataset"]["revision"],
            "split": cfg["dataset"]["split"],
            "min_annotators": int(cfg["dataset"]["min_annotators"]),
        },
        "sampling": {
            "seed": int(cfg["seed"]),
            "dev_items_target": int(cfg["sampling"]["dev_items"]),
            "sensitivity_items_target": int(cfg["sampling"]["sensitivity_items"]),
            "exact_text_grouped_across_splits": True,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _labels(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _stable_item_id(bucket: str, subreddit: str, text: str) -> str:
    # JSON array encoding preserves tuple boundaries even if a value contains a NUL.
    raw = json.dumps([bucket, subreddit, text], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rank(value: str, seed: int, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}:{seed}:{value}".encode("utf-8")).hexdigest()


def _expected_source(cfg: dict) -> dict:
    dcfg = cfg["dataset"]
    return {"repo_id": dcfg["repo_id"], "requested_revision": dcfg["revision"], "split": dcfg["split"]}


def _verify_raw_snapshot(raw_path: Path, meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = meta.get("sha256")
    if not expected:
        raise RuntimeError("Raw metadata predates checksum validation; delete both raw snapshot files and re-download deliberately.")
    if not meta.get("resolved_revision"):
        raise RuntimeError("Raw metadata is missing the resolved dataset revision; restore or deliberately re-download both raw files.")
    actual = _sha256_file(raw_path)
    if actual != expected:
        raise RuntimeError("Raw MFRC snapshot checksum changed. Raw data are immutable; restore or deliberately re-download both raw files.")
    return meta


def download_raw(cfg: dict) -> tuple[Path, Path]:
    raw_path = path("data", "raw", "mfrc.csv.gz")
    meta_path = path("data", "raw", "mfrc_meta.json")
    if raw_path.exists() != meta_path.exists():
        raise RuntimeError("Raw snapshot is incomplete: expected both mfrc.csv.gz and mfrc_meta.json")
    if raw_path.exists():
        meta = _verify_raw_snapshot(raw_path, meta_path)
        expected_source = _expected_source(cfg)
        observed_source = {k: meta.get(k) for k in expected_source}
        if observed_source != expected_source:
            raise RuntimeError(f"Cached raw snapshot provenance does not match config: {observed_source} != {expected_source}")
        return raw_path, meta_path

    from datasets import load_dataset
    from huggingface_hub import HfApi

    dcfg = cfg["dataset"]
    info = HfApi().dataset_info(dcfg["repo_id"], revision=dcfg["revision"])
    resolved = info.sha
    ds = load_dataset(dcfg["repo_id"], revision=resolved, split=dcfg["split"])
    df = ds.to_pandas()
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"MFRC missing expected columns: {sorted(missing)}")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False, compression={"method": "gzip", "mtime": 0})
    meta = {
        "repo_id": dcfg["repo_id"],
        "requested_revision": dcfg["revision"],
        "resolved_revision": resolved,
        "split": dcfg["split"],
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sha256": _sha256_file(raw_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return raw_path, meta_path


def prepared_data_provenance(cfg: dict) -> dict:
    raw_path = path("data", "raw", "mfrc.csv.gz")
    meta_path = path("data", "raw", "mfrc_meta.json")
    items_path = path("data", "processed", "items.csv.gz")
    audit_path = path("data", "processed", "audit.json")
    required = [raw_path, meta_path, items_path, audit_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Prepared-data provenance is incomplete; missing: {missing}")

    meta = _verify_raw_snapshot(raw_path, meta_path)
    expected_source = _expected_source(cfg)
    observed_source = {k: meta.get(k) for k in expected_source}
    if observed_source != expected_source:
        raise RuntimeError(f"Raw snapshot provenance does not match config: {observed_source} != {expected_source}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_sha = _sha256_file(raw_path)
    config_fp = _data_config_fingerprint(cfg)
    items_sha = _sha256_file(items_path)
    if audit.get("raw_sha256") != raw_sha or audit.get("data_config_fingerprint") != config_fp:
        raise RuntimeError("Processed MFRC data do not match the current raw snapshot or scientific split/inclusion configuration; rebuild them.")
    if audit.get("processed_items_sha256") != items_sha:
        raise RuntimeError("Processed MFRC items no longer match their recorded content hash; rebuild before continuing.")

    return {
        "repo_id": meta.get("repo_id"),
        "requested_revision": meta.get("requested_revision"),
        "resolved_revision": meta.get("resolved_revision"),
        "split": meta.get("split"),
        "raw_sha256": raw_sha,
        "data_config_fingerprint": config_fp,
        "processed_items_sha256": items_sha,
    }


def experiment_fingerprint(cfg: dict) -> str:
    provenance = prepared_data_provenance(cfg)
    return hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()


def _choose_grouped_ids(frame: pd.DataFrame, target: int, seed: int, namespace: str) -> set[str]:
    """Choose approximately target items while keeping identical text together."""
    if target < 1 or target > len(frame):
        raise ValueError(f"{namespace}_items must be between 1 and {len(frame)}")
    groups = (
        frame.groupby("text", as_index=False)["item_id"]
        .agg(list)
        .assign(rank=lambda x: x["text"].map(lambda z: _rank(z, seed, namespace)))
        .sort_values(["rank", "text"], kind="mergesort")
    )
    chosen: set[str] = set()
    for ids in groups["item_id"]:
        if chosen and len(chosen) >= target:
            break
        chosen.update(ids)
    return chosen


def prepare_items(cfg: dict, *, force: bool = False) -> Path:
    """Rebuild the small derived table deterministically from the immutable raw snapshot.

    `force` is retained for backward compatibility but derived data are intentionally
    cheap enough to rebuild every call. This removes stale-cache state from the study.
    """
    del force
    out = path("data", "processed", "items.csv.gz")
    audit_path = path("data", "processed", "audit.json")
    excluded_path = path("data", "processed", "excluded_lt_min_annotators.csv.gz")
    raw_path = path("data", "raw", "mfrc.csv.gz")
    meta_path = path("data", "raw", "mfrc_meta.json")
    if not raw_path.exists() or not meta_path.exists():
        raise RuntimeError("Raw MFRC snapshot missing; run the data stage first")
    meta = _verify_raw_snapshot(raw_path, meta_path)
    expected_source = _expected_source(cfg)
    observed_source = {k: meta.get(k) for k in expected_source}
    if observed_source != expected_source:
        raise RuntimeError(f"Raw snapshot provenance does not match config: {observed_source} != {expected_source}")

    min_annotators = int(cfg["dataset"]["min_annotators"])
    if min_annotators < PRIMARY_MIN_ANNOTATORS:
        raise ValueError(
            f"dataset.min_annotators must be at least {PRIMARY_MIN_ANNOTATORS} for the prespecified MFRC study"
        )

    raw_sha = _sha256_file(raw_path)
    config_fp = _data_config_fingerprint(cfg)
    # Preserve literal strings (including values such as "NA") exactly as released.
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    missing = EXPECTED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Raw MFRC missing expected columns: {sorted(missing)}")
    null_identity = raw[ANNOTATION_RECORD_COLUMNS].isna().any(axis=1)
    if null_identity.any():
        raise ValueError(f"MFRC has {int(null_identity.sum())} rows with missing item identity/annotator fields")
    blank_identity = raw[ANNOTATION_RECORD_COLUMNS].apply(lambda col: col.astype(str).str.strip().eq("")).any(axis=1)
    if blank_identity.any():
        raise ValueError(f"MFRC has {int(blank_identity.sum())} rows with blank item identity/annotator fields")
    if raw["annotation"].eq("").any():
        raise ValueError("MFRC contains missing annotation values; do not guess how to code them")

    original_rows = len(raw)
    exact_duplicate_rows = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()
    raw["item_id"] = [
        _stable_item_id(b, s, t)
        for b, s, t in zip(raw["bucket"], raw["subreddit"], raw["text"])
    ]

    dup = raw.duplicated(["item_id", "annotator"], keep=False)
    if dup.any():
        bad = raw.loc[dup, ["item_id", "annotator", "annotation", "confidence"]].sort_values(["item_id", "annotator"])
        raise ValueError(
            "Conflicting or repeated item/annotator records remain after exact deduplication:\n"
            + bad.head(12).to_string(index=False)
        )

    observed_conf = set(raw["confidence"].astype(str).unique()) - {""}
    unknown_conf = observed_conf - set(CONFIDENCE)
    if unknown_conf:
        raise ValueError(f"Unknown confidence values: {sorted(unknown_conf)}")
    raw["confidence_num"] = raw["confidence"].map(CONFIDENCE)

    atomic_labels = sorted({label for value in raw["annotation"] for label in _labels(value)})
    if not atomic_labels:
        raise ValueError("No annotation labels found")
    lower_lookup: dict[str, set[str]] = {}
    for token in atomic_labels:
        lower_lookup.setdefault(token.casefold(), set()).add(token)
    for f in FOUNDATIONS:
        variants = lower_lookup.get(f.casefold(), set())
        if not variants:
            raise ValueError(f"Focal label {f.title()!r} is absent from the MFRC snapshot")
        if variants != {f.title()}:
            raise ValueError(f"Unexpected case/spelling variant for focal label {f}: {sorted(variants)}")
        label = f.title()
        raw[f"human_{f}"] = raw["annotation"].map(lambda x, lab=label: float(lab in _labels(x)))

    agg = {
        "text": "first",
        "subreddit": "first",
        "bucket": "first",
        "annotator": "nunique",
        "confidence_num": "mean",
        **{f"human_{f}": "mean" for f in FOUNDATIONS},
    }
    items = (
        raw.groupby("item_id", as_index=False)
        .agg(agg)
        .rename(columns={"annotator": "n_annotators", "confidence_num": "human_confidence_mean"})
    )
    items["content_id"] = items["text"].astype(str).map(_content_id)

    eligible = items.loc[items["n_annotators"] >= min_annotators].copy()
    excluded = items.loc[items["n_annotators"] < min_annotators].copy()
    if eligible.empty:
        raise ValueError("No eligible MFRC items after annotator-count filter")

    seed = int(cfg["seed"])
    dev_target = int(cfg["sampling"]["dev_items"])
    if not 0 < dev_target < len(eligible):
        raise ValueError(f"dev_items must be between 1 and {len(eligible)-1}")
    dev_ids = _choose_grouped_ids(eligible, dev_target, seed, "dev")
    if len(dev_ids) >= len(eligible):
        raise ValueError("Grouped dev split consumed all eligible items; reduce dev_items")
    eligible["split"] = eligible["item_id"].map(lambda x: "dev" if x in dev_ids else "test")

    test = eligible.loc[eligible["split"] == "test"].copy()
    sens_target = int(cfg["sampling"]["sensitivity_items"])
    if not 0 < sens_target <= len(test):
        raise ValueError(f"sensitivity_items must be between 1 and {len(test)}")
    sens_ids = _choose_grouped_ids(test, sens_target, seed, "sensitivity")
    eligible["sensitivity"] = eligible["item_id"].isin(sens_ids)

    # A leak-prevention invariant worth enforcing: exact text never crosses dev/test.
    split_counts = eligible.groupby("text")["split"].nunique()
    if (split_counts > 1).any():
        raise AssertionError("Exact-text grouping failed: identical text crossed dev/test")

    eligible = eligible.sort_values("item_id").reset_index(drop=True)
    excluded = excluded.sort_values("item_id").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0}
    eligible.to_csv(out, index=False, compression=compression)
    excluded.to_csv(excluded_path, index=False, compression=compression)
    items_sha = _sha256_file(out)

    text_source_counts = (
        raw[["text", "bucket", "subreddit"]]
        .drop_duplicates()
        .groupby("text", dropna=False)
        .size()
    )
    cross_source_texts = text_source_counts[text_source_counts > 1]
    ties = {f: int((eligible[f"human_{f}"] == 0.5).sum()) for f in FOUNDATIONS}
    audit = {
        "data_pipeline_version": DATA_PIPELINE_VERSION,
        "raw_sha256": raw_sha,
        "data_config_fingerprint": config_fp,
        "processed_items_sha256": items_sha,
        "raw_rows_before_exact_dedup": int(original_rows),
        "raw_rows_after_exact_dedup": int(len(raw)),
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "unique_items_all": int(len(items)),
        "eligible_items": int(len(eligible)),
        "excluded_lt_min_annotators": int(len(excluded)),
        "min_annotators_required": min_annotators,
        "annotator_count_distribution_all": {str(k): int(v) for k, v in sorted(Counter(items["n_annotators"]).items())},
        "dev_items_target": dev_target,
        "dev_items": int((eligible["split"] == "dev").sum()),
        "test_items": int((eligible["split"] == "test").sum()),
        "sensitivity_items_target": sens_target,
        "sensitivity_items": int(eligible["sensitivity"].sum()),
        "cross_source_exact_text_groups": int(len(cross_source_texts)),
        "exact_text_grouped_across_splits": True,
        "subreddits": sorted(raw["subreddit"].dropna().astype(str).unique()),
        "buckets": sorted(raw["bucket"].dropna().astype(str).unique()),
        "source_cells": sorted({f"{b} :: {s}" for b, s in zip(raw["bucket"].astype(str), raw["subreddit"].astype(str))}),
        "source_cell_count": int(raw[["bucket", "subreddit"]].drop_duplicates().shape[0]),
        "confidence_values": sorted(observed_conf),
        "atomic_annotation_labels": atomic_labels,
        "exact_half_ties_by_foundation": ties,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prepared_data_provenance(cfg)
    return out
