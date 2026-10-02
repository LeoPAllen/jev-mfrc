#!/usr/bin/env python3
"""Build and validate a compact, text-free archive of the completed study."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_mfrc import FOUNDATIONS  # noqa: E402
from jev_mfrc.config import load_config  # noqa: E402
from jev_mfrc.data import experiment_fingerprint, prepared_data_provenance  # noqa: E402
from jev_mfrc.jev import _validate_approval  # noqa: E402
from jev_mfrc.prompts import instrument_bundle_hash, prompt_hash  # noqa: E402

RELEASE = ROOT / "release"
EXPECTED_COUNTS = {
    "development_canonical_predictions": 1_000,
    "canonical_heldout_predictions": 16_709,
    "strict_sensitivity_predictions": 1_000,
}
EXPECTED_RETURNED_MODEL = "jev-1.13.0"

DATA_COLUMNS = [
    "item_id",
    "content_id",
    "bucket",
    "subreddit",
    "n_annotators",
    "human_confidence_mean",
    *(f"human_{foundation}" for foundation in FOUNDATIONS),
    *(f"p_{foundation}" for foundation in FOUNDATIONS),
    "variant",
    "prompt_hash",
    "requested_model",
    "returned_model",
    "sensitivity",
]
ITEM_COLUMNS = [
    "item_id", "content_id", "bucket", "subreddit", "n_annotators",
    "human_confidence_mean", *(f"human_{f}" for f in FOUNDATIONS),
    "split", "sensitivity",
]
PREDICTION_COLUMNS = [
    "item_id", "variant", "prompt_hash", "requested_model", "returned_model",
    *(f"p_{f}" for f in FOUNDATIONS),
]
FORBIDDEN_NAME_PARTS = {
    "text", "comment", "annotator", "coder", "secret", "credential",
    "authorization", "api", "token",
}
RELEASE_CONTENTS = {
    "README.md",
    "analysis.json",
    "summary.md",
    "tables/foundation_metrics.csv",
    "tables/selective_review.csv",
    "tables/source_cell_recovery.csv",
    "tables/source_cell_recovery_summary.csv",
    "tables/wording_sensitivity.csv",
    "tables/jev_usage.csv",
    "figures/selective_review.png",
    "figures/cell_recovery.png",
    "data/canonical_heldout.csv.gz",
    "data/strict_sensitivity.csv.gz",
}
RESULT_COPIES = {
    "analysis.json": "results/analysis.json",
    "summary.md": "results/summary.md",
    "tables/foundation_metrics.csv": "results/tables/foundation_metrics.csv",
    "tables/selective_review.csv": "results/tables/selective_review.csv",
    "tables/source_cell_recovery.csv": "results/tables/source_cell_recovery.csv",
    "tables/source_cell_recovery_summary.csv": "results/tables/source_cell_recovery_summary.csv",
    "tables/wording_sensitivity.csv": "results/tables/wording_sensitivity.csv",
    "tables/jev_usage.csv": "results/tables/jev_usage.csv",
    "figures/selective_review.png": "results/figures/selective_review.png",
    "figures/cell_recovery.png": "results/figures/cell_recovery.png",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_parts(name: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", name.lower()) if part}


def _validate_data_frame(
    frame: pd.DataFrame,
    *,
    variant: str,
    expected_count: int,
    expected_sensitivity_count: int,
    expected_prompt_hash: str,
    requested_model: str,
    returned_model: str,
) -> None:
    forbidden = [
        column for column in frame.columns
        if _column_parts(str(column)) & FORBIDDEN_NAME_PARTS
    ]
    if forbidden:
        raise ValueError(f"Release data has forbidden text, identifier, or credential fields: {forbidden}")
    if list(frame.columns) != DATA_COLUMNS:
        raise ValueError(f"Unexpected release data columns: {list(frame.columns)}")
    if len(frame) != expected_count:
        raise ValueError(f"{variant} release has {len(frame)} rows; expected {expected_count}")
    if frame["item_id"].isna().any() or frame["item_id"].duplicated().any():
        raise ValueError(f"{variant} release item IDs must be present and unique")
    if frame[["content_id", "bucket", "subreddit"]].isna().any().any():
        raise ValueError(f"{variant} release is missing source identity fields")

    for column in [*(f"p_{f}" for f in FOUNDATIONS), *(f"human_{f}" for f in FOUNDATIONS), "human_confidence_mean"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{variant} release field {column} is missing or outside [0, 1]")
    n = pd.to_numeric(frame["n_annotators"], errors="coerce")
    if n.isna().any() or (n < 1).any() or (n % 1 != 0).any():
        raise ValueError(f"{variant} release annotator counts must be positive integers")

    sensitivity = frame["sensitivity"].map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() == "true"
    )
    if variant == "strict" and not sensitivity.all():
        raise ValueError("Strict sensitivity release contains rows outside the prespecified subset")
    if variant == "canonical" and int(sensitivity.sum()) != expected_sensitivity_count:
        raise ValueError(
            f"Canonical release marks {int(sensitivity.sum())} sensitivity rows; "
            f"expected {expected_sensitivity_count}"
        )
    if set(frame["variant"].dropna().astype(str)) != {variant}:
        raise ValueError(f"{variant} release contains a different prediction variant")
    if set(frame["prompt_hash"].dropna().astype(str)) != {expected_prompt_hash}:
        raise ValueError(f"{variant} release prompt hash differs from the frozen instrument")
    if set(frame["requested_model"].dropna().astype(str)) != {requested_model}:
        raise ValueError(f"{variant} release requested model identity differs")
    if set(frame["returned_model"].dropna().astype(str)) != {returned_model}:
        raise ValueError(f"{variant} release returned model identity differs")


def _validate_file_hashes(release_dir: Path, hashes: dict[str, str]) -> None:
    for relative_path, expected in hashes.items():
        target = release_dir / relative_path
        if not target.is_file():
            raise ValueError(f"Manifest file is missing: {relative_path}")
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"Manifest hash mismatch for {relative_path}")


def validate_release_bundle(release_dir: Path = RELEASE) -> None:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = set(manifest.get("file_sha256", {}))
    actual_files = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*") if path.is_file() and path != manifest_path
    }
    if listed != RELEASE_CONTENTS or actual_files != RELEASE_CONTENTS:
        raise ValueError("Release file set does not match the required archival bundle")
    _validate_file_hashes(release_dir, manifest["file_sha256"])

    counts = manifest["counts"]
    if {key: counts.get(key) for key in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise ValueError("Release manifest item counts differ from the completed experiment")
    models = manifest["jev_models"]
    if models.get("requested") != "jev-latest" or models.get("returned") != EXPECTED_RETURNED_MODEL:
        raise ValueError("Release manifest JEV model identity differs from the completed experiment")
    prompts = manifest["instrument"]
    if prompts.get("canonical_prompt_hash") != prompt_hash("canonical"):
        raise ValueError("Manifest canonical prompt hash differs from the frozen instrument")
    if prompts.get("strict_prompt_hash") != prompt_hash("strict"):
        raise ValueError("Manifest strict prompt hash differs from the frozen instrument")

    expected_sensitivity = EXPECTED_COUNTS["strict_sensitivity_predictions"]
    _validate_data_frame(
        pd.read_csv(release_dir / "data/canonical_heldout.csv.gz"),
        variant="canonical",
        expected_count=EXPECTED_COUNTS["canonical_heldout_predictions"],
        expected_sensitivity_count=expected_sensitivity,
        expected_prompt_hash=prompts["canonical_prompt_hash"],
        requested_model=models["requested"],
        returned_model=models["returned"],
    )
    _validate_data_frame(
        pd.read_csv(release_dir / "data/strict_sensitivity.csv.gz"),
        variant="strict",
        expected_count=expected_sensitivity,
        expected_sensitivity_count=expected_sensitivity,
        expected_prompt_hash=prompts["strict_prompt_hash"],
        requested_model=models["requested"],
        returned_model=models["returned"],
    )

    analysis = json.loads((release_dir / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("test_items") != counts["canonical_heldout_predictions"]:
        raise ValueError("Released analysis count does not match canonical data")
    if analysis.get("sensitivity_items") != counts["strict_sensitivity_predictions"]:
        raise ValueError("Released analysis count does not match strict sensitivity data")
    if analysis.get("provenance", {}).get("experiment_fingerprint") != manifest["experiment_fingerprint"]:
        raise ValueError("Released analysis experiment fingerprint differs from manifest")


def _write_gzip_csv(frame: pd.DataFrame, destination: Path) -> None:
    csv_bytes = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            compressed.write(csv_bytes)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def build_release_bundle() -> Path:
    cfg = load_config()
    approval_path = ROOT / "approvals/instrument.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    _validate_approval(cfg)
    provenance = prepared_data_provenance(cfg)
    analysis = _read_json("results/analysis.json")
    audit = _read_json("data/processed/audit.json")
    raw_meta = _read_json("data/raw/mfrc_meta.json")

    if analysis.get("provenance", {}).get("experiment_fingerprint") != experiment_fingerprint(cfg):
        raise ValueError("Analysis output belongs to a different experiment")
    if analysis.get("provenance", {}).get("data") != provenance:
        raise ValueError("Analysis data provenance does not match the verified processed snapshot")
    if raw_meta.get("sha256") != provenance.get("raw_sha256"):
        raise ValueError("Pinned raw MFRC checksum does not match the processed-data provenance")
    if audit.get("processed_items_sha256") != provenance.get("processed_items_sha256"):
        raise ValueError("Processed-data audit checksum does not match prepared data")

    requested_model = str(cfg["jev"]["model"])
    returned_model = EXPECTED_RETURNED_MODEL
    canonical_prompt_hash = prompt_hash("canonical")
    strict_prompt_hash = prompt_hash("strict")
    if approval.get("instrument_bundle_hash") != instrument_bundle_hash():
        raise ValueError("Human approval does not match the current frozen instrument")
    if approval.get("experiment_fingerprint") != experiment_fingerprint(cfg):
        raise ValueError("Human instrument approval belongs to a different experiment")
    if analysis["provenance"].get("requested_models_in_export") != [requested_model]:
        raise ValueError("Analysis requested-model identity differs from the configured model")
    if analysis["provenance"].get("returned_models_in_export") != [returned_model]:
        raise ValueError("Analysis returned-model identity differs from the completed experiment")
    if analysis["provenance"].get("prompt_hashes") != {
        "canonical": canonical_prompt_hash, "strict": strict_prompt_hash,
    }:
        raise ValueError("Analysis prompt hashes differ from the frozen instrument")

    item_path = ROOT / "data/processed/items.csv.gz"
    items = pd.read_csv(item_path, usecols=ITEM_COLUMNS)
    test = items.loc[items["split"].eq("test")].copy()
    predictions_path = ROOT / "data/processed/jev_predictions.csv.gz"
    predictions = pd.read_csv(predictions_path, usecols=PREDICTION_COLUMNS)
    dev_predictions = pd.read_csv(
        ROOT / "data/processed/dev_predictions.csv.gz", usecols=PREDICTION_COLUMNS,
    )

    if int(audit.get("dev_items", -1)) != EXPECTED_COUNTS["development_canonical_predictions"]:
        raise ValueError("Development count differs from the completed experiment")
    if int(audit.get("test_items", -1)) != EXPECTED_COUNTS["canonical_heldout_predictions"]:
        raise ValueError("Held-out count differs from the completed experiment")
    if int(audit.get("sensitivity_items", -1)) != EXPECTED_COUNTS["strict_sensitivity_predictions"]:
        raise ValueError("Sensitivity count differs from the completed experiment")
    if len(dev_predictions) != EXPECTED_COUNTS["development_canonical_predictions"] or set(dev_predictions["variant"]) != {"canonical"}:
        raise ValueError("Canonical development predictions are incomplete or include another variant")

    for frame in (dev_predictions, predictions):
        if set(frame["requested_model"].dropna().astype(str)) != {requested_model}:
            raise ValueError("Exported predictions contain a different requested model")
        if set(frame["returned_model"].dropna().astype(str)) != {returned_model}:
            raise ValueError("Exported predictions contain a different returned model")

    canonical = predictions.loc[predictions["variant"].eq("canonical")].merge(
        test, on="item_id", how="inner", validate="one_to_one",
    )
    strict_expected = test.loc[test["sensitivity"].astype(bool)].copy()
    strict = predictions.loc[predictions["variant"].eq("strict")].merge(
        strict_expected, on="item_id", how="inner", validate="one_to_one",
    )
    if len(canonical) != EXPECTED_COUNTS["canonical_heldout_predictions"]:
        raise ValueError("Canonical held-out prediction export is incomplete")
    if len(strict) != EXPECTED_COUNTS["strict_sensitivity_predictions"]:
        raise ValueError("Strict sensitivity prediction export is incomplete")
    if set(canonical["prompt_hash"]) != {canonical_prompt_hash} or set(strict["prompt_hash"]) != {strict_prompt_hash}:
        raise ValueError("Prediction prompt hashes differ from the frozen instrument")

    def release_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.loc[:, DATA_COLUMNS].copy()
        out["sensitivity"] = out["sensitivity"].astype(bool)
        return out.sort_values("item_id", kind="mergesort").reset_index(drop=True)

    canonical_data = release_frame(canonical)
    strict_data = release_frame(strict)
    _validate_data_frame(
        canonical_data, variant="canonical",
        expected_count=EXPECTED_COUNTS["canonical_heldout_predictions"],
        expected_sensitivity_count=EXPECTED_COUNTS["strict_sensitivity_predictions"],
        expected_prompt_hash=canonical_prompt_hash,
        requested_model=requested_model, returned_model=returned_model,
    )
    _validate_data_frame(
        strict_data, variant="strict",
        expected_count=EXPECTED_COUNTS["strict_sensitivity_predictions"],
        expected_sensitivity_count=EXPECTED_COUNTS["strict_sensitivity_predictions"],
        expected_prompt_hash=strict_prompt_hash,
        requested_model=requested_model, returned_model=returned_model,
    )

    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    (RELEASE / "tables").mkdir(parents=True)
    (RELEASE / "figures").mkdir(parents=True)
    (RELEASE / "data").mkdir(parents=True)
    for target_relative, source_relative in RESULT_COPIES.items():
        shutil.copyfile(ROOT / source_relative, RELEASE / target_relative)
    _write_gzip_csv(canonical_data, RELEASE / "data/canonical_heldout.csv.gz")
    _write_gzip_csv(strict_data, RELEASE / "data/strict_sensitivity.csv.gz")

    mfrc = {
        "repo_id": provenance["repo_id"],
        "revision": provenance["resolved_revision"],
        "split": provenance["split"],
        "raw_dataset_sha256": provenance["raw_sha256"],
        "processed_data_sha256": provenance["processed_items_sha256"],
        "data_config_fingerprint": provenance["data_config_fingerprint"],
    }
    experiment = experiment_fingerprint(cfg)
    manifest = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "mfrc": mfrc,
        "experiment_fingerprint": experiment,
        "instrument": {
            "approval_file_sha256": sha256_file(approval_path),
            "instrument_bundle_hash": instrument_bundle_hash(),
            "canonical_prompt_hash": canonical_prompt_hash,
            "strict_prompt_hash": strict_prompt_hash,
        },
        "jev_models": {"requested": requested_model, "returned": returned_model},
        "counts": {
            **EXPECTED_COUNTS,
            "eligible_items": int(audit["eligible_items"]),
        },
        "analysis_configuration": {
            "seed": int(cfg["seed"]),
            **cfg["analysis"],
        },
        "file_sha256": {},
    }
    readme = f"""# JEV × MFRC reproducibility bundle

This archive contains the completed prespecified held-out analysis and the non-text data needed to recalculate it. It includes {EXPECTED_COUNTS['canonical_heldout_predictions']:,} canonical held-out rows and {EXPECTED_COUNTS['strict_sensitivity_predictions']:,} strict-wording sensitivity rows. The canonical file also flags the {EXPECTED_COUNTS['strict_sensitivity_predictions']:,} rows selected for wording sensitivity.

## Contents and use

- `analysis.json`, `summary.md`, `tables/`, and `figures/` are the archived analysis outputs.
- `data/canonical_heldout.csv.gz` contains the six JEV probabilities and six trained-rater vote shares for every held-out item.
- `data/strict_sensitivity.csv.gz` contains strict-wording predictions for the prespecified sensitivity subset, alongside the canonical comparison fields.
- `manifest.json` records data and instrument provenance, model identity, analysis settings, the generating Git commit, and SHA-256 hashes for every other release file.

No Reddit comment text, annotator identifiers, raw MFRC files, JEV cache, or API credentials are included. `item_id` and `content_id` are one-way identifiers; the bucket and subreddit fields preserve the specified source-cell analyses.

The data use `human_<foundation>` for trained-rater shares and `p_<foundation>` for the probability of the bounded JEV yes/no judgment. These are not moral-intensity scores or ontological truth labels. Source-cell estimates describe this selected MFRC sample, not Reddit population prevalence.

The pinned analysis settings and seed are in `manifest.json`. The methods are specified in [`../docs/METHODS.md`](../docs/METHODS.md), with implementation in [`../src/jev_mfrc/metrics.py`](../src/jev_mfrc/metrics.py). Recalculation from this bundle requires no JEV request, local call cache, raw MFRC file, or comment text.
"""
    (RELEASE / "README.md").write_text(readme, encoding="utf-8")
    manifest["file_sha256"] = {
        path.relative_to(RELEASE).as_posix(): sha256_file(path)
        for path in sorted(RELEASE.rglob("*")) if path.is_file()
    }
    (RELEASE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_release_bundle(RELEASE)
    return RELEASE


if __name__ == "__main__":
    output = build_release_bundle()
    print(f"Built and validated release bundle: {output.relative_to(ROOT)}")
