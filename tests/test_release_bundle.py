from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_release_bundle import (
    DATA_COLUMNS,
    _validate_data_frame,
    _validate_file_hashes,
    _write_gzip_csv,
    validate_release_bundle,
)


def _frame() -> pd.DataFrame:
    rows = []
    for i, sensitivity in enumerate((True, False)):
        row = {
            "item_id": f"item-{i}",
            "content_id": f"content-{i}",
            "bucket": "bucket-a",
            "subreddit": "subreddit-a",
            "n_annotators": 3,
            "human_confidence_mean": 0.5,
            "variant": "canonical",
            "prompt_hash": "canonical-hash",
            "requested_model": "jev-latest",
            "returned_model": "jev-1.13.0",
            "sensitivity": sensitivity,
        }
        row.update({f"human_{foundation}": 0.25 for foundation in (
            "care", "equality", "proportionality", "loyalty", "authority", "purity",
        )})
        row.update({f"p_{foundation}": 0.5 for foundation in (
            "care", "equality", "proportionality", "loyalty", "authority", "purity",
        )})
        rows.append(row)
    return pd.DataFrame(rows, columns=DATA_COLUMNS)


def _check(frame: pd.DataFrame, *, count: int = 2) -> None:
    _validate_data_frame(
        frame,
        variant="canonical",
        expected_count=count,
        expected_sensitivity_count=1,
        expected_prompt_hash="canonical-hash",
        requested_model="jev-latest",
        returned_model="jev-1.13.0",
    )


def test_checked_in_release_bundle_passes_all_validations():
    validate_release_bundle()


@pytest.mark.parametrize("field", ["text", "raw_comment", "annotator_id"])
def test_release_data_rejects_text_and_annotator_identifiers(field):
    frame = _frame()
    frame[field] = "must not be released"
    with pytest.raises(ValueError, match="forbidden"):
        _check(frame)


def test_release_data_rejects_wrong_item_count():
    with pytest.raises(ValueError, match="expected 3"):
        _check(_frame(), count=3)


@pytest.mark.parametrize("field,value", [
    ("p_care", None),
    ("p_care", 1.01),
    ("p_care", -0.01),
    ("human_care", None),
    ("human_care", 1.01),
    ("human_care", -0.01),
])
def test_release_data_rejects_missing_or_out_of_range_probabilities_and_shares(field, value):
    frame = _frame()
    frame.loc[0, field] = value
    with pytest.raises(ValueError, match=r"missing or outside \[0, 1\]"):
        _check(frame)


@pytest.mark.parametrize("field,value", [
    ("requested_model", "different-request"),
    ("returned_model", "jev-other-version"),
])
def test_release_data_rejects_model_identity_drift(field, value):
    frame = _frame()
    frame.loc[0, field] = value
    with pytest.raises(ValueError, match="model identity differs"):
        _check(frame)


def test_release_data_rejects_prompt_variant_or_sensitive_subset_drift():
    wrong_prompt = _frame()
    wrong_prompt.loc[0, "prompt_hash"] = "other-hash"
    with pytest.raises(ValueError, match="prompt hash"):
        _check(wrong_prompt)

    wrong_variant = _frame()
    wrong_variant.loc[0, "variant"] = "strict"
    with pytest.raises(ValueError, match="variant"):
        _check(wrong_variant)

    wrong_sensitivity = _frame()
    wrong_sensitivity.loc[0, "sensitivity"] = False
    with pytest.raises(ValueError, match="sensitivity rows"):
        _check(wrong_sensitivity)


def test_manifest_hash_mismatch_is_rejected(tmp_path):
    payload = tmp_path / "table.csv"
    payload.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_file_hashes(tmp_path, {"table.csv": "not-the-file-hash"})


def test_release_csv_gzip_bytes_are_deterministic(tmp_path):
    frame = _frame()
    first, second = tmp_path / "first.csv.gz", tmp_path / "second.csv.gz"
    _write_gzip_csv(frame, first)
    _write_gzip_csv(frame, second)
    assert first.read_bytes() == second.read_bytes()
