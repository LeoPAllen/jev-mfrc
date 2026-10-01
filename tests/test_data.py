from __future__ import annotations

import json
import pandas as pd
import pytest

from jev_mfrc import data


def _raw_rows():
    rows = []
    comments = [
        ("a", "s1", "b1", ["Care", "Care", "Non-Moral"]),
        ("b", "s1", "b1", ["Equality", "Non-Moral", "Equality"]),
        ("c", "s2", "b2", ["Purity", "Non-Moral"]),
        ("d", "s2", "b2", ["Loyalty", "Authority", "Non-Moral"]),
        ("e", "s3", "b3", ["Proportionality", "Non-Moral", "Proportionality"]),
        ("f", "s3", "b3", ["Care,Equality", "Care", "Equality"]),
    ]
    for text, subreddit, bucket, labels in comments:
        for i, label in enumerate(labels):
            rows.append({"text": text, "subreddit": subreddit, "bucket": bucket, "annotator": f"ann{i}", "annotation": label, "confidence": "Confident"})
    return pd.DataFrame(rows)


def _cfg(seed=42):
    return {
        "seed": seed,
        "dataset": {"repo_id": "x", "revision": "main", "split": "train_dedup", "min_annotators": 3},
        "sampling": {"dev_items": 2, "sensitivity_items": 2},
    }


def _write_snapshot(root, df):
    (root / "data/raw").mkdir(parents=True, exist_ok=True)
    raw = root / "data/raw/mfrc.csv.gz"
    df.to_csv(raw, index=False, compression="gzip")
    meta = {
        "sha256": data._sha256_file(raw), "repo_id": "x", "requested_revision": "main",
        "resolved_revision": "sha", "split": "train_dedup",
    }
    (root / "data/raw/mfrc_meta.json").write_text(json.dumps(meta))


def test_item_identity_hash_preserves_tuple_boundaries():
    left = data._stable_item_id("c", "a\0b", "d")
    right = data._stable_item_id("b\0c", "a", "d")
    assert left != right


def test_data_config_fingerprint_does_not_depend_on_pipeline_version(monkeypatch):
    cfg = _cfg()
    fingerprint = data._data_config_fingerprint(cfg)
    monkeypatch.setattr(data, "DATA_PIPELINE_VERSION", data.DATA_PIPELINE_VERSION + 1)
    assert data._data_config_fingerprint(cfg) == fingerprint


def test_prepare_items_excludes_two_annotator_and_is_row_order_stable(tmp_path, monkeypatch):
    a = tmp_path / "a"; b = tmp_path / "b"
    raw = _raw_rows()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    _write_snapshot(a, raw); _write_snapshot(b, raw.sample(frac=1, random_state=99))

    monkeypatch.setattr(data, "path", lambda *p: a.joinpath(*p))
    data.prepare_items(_cfg(), force=True)
    first = pd.read_csv(a / "data/processed/items.csv.gz").sort_values("item_id").reset_index(drop=True)
    audit = json.loads((a / "data/processed/audit.json").read_text())
    assert len(first) == 5
    assert audit["items_excluded_lt_min_annotators"] == 1
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["semantic_annotation_duplicates_removed"] == 0
    assert audit["raw_rows_before_exact_dedup"] == len(raw)
    assert audit["raw_rows_after_exact_dedup"] == len(raw) - 1
    excluded = pd.read_csv(a / "data/processed/excluded_lt_min_annotators.csv.gz")
    assert len(excluded) == 1
    assert excluded["text"].tolist() == ["c"]
    assert (first["n_annotators"] >= 3).all()
    assert (first["split"] == "dev").sum() == 2
    assert first["sensitivity"].sum() == 2
    assert first.loc[first["text"] == "a", "human_care"].iloc[0] == pytest.approx(2/3)
    assert first.loc[first["text"] == "f", "human_equality"].iloc[0] == pytest.approx(2/3)

    monkeypatch.setattr(data, "path", lambda *p: b.joinpath(*p))
    data.prepare_items(_cfg(), force=True)
    second = pd.read_csv(b / "data/processed/items.csv.gz").sort_values("item_id").reset_index(drop=True)
    compare_cols = [c for c in first.columns]
    pd.testing.assert_frame_equal(first[compare_cols], second[compare_cols])


def test_processed_cache_rebuilds_on_config_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    _write_snapshot(tmp_path, _raw_rows())
    first_cfg = _cfg()
    data.prepare_items(first_cfg, force=True)
    before = json.loads((tmp_path / "data/processed/audit.json").read_text())
    changed_cfg = _cfg(seed=43)
    data.prepare_items(changed_cfg, force=False)
    after = json.loads((tmp_path / "data/processed/audit.json").read_text())
    assert before["data_config_fingerprint"] != after["data_config_fingerprint"]
    assert after["data_config_fingerprint"] == data._data_config_fingerprint(changed_cfg)

def test_raw_checksum_drift_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    _write_snapshot(tmp_path, _raw_rows())
    raw = pd.read_csv(tmp_path / "data/raw/mfrc.csv.gz")
    raw.iloc[:-1].to_csv(tmp_path / "data/raw/mfrc.csv.gz", index=False, compression="gzip")
    with pytest.raises(RuntimeError, match="checksum"):
        data.prepare_items(_cfg(), force=True)


def test_conflicts_exclude_whole_items_preserve_other_items_and_audit_counts(tmp_path, monkeypatch):
    baseline_root = tmp_path / "baseline"
    ordered_root = tmp_path / "ordered"
    shuffled_root = tmp_path / "shuffled"
    baseline_raw = _raw_rows()

    literal_duplicate = baseline_raw.loc[
        (baseline_raw["text"] == "a") & (baseline_raw["annotator"] == "ann0")
    ].iloc[[0]]
    reordered_semantic_duplicate = baseline_raw.loc[
        (baseline_raw["text"] == "f") & (baseline_raw["annotator"] == "ann0")
    ].iloc[[0]].copy()
    reordered_semantic_duplicate["annotation"] = "Equality,Care"
    conflicting_duplicate = reordered_semantic_duplicate.copy()
    conflicting_duplicate["annotation"] = "Care"
    raw = pd.concat(
        [baseline_raw, literal_duplicate, reordered_semantic_duplicate, conflicting_duplicate],
        ignore_index=True,
    )
    _write_snapshot(baseline_root, baseline_raw)
    _write_snapshot(ordered_root, raw)
    _write_snapshot(shuffled_root, raw.sample(frac=1, random_state=811))

    outputs = {}
    audits = {}
    for name, root in (("baseline", baseline_root), ("ordered", ordered_root), ("shuffled", shuffled_root)):
        monkeypatch.setattr(data, "path", lambda *p, root=root: root.joinpath(*p))
        output_path = data.prepare_items(_cfg(), force=True)
        outputs[name] = pd.read_csv(output_path).sort_values("item_id").reset_index(drop=True)
        audits[name] = json.loads((root / "data/processed/audit.json").read_text())

    audit = audits["ordered"]
    assert audit["raw_rows_before_exact_dedup"] == 20
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["raw_rows_after_exact_dedup"] == 19
    assert audit["semantic_annotation_duplicate_groups"] == 1
    assert audit["semantic_annotation_duplicates_removed"] == 1
    assert audit["genuine_conflicting_item_annotator_groups"] == 1
    assert audit["items_excluded_for_conflicting_same_annotator_records"] == 1
    assert audit["rows_excluded_for_conflicting_items"] == 4
    assert audit["items_remaining_after_conflict_exclusion"] == 5
    assert audit["retained_item_annotator_pairs_unique"] is True
    assert audit["items_excluded_lt_min_annotators"] == 1
    assert audit["unique_items_all"] == 5
    assert audit["eligible_items"] == 4

    processed = outputs["ordered"]
    assert len(processed) == 4
    assert processed["item_id"].is_unique
    assert "f" not in set(processed["text"])
    assert processed.loc[processed["text"] == "a", "n_annotators"].iloc[0] == 3
    # Disagreement between different annotators remains in the human vote share.
    assert processed.loc[processed["text"] == "a", "human_care"].iloc[0] == pytest.approx(2 / 3)
    assert processed.loc[processed["text"] == "b", "human_equality"].iloc[0] == pytest.approx(2 / 3)
    excluded_min = pd.read_csv(ordered_root / "data/processed/excluded_lt_min_annotators.csv.gz")
    assert excluded_min["text"].tolist() == ["c"]

    # Conflict exclusion changes only the affected item's presence; other item
    # summaries stay the same before and after the fix.
    summary_cols = [
        "item_id", "text", "subreddit", "bucket", "n_annotators",
        "human_confidence_mean", "content_id",
        "human_care", "human_equality", "human_proportionality",
        "human_loyalty", "human_authority", "human_purity",
    ]
    baseline_all = pd.concat([
        outputs["baseline"],
        pd.read_csv(baseline_root / "data/processed/excluded_lt_min_annotators.csv.gz"),
    ], ignore_index=True)[summary_cols].sort_values("item_id").reset_index(drop=True)
    fixed_all = pd.concat([
        outputs["ordered"],
        pd.read_csv(ordered_root / "data/processed/excluded_lt_min_annotators.csv.gz"),
    ], ignore_index=True)[summary_cols].sort_values("item_id").reset_index(drop=True)
    affected_id = data._stable_item_id("b3", "s3", "f")
    baseline_all = baseline_all.loc[baseline_all["item_id"] != affected_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_all, fixed_all)

    pd.testing.assert_frame_equal(outputs["ordered"], outputs["shuffled"])
    for field in (
        "raw_rows_before_exact_dedup", "exact_duplicate_rows_removed",
        "semantic_annotation_duplicate_groups", "semantic_annotation_duplicates_removed",
        "genuine_conflicting_item_annotator_groups",
        "items_excluded_for_conflicting_same_annotator_records",
        "rows_excluded_for_conflicting_items", "items_remaining_after_conflict_exclusion",
        "retained_item_annotator_pairs_unique", "items_excluded_lt_min_annotators",
        "eligible_items", "dev_items", "test_items",
    ):
        assert audits["ordered"][field] == audits["shuffled"][field]


def test_semantic_annotation_duplicates_collapse_and_are_row_order_stable(tmp_path, monkeypatch):
    a = tmp_path / "a"; b = tmp_path / "b"
    raw = _raw_rows()
    duplicate = raw.loc[(raw["text"] == "f") & (raw["annotation"] == "Care,Equality")].iloc[[0]].copy()
    duplicate["annotation"] = "Equality,Care"
    raw_with_semantic_duplicate = pd.concat([raw, duplicate], ignore_index=True)
    _write_snapshot(a, raw_with_semantic_duplicate)
    _write_snapshot(b, raw_with_semantic_duplicate.sample(frac=1, random_state=17))

    monkeypatch.setattr(data, "path", lambda *p: a.joinpath(*p))
    first_path = data.prepare_items(_cfg(), force=True)
    first = pd.read_csv(first_path).sort_values("item_id").reset_index(drop=True)
    audit = json.loads((a / "data/processed/audit.json").read_text())
    assert audit["exact_duplicate_rows_removed"] == 0
    assert audit["semantic_annotation_duplicates_removed"] == 1
    assert audit["semantic_annotation_duplicate_groups"] == 1
    assert audit["genuine_conflicting_item_annotator_groups"] == 0
    assert audit["items_excluded_for_conflicting_same_annotator_records"] == 0
    assert audit["raw_rows_after_exact_dedup"] == len(raw_with_semantic_duplicate)
    assert first.loc[first["text"] == "f", "n_annotators"].iloc[0] == 3
    assert first.loc[first["text"] == "f", "human_care"].iloc[0] == pytest.approx(2 / 3)
    assert first.loc[first["text"] == "f", "human_equality"].iloc[0] == pytest.approx(2 / 3)

    monkeypatch.setattr(data, "path", lambda *p: b.joinpath(*p))
    second_path = data.prepare_items(_cfg(), force=True)
    second = pd.read_csv(second_path).sort_values("item_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_same_annotation_with_different_confidence_excludes_the_item(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows()
    duplicate = raw.iloc[[0]].copy()
    duplicate["confidence"] = "Somewhat Confident"
    _write_snapshot(tmp_path, pd.concat([raw, duplicate], ignore_index=True))
    data.prepare_items(_cfg(), force=True)
    audit = json.loads((tmp_path / "data/processed/audit.json").read_text())
    assert audit["semantic_annotation_duplicates_removed"] == 0
    assert audit["genuine_conflicting_item_annotator_groups"] == 1
    assert audit["items_excluded_for_conflicting_same_annotator_records"] == 1
    assert audit["rows_excluded_for_conflicting_items"] == 4
    items = pd.read_csv(tmp_path / "data/processed/items.csv.gz")
    assert "a" not in set(items["text"])


def test_primary_sample_requires_at_least_three_annotators(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    _write_snapshot(tmp_path, _raw_rows())
    cfg = _cfg()
    cfg["dataset"]["min_annotators"] = 2
    with pytest.raises(ValueError, match="must be at least 3"):
        data.prepare_items(cfg)


def test_unknown_confidence_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows(); raw.loc[0, "confidence"] = "Maybe"
    _write_snapshot(tmp_path, raw)
    with pytest.raises(ValueError, match="Unknown confidence"):
        data.prepare_items(_cfg(), force=True)


def test_blank_confidence_remains_missing_not_an_unknown_label(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows()
    raw.loc[0, "confidence"] = ""
    _write_snapshot(tmp_path, raw)
    out = data.prepare_items(_cfg())
    items = pd.read_csv(out)
    assert items.loc[items["text"] == "a", "human_confidence_mean"].iloc[0] == pytest.approx(1.0)


def test_processed_content_mutation_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    _write_snapshot(tmp_path, _raw_rows())
    data.prepare_items(_cfg(), force=True)
    p = tmp_path / "data/processed/items.csv.gz"
    audit = json.loads((tmp_path / "data/processed/audit.json").read_text())
    assert audit["processed_items_sha256"] == data._sha256_file(p)
    df = pd.read_csv(p)
    df.loc[0, "text"] = "tampered"
    df.to_csv(p, index=False, compression="gzip")
    with pytest.raises(RuntimeError, match="content hash"):
        data.prepared_data_provenance(_cfg())


def test_revision_drift_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    _write_snapshot(tmp_path, _raw_rows())
    cfg = _cfg(); cfg["dataset"]["revision"] = "other"
    with pytest.raises(RuntimeError, match="provenance"):
        data.prepare_items(cfg, force=True)


def test_literal_na_text_is_preserved_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows()
    raw.loc[raw["text"] == "a", "text"] = "NA"
    _write_snapshot(tmp_path, raw)
    out = data.prepare_items(_cfg())
    items = pd.read_csv(out, dtype={"text": str}, keep_default_na=False)
    assert "NA" in set(items["text"])


def test_blank_identity_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows(); raw.loc[0, "text"] = "   "
    _write_snapshot(tmp_path, raw)
    with pytest.raises(ValueError, match="blank item identity"):
        data.prepare_items(_cfg(), force=True)


def test_absent_focal_label_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows(); raw["annotation"] = raw["annotation"].str.replace("Authority", "Non-Moral", regex=False)
    _write_snapshot(tmp_path, raw)
    with pytest.raises(ValueError, match="Focal label 'Authority' is absent"):
        data.prepare_items(_cfg(), force=True)


def test_audit_reports_bucket_subreddit_source_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows()
    # Same subreddit string in another bucket must count as a separate sampled source cell.
    extra = raw.loc[raw["text"] == "a"].copy()
    extra["text"] = "a-other-bucket"; extra["bucket"] = "b9"
    _write_snapshot(tmp_path, pd.concat([raw, extra], ignore_index=True))
    data.prepare_items(_cfg(), force=True)
    audit = json.loads((tmp_path / "data/processed/audit.json").read_text())
    expected = raw[["bucket", "subreddit"]].drop_duplicates().shape[0] + 1
    assert audit["source_cell_count"] == expected
    assert any(x == "b9 :: s1" for x in audit["source_cells"])


def test_prepare_items_rebuilds_stale_derived_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    cfg = _cfg()
    raw = _raw_rows()
    raw_path = tmp_path / "data/raw/mfrc.csv.gz"
    meta_path = tmp_path / "data/raw/mfrc_meta.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False, compression={"method":"gzip", "mtime":0})
    meta_path.write_text(json.dumps({
        "repo_id": cfg["dataset"]["repo_id"], "requested_revision": cfg["dataset"]["revision"],
        "resolved_revision": "sha", "split": cfg["dataset"]["split"], "sha256": data._sha256_file(raw_path)
    }))
    out = data.prepare_items(cfg)
    before = pd.read_csv(out)
    audit_path = tmp_path / "data/processed/audit.json"
    audit = json.loads(audit_path.read_text())
    audit["data_config_fingerprint"] = "stale"
    audit_path.write_text(json.dumps(audit))
    out2 = data.prepare_items(cfg)
    after = pd.read_csv(out2)
    pd.testing.assert_frame_equal(before, after)
    assert json.loads(audit_path.read_text())["data_config_fingerprint"] == data._data_config_fingerprint(cfg)


def test_cross_source_identical_text_is_grouped_to_same_split(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows()
    # Reuse exact text in another source cell. Preserve both source-specific items,
    # but never let identical content cross development/held-out splits.
    extra = raw.loc[raw["text"] == "a"].copy()
    extra["bucket"] = "other_bucket"
    extra["subreddit"] = "other_subreddit"
    _write_snapshot(tmp_path, pd.concat([raw, extra], ignore_index=True))
    out = data.prepare_items(_cfg(), force=True)
    items = pd.read_csv(out)
    same = items.loc[items["text"] == "a"]
    assert len(same) == 2
    assert same["item_id"].nunique() == 2
    assert same["split"].nunique() == 1
    audit = json.loads((tmp_path / "data/processed/audit.json").read_text())
    assert audit["cross_source_exact_text_groups"] >= 1
    assert audit["exact_text_grouped_across_splits"] is True


def test_download_raw_reuses_verified_snapshot_without_replacing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    cfg = _cfg()
    _write_snapshot(tmp_path, _raw_rows())
    raw_path = tmp_path / "data/raw/mfrc.csv.gz"
    before = raw_path.read_bytes()
    returned_raw, returned_meta = data.download_raw(cfg)
    assert returned_raw == raw_path
    assert returned_meta == tmp_path / "data/raw/mfrc_meta.json"
    assert raw_path.read_bytes() == before


def test_raw_metadata_requires_resolved_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    _write_snapshot(tmp_path, _raw_rows())
    meta_path = tmp_path / "data/raw/mfrc_meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["resolved_revision"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="resolved dataset revision"):
        data.download_raw(_cfg())
