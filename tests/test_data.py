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
    meta = {"sha256": data._sha256_file(raw), "repo_id": "x", "requested_revision": "main", "split": "train_dedup"}
    (root / "data/raw/mfrc_meta.json").write_text(json.dumps(meta))


def test_prepare_items_excludes_two_annotator_and_is_row_order_stable(tmp_path, monkeypatch):
    a = tmp_path / "a"; b = tmp_path / "b"
    raw = _raw_rows()
    _write_snapshot(a, raw); _write_snapshot(b, raw.sample(frac=1, random_state=99))

    monkeypatch.setattr(data, "path", lambda *p: a.joinpath(*p))
    data.prepare_items(_cfg(), force=True)
    first = pd.read_csv(a / "data/processed/items.csv.gz").sort_values("item_id").reset_index(drop=True)
    audit = json.loads((a / "data/processed/audit.json").read_text())
    assert len(first) == 5
    assert audit["excluded_lt_min_annotators"] == 1
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


def test_conflicting_item_annotator_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows()
    duplicate = raw.iloc[[0]].copy(); duplicate["annotation"] = "Purity"
    _write_snapshot(tmp_path, pd.concat([raw, duplicate], ignore_index=True))
    with pytest.raises(ValueError, match="item/annotator"):
        data.prepare_items(_cfg(), force=True)


def test_unknown_confidence_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    raw = _raw_rows(); raw.loc[0, "confidence"] = "Maybe"
    _write_snapshot(tmp_path, raw)
    with pytest.raises(ValueError, match="Unknown confidence"):
        data.prepare_items(_cfg(), force=True)


def test_processed_content_mutation_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "path", lambda *p: tmp_path.joinpath(*p))
    _write_snapshot(tmp_path, _raw_rows())
    data.prepare_items(_cfg(), force=True)
    p = tmp_path / "data/processed/items.csv.gz"
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
    assert same["split"].nunique() == 1
    audit = json.loads((tmp_path / "data/processed/audit.json").read_text())
    assert audit["cross_source_exact_text_groups"] >= 1
    assert audit["exact_text_grouped_across_splits"] is True
