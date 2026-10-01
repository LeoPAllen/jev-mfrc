from jev_mfrc import FOUNDATIONS
from jev_mfrc.prompts import instrument_bundle_hash, prompt_hash, question_set


def test_question_sets_are_six_nouls():
    for variant in ("canonical", "strict"):
        q = question_set(variant)
        assert list(q) == FOUNDATIONS
        assert all(v["type"] == "noul" for v in q.values())
        assert all(set(v["criteria"]) == {"true", "false"} for v in q.values())


def test_prompt_hashes_and_bundle_are_stable_and_distinct():
    assert prompt_hash("canonical") == prompt_hash("canonical")
    assert prompt_hash("canonical") != prompt_hash("strict")
    assert instrument_bundle_hash() == instrument_bundle_hash()
    assert len(instrument_bundle_hash()) == 64


def test_canonical_tracks_mfrc_construct_language_and_thin_morality_boundary():
    q = question_set("canonical")
    for question in q.values():
        assert "concern, belief, attitude, or emotion" in question["instructions"]
        assert "Thin Morality" in question["criteria"]["false"]
        assert "explicit or implicit" in question["criteria"]["true"]
