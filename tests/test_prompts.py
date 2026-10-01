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
        assert "word or topic mention alone is not enough" in question["instructions"]
        assert "Thin Morality" in question["criteria"]["false"]
        assert "explicit or implicit" in question["criteria"]["true"]


def test_canonical_distinguishes_mfrc_equality_from_proportionality():
    q = question_set("canonical")
    assert "egalitarian treatment and equal outcomes" in q["equality"]["instructions"]
    assert "rewarded in proportion to their merit" in q["proportionality"]["instructions"]
    assert "treated in proportion" not in q["proportionality"]["instructions"]


def test_all_six_canonical_domains_follow_the_coding_guide():
    q = question_set("canonical")
    expected = {
        "care": "caring for, protecting, or loving people, animals, or other living things",
        "equality": "egalitarian treatment and equal outcomes for individuals and groups",
        "proportionality": "individuals being rewarded in proportion to their merit",
        "loyalty": "loyalty to or cooperation with family, community, region, nation",
        "authority": "deference toward legitimate authorities and high-status individuals",
        "purity": "avoiding bodily or spiritual contamination and degradation",
    }
    assert all(expected[foundation] in q[foundation]["instructions"] for foundation in FOUNDATIONS)
