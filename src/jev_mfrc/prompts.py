from __future__ import annotations

import hashlib
import json

from . import FOUNDATIONS

# MFRC Coding Guide-2 domain descriptions. Canonical asks for a voiced concern in
# each domain; strict is the one prespecified wording sensitivity check.
DEFINITIONS = {
    "care": "caring for, protecting, or loving people, animals, or other living things, including concern about emotional or physical harm to others",
    "equality": "egalitarian treatment and equal outcomes for individuals and groups, including social justice, inequality, discrimination, or prejudice",
    "proportionality": "individuals being rewarded in proportion to their merit, such as effort, talent, or input, including meritocracy, deservingness, corruption, or nepotism",
    "loyalty": "loyalty to or cooperation with family, community, region, nation, or another in-group, including patriotism, self-sacrifice, abandonment, betrayal, cheating, or treason",
    "authority": "deference toward legitimate authorities and high-status individuals, including leadership, respect for tradition, duty, obedience, and respect for family, social, or government institutions",
    "purity": "avoiding bodily or spiritual contamination and degradation, including holiness, sanctity, purity, nobility, cleanliness, impurity, sinfulness, or disgust",
}


def _question(foundation: str, *, strict: bool) -> dict:
    definition = DEFINITIONS[foundation]
    if strict:
        instructions = (
            f"Does the text itself clearly express a concern, belief, attitude, or emotion about {definition}? "
            "Count yes only when the foundation-specific moral meaning is clear from the text. Mere topic mention, neutral description, "
            "generic good/bad/right/wrong language, unresolved sarcasm, or meaning that depends on missing context does not count. "
            "Do not infer unsupported intent."
        )
    else:
        instructions = (
            f"Does the text express a concern, belief, attitude, or emotion about {definition}? "
            "A related word or topic mention alone is not enough; the text must communicate a concern grounded in this domain."
        )
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {
            "true": (
                f"The text communicates a positive or negative concern, belief, attitude, or emotion grounded in {definition}. "
                "The expression may be explicit or implicit."
            ),
            "false": (
                f"The text does not communicate a concern, belief, attitude, or emotion grounded in {definition}. "
                "Generic evaluation such as simply calling something good, bad, right, or wrong without this domain is Thin Morality, not a positive label for this foundation."
            ),
        },
    }


def question_set(variant: str) -> dict:
    if variant not in {"canonical", "strict"}:
        raise ValueError(f"Unknown prompt variant: {variant}")
    return {f: _question(f, strict=(variant == "strict")) for f in FOUNDATIONS}


def prompt_hash(variant: str) -> str:
    payload = json.dumps(question_set(variant), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def instrument_bundle_hash() -> str:
    payload = json.dumps({v: question_set(v) for v in ("canonical", "strict")}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
