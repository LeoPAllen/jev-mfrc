from __future__ import annotations

import hashlib
import json

from . import FOUNDATIONS

# Close paraphrases of MFRC Coding Guide-2. Keep the canonical instrument anchored
# to the annotation construct; the strict variant is a prespecified sensitivity check.
DEFINITIONS = {
    "care": "caring for, protecting, or loving people, animals, or other living things, including avoiding emotional or physical harm",
    "equality": "egalitarian treatment, equal standing, opportunity, or outcomes, including inequality, discrimination, or prejudice",
    "proportionality": "people being rewarded or treated in proportion to effort, talent, input, merit, or deservingness, including corruption, nepotism, or disproportionate reward",
    "loyalty": "loyalty to or cooperation with family, community, region, nation, or another in-group, including betrayal or abandonment",
    "authority": "respect for or deference to legitimate authority, hierarchy, duty, obedience, or tradition in family, social, religious, or government institutions",
    "purity": "holiness, sanctity, purity, contamination, or degradation of people, bodies, objects, attributes, or practices",
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
        instructions = f"Does the text express a concern, belief, attitude, or emotion about {definition}?"
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
                "Generic moral evaluation without this domain is Thin Morality, not a positive label for this foundation."
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
