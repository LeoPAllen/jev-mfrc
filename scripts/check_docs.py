#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
required = {
    "docs/RESEARCH_SPEC.md": ["Splits fixed before JEV inference", "Claims outside scope"],
    "docs/METHODS.md": ["Uncertainty validity", "Information retained", "Selective review", "Wording sensitivity"],
    "docs/LITERATURE.md": ["Carlson", "MFRC", "Engineering / agent workflow guidance"],
    "docs/HUMAN_GATES.md": ["instrument meaning", "manuscript claims"],
    "paper/OUTLINE.md": ["Introduction", "Methods", "Results", "Discussion"],
}
for name, needles in required.items():
    p = ROOT / name
    if not p.exists():
        raise SystemExit(f"missing required document: {name}")
    text = p.read_text(encoding="utf-8").lower()
    missing = [x for x in needles if x.lower() not in text]
    if missing:
        raise SystemExit(f"{name} missing required terms: {missing}")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
if "approve_instrument.py --approve" not in readme:
    raise SystemExit("README must show the explicit human approval command before held-out inference")
print("documentation gate passed")
