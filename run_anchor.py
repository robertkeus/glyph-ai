"""Phase 2 anchor run (PLAN): English-in robustness + English-out translation.

Trains Speaker on paraphrase variants 0-2 (variant 3 NEVER trained) and the
glyphs→English translator, then measures:
  - regression: grounding + held-out compositional decode (test 2)
  - NEW speaker robustness: encode UNSEEN phrasings (variant 3) → canonical glyphs
  - NEW translation round-trip: glyphs → English → Speaker re-encodes → same glyphs
    (automatic, ungameable fidelity check; also prints samples to eyeball)
"""
import json

from glyph.agents import _extract_code, builder_prompt, grade, speaker_prompt
from glyph.channel import Native
from glyph.paraphrase import HELDOUT_VARIANT, english
from glyph.policy import LoraPolicy
from glyph.seed import canonical_message
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


def main():
    ch = Native()
    train = load_tasks(split="train")
    held = load_tasks(split="heldout")
    policy = LoraPolicy(MODEL, channel=ch)
    policy.warmup_seeded(train, rounds=9,
                         english=lambda t, rd: english(t, rd % HELDOUT_VARIANT),
                         translate=True)

    dec = sum(grade(_extract_code(policy.build(builder_prompt(
        ch.builder_text(canonical_message(t))))), t)["passed"] for t in train[:16])
    print(f"grounding (train[:16] decode): {dec}/16")

    t2 = sum(grade(_extract_code(policy.build(builder_prompt(
        ch.builder_text(canonical_message(t))))), t)["passed"] for t in held)
    print(f"test 2 regression (held-out decode): {t2}/{len(held)}")

    rob = 0
    for t in train:
        got = policy.sample(speaker_prompt(dict(t, prompt=english(t, HELDOUT_VARIANT)),
                                           ch), 1, greedy=True)[0]
        rob += got == canonical_message(t)
    print(f"SPEAKER robustness (unseen phrasing → glyphs): {rob}/{len(train)}")

    rt = 0
    samples = []
    for t in held:
        cm = canonical_message(t)
        en = policy.translate(cm)
        back = policy.sample(speaker_prompt({"prompt": en}, ch), 1, greedy=True)[0]
        rt += back == cm
        if len(samples) < 4:
            samples.append(f"  {cm} -> {en!r} -> {back} {'OK' if back == cm else 'X'}")
    print(f"TRANSLATION round-trip (glyphs→English→glyphs): {rt}/{len(held)}")
    print("\n".join(samples))


if __name__ == "__main__":
    main()
