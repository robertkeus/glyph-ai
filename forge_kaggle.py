"""Phase 1 forge run — GPU/Kaggle entry (PLAN §A–E).

Wires the real two-adapter LoRA policy into the forge loop:
  1. warmup_builder — co-adapt the Builder so the reward is reachable (§B).
  2. cold-start probe — does the easy curriculum yield reward variance? (§A) The
     gate that decides from-scratch vs seeded vocabulary.
  3. forge_run — GRPO-forge the Speaker until the Pareto plateau (§E).

Checkpoints adapters (free tiers wipe disk + cap 12h — re-run resumes from CKPT).

    pip install -r requirements.txt
    python forge_kaggle.py
"""
import json
import os
from itertools import cycle

from glyph.agents import builder_prompt, speaker_prompt
from glyph.channel import Native
from glyph.forge import forge_run, reward
from glyph.policy import LoraPolicy
from glyph.probe import probe_robust
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CKPT = "glyph_ckpt"


FAST = bool(os.environ.get("FAST"))  # FAST=1 → ~15-min end-to-end smoke (noisy but real)


def main():
    ch = Native()
    train = load_tasks(split="train")
    easy = [t for t in train if t["difficulty"] == 0]
    policy = LoraPolicy(MODEL, channel=ch, max_code=96 if FAST else 256)

    if os.path.isdir(CKPT):
        policy.load(CKPT)
        print("resumed from", CKPT)
    else:
        policy.warmup_builder(easy if FAST else train, rounds=1 if FAST else 3)  # §B
        policy.save(CKPT)

    pool = cycle(easy)

    def sample_reward():
        t = next(pool)
        msg = policy.sample(speaker_prompt(t, ch), 1)[0]
        code = policy.build(builder_prompt(ch.builder_text(msg)))
        return float(reward(msg, code, t, ch, lam=0.0)[1])  # pure pass signal

    g = dict(runs=2, groups=4, group_size=4) if FAST else dict(runs=5, groups=16, group_size=8)
    gate = probe_robust(sample_reward, **g)
    print("cold-start gate (§A):", json.dumps(gate, indent=2))
    if gate["verdict"] == "seed the vocabulary":
        print("STOP: no reward variance — the Builder cannot decode the message, so "
              "from-scratch can't start. Seed primitive meanings before forging.")
        return

    def ckpt(step, _m):
        if step % 25 == 24:
            policy.save(CKPT)

    history = forge_run(train, policy, channel=ch, steps=20 if FAST else 300,
                        group_size=8, on_step=ckpt)
    policy.save(CKPT)
    print(json.dumps(history[-5:], indent=2))


if __name__ == "__main__":
    main()
