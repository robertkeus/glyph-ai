"""Reduced-cost cold-start gate for a LOCAL CPU run (no GPU).

Cheap warmup (easy tasks, 1 round) + tiny probe so it finishes on CPU. NOISY,
early read only — the real gate is forge_kaggle.py on a GPU. Use it to sniff
whether the easy curriculum produces any reward variance at all.
"""
import json
import os
import sys
from itertools import cycle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glyph.agents import builder_prompt, speaker_prompt
from glyph.channel import Native
from glyph.forge import reward
from glyph.policy import LoraPolicy
from glyph.probe import probe_robust
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main():
    ch = Native()
    easy = [t for t in load_tasks(split="train") if t["difficulty"] == 0]
    p = LoraPolicy(MODEL, channel=ch, max_msg=8, max_code=64, lora_r=4)
    p.warmup_builder(easy, rounds=1)
    it = cycle(easy)

    def sr():
        t = next(it)
        msg = p.sample(speaker_prompt(t, ch), 1)[0]
        code = p.build(builder_prompt(ch.builder_text(msg)))
        return float(reward(msg, code, t, ch, lam=0.0)[1])

    print("LOCAL reduced gate:",
          json.dumps(probe_robust(sr, runs=2, groups=4, group_size=4)))


if __name__ == "__main__":
    main()
