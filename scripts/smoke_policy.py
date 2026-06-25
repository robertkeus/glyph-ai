"""CPU smoke: LoraPolicy executes sample/build/learn/warmup end-to-end.

Not a convergence test — only that the training code runs (forward, symbol mask,
GRPO backward, optimizer step, SFT warmup, checkpoint) without shape/wiring bugs.
Tiny config; real training needs a GPU.

    .venv/bin/python scripts/smoke_policy.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glyph.agents import speaker_prompt
from glyph.channel import Native
from glyph.forge import forge_step
from glyph.policy import LoraPolicy
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main():
    ch = Native()
    tasks = load_tasks(split="train")[:2]
    p = LoraPolicy(MODEL, channel=ch, max_msg=4, max_code=16, lora_r=4)
    print("init OK — vocab", len(p.tok), "symbols", len(p.sym_ids))

    p.warmup_builder(tasks[:1], rounds=1)
    print("warmup_builder OK")

    sp = speaker_prompt(tasks[0], ch)
    msgs = p.sample(sp, 2)
    print("sample OK —", [len(m) for m in msgs], "glyphs")
    print("build OK —", repr(p.build("You are the Builder. Message:\n" + msgs[0])[:40]))

    used = p.learn(sp, msgs, [0.5, -0.5])   # forced advantages → exercises backward
    print("learn OK — updated on", used, "messages")

    m = forge_step(tasks[0], p, ch, lam=1e-4, group_size=2)
    print("forge_step OK —", {k: m[k] for k in ("pass_rate", "had_signal", "mean_bytes")})

    p.save("/tmp/glyph_ckpt")
    print("save OK\nSMOKE PASS")


if __name__ == "__main__":
    main()
