"""Phase 1 forge run — GPU/Kaggle entry (PLAN §A–E).

Default is SEEDED (PLAN §A fallback — from-scratch won't bootstrap: a random
Speaker emits undecodable symbols, so GRPO gets no within-task gradient). Seeding
grounds each primitive to a symbol and SFT-warms BOTH agents to that shared code;
RL then refines/compresses. Set FROMSCRATCH=1 to skip seeding (expect no signal).

  1. warmup_seeded — ground both agents (§A/§B).
  2. honest within-task gate — can the agents actually solve after warmup?
  3. forge_run — GRPO-compress the Speaker until the Pareto plateau (§E).

Checkpoints adapters (free tiers wipe disk + cap 12h — re-run resumes from CKPT).

    pip install -q peft
    python forge_kaggle.py          # full
    FAST=1 python forge_kaggle.py   # ~15-min smoke
"""
import json
import os
from itertools import cycle

from glyph.agents import _extract_code, builder_prompt, grade, speaker_prompt
from glyph.channel import Native
from glyph.forge import forge_run, reward
from glyph.policy import LoraPolicy
from glyph.probe import probe_grouped_robust
from glyph.seed import canonical_message
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CKPT = "glyph_ckpt"
FAST = bool(os.environ.get("FAST"))
SEEDED = os.environ.get("FROMSCRATCH") is None


def main():
    ch = Native()
    train = load_tasks(split="train")
    easy = [t for t in train if t["difficulty"] == 0]
    policy = LoraPolicy(MODEL, channel=ch, max_code=96 if FAST else 256)

    warm = easy if FAST else train
    if os.path.isdir(CKPT):
        policy.load(CKPT)
        print("resumed from", CKPT)
    elif SEEDED:
        policy.warmup_seeded(warm, rounds=30 if FAST else 12)  # SFT memorizes tiny mappings → needs epochs
        policy.save(CKPT)
    else:
        policy.warmup_builder(warm, rounds=1 if FAST else 3)
        policy.save(CKPT)

    # Diagnostic: isolate the two grounding halves (greedy, on the canonical msg).
    dec = sum(grade(_extract_code(policy.build(builder_prompt(
                ch.builder_text(canonical_message(t))))), t)["passed"] for t in easy)
    fid = sum(policy.sample(speaker_prompt(t, ch), 1, greedy=True)[0] == canonical_message(t)
              for t in easy)
    print(f"grounding: builder_decodes_canonical={dec}/{len(easy)}  "
          f"speaker_emits_canonical={fid}/{len(easy)}")

    pool = cycle(easy)

    def draw_group():
        t = next(pool)  # one task per group → honest within-task variance
        gs = 4 if FAST else 8
        out = []
        for _ in range(gs):
            msg = policy.sample(speaker_prompt(t, ch), 1)[0]
            code = policy.build(builder_prompt(ch.builder_text(msg)))
            out.append(float(reward(msg, code, t, ch, lam=0.0)[1]))  # pure pass
        return out

    gate = probe_grouped_robust(draw_group, runs=2 if FAST else 4,
                                groups=4 if FAST else 12)
    print("warmup gate:", json.dumps(gate, indent=2))
    if gate["pass_rate"]["mean"] < 0.3:
        print("STOP: agents can't solve even after warmup — grounding failed; "
              "the Builder can't decode the messages.")
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
