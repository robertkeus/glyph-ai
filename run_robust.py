"""A/B: does base-model normalization make free-English input robust?

Loads published adapters (no retrain). For each held-out task, feeds an UNSEEN
phrasing (variant 5) two ways — direct to Speaker vs normalize→Speaker — and
checks whether the emitted glyphs match the canonical message.
"""
import chat_app
from glyph.agents import speaker_prompt
from glyph.channel import Native
from glyph.paraphrase import HELDOUT_VARIANT, english
from glyph.policy import LoraPolicy
from glyph.seed import canonical_message
from glyph.tasks import load_tasks

CH = Native()
p = LoraPolicy("Qwen/Qwen2.5-Coder-3B-Instruct", channel=CH)
p.load(chat_app._resolve_adapters("/kaggle/input"))
chat_app.P = p

held = load_tasks(split="heldout")
direct = norm = 0
for t in held:
    free = english(t, HELDOUT_VARIANT)
    cm = canonical_message(t)
    d = p.sample(speaker_prompt({"prompt": free}, CH), 1, greedy=True)[0]
    n = p.sample(speaker_prompt({"prompt": chat_app._normalize(free)}, CH), 1, greedy=True)[0]
    direct += d == cm
    norm += n == cm
print(f"ROBUST(unseen phrasing → correct glyphs)  direct {direct}/{len(held)}  "
      f"normalized {norm}/{len(held)}")

print("=== truly free phrasings (eyeball) ===")
for free in ["get rid of the odd ones and double what's left",
             "biggest value after sorting big to small",
             "flip signs then put it backwards"]:
    print(f"\nUSER: {free}\n{chat_app._respond(free, None)}")
