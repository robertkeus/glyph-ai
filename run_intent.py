"""A/B/C on free-English → correct glyphs: Speaker vs normalize vs intent-extract.

Loads published adapters (no retrain). Tests unseen phrasings (variant 5) on all
held-out tasks + a few genuinely loose prompts.
"""
import chat_app
from glyph.agents import speaker_prompt
from glyph.channel import Native
from glyph.paraphrase import HELDOUT_VARIANT, english
from glyph.policy import LoraPolicy
from glyph.seed import PRIM_ORDER, canonical_message, prim_symbol
from glyph.tasks import load_tasks

CH = Native()
p = LoraPolicy("Qwen/Qwen2.5-Coder-3B-Instruct", channel=CH)
p.load(chat_app._resolve_adapters("/kaggle/input"))
chat_app.P = p

held = load_tasks(split="heldout")
direct = norm = intent = 0
for t in held:
    free = english(t, HELDOUT_VARIANT)
    cm = canonical_message(t)
    direct += p.sample(speaker_prompt({"prompt": free}, CH), 1, greedy=True)[0] == cm
    norm += p.sample(speaker_prompt({"prompt": chat_app._normalize(free)}, CH), 1, greedy=True)[0] == cm
    intent += "".join(prim_symbol(k) for k in chat_app._intent(free)) == cm
n = len(held)
print(f"FREE-ENGLISH → correct glyphs (n={n}):  speaker {direct}  normalize {norm}  intent {intent}")

print("=== genuinely free (eyeball) ===")
for free in ["get rid of the odd ones and double what's left",
             "biggest value after sorting big to small",
             "flip signs then put it backwards",
             "dedupe then add up everything"]:
    print(f"{free!r}\n  intent={chat_app._intent(free)}")
