"""Production: train once → save adapters → verify load path + chat boots.

Kaggle output `/kaggle/working/glyph_adapters` becomes the reusable checkpoint
(publish as a Kaggle Dataset; then chat_app.launch(adapters=...) starts in seconds).
Also boot-tests the chat headless so a live launch can't surprise you.
"""
import chat_app
from glyph.agents import _extract_code, builder_prompt, grade, speaker_prompt
from glyph.channel import Native
from glyph.paraphrase import HELDOUT_VARIANT, english
from glyph.policy import LoraPolicy
from glyph.seed import canonical_message
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTERS = "/kaggle/working/glyph_adapters"
CH = Native()

p = LoraPolicy(MODEL, channel=CH)
p.warmup_seeded(load_tasks(split="train"), rounds=10,
                english=lambda t, rd: english(t, rd % HELDOUT_VARIANT), translate=True)
p.save(ADAPTERS)
print("SAVED adapters ->", ADAPTERS)

# Production load path: a FRESH policy loads the saved adapters (no warmup) — this
# is exactly what chat_app.launch(adapters=...) does.
q = LoraPolicy(MODEL, channel=CH)
q.load(ADAPTERS)
chat_app.P = q

# metrics on the LOADED adapters (also proves save/load fidelity)
train, held = load_tasks(split="train"), load_tasks(split="heldout")
t2 = sum(grade(_extract_code(q.build(builder_prompt(
    CH.builder_text(canonical_message(t))))), t)["passed"] for t in held)
rob = sum(q.sample(speaker_prompt(dict(t, prompt=english(t, HELDOUT_VARIANT)), CH),
                   1, greedy=True)[0] == canonical_message(t) for t in train)
print(f"METRICS  test2(compositional) {t2}/{len(held)}  speaker_robust(unseen phrasing) {rob}/{len(train)}")

print("=== BOOT TEST (loaded, not warmed) ===")
for m in ["keep the even numbers, then double each",
          "sort descending, then return the maximum",
          "reverse the order, then drop duplicates",
          "三下"]:
    print(f"\nUSER: {m}\n{chat_app._respond(m, None)}")
print("\nBOOT OK")
