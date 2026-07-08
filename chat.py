"""Talk to Glyph — REPL for a Kaggle interactive notebook (GPU T4).

    !git clone https://github.com/robertkeus/glyph-ai && %cd glyph-ai
    !pip -q install peft
    from chat import start, say
    start()                      # ~15 min warmup, once
    say("三下")                  # glyphs in → code out (Builder decodes)
    say("keep the even numbers, then double each")   # English in → glyphs + code

Scope: the model only speaks list-operations (16 symbols, see say("help")).
English must stay close to the task-bank phrasing — it's a trained encoder,
not a general chatbot.
"""
from glyph.agents import _extract_code, builder_prompt, speaker_prompt
from glyph.channel import Native
from glyph.policy import LoraPolicy
from glyph.seed import PRIM_ORDER, prim_symbol
from glyph.taskgen import CHAIN, REDUCE
from glyph.tasks import load_tasks
from glyph.verifier import run_tests

CH = Native()
P = None


def start(model="Qwen/Qwen2.5-Coder-3B-Instruct", rounds=10):
    global P
    P = LoraPolicy(model, channel=CH)
    P.warmup_seeded(load_tasks(split="train"), rounds=rounds)
    print("ready — say('三下') or say('keep the even numbers, then double each')")


def _vocab():
    return "\n".join(f"  {prim_symbol(p)}  {(CHAIN.get(p) or REDUCE.get(p))[2]}"
                     for p in PRIM_ORDER)


def say(text: str, demo_input=(3, -1, 2, 2, -5)):
    if text.strip() == "help":
        print(_vocab())
        return
    is_native = all(CH.is_symbol(c) for c in text.strip())
    if is_native:
        msg = text.strip()
    else:  # English → Speaker encodes
        task = {"prompt": f"Take a list of integers; {text.strip().rstrip('.')}."}
        msg = P.sample(speaker_prompt(task, CH), 1, greedy=True)[0]
        print("GLYPH :", msg, f"({CH.bytes(msg)} bytes)")
    code = _extract_code(P.build(builder_prompt(CH.builder_text(msg))))
    print("CODE  :", "\n        ".join(code.splitlines()))
    probe = f"print(solve(list({list(demo_input)!r})))"
    r = run_tests(code, probe)
    out = r["stdout"] if r["passed"] else "error: " + r["detail"][:80]
    print("RUN   :", f"solve({list(demo_input)}) = {out}")
