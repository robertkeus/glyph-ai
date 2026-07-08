"""Interactive walkthrough — see the forged language work, per task.

On Kaggle (GPU T4), in a notebook:

    !git clone https://github.com/robertkeus/glyph-ai && %cd glyph-ai
    !pip -q install peft
    from demo_interactive import setup, show, held
    setup()                    # ~15 min: load 3B + seed-warm both agents
    show(held()[0])            # full round-trip for a HELD-OUT task
    for t in held()[:5]: show(t)

Each show() prints: the English task → the Speaker's native glyph message →
the Builder's code (test 2), then the model's own symbol explanation and a COLD
model decoding the SAME message from that explanation alone (test 3) → pass/fail.
"""
from glyph.agents import (_extract_code, builder_prompt, grade, solve_solution,
                          speaker_prompt)
from glyph.channel import Native
from glyph.policy import LoraPolicy
from glyph.seed import PRIM_ORDER, canonical_message, prim_symbol
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
CH = Native()
P = None


def setup(model=MODEL, rounds=10):
    global P
    P = LoraPolicy(model, channel=CH)
    P.warmup_seeded(load_tasks(split="train"), rounds=rounds)
    print("ready.")


def held():
    return load_tasks(split="heldout")


def _operation(code):
    ops = [l.strip() for l in code.splitlines()
           if l.strip() and not l.strip().startswith("def ")
           and l.strip() not in ("r = list(xs)", "return r")]
    return ops[0] if ops else "pass"


def _explanation():
    return "\n".join(
        f"{prim_symbol(p)} : {_operation(_extract_code(P.build(builder_prompt(prim_symbol(p)))))}"
        for p in PRIM_ORDER)


def _cold_prompt(expl, ex, msg):
    return ("Each symbol maps to one Python statement on a list `r`:\n\n" + expl +
            f"\n\nExample — Symbols: {canonical_message(ex)}\n```python\n{solve_solution(ex)}\n```"
            "\n\nNow do the same: define `solve(xs)` starting with `r = list(xs)`, apply "
            "the statements for the symbols below IN ORDER (a `return` statement is the "
            f"final result; else end with `return r`). One ```python block.\nSymbols: {msg}")


def show(task):
    assert P is not None, "call setup() first"
    msg = P.sample(speaker_prompt(task, CH), 1, greedy=True)[0]
    code = _extract_code(P.build(builder_prompt(CH.builder_text(msg))))
    print("=" * 60)
    print("TASK   :", task["prompt"])
    print("NATIVE :", msg, f"  ({CH.bytes(msg)} bytes)")
    print("CODE   :", code.replace("\n", " | "))
    print("BUILDER PASSES:", grade(code, task)["passed"])

    train = load_tasks(split="train")
    ex = next(t for t in train if len(t["primitives"]) == 2)
    cold = _extract_code(P.build(_cold_prompt(_explanation(), ex, msg), cold=True))
    print("COLD DECODE (from explanation, model never learned the language):")
    print("       ", cold.replace("\n", " | "))
    print("COLD PASSES   :", grade(cold, task)["passed"])
