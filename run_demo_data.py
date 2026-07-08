"""Demo data run: same tasks through BOTH channels, honest byte comparison.

English A2A baseline = the plain base model (adapters off, chat template) as
Speaker and Builder — "normal agent to agent". Native = glyph message → trained
Builder. Emits events.json for demo/index.html + the headline reduction stat.
"""
import json

from glyph.agents import (_extract_code, builder_prompt, episode, grade,
                          speaker_prompt)
from glyph.channel import English, Native
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
    policy.warmup_seeded(train, rounds=10,
                         english=lambda t, rd: english(t, rd % HELDOUT_VARIANT),
                         translate=True)

    tasks = train[:6] + held  # mix; held-out are the honest ones
    events = []

    cold_gen = lambda p: policy.build(p, cold=True)  # base model, chat template
    for t in tasks:
        e = episode(t, cold_gen, channel=English())   # normal English A2A
        events.append(e)

    for t in tasks:                                   # native channel
        msg = canonical_message(t)
        code = _extract_code(policy.build(builder_prompt(ch.builder_text(msg))))
        v = grade(code, t)
        events.append({
            "task": t["id"], "split": t["split"], "channel": "native",
            "prompt": t["prompt"], "message": msg, "code": code,
            "passed": v["passed"], "message_bytes": ch.bytes(msg),
            "detail": v["detail"], "english": policy.translate(msg),
        })

    def stat(name):
        ev = [e for e in events if e["channel"] == name]
        ok = [e for e in ev if e["passed"]]
        bps = sum(e["message_bytes"] for e in ok) / len(ok) if ok else None
        return {"n": len(ev), "passed": len(ok), "bytes_per_solved": bps}

    s_en, s_gl = stat("english"), stat("native")
    red = (1 - s_gl["bytes_per_solved"] / s_en["bytes_per_solved"]) * 100 \
        if s_en["bytes_per_solved"] and s_gl["bytes_per_solved"] else None
    summary = {"english": s_en, "native": s_gl, "byte_reduction_pct": red}
    print("SUMMARY:", json.dumps(summary))
    with open("events.json", "w") as f:
        json.dump({"summary": summary, "events": events}, f, ensure_ascii=False, indent=1)
    print("events.json written:", len(events), "events")


if __name__ == "__main__":
    main()
