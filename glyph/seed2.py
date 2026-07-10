"""v2 symbol assignment + SFT training-pair emitter (Phase 2 input).

68 primitives → first 68 glyphs of the 256-glyph inventory, in registry order.
Emits speaker pairs (English → glyph message), builder pairs (glyphs → Python),
and translator pairs (glyphs → English) from bank2, cycling phrasing variants.
"""
import json

from glyph.channel import glyph
from glyph.lang import BY_KEY, english
from glyph.tasks import ROOT

SYM = {k: glyph(i) for i, k in enumerate(BY_KEY)}


def message(keys):
    return "".join(SYM[k] for k in keys)


def emit(path="tasks/sft2.jsonl", variants=3):
    tasks = [json.loads(l) for l in (ROOT / "tasks" / "bank2.jsonl").read_text().splitlines()]
    rows = []
    for t in tasks:
        if t["split"] != "train":
            continue
        keys, msg = t["primitives"], message(t["primitives"])
        for v in range(variants):
            rows.append({"kind": "speaker", "in": english(keys, v), "out": msg})
        rows.append({"kind": "builder", "in": msg, "out": t["solution"].replace(t["entry_point"], "solve", 1)})
        rows.append({"kind": "translator", "in": msg, "out": t["prompt"]})
    (ROOT / path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return len(rows)


if __name__ == "__main__":
    print(emit(), "training pairs -> tasks/sft2.jsonl")
