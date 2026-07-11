"""v2 symbol assignment + SFT training-pair emitter (Phase 2 input).

68 primitives → first 68 glyphs of the 256-glyph inventory, in registry order.
Emits speaker pairs (English → glyph message), builder pairs (glyphs → Python),
and translator pairs (glyphs → English) from bank2, cycling phrasing variants.
"""
import json

from glyph.channel import glyph
from glyph.lang import BY_KEY, NOUN
from glyph.tasks import ROOT

SYM = {k: glyph(i) for i, k in enumerate(BY_KEY)}
_PARA = json.loads((ROOT / "glyph" / "paraphrases.json").read_text())
# hold out the LAST 20% of each pool for the unseen-phrasing eval (never trained)
TRAIN_PARA = {k: v[:max(3, int(len(v) * 0.8))] for k, v in _PARA.items()}
EVAL_PARA = {k: v[max(3, int(len(v) * 0.8)):] or v[-1:] for k, v in _PARA.items()}


def message(keys):
    return "".join(SYM[k] for k in keys)


def phrased(keys, variant, pools=TRAIN_PARA):
    parts = [pools[k][(variant * 7 + i) % len(pools[k])] for i, k in enumerate(keys)]
    return f"Take {NOUN[BY_KEY[keys[0]].tin]}; " + ", then ".join(parts) + "."


def emit(path="tasks/sft2.jsonl", variants=6):
    tasks = [json.loads(l) for l in (ROOT / "tasks" / "bank2.jsonl").read_text().splitlines()]
    rows = []
    for t in tasks:
        if t["split"] != "train":
            continue
        keys, msg = t["primitives"], message(t["primitives"])
        seen = set()
        for v in range(variants):
            p = phrased(keys, v)
            if p not in seen:                        # pools vary in size; dedupe
                seen.add(p)
                rows.append({"kind": "speaker", "in": p, "out": msg})
        rows.append({"kind": "builder", "in": msg, "out": t["solution"].replace(t["entry_point"], "solve", 1)})
        rows.append({"kind": "translator", "in": msg, "out": t["prompt"]})
    (ROOT / path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return len(rows)


if __name__ == "__main__":
    print(emit(), "training pairs -> tasks/sft2.jsonl")
