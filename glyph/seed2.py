"""v2 symbol assignment + SFT training-pair emitter (Phase 2 input).

68 primitives → first 68 glyphs of the 256-glyph inventory, in registry order.
Emits speaker pairs (English → glyph message), builder pairs (glyphs → Python),
and translator pairs (glyphs → English) from bank2, cycling phrasing variants.
"""
import json

from glyph.channel import glyph
from glyph.lang import BY_KEY, NOUN, parse
from glyph.tasks import ROOT

SYM = {k: glyph(i) for i, k in enumerate(BY_KEY)}
DIGIT_BASE = 200                             # inventory 200-209 = operand digits 0-9
DIGIT = {str(d): glyph(DIGIT_BASE + d) for d in range(10)}
_UNDIGIT = {v: k for k, v in DIGIT.items()}
_UNSYM = {v: k for k, v in SYM.items()}
_PARA = json.loads((ROOT / "glyph" / "paraphrases.json").read_text())
# hold out the LAST 20% of each pool for the unseen-phrasing eval (never trained)
TRAIN_PARA = {k: v[:max(3, int(len(v) * 0.8))] for k, v in _PARA.items()}
EVAL_PARA = {k: v[max(3, int(len(v) * 0.8)):] or v[-1:] for k, v in _PARA.items()}


def message(keys):
    out = []
    for item in keys:
        k, a = parse(item)
        out.append(SYM[k] + ("".join(DIGIT[c] for c in str(a)) if a is not None else ""))
    return "".join(out)


def decode_items(msg):
    """Inverse of message(): glyph string -> chain items ('gtn:7' / 'evens')."""
    items = []
    for ch in msg:
        if ch in _UNDIGIT and items:
            items[-1] += ("" if ":" in items[-1] else ":") + _UNDIGIT[ch]
        elif ch in _UNSYM:
            items.append(_UNSYM[ch])
    return items


def phrased(keys, variant, pools=TRAIN_PARA):
    parts = []
    for i, item in enumerate(keys):
        k, a = parse(item)
        pool = pools.get(k) or BY_KEY[k].en   # slotted prims: registry templates until paraphrased
        t = pool[(variant * 7 + i) % len(pool)]
        parts.append(t.format(a=a) if a is not None else t)
    return f"Take {NOUN[BY_KEY[parse(keys[0])[0]].tin]}; " + ", then ".join(parts) + "."


def emit(path="tasks/sft2.jsonl", variants=6, bank="tasks/bank2.jsonl"):
    tasks = [json.loads(l) for l in (ROOT / bank).read_text().splitlines()]
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
    print(emit("tasks/sft3.jsonl", bank="tasks/bank3.jsonl"),
          "training pairs -> tasks/sft3.jsonl")
