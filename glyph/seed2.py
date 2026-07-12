"""v2 symbol assignment + SFT training-pair emitter (Phase 2 input).

68 primitives → first 68 glyphs of the 256-glyph inventory, in registry order.
Emits speaker pairs (English → glyph message), builder pairs (glyphs → Python),
and translator pairs (glyphs → English) from bank2, cycling phrasing variants.
"""
import json

from glyph.channel import glyph
from glyph.lang import BY_KEY, NOUN, _args, _fmt, parse
from glyph.tasks import ROOT

SYM = {k: glyph(i) for i, k in enumerate(BY_KEY)}
DIGIT_BASE = 200                             # inventory 200-209 = operand digits 0-9
DIGIT = {str(d): glyph(DIGIT_BASE + d) for d in range(10)}
VOCAB_BASE = 210                             # inventory 210+ = string-operand vocab (append-only)
VOCAB = ("#", "-", "_", "!", ",", ";", "a", "e", "o", "x",
         "name", "email", "age", "@", ".")
VGLYPH = {w: glyph(VOCAB_BASE + i) for i, w in enumerate(VOCAB)}
_UNDIGIT = {v: k for k, v in DIGIT.items()}
_UNVOCAB = {v: k for k, v in VGLYPH.items()}
_UNSYM = {v: k for k, v in SYM.items()}
_PARA = json.loads((ROOT / "glyph" / "paraphrases.json").read_text())
# hold out the LAST 20% of each pool for the unseen-phrasing eval (never trained)
TRAIN_PARA = {k: v[:max(3, int(len(v) * 0.8))] for k, v in _PARA.items()}
EVAL_PARA = {k: v[max(3, int(len(v) * 0.8)):] or v[-1:] for k, v in _PARA.items()}


def message(keys):
    out = []
    for item in keys:
        k, a = parse(item)
        out.append(SYM[k])
        for v in _args(a):
            out.append("".join(DIGIT[c] for c in str(v)) if isinstance(v, int) else VGLYPH[v])
    return "".join(out)


def decode_items(msg):
    """Inverse of message(): glyph string -> chain items ('gtn:7' / 'fminlen:name:3' /
    'evens'). A vocab glyph always starts a new operand; consecutive digits merge
    into one int (the registry never puts two int operands adjacent)."""
    items = []
    for ch in msg:
        if ch in _UNSYM:
            items.append(_UNSYM[ch])
        elif ch in _UNVOCAB and items:
            items[-1] += ":" + _UNVOCAB[ch]
        elif ch in _UNDIGIT and items:
            items[-1] += ("" if items[-1][-1].isdigit() else ":") + _UNDIGIT[ch]
    return items


def phrased(keys, variant, pools=TRAIN_PARA):
    parts = []
    for i, item in enumerate(keys):
        k, a = parse(item)
        pool = pools.get(k) or BY_KEY[k].en   # slotted prims: registry templates until paraphrased
        t = pool[(variant * 7 + i) % len(pool)]
        parts.append(_fmt(t, a) if a is not None else t)
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
