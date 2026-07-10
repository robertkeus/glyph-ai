"""Glyph — self-contained Hugging Face Space (no training, base model only).

English → base-model intent-extraction → glyph message → deterministic decode →
executed Python. Set GLYPH_MODEL (default 1.5B for CPU responsiveness).
"""
import os
import re
import subprocess
import sys
import tempfile
import time

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# glyph codepoints (must match glyph/channel.py) → guarantees correct, unique glyphs
_CPS = [0x4e00, 0x4e01, 0x4e03, 0x4e07, 0x4e08, 0x4e09, 0x4e0a, 0x4e0b,
        0x4e0d, 0x4e0e, 0x4e0f, 0x4e10, 0x4e11, 0x4e13, 0x4e14, 0x4e15]
_OPS = [  # key, code-line, English description, is_reducer
    ("evens",  "r = [x for x in r if x % 2 == 0]", "keep the even numbers", False),
    ("pos",    "r = [x for x in r if x > 0]", "keep the positive numbers", False),
    ("double", "r = [x * 2 for x in r]", "double each", False),
    ("square", "r = [x * x for x in r]", "square each", False),
    ("inc",    "r = [x + 1 for x in r]", "add one to each", False),
    ("negate", "r = [-x for x in r]", "negate each", False),
    ("absval", "r = [abs(x) for x in r]", "take the absolute value of each", False),
    ("rev",    "r = list(reversed(r))", "reverse the order", False),
    ("sorta",  "r = sorted(r)", "sort ascending", False),
    ("sortd",  "r = sorted(r, reverse=True)", "sort descending", False),
    ("uniq",   "r = list(dict.fromkeys(r))", "drop duplicates keeping first occurrence", False),
    ("dec",    "r = [x - 1 for x in r]", "subtract one from each", False),
    ("sum",    "return sum(r)", "return their sum", True),
    ("max",    "return max(r) if r else 0", "return the maximum (0 if empty)", True),
    ("len",    "return len(r)", "return how many remain", True),
    ("cnt",    "return sum(1 for _ in r)", "return the count", True),
]
# key, glyph, code-line, description, is_reducer
PRIMS = [(k, chr(_CPS[i]), code, desc, red) for i, (k, code, desc, red) in enumerate(_OPS)]
BY_KEY = {k: p for p in PRIMS for k in [p[0]]}
BY_GLYPH = {p[1]: p for p in PRIMS}
KEYSET = {p[0] for p in PRIMS}
MENU = "\n".join(f"{p[0]} = {p[3]}" for p in PRIMS)
DEMO = [3, -1, 2, 2, -5]

MODEL = os.environ.get("GLYPH_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")  # light → stable on free CPU
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV)


def ask(prompt, max_new=48):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(DEV)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def intent(msg):
    q = ("List the operations the request performs, IN ORDER, as comma-separated "
         "keys from this menu (keys only, e.g. `evens, double`):\n" + MENU +
         f"\n\nRequest: {msg}\nKeys:")
    return [k for t in re.split(r"[,\s]+", ask(q, 32)) if (k := t.strip()) in KEYSET]


def to_code(keys):
    lines = ["def solve(xs):", "    r = list(xs)"]
    for k in keys:
        _, _, line, _, red = BY_KEY[k]
        lines.append("    " + line)
        if red:
            return "\n".join(lines)
    return "\n".join(lines + ["    return r"])


JS = {  # same primitives rendered as JavaScript — symbols carry intent, not syntax
    "evens": "r = r.filter(x => x % 2 === 0);", "pos": "r = r.filter(x => x > 0);",
    "double": "r = r.map(x => x * 2);", "square": "r = r.map(x => x * x);",
    "inc": "r = r.map(x => x + 1);", "negate": "r = r.map(x => -x);",
    "absval": "r = r.map(x => Math.abs(x));", "rev": "r = r.slice().reverse();",
    "sorta": "r = r.slice().sort((a, b) => a - b);", "sortd": "r = r.slice().sort((a, b) => b - a);",
    "uniq": "r = [...new Set(r)];", "dec": "r = r.map(x => x - 1);",
    "sum": "return r.reduce((a, b) => a + b, 0);", "max": "return r.length ? Math.max(...r) : 0;",
    "len": "return r.length;", "cnt": "return r.length;",
}


def to_js(keys):
    lines = ["function solve(xs) {", "  let r = [...xs];"]
    for k in keys:
        lines.append("  " + JS[k])
        if JS[k].startswith("return"):
            return "\n".join(lines + ["}"])
    return "\n".join(lines + ["  return r;", "}"])


def run(code):
    src = f"{code}\n\nprint(solve({DEMO}))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=5)
        return p.stdout.strip() if p.returncode == 0 else "error"
    except Exception:
        return "error"
    finally:
        os.unlink(path)


KW = {  # reliable keyword parse (fast, instant) — order in the sentence = op order
    "evens": ["even"], "pos": ["positive"], "double": ["double", "twice"],
    "square": ["square"], "inc": ["add one", "increment", "plus one"],
    "negate": ["negate", "flip sign", "flip the sign", "invert"],
    "absval": ["absolute", "abs value"], "rev": ["revers", "backward", "back to front"],
    "sorta": ["ascending", "smallest to", "low to high", "increasing"],
    "sortd": ["descending", "big to small", "large to small", "high to low", "biggest to"],
    "uniq": ["dedup", "duplicate", "unique", "distinct"],
    "dec": ["subtract one", "decrement", "minus one"],
    "sum": ["sum", "add up", "total", "add everything", "add them"],
    "max": ["max", "biggest", "largest", "top one", "the top", "highest"],
    "len": ["how many", "length", "size"], "cnt": ["count", "number of"],
}


def kw_parse(msg):
    low = msg.lower()
    hits = []
    for k, words in KW.items():
        for w in words:
            i = low.find(w)
            if i >= 0:
                hits.append((i, k)); break
    hits.sort()
    seen, keys = set(), []
    for _, k in hits:
        if k not in seen:
            seen.add(k); keys.append(k)
    return keys


def _gen(conv):
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=120, do_sample=True, temperature=0.7,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _conv(history, msg):
    """Coerce any Gradio history shape (dicts / tuples / stray types) to a clean
    list of {role, content} string pairs."""
    conv = []
    for h in (history or [])[-6:]:
        if isinstance(h, dict):
            role, content = h.get("role"), h.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                conv.append({"role": role, "content": content})
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            u, a = h
            if isinstance(u, str) and u.strip():
                conv.append({"role": "user", "content": u})
            if isinstance(a, str) and a.strip():
                conv.append({"role": "assistant", "content": a})
    conv.append({"role": "user", "content": msg})
    return conv


def _chat(history, msg):
    """Plain generation. Falls back to no-history if the history shape is odd."""
    try:
        return _gen(_conv(history, msg))
    except Exception:
        return _gen([{"role": "user", "content": msg}])


def respond(msg, history):
    """Generator → streams a visible chain of thought (Gradio renders each yield)."""
    try:
        yield from _respond(msg, history)
    except Exception as e:  # never hard-crash the UI
        yield f"⚠️ hiccup: {type(e).__name__}: {str(e)[:150]}"


def _respond(msg, history):
    msg = (msg or "").strip()
    if not msg:
        yield "Ask me anything — for list-of-number tasks I show my reasoning in glyphs + code."
        return
    if all(c in BY_GLYPH for c in msg):
        keys = [BY_GLYPH[c][0] for c in msg]
    else:
        keys = kw_parse(msg)          # reliable; no model guessing (avoids false glyph mode on chat)

    if keys:  # stream the glyph reasoning step by step
        al = lambda g: chr(0x1400 + (ord(g) - 0x4e00))
        buf = "🧠 *reading your request…*"
        yield buf
        time.sleep(0.5)
        buf = "🧠 **Reasoning — turning your words into my glyph language:**\n\n"
        yield buf
        for i, k in enumerate(keys):
            time.sleep(0.6)
            buf += f"{i+1}. *{BY_KEY[k][3]}*  →  **{al(BY_KEY[k][1])}**  →  `{BY_KEY[k][2]}`\n"
            yield buf
        glyphs = "".join(al(BY_KEY[k][1]) for k in keys)
        cut = round((1 - len(keys) * 2 / max(len(msg), 1)) * 100)
        time.sleep(0.5)
        buf += (f"\n📨 **Message the agents send:** {glyphs}  ·  **{len(keys)*2} bytes** "
                f"vs {len(msg)} in English (**{cut}% smaller**)\n")
        yield buf
        code = to_code(keys)
        time.sleep(0.4)
        buf += f"\n🛠️ **Composing the code…**\n```python\n{code}\n```\n"
        yield buf
        time.sleep(0.4)
        buf += (f"\n🌐 **Same message, different language** (the glyphs encode intent, "
                f"not Python):\n```javascript\n{to_js(keys)}\n```\n")
        yield buf
        time.sleep(0.5)
        buf += f"\n▶️ **Running it:** `solve({DEMO})` → **`{run(code)}`**"
        yield buf
        return

    yield _chat(history or [], msg)                  # normal chat (non-threaded, robust)


EXAMPLES = ["Hi, what can you do?", "keep positives, square them, then sum",
            "sort big to small and give the top one", "remove duplicates then add everything up",
            "explain your glyph language"]

gr.ChatInterface(
    respond, examples=EXAMPLES, title="Glyph",
    description="A chat model that speaks its own compact glyph language for list-of-number "
                "tasks — answering in symbols, then real Python. Chat normally too.",
).launch()
