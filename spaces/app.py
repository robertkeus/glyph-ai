"""Glyph — self-contained Hugging Face Space (no training, base model only).

English → base-model intent-extraction → glyph message → deterministic decode →
executed Python. Set GLYPH_MODEL (default 1.5B for CPU responsiveness).
"""
import os
import re
import subprocess
import sys
import tempfile

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

MODEL = os.environ.get("GLYPH_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
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


_HINTS = tuple(w for p in PRIMS for w in [p[0]]) + (
    "list", "number", "even", "odd", "positive", "sort", "double", "square",
    "sum", "max", "reverse", "duplicate", "negate", "absolute", "count")


def chat(history, msg):
    """Normal multi-turn conversation via the base model. Accepts Gradio history in
    either format: message-dicts ({role,content}) or (user, assistant) tuples."""
    conv = []
    for h in history or []:
        if isinstance(h, dict) and h.get("content"):
            conv.append({"role": h.get("role", "user"), "content": h["content"]})
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            u, a = h
            if u:
                conv.append({"role": "user", "content": u})
            if a:
                conv.append({"role": "assistant", "content": a})
    conv.append({"role": "user", "content": msg})
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(DEV)
    out = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def respond(msg, history):
    msg = (msg or "").strip()
    if not msg:
        return "Ask me anything — and for list-of-numbers tasks I'll answer in my glyph language + code."
    # glyph mode only when it looks like a list-operation request
    is_glyph = all(c in BY_GLYPH for c in msg) or any(h in msg.lower() for h in _HINTS)
    if is_glyph:
        keys = [BY_GLYPH[c][0] for c in msg] if all(c in BY_GLYPH for c in msg) else intent(msg)
        if keys:
            al = lambda g: chr(0x1400 + (ord(g) - 0x4e00))          # alien display glyph
            glyphs = "".join(al(BY_KEY[k][1]) for k in keys)
            code = to_code(keys)
            steps = "\n".join(f"{i+1}. *{BY_KEY[k][3]}*  →  {al(BY_KEY[k][1])}  →  `{BY_KEY[k][2]}`"
                              for i, k in enumerate(keys))
            return (f"**🧠 Reasoning — I break the request into operations, encode each as one "
                    f"glyph, and compose the code:**\n{steps}\n\n"
                    f"**Message the agents actually send:** {glyphs} — "
                    f"**{len(keys) * 2} bytes** vs {len(msg)} in English "
                    f"({round((1 - len(keys)*2/max(len(msg),1))*100)}% smaller)\n\n"
                    f"```python\n{code}\n```\n\nRunning it: `solve({DEMO})` → `{run(code)}`")
    return chat(history or [], msg)          # otherwise, just converse


EXAMPLES = ["Hi, what can you do?", "keep positives, square them, then sum",
            "sort big to small and give the top one", "remove duplicates then add everything up",
            "explain your glyph language"]

gr.ChatInterface(
    respond, type="messages", examples=EXAMPLES, title="Glyph",
    description="A chat model that speaks its own compact glyph language for list-of-number "
                "tasks — answering in symbols, then real Python. Chat normally too.",
).launch()
