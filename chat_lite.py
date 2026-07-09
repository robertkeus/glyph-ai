"""Lite chat — base model only, NO peft/adapters/torchao (avoids Kaggle env issues
and OOM). English → base-model intent-extraction → glyphs → reference decode → code.
Launches Gradio with a public share URL.

    !git clone https://github.com/robertkeus/glyph-ai && cd glyph-ai
    !pip -q install gradio
    !python chat_lite.py        # prints a public https://xxxx.gradio.live URL
"""
import os
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from glyph.channel import Native
from glyph.decode import decode
from glyph.paraphrase import V
from glyph.seed import PRIM_ORDER, prim_symbol
from glyph.tasks import load_tasks
from glyph.verifier import run_tests

MODEL = os.environ.get("GLYPH_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")
CH = Native()
DEMO = [3, -1, 2, 2, -5]
_KEYS = "\n".join(f"{p} = {V[p][0]}" for p in PRIM_ORDER)
_PSET = set(PRIM_ORDER)

_dev = "cuda" if torch.cuda.is_available() else "cpu"
_tok = AutoTokenizer.from_pretrained(MODEL)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if _dev == "cuda" else torch.float32).to(_dev)


def _ask(prompt, max_new=48):
    text = _tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True)
    enc = _tok(text, return_tensors="pt").to(_dev)
    out = _model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=_tok.eos_token_id)
    return _tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _intent(msg):
    q = ("List the operations the request performs, IN ORDER, as comma-separated "
         "keys from this menu (keys only, e.g. `evens, double`):\n" + _KEYS +
         f"\n\nRequest: {msg}\nKeys:")
    return [k for t in re.split(r"[,\s]+", _ask(q, 32)) if (k := t.strip()) in _PSET]


_HINTS = tuple(PRIM_ORDER) + ("list", "number", "even", "odd", "positive", "sort",
    "double", "square", "sum", "max", "reverse", "duplicate", "negate", "absolute", "count")


def _chat(history, msg):
    conv = []
    for u, a in history or []:
        conv += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    conv.append({"role": "user", "content": msg})
    text = _tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    enc = _tok(text, return_tensors="pt").to(_dev)
    out = _model.generate(**enc, max_new_tokens=220, do_sample=True, temperature=0.7,
                          pad_token_id=_tok.eos_token_id)
    return _tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def respond(msg, history):
    msg = (msg or "").strip()
    if not msg:
        return "Ask me anything — for list-of-number tasks I answer in glyphs + code."
    try:
        glyphed = all(CH.is_symbol(c) for c in msg)
        if glyphed or any(h in msg.lower() for h in _HINTS):     # list-op → glyph mode
            glyphs = msg if glyphed else "".join(prim_symbol(k) for k in _intent(msg))
            if glyphs:
                code = decode(glyphs)
                r = run_tests(code, f"print(solve({DEMO}))")
                ran = r["stdout"] if r["passed"] else "error"
                return (f"In my language: **{glyphs}**  ·  {CH.bytes(glyphs)} bytes "
                        f"(vs {len(msg)} in English)\n\n```python\n{code}\n```\n\n"
                        f"`solve({DEMO})` → `{ran}`")
        return _chat(history, msg)                                 # else → converse
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


if __name__ == "__main__":
    ex = ["Hi, what can you do?", "keep positives, square them, then sum",
          "sort big to small and give the top one", "remove duplicates then add everything up"]
    gr.ChatInterface(respond, examples=ex, title="Glyph",
                     description="Chat normally; for list-of-number tasks I reply in my "
                     "glyph language, then working code.").launch(share=True)
