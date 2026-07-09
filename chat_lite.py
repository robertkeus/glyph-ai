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


def respond(msg, _history):
    msg = (msg or "").strip()[:200]
    if not msg:
        return "Ask for a list-of-integers operation, e.g. *keep positives, square them, then sum*."
    try:
        if all(CH.is_symbol(c) for c in msg):
            glyphs = msg
        else:
            glyphs = "".join(prim_symbol(k) for k in _intent(msg))
        if not glyphs:
            return "Couldn't map that to the 16 operations — try phrasing like the examples."
        code = decode(glyphs)
        r = run_tests(code, f"print(solve({DEMO}))")
        ran = r["stdout"] if r["passed"] else "error"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    return (f"**glyph** `{glyphs}`  ·  {CH.bytes(glyphs)} bytes\n\n"
            f"```python\n{code}\n```\n\n`solve({DEMO})` → `{ran}`")


if __name__ == "__main__":
    ex = [t["prompt"].split("; ", 1)[1].rstrip(".") for t in load_tasks(split="heldout")[:6]]
    gr.ChatInterface(respond, examples=ex, title="Glyph — ask in English, it answers in "
                     "its own symbol language, then working code").launch(share=True)
