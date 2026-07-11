"""Glyph v2 Space — the REAL finetuned model (ZeroGPU).

Pipeline per message: English → speaker (LoRA) → glyph message → builder (LoRA)
→ Python → executed. Glyph input goes straight to builder + translator.
"""
import os
import subprocess
import sys
import tempfile

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER = "robertkeus/glyph-adapters"
DEMO = [3, -1, 2, 2, -5]

PROMPT = {
    "speaker": "Encode this task as glyph symbols.\nTask: {x}\nSymbols:",
    "builder": "Write Python for this glyph message.\nSymbols: {x}\nCode:",
    "translator": "Translate this glyph message into English.\nSymbols: {x}\nEnglish:",
}

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="cuda")
model = PeftModel.from_pretrained(model, ADAPTER, subfolder="v2")
model.eval()

# wire glyphs are CJK (0x4E00 block); display remap to syllabics (0x1400) = alien look
_A, _C = 0x1400, 0x4E00
alien = lambda s: "".join(chr(_A + ord(c) - _C) if 0x4E00 <= ord(c) <= 0x9FFF else c for c in s)
unalien = lambda s: "".join(chr(_C + ord(c) - _A) if 0x1400 <= ord(c) <= 0x167F else c for c in s)
is_glyphs = lambda s: all(0x4E00 <= ord(c) <= 0x9FFF for c in s)


@spaces.GPU
def gen(prompt, max_new):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_code(code):
    src = f"{code}\n\nprint(solve({DEMO}))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], capture_output=True,
                           text=True, timeout=5)
        return p.stdout.strip() if p.returncode == 0 else "error: " + p.stderr.strip()[-120:]
    except Exception as e:
        return f"error: {type(e).__name__}"
    finally:
        os.unlink(path)


def respond(msg, _history):
    msg = (msg or "").strip()[:300]
    if not msg:
        return "Ask a list/string/record task — I answer in my glyph language, then code."
    try:
        raw = unalien(msg)
        if is_glyphs(raw):                       # user typed glyphs directly
            glyphs, note = raw, "you spoke glyph — decoding directly"
        else:                                    # REAL trained speaker encodes
            glyphs = gen(PROMPT["speaker"].format(x=msg), 24)
            note = "encoded by the finetuned speaker"
        if not glyphs or not is_glyphs(glyphs):
            return f"Speaker produced no valid glyphs for that (got: `{glyphs[:40]}`) — try an in-domain task."
        english = gen(PROMPT["translator"].format(x=glyphs), 60)
        code = gen(PROMPT["builder"].format(x=glyphs), 200)
        return (f"**glyph message** ({note}): {alien(glyphs)}  ·  {2*len(glyphs)} bytes\n\n"
                f"**model reads it as:** {english}\n\n"
                f"```python\n{code}\n```\n\n`solve({DEMO})` → `{run_code(code)}`")
    except Exception as e:
        return f"⚠️ {type(e).__name__}: {str(e)[:150]}"


EX = ["keep the positive numbers, square each, then return their sum",
      "drop duplicates, sort descending",
      "keep multiples of three, then count them",
      "take the absolute value of each, sort ascending, return the maximum (0 if empty)"]

gr.ChatInterface(respond, examples=EX, title="Glyph v2 — live finetuned model",
                 description="Every reply is the real trained pipeline: speaker → glyphs "
                             "→ builder → executed Python. (builder 99.5% held-out)").launch()
