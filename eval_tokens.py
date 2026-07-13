"""Output-token eval: glyph v4 pipeline vs vanilla Qwen at EQUAL PASS RATE.

Second headline beside the 98.8% wire savings: how many output tokens does
each system spend to produce a PASSING solution for the same held-out task?
  vanilla — base model (adapter disabled), instruct-style English prompt
  glyph   — v4 speaker (English -> glyphs) + builder (glyphs -> code)
Honest split (CaveGemma lesson): vanilla chattiness (markdown/prose) vs the
code itself are reported separately (total vs code-only tokens). Savings are
computed on the BOTH-SOLVED subset so correctness is held equal.

    GLYPH_V=v4 EVAL_N=150 python eval_tokens.py   # Kaggle T4, ~1.5h
"""
import json
import os
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from glyph.agents import grade
from glyph.lang import NOUN
from glyph.seed2 import message
from glyph.tasks import ROOT

MODEL = os.environ.get("GLYPH_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")
ADAPTER = "robertkeus/glyph-adapters"
SUB = os.environ.get("GLYPH_V", "v4")
N = int(os.environ.get("EVAL_N", 150))
BANK = os.environ.get("BANK", "bank3.jsonl")

PROMPT = {
    "speaker": "Encode this task as glyph symbols.\nTask: {x}\nSymbols:",
    "builder": "Write Python for this glyph message.\nSymbols: {x}\nCode:",
}

random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
model = PeftModel.from_pretrained(base, ADAPTER, subfolder=SUB).to(dev)
model.eval()


@torch.no_grad()
def gen(prompt, max_new, chat=False, vanilla=False):
    """Returns (text, generated_token_count)."""
    if chat:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(dev)
    else:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    ids = enc["input_ids"]
    ctx = model.disable_adapter() if vanilla else torch.no_grad()
    with ctx:
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new = out[0][ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), int(new.shape[0])


def extract_code(text):
    if "```" in text:
        block = text.split("```", 2)[1]
        if block.startswith(("python", "py")):
            block = block.split("\n", 1)[-1]
        return block
    return text


def main():
    tasks = [json.loads(l) for l in (ROOT / "tasks" / BANK).read_text().splitlines()]
    held = [t for t in tasks if t["split"] == "heldout_comp"]
    sample = random.sample(held, min(N, len(held)))
    rows = []
    for i, t in enumerate(sample):
        noun = NOUN[t["type"]]
        # vanilla arm: base model, normal instruct usage
        vp = (f"Write a Python function solve(xs) that takes {noun} and does the "
              f"following: {t['prompt']}\nReturn only the code.")
        vtext, vtotal = gen(vp, 300, chat=True, vanilla=True)
        vcode = extract_code(vtext)
        vpass = grade(vcode, t)["passed"]
        # glyph arm: v4 speaker -> builder
        glyphs, stok = gen(PROMPT["speaker"].format(x=t["prompt"]), 24)
        gcode, gtok = gen(PROMPT["builder"].format(x=glyphs), 200)
        gpass = grade(gcode, t)["passed"]
        rows.append({"id": t["id"], "vpass": vpass, "vtotal": vtotal,
                     "vcode": len(tok(vcode, add_special_tokens=False)["input_ids"]),
                     "gpass": gpass, "gtok": gtok, "stok": stok,
                     "wire_glyphs": glyphs == message(t["primitives"])})
        if i % 20 == 0:
            print(f"{i}/{len(sample)}", flush=True)

    both = [r for r in rows if r["vpass"] and r["gpass"]]
    mean = lambda k, rs: round(sum(r[k] for r in rs) / max(len(rs), 1), 1)
    res = {
        "n": len(rows), "pass_vanilla": round(sum(r["vpass"] for r in rows) / len(rows), 3),
        "pass_glyph": round(sum(r["gpass"] for r in rows) / len(rows), 3),
        "both_solved": len(both),
        "vanilla_total_tokens": mean("vtotal", both),
        "vanilla_code_tokens": mean("vcode", both),
        "glyph_builder_tokens": mean("gtok", both),
        "glyph_speaker_tokens": mean("stok", both),
        "speaker_exact_wire": round(sum(r["wire_glyphs"] for r in rows) / len(rows), 3),
    }
    if both:
        res["savings_vs_total_pct"] = round(100 * (1 - res["glyph_builder_tokens"] / res["vanilla_total_tokens"]), 1)
        res["savings_vs_code_pct"] = round(100 * (1 - res["glyph_builder_tokens"] / res["vanilla_code_tokens"]), 1)
    print("RESULT " + json.dumps(res), flush=True)
    (ROOT / "eval_tokens_result.json").write_text(json.dumps({"rows": rows, "summary": res}, indent=1))


if __name__ == "__main__":
    main()
