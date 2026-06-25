"""Kaggle entrypoint — real-model English baseline (PLAN Phase 0).

Run on Kaggle Notebooks (GPU T4 x2). Produces the bytes-per-solved-task baseline
that the native channel must later beat (PLAN test 1). Inference only — no
checkpointing needed; safe under the 12h session cap.

Setup cell:
    !pip -q install transformers accelerate torch
    !git clone <repo> && cd glyph-ai
Then: !python kaggle_entry.py

Swap MODEL to the 3B only if 1.5B's baseline is too weak — smaller iterates faster
and is the better de-risking choice for cold-start.
"""
import json

from glyph.agents import run
from glyph.events import write_events
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def make_generate(model_name=MODEL, max_new_tokens=256):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    def generate(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    return generate


def main():
    generate = make_generate()
    events, summary = run(load_tasks(), generate)  # channel="english" (baseline)
    write_events(events)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
