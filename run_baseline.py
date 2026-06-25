"""English two-agent baseline — real model (PLAN Phase 0, test 1).

Runs anywhere: CUDA (Kaggle T4) → bf16, Apple MPS → fp16, else CPU → fp32.
Produces bytes-per-solved-task that the forged native channel must later beat.
Inference only — no checkpoint needed.

    pip install -r requirements.txt
    python run_baseline.py            # → events.json + pass_rate / bytes_per_solved
"""
import json

from glyph.agents import run
from glyph.events import write_events
from glyph.tasks import load_tasks

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _device_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def make_generate(model_name=MODEL, max_new_tokens=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = _device_dtype()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    def generate(prompt: str) -> str:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][ids.input_ids.shape[1]:].cpu()  # decode on CPU (CUDA/MPS safe)
        return tok.decode(gen, skip_special_tokens=True)

    return generate


def main():
    generate = make_generate()
    events, summary = run(load_tasks(), generate)  # English channel = baseline
    write_events(events)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
