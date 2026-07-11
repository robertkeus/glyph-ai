"""Phase-2 finetune: batched multi-task SFT over tasks/sft2.jsonl (40k pairs).

ONE LoRA adapter, three roles by instruction (product = one model):
  speaker    "Encode this task as glyph symbols."
  builder    "Write Python for this glyph message."
  translator "Translate this glyph message into English."

Eval each epoch (greedy): speaker on UNSEEN phrasings (EVAL_PARA pool), builder
decode on held-out compositional tasks (run the hidden tests), zero-shot symbols.
Saves best adapter. SMOKE=1 → small caps, 1 epoch (~15 min pipeline proof).

    SMOKE=1 python train2.py      # smoke
    python train2.py              # full (~3 epochs, fits Kaggle 12h)
"""
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

from glyph.agents import grade
from glyph.seed2 import EVAL_PARA, message, phrased
from glyph.tasks import ROOT

MODEL = os.environ.get("GLYPH_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")
SMOKE = bool(os.environ.get("SMOKE"))
BANK = os.environ.get("BANK", "bank2.jsonl")   # BANK=bank3.jsonl SFT=sft3.jsonl -> v3 (slots)
SFT = os.environ.get("SFT", "sft2.jsonl")
OUT = "/kaggle/working/glyph_v2_adapter" if os.path.isdir("/kaggle") else "/tmp/glyph_v2_adapter"
EPOCHS, BATCH, LR, EVAL_N = (1, 8, 2e-4, 40) if SMOKE else (3, 8, 2e-4, 200)
ACCUM = 1 if SMOKE else 2
MAXLEN = 384
MAXPAIRS = 3000 if SMOKE else None

PROMPT = {
    "speaker": "Encode this task as glyph symbols.\nTask: {x}\nSymbols:",
    "builder": "Write Python for this glyph message.\nSymbols: {x}\nCode:",
    "translator": "Translate this glyph message into English.\nSymbols: {x}\nEnglish:",
}

torch.manual_seed(int(os.environ.get("SEED", 0)))
random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
model = get_peft_model(base, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def encode(row):
    p = tok(PROMPT[row["kind"]].format(x=row["in"]), add_special_tokens=False)["input_ids"]
    t = tok(" " + row["out"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return p + t, [-100] * len(p) + t


def collate(batch):
    ids, labs = zip(*batch)
    n = max(len(i) for i in ids)
    pad = tok.pad_token_id
    return (torch.tensor([i + [pad] * (n - len(i)) for i in ids]),
            torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids]),
            torch.tensor([l + [-100] * (n - len(l)) for l in labs]))


@torch.no_grad()
def gen(prompt, max_new=64):
    model.eval()
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate(tasks_by_split):
    comp = random.sample(tasks_by_split["heldout_comp"],
                         min(EVAL_N, len(tasks_by_split["heldout_comp"])))
    zs = random.sample(tasks_by_split["heldout_zeroshot"],
                       min(EVAL_N // 2, len(tasks_by_split["heldout_zeroshot"])))
    spk = sum(gen(PROMPT["speaker"].format(x=phrased(t["primitives"], v, EVAL_PARA)), 24)
              == message(t["primitives"])
              for v, t in enumerate(comp))
    bld = sum(grade(gen(PROMPT["builder"].format(x=message(t["primitives"])), 180), t)["passed"]
              for t in comp)
    zsb = sum(grade(gen(PROMPT["builder"].format(x=message(t["primitives"])), 180), t)["passed"]
              for t in zs)
    return {"speaker_unseen": spk / len(comp), "builder_heldout": bld / len(comp),
            "zeroshot": zsb / len(zs) if zs else 0.0}


def main():
    rows = [json.loads(l) for l in (ROOT / "tasks" / SFT).read_text().splitlines()]
    random.shuffle(rows)
    rows = rows[:MAXPAIRS] if MAXPAIRS else rows
    tasks = [json.loads(l) for l in (ROOT / "tasks" / BANK).read_text().splitlines()]
    by_split = {}
    for t in tasks:
        by_split.setdefault(t["split"], []).append(t)

    data = [encode(r) for r in rows]
    n0 = len(data)
    data = [d for d in data if len(d[0]) <= MAXLEN]
    if len(data) < n0:
        print(f"dropped {n0 - len(data)} pairs over {MAXLEN} tokens")
    dl = DataLoader(data, batch_size=BATCH, shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = get_cosine_schedule_with_warmup(opt, 100, EPOCHS * len(dl))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    print(f"pairs={len(data)} steps/epoch={len(dl)} epochs={EPOCHS}")
    best = -1.0
    for ep in range(EPOCHS):
        model.train()
        for step, (ids, mask, labs) in enumerate(dl):
            ids, mask, labs = ids.to(dev), mask.to(dev), labs.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
            scaler.scale(loss / ACCUM).backward()
            if (step + 1) % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            if step % 100 == 0:
                print(f"ep{ep} step{step}/{len(dl)} loss {loss.item():.3f}", flush=True)
        m = evaluate(by_split)
        score = m["speaker_unseen"] + m["builder_heldout"]
        print(f"EPOCH {ep} METRICS {json.dumps(m)}", flush=True)
        if score > best:
            best = score
            model.save_pretrained(OUT)
            print(f"saved best -> {OUT}")
    print("DONE best_score", round(best, 3))


if __name__ == "__main__":
    main()
