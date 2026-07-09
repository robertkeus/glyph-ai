"""Experiment: train the Builder as a READER (glyphs → operation keys) and measure
held-out accuracy. If high, the model reliably DECODES its language (classification),
and code is assembled deterministically — model does the linguistic work, reliably.
"""
from glyph.agents import reader_prompt
from glyph.channel import Native
from glyph.policy import LoraPolicy
from glyph.seed import PRIM_ORDER, canonical_message
from glyph.tasks import load_tasks

CH = Native()
KEYS = " ".join(PRIM_ORDER)
tok_eos = None


def train_reader(p, tasks, rounds=12):
    global tok_eos
    tok_eos = p.tok.eos_token
    for _ in range(rounds):
        for t in tasks:
            p._sft(reader_prompt(canonical_message(t), KEYS),
                   " " + " ".join(t["primitives"]) + tok_eos, "builder", p.opt_builder)


def read(p, glyphs):
    p.model.set_adapter("builder"); p.model.eval()
    import torch
    enc = p.tok(reader_prompt(glyphs, KEYS), return_tensors="pt").to(p.device)
    with torch.no_grad():
        out = p.model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=p.eos)
    txt = p.tok.decode(out[0][enc.input_ids.shape[1]:].cpu(), skip_special_tokens=True)
    return [k for k in txt.split() if k in set(PRIM_ORDER)]


def main():
    train = load_tasks(split="train")
    held = load_tasks(split="heldout")
    p = LoraPolicy("Qwen/Qwen2.5-Coder-3B-Instruct", channel=CH)
    train_reader(p, train, rounds=12)
    ok = sum(read(p, canonical_message(t)) == t["primitives"] for t in held)
    print(f"READER held-out (glyphs → correct keys): {ok}/{len(held)}")
    for t in held[:5]:
        print(f"  {canonical_message(t)} -> {read(p, canonical_message(t))} (want {t['primitives']})")


if __name__ == "__main__":
    main()
