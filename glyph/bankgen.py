"""Scaled bank generator (TRAINING_DATA_PLAN §2) — typed chains from glyph.lang.

Splits, carved before any training:
  train            — all singletons (except zero-shot) + sampled 2/3-chains
  heldout_comp     — well-typed 2/3-chains whose exact combo never appears in train
  heldout_zeroshot — chains containing a primitive NO training chain ever uses
Solutions (Py+JS) and hidden tests are derived by executing the chain semantics.
Deterministic (seeded); verify with smoke gate before use.
"""
import itertools
import json
import random

from glyph.lang import BY_KEY, INPUTS, compose, english, run_chain, solution_py
from glyph.tasks import ROOT

ZEROSHOT = ("halve", "revstr", "range_")   # symbols never seen in training
RNG = random.Random(0)


def _task(keys, split):
    tin = compose(keys)
    name = "op_" + "_".join(keys)
    tests = "\n".join(
        f"assert {name}({inp!r}) == {run_chain(keys, inp)!r}" for inp in INPUTS[tin]
    )
    return {
        "id": name, "entry_point": name, "primitives": list(keys),
        "difficulty": len(keys) - 1, "split": split, "type": tin,
        "prompt": english(keys),
        "solution": solution_py(keys, name), "tests": tests,
    }


def chains(max_len=3):
    keys = list(BY_KEY)
    out = []
    for n in range(1, max_len + 1):
        for combo in itertools.product(keys, repeat=n):
            if len(set(combo)) == n and compose(combo):
                out.append(combo)
    return out


def build(train_frac=0.7, cap_per_len=(None, 1500, 3000)):
    """Returns (tasks, stats). Zero-shot chains split out first; the rest split
    train/heldout_comp by combo. Per-length caps keep the bank tractable."""
    all_chains = chains()
    zs = [c for c in all_chains if any(k in ZEROSHOT for k in c)]
    rest = [c for c in all_chains if c not in set(zs)]

    by_len = {}
    for c in rest:
        by_len.setdefault(len(c), []).append(c)
    tasks = []
    for n, cs in sorted(by_len.items()):
        RNG.shuffle(cs)
        cap = cap_per_len[n - 1]
        cs = cs[:cap] if cap else cs
        if n == 1:
            tasks += [_task(c, "train") for c in cs]
        else:
            cut = int(len(cs) * train_frac)
            tasks += [_task(c, "train") for c in cs[:cut]]
            tasks += [_task(c, "heldout_comp") for c in cs[cut:]]

    RNG.shuffle(zs)
    tasks += [_task(c, "heldout_zeroshot") for c in zs[:300]]

    stats = {}
    for t in tasks:
        stats[t["split"]] = stats.get(t["split"], 0) + 1
    return tasks, stats


def main():
    tasks, stats = build()
    path = ROOT / "tasks" / "bank2.jsonl"
    path.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n")
    print(f"{len(tasks)} tasks -> {path.name}  {stats}")


if __name__ == "__main__":
    main()
