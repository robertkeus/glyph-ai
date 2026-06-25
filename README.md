# Glyph

A model that forges its own native logographic agent-language on coding tasks,
then explains it well enough that a cold model can decode it. See [PLAN.md](PLAN.md).

## Status: Phase 0 (harness) — complete, GPU-free parts green

Zero dependencies, stdlib only — runs on any `python3`, including Kaggle with no installs.

| Piece | File | State |
|---|---|---|
| Run-the-tests verifier (= RL reward), hardened | `glyph/verifier.py` | ✅ |
| Task-bank loader + held-out split | `glyph/tasks.py` | ✅ |
| Byte-counted channel: English + Native + mask | `glyph/channel.py` | ✅ |
| Seed task bank (curriculum + 1 held-out) | `tasks/seed.jsonl` | ✅ |
| Curriculum tagger / compositional-novelty check | `glyph/curriculum.py` | ✅ |
| Two-agent runtime (Builder sees message only) | `glyph/agents.py` | ✅ |
| Cold-start reward-variance probe (the gate) | `glyph/probe.py` | ✅ |
| Verifier-soundness diagnostic | `smoke.py` | ✅ |
| Test suite (25 tests, unittest) | `tests/` | ✅ |
| **English baseline, real model** | `kaggle_entry.py` | ⏳ **needs Kaggle GPU run** |

```bash
./check.sh                    # full gate: 25 tests + verifier soundness + curriculum
python3 -m unittest discover -s tests -t .
python3 smoke.py
python3 -m glyph.curriculum
```

## The one remaining step needs your GPU

Everything that can be validated on a laptop is green. The English baseline needs
a real model — run on Kaggle (GPU T4 x2):

```bash
!pip -q install transformers accelerate torch
!python kaggle_entry.py        # → writes events.json, prints pass_rate + bytes_per_solved
```

`bytes_per_solved` is the number the forged native channel must later beat
(PLAN test 1). `pass_rate` decides what's next:
- ≳0.7 → build the cold-start RL loop on the native channel (probe instrument is ready).
- weak → widen the easy curriculum or move to Qwen2.5-Coder-3B before any RL.

Phase 0 is the gate that decides **from-scratch vs seeded** vocabulary (PLAN §A).
