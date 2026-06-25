# Glyph

A model that forges its own native logographic agent-language on coding tasks,
then explains it well enough that a cold model can decode it. See [PLAN.md](PLAN.md).

## Status: Phase 0 (harness)

| Piece | File | Done |
|---|---|---|
| Run-the-tests verifier (= RL reward) | `glyph/verifier.py` | ✅ |
| Task-bank loader + held-out split | `glyph/tasks.py` | ✅ |
| Byte-counted channel | `glyph/channel.py` | ✅ |
| Seed task bank (curriculum + 1 held-out) | `tasks/seed.jsonl` | ✅ |
| Harness-integrity smoke test | `smoke.py` | ✅ |
| Curriculum tagger / novelty check | `glyph/curriculum.py` | ✅ |
| Two-agent runtime (Builder sees message only) | `glyph/agents.py` | ✅ |
| English-baseline run (real model) | `kaggle_entry.py` | ⏳ needs Kaggle run |
| Cold-start reward-variance probe | — | ☐ |

```bash
python3 smoke.py              # verifier sound: N/N satisfiable & discriminating
python3 -m glyph.curriculum   # curriculum shape + held-out split novelty
python3 test_runtime.py       # two-agent loop, fake model, no GPU
```

Real English baseline (on Kaggle GPU): `python kaggle_entry.py` → writes
`events.json`, prints `bytes_per_solved` — the number the native channel must beat.

No model and no deps yet — stdlib only. Phase 0 is the gate that decides
from-scratch vs seeded vocabulary (PLAN §A, Phase 0).
