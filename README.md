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
| Curriculum tagger / novelty check | — | ☐ |
| English-baseline two-agent smoke | — | ☐ |
| Cold-start reward-variance probe | — | ☐ |

```bash
python3 smoke.py   # must print "N/N tasks have a satisfiable, discriminating test"
```

No model and no deps yet — stdlib only. Phase 0 is the gate that decides
from-scratch vs seeded vocabulary (PLAN §A, Phase 0).
