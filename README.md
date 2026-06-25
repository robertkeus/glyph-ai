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
| Test suite (32 tests, unittest) | `tests/` | ✅ |
| Phase 1 forge loop (GRPO, model-agnostic) | `glyph/forge.py` | ✅ |
| **English baseline, real model** | `run_baseline.py` | ✅ smoke: 0.833 / 676 B ([RESULTS.md](RESULTS.md)) |

```bash
./check.sh                    # full gate: 32 tests + verifier soundness + curriculum
python run_baseline.py        # real model → pass_rate + bytes_per_solved (see RESULTS.md)
```

Baseline cleared the bar (0.833 ≥ 0.7) — the harness is proven end-to-end with a
real model. `bytes_per_solved=676` is the number the forged native channel must beat.

## Next step needs a GPU (Phase 1 training)

`glyph/forge.py` has the full GRPO forge loop, unit-tested with a fake policy. The
only missing piece is a real two-adapter LoRA policy (`sample`/`build`/`learn`) —
which needs a GPU to train (CPU RL over thousands of rollouts is infeasible). On
Kaggle (T4 x2): wire the policy, run `forge_run` on the curriculum; the cold-start
probe (`glyph/probe.py`) then decides **from-scratch vs seeded** vocabulary (PLAN §A).
