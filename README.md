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
| Test suite (37 tests, unittest) | `tests/` | ✅ |
| Generated curriculum (67 tasks, 22 held-out) | `glyph/taskgen.py` | ✅ |
| Interactions-view demo (keynote artifact) | `demo/index.html` | ✅ |
| Phase 1 forge loop (GRPO, model-agnostic) | `glyph/forge.py` | ✅ |
| Phase 1 two-adapter LoRA policy + masked channel | `glyph/policy.py` | ✅ executes (CPU smoke) |
| **English baseline, real model** | `run_baseline.py` | ✅ smoke: 0.833 / 676 B ([RESULTS.md](RESULTS.md)) |

```bash
./check.sh                    # full gate: 32 tests + verifier soundness + curriculum
python run_baseline.py        # real model → pass_rate + bytes_per_solved (see RESULTS.md)
```

Baseline cleared the bar (0.833 ≥ 0.7) — the harness is proven end-to-end with a
real model. `bytes_per_solved=676` is the number the forged native channel must beat.

## Phase 1 — run the forge on a GPU

The full pipeline is built and the training code is **verified to execute** end to
end (CPU smoke, `scripts/smoke_policy.py`): symbol-masked sampling, builder SFT
warmup, GRPO backward, checkpointing. Convergence and the §A/§D tuning are the open
research — they need a GPU (CPU RL over thousands of rollouts is infeasible).

```bash
pip install -r requirements.txt
python forge_kaggle.py     # warmup builder (§B) → cold-start probe (§A) → forge_run (§E)
```

`forge_kaggle.py` warms up the Builder so reward is reachable (§B), runs the
cold-start probe that decides **from-scratch vs seeded** vocabulary (§A), then
GRPO-forges the Speaker, checkpointing every 25 steps (free tiers cap 12h).
