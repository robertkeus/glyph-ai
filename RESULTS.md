# Results

## English baseline — smoke (PLAN Phase 0, test 1)

Single deterministic (greedy) run. **Not** a reported result — PLAN requires
fixed seeds and reported variance over multiple runs; this is a harness-proof,
not a headline number.

| | |
|---|---|
| Model | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| Device | CPU (macOS-x86, torch 2.2.2), greedy decode, max_new_tokens=512 |
| Tasks | 6 seed (5 train, 1 held-out compositional) |
| **pass_rate** | **0.833** (5/6) |
| **bytes_per_solved** | **676** (English message, the number the native channel must beat) |

Per task: `sum_list` 146B · `sort_desc` 270B · `double_evens` (held-out) 619B ·
`word_count` 693B · `keep_even` 1652B — all PASS. `double_list` FAIL.

**`double_list` failure is a 1.5B instruction-following limit, not a harness bug:**
the Speaker emitted code naming the function `double_integers`; the Builder echoed
that name despite the prompt's "MUST be named exactly `double_list`", so the test's
`double_list(...)` call raises `NameError`. A 3B base or a few-shot name anchor
would likely close it. Left honest rather than papered over with rename-aliasing,
which would mask real Builder failures.

Message bytes span 146–1652 — wide, because greedy English is verbose and
un-sampled. This is exactly the slack the forged native channel exists to remove;
the real comparison is native vs English bytes_per_solved at equal pass_rate,
over seeded multi-run distributions (PLAN test 1).

Reproduce: `pip install -r requirements.txt && python run_baseline.py`
