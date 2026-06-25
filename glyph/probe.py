"""Cold-start reward-variance probe (PLAN §A) — the gate instrument.

Decides from-scratch vs seeded vocabulary. Under GRPO the advantage of a sampled
group is `r_i - mean(r)`; if every sample in a group gets the SAME reward, that
vector is all-zeros → no gradient → no learning. Cold-start isn't "weak signal,"
it's *no* signal. This measures, model-agnostically, whether the easy curriculum
produces reward VARIANCE often enough for the loop to ever turn over.

`sample_reward()` draws one fresh episode under the current (stochastic) policy on
an easy task and returns its reward in [0,1] (1 = Builder passed). Inject a real
rollout on Kaggle; a Bernoulli stub in tests.
"""
from statistics import mean, pstdev

SIGNAL_EPS = 1e-9
VIABLE_FRACTION = 0.10  # ≥10% of groups must carry a gradient (placeholder threshold)


def group_advantages(rewards):
    """GRPO-style advantages. All-zero iff the group has no reward variance."""
    m = mean(rewards)
    return [r - m for r in rewards]


def has_signal(rewards) -> bool:
    return len(rewards) > 1 and pstdev(rewards) > SIGNAL_EPS


def probe(sample_reward, groups: int = 32, group_size: int = 8) -> dict:
    """One diagnostic run. `signal_fraction` = share of groups with a live
    gradient; `mean_reward` = easy-task pass-rate. Prefer probe_robust for the
    actual gate — a single run can be a lucky (or unlucky) seed."""
    rewards = [[sample_reward() for _ in range(group_size)] for _ in range(groups)]
    return {
        "groups": groups, "group_size": group_size,
        "mean_reward": mean(r for g in rewards for r in g),
        "signal_fraction": sum(has_signal(g) for g in rewards) / groups,
    }


def probe_robust(sample_reward, runs: int = 5, groups: int = 16,
                 group_size: int = 8) -> dict:
    """The gate (PLAN §A, "never single-run"). Repeat the probe `runs` times and
    gate the verdict on the WORST run, so a lucky seed can't green-light a forge.

    Reads: signal≈0 with pass-rate 0 or 1 → saturated, no variance to learn from
    (seed or steepen). signal>0 → a gradient exists → from-scratch can start.
    """
    rs = [probe(sample_reward, groups, group_size) for _ in range(runs)]
    fr = [r["signal_fraction"] for r in rs]
    pr = [r["mean_reward"] for r in rs]
    worst = min(fr)
    if worst >= VIABLE_FRACTION:
        verdict = "from-scratch viable"
    elif mean(fr) >= VIABLE_FRACTION:
        verdict = "borderline — steepen the easy curriculum, re-probe"
    else:
        verdict = "seed the vocabulary"
    return {
        "runs": runs, "groups": groups, "group_size": group_size,
        "signal_fraction": {"mean": mean(fr), "min": worst, "max": max(fr)},
        "easy_pass_rate": {"mean": mean(pr), "min": min(pr), "max": max(pr)},
        "verdict": verdict,
    }
