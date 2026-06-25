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
    """Run the diagnostic. `signal_fraction` = share of groups with a live
    gradient. Near 0 → from-scratch RL cannot start; seed the vocabulary."""
    rewards = [[sample_reward() for _ in range(group_size)] for _ in range(groups)]
    live = sum(has_signal(g) for g in rewards)
    overall = mean(r for g in rewards for r in g)
    frac = live / groups
    return {
        "groups": groups, "group_size": group_size,
        "mean_reward": overall,
        "signal_fraction": frac,
        "verdict": "from-scratch viable" if frac >= VIABLE_FRACTION
                   else "seed the vocabulary",
    }
