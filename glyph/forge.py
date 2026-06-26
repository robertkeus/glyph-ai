"""Phase 1 forge loop — GRPO over the native channel (PLAN §A–E).

Model-agnostic: the `policy` is injected, so the loop LOGIC (group sampling,
run-the-tests reward, byte penalty, GRPO advantages, λ schedule, Pareto stop) is
GPU-free and unit-tested with a fake policy. Wire a real two-adapter LoRA policy
on GPU; nothing here changes.

policy protocol:
    sample(speaker_prompt: str, n: int) -> list[str]   # n native messages, mask-restricted
    build(builder_prompt: str) -> str                  # code (Builder; frozen after warmup, PLAN §B)
    learn(speaker_prompt: str, messages, advantages)   # one GRPO update on the Speaker
"""
from glyph.agents import _extract_code, builder_prompt, grade, speaker_prompt
from glyph.channel import Native
from glyph.probe import group_advantages, has_signal


def reward(message, code_text, task, channel, lam):
    """task_success − λ·bytes (PLAN Phase 1). Returns (reward, passed). Grades via
    the neutral-name alias so the Builder can't read the task from the function name."""
    passed = grade(_extract_code(code_text), task)["passed"]
    return (1.0 if passed else 0.0) - lam * channel.bytes(message), passed


def forge_step(task, policy, channel, lam, group_size=8) -> dict:
    """One GRPO step on one task. Updates the Speaker only when the group carries
    a gradient (PLAN §A: equal rewards → zero advantage → skip)."""
    sp = speaker_prompt(task, channel)
    messages = policy.sample(sp, group_size)
    pairs = [reward(m, policy.build(builder_prompt(channel.builder_text(m))),
                    task, channel, lam) for m in messages]
    rewards = [r for r, _ in pairs]
    signal = has_signal(rewards)
    if signal:
        policy.learn(sp, messages, group_advantages(rewards))
    n = len(messages)
    return {
        "task": task["id"],
        "pass_rate": sum(ok for _, ok in pairs) / n,
        "mean_reward": sum(rewards) / n,
        "had_signal": signal,
        "mean_bytes": sum(channel.bytes(m) for m in messages) / n,
    }


def lam_for(pass_rate, floor=1e-4, full=1e-2, knee=0.5) -> float:
    """λ floor from step one, ramp after competence (PLAN §D — never start at 0,
    or the Speaker stays a verbose transliterator and nothing condenses)."""
    return full if pass_rate >= knee else floor


def should_stop(byte_history, window=5) -> bool:
    """Pareto plateau on bytes (PLAN §E). True when mean message bytes have not
    improved over the last `window` steps. The caller gates on success having
    plateaued first — bytes alone is half the Pareto front."""
    if len(byte_history) <= window:
        return False
    return min(byte_history[-window:]) >= min(byte_history[:-window])


def forge_run(tasks, policy, channel=None, steps=200, group_size=8,
              knee=0.5, stop_window=5, on_step=None) -> list:
    """Illustrative driver: cycle the curriculum, schedule λ by competence, stop
    on the Pareto plateau once competent. Returns per-step metrics. `on_step(step,
    metrics)` runs after each step — use it to checkpoint (free tiers cap 12h)."""
    channel = channel or Native()
    history, bytes_hist, competent = [], [], False
    for step in range(steps):
        pass_rate = history[-1]["pass_rate"] if history else 0.0
        competent = competent or pass_rate >= knee  # latch: λ never flips back (avoids §D oscillation)
        m = forge_step(tasks[step % len(tasks)], policy, channel,
                       lam_for(1.0 if competent else 0.0, knee=knee), group_size)
        history.append(m)
        bytes_hist.append(m["mean_bytes"])
        if on_step:
            on_step(step, m)
        if competent and should_stop(bytes_hist, stop_window):
            break
    return history
