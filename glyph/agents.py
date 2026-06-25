"""Two-agent forge loop, channel-agnostic (PLAN: Speaker/Builder).

Invariant that makes the channel load-bearing: the Builder sees ONLY the
Speaker's message — never the English task, never the hidden tests. If the
message doesn't carry the task, the Builder cannot pass. That's the whole point.

`generate(prompt) -> str` is injected: a fake for local plumbing tests, real
transformers on Kaggle. Keeps this file GPU-free and unit-testable.

channel="english" is the BASELINE (message is English text). channel="native"
is the forged path (message is symbol-IDs, masked) — wired in the Phase 1 probe.
"""
import re

from glyph.verifier import run_tests


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def speaker_prompt(task) -> str:
    return (f"You are the Speaker. Convey this programming task so another agent "
            f"can implement it. Do NOT write code.\n\nTask: {task['prompt']}")


def builder_prompt(message: str, entry_point: str) -> str:
    return (f"You are the Builder. Using ONLY the message below, write Python.\n"
            f"Return one ```python code block defining `{entry_point}`.\n\n"
            f"Message:\n{message}")


def episode(task, generate, channel: str = "english") -> dict:
    """One forge episode. Returns an event record (also the demo/log unit)."""
    if channel != "english":
        raise NotImplementedError("native channel lands in the Phase 1 probe")
    message = generate(speaker_prompt(task)).strip()
    code = _extract_code(generate(builder_prompt(message, task["entry_point"])))
    v = run_tests(code, task["tests"])
    return {
        "task": task["id"], "split": task["split"], "channel": channel,
        "prompt": task["prompt"], "message": message, "code": code,
        "passed": v["passed"],
        "message_bytes": len(message.encode()),  # the density number (PLAN test 1)
        "detail": v["detail"],
    }


def run(tasks, generate, channel: str = "english"):
    """Returns (events, summary). pass_rate + mean bytes over SOLVED tasks."""
    events = [episode(t, generate, channel) for t in tasks]
    solved = [e for e in events if e["passed"]]
    summary = {
        "n": len(events), "passed": len(solved),
        "pass_rate": len(solved) / len(events) if events else 0.0,
        "bytes_per_solved": (sum(e["message_bytes"] for e in solved) / len(solved))
                            if solved else None,
    }
    return events, summary
