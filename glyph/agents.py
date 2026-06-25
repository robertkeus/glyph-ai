"""Two-agent forge loop, channel-agnostic (PLAN: Speaker/Builder).

Invariant that makes the channel load-bearing: the Builder sees ONLY the
Speaker's message — never the English task, never the hidden tests. If the
message doesn't carry the task, the Builder cannot pass. That's the whole point.

`generate(prompt) -> str` is injected: a fake for local plumbing tests, real
transformers on Kaggle. Keeps this file GPU-free and unit-testable. The channel
(English | Native) decides how the message is phrased, read, and byte-counted.
"""
import re

from glyph.channel import English, Channel
from glyph.verifier import run_tests

_CODE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def _extract_code(text: str) -> str:
    m = _CODE.search(text)
    return (m.group(1) if m else text).strip()


def speaker_prompt(task, channel: Channel) -> str:
    return (f"You are the Speaker. Convey this programming task so another agent "
            f"can implement it. {channel.speaker_hint()}\n\nTask: {task['prompt']}")


def builder_prompt(message: str, entry_point: str) -> str:
    return (f"You are the Builder. Using ONLY the message below, write Python.\n"
            f"Return one ```python code block defining `{entry_point}`.\n\n"
            f"Message:\n{message}")


def episode(task, generate, channel: Channel = None) -> dict:
    """One forge episode. Returns an event record (also the demo/log unit)."""
    channel = channel or English()
    message = generate(speaker_prompt(task, channel)).strip()
    builder_in = channel.builder_text(message)
    code = _extract_code(generate(builder_prompt(builder_in, task["entry_point"])))
    v = run_tests(code, task["tests"])
    return {
        "task": task["id"], "split": task["split"], "channel": channel.name,
        "prompt": task["prompt"], "message": message, "code": code,
        "passed": v["passed"],
        "message_bytes": channel.bytes(message),  # density number (PLAN test 1)
        "detail": v["detail"],
    }


def run(tasks, generate, channel: Channel = None):
    """Returns (events, summary). bytes_per_solved is the comparison metric."""
    events = [episode(t, generate, channel) for t in tasks]
    solved = [e for e in events if e["passed"]]
    summary = {
        "channel": events[0]["channel"] if events else None,
        "n": len(events), "passed": len(solved),
        "pass_rate": len(solved) / len(events) if events else 0.0,
        "bytes_per_solved": (sum(e["message_bytes"] for e in solved) / len(solved))
                            if solved else None,
    }
    return events, summary
