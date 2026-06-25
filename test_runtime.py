"""Local plumbing test for the two-agent loop — no GPU, fake model.

Proves: roles are invoked, the Builder solves from the message alone, the
verifier gates the event, and bytes/pass_rate aggregate. Model *quality* is not
under test here — the harness is.
"""
from glyph.agents import run
from glyph.tasks import load_tasks


def fake_generate_for(task):
    """Speaker echoes the task as its message; Builder returns the reference
    solution. A real model must EARN what this fake is handed."""
    def g(prompt):
        if prompt.startswith("You are the Builder"):
            return f"```python\n{task['solution']}\n```"
        return task["prompt"]
    return g


def main():
    flat = [run([t], fake_generate_for(t))[0][0] for t in load_tasks()]
    for e in flat:
        print(f"{'ok ' if e['passed'] else 'XX '} {e['task']:14} "
              f"bytes={e['message_bytes']:3} passed={e['passed']}")
    passed = sum(e["passed"] for e in flat)
    print(f"\n{passed}/{len(flat)} episodes solved from message alone")
    assert all("assert" not in e["message"] for e in flat), "tests leaked into message"
    return 0 if passed == len(flat) else 1


if __name__ == "__main__":
    raise SystemExit(main())
