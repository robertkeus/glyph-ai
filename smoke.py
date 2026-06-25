"""Harness-integrity check (PLAN Phase 0, step 4 prerequisite).

For each task, confirm the verifier is sound:
  - the reference solution PASSES the hidden tests (tests are satisfiable)
  - a do-nothing stub FAILS them (tests actually discriminate)

If this is not green, the harness is broken — not the language. Find out here,
before any model exists.
"""
from glyph.tasks import load_tasks
from glyph.verifier import run_tests


def main():
    tasks = load_tasks()
    ok = 0
    for t in tasks:
        good = run_tests(t["solution"], t["tests"])
        stub = run_tests(f"def {t['entry_point']}(*a, **k):\n    return None", t["tests"])
        sound = good["passed"] and not stub["passed"]
        ok += sound
        flag = "ok " if sound else "BROKEN"
        print(f"{flag}  {t['id']:14} {t['split']:7} ref={good['passed']!s:5} "
              f"stub={stub['passed']!s:5} {good['detail'][:50]}")
    print(f"\n{ok}/{len(tasks)} tasks have a satisfiable, discriminating test")
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
