"""Regenerate tasks/seed.jsonl from the primitive-composition generator.

    .venv/bin/python scripts/gen_tasks.py   (or set PYTHONPATH=.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glyph.taskgen import build_bank
from glyph.tasks import ROOT


def main():
    train, held, dropped = build_bank()
    seen, rows = set(), []
    for t in train + held:
        if t["id"] not in seen:
            seen.add(t["id"])
            rows.append(t)
    (ROOT / "tasks" / "seed.jsonl").write_text(
        "\n".join(json.dumps(t) for t in rows) + "\n")
    n_train = sum(t["split"] == "train" for t in rows)
    print(f"{len(rows)} tasks: {n_train} train, {len(rows) - n_train} heldout "
          f"({dropped} held candidates dropped as non-novel)")


if __name__ == "__main__":
    main()
