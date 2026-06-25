import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_tasks(path="tasks/seed.jsonl", split=None):
    """Load single-function tasks. `split`: 'train' | 'heldout' | None (all).

    The held-out compositional split is carved here, before any training touches
    it — recombines primitives seen only separately in train. See PLAN §G.
    """
    lines = (ROOT / path).read_text().splitlines()
    tasks = [json.loads(l) for l in lines if l.strip()]
    return [t for t in tasks if split is None or t.get("split") == split]
