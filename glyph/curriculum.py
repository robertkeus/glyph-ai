"""Curriculum tagger + held-out novelty check (PLAN Phase 0).

Two jobs:
  1. Report the curriculum shape — is the easy end wide enough to give cold-start
     reward *variance* (PLAN §A)? Variance, not low difficulty, is what makes the
     gradient exist.
  2. Verify the held-out split is genuinely *compositional* (PLAN §G): each
     held-out task must recombine primitives that appear in train ONLY separately
     — never a primitive train has not seen (that's novel-primitive, not
     compositional and not expected to generalize).
"""
from collections import Counter
from itertools import combinations

from glyph.tasks import load_tasks


def train_primitives(train):
    """Set of primitives, and set of primitive-combos, seen in train."""
    seen = set()
    combos = set()
    for t in train:
        prims = t["primitives"]
        seen.update(prims)
        for r in range(1, len(prims) + 1):
            combos.update(frozenset(c) for c in combinations(prims, r))
    return seen, combos


def check_heldout_novelty(train, heldout):
    """Return list of issues; empty list == split is sound.

    issue kinds: 'combo-seen' (not novel), 'unknown-primitive' (not compositional).
    """
    seen, combos = train_primitives(train)
    issues = []
    for t in heldout:
        prims = t["primitives"]
        unknown = [p for p in prims if p not in seen]
        if unknown:
            issues.append((t["id"], "unknown-primitive", unknown))
        elif frozenset(prims) in combos:
            issues.append((t["id"], "combo-seen", prims))
    return issues


def report(tasks):
    by_diff = Counter(t["difficulty"] for t in tasks)
    by_prim = Counter(p for t in tasks for p in t["primitives"])
    return by_diff, by_prim


def main():
    train = load_tasks(split="train")
    heldout = load_tasks(split="heldout")
    by_diff, by_prim = report(train + heldout)

    print("difficulty histogram:", dict(sorted(by_diff.items())))
    print("primitive coverage:  ", dict(by_prim))
    easy = by_diff.get(0, 0)
    print(f"easy end (difficulty 0): {easy} tasks "
          f"{'— OK for cold-start variance' if easy >= 3 else '— TOO THIN (PLAN §A)'}")

    issues = check_heldout_novelty(train, heldout)
    if not issues:
        print(f"\nheld-out split SOUND: {len(heldout)} compositional task(s), "
              "each recombines train-only primitives")
        return 0
    print("\nheld-out split BROKEN:")
    for tid, kind, detail in issues:
        print(f"  {tid}: {kind} {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
