"""Primitive-composition curriculum generator (PLAN data needs 2 & 3).

A task is a chain of list→list primitives, optionally ending in a list→scalar
reducer, applied to a list of ints. The reference solution AND its tests are
DERIVED BY EXECUTING the true composition — ungameable, no judge, no hand-written
expected values to get wrong.

The held-out split is built by *validation*, not assertion: candidate held-out
tasks are filtered through glyph.curriculum.check_heldout_novelty, so every
held-out task provably recombines primitives seen in train only separately
(PLAN §G). Function names are `op_*` to avoid shadowing builtins (e.g. `sum`).
"""
from glyph.curriculum import check_heldout_novelty

# name -> (fn: list->list, code line transforming `r`, English phrase)
CHAIN = {
    "evens":  (lambda r: [x for x in r if x % 2 == 0], "r = [x for x in r if x % 2 == 0]", "keep the even numbers"),
    "pos":    (lambda r: [x for x in r if x > 0],       "r = [x for x in r if x > 0]",       "keep the positive numbers"),
    "double": (lambda r: [x * 2 for x in r],            "r = [x * 2 for x in r]",            "double each"),
    "square": (lambda r: [x * x for x in r],            "r = [x * x for x in r]",            "square each"),
    "inc":    (lambda r: [x + 1 for x in r],            "r = [x + 1 for x in r]",            "add one to each"),
    "negate": (lambda r: [-x for x in r],               "r = [-x for x in r]",               "negate each"),
    "absval": (lambda r: [abs(x) for x in r],           "r = [abs(x) for x in r]",           "take the absolute value of each"),
    "rev":    (lambda r: list(reversed(r)),             "r = list(reversed(r))",             "reverse the order"),
    "sorta":  (lambda r: sorted(r),                     "r = sorted(r)",                     "sort ascending"),
    "sortd":  (lambda r: sorted(r, reverse=True),       "r = sorted(r, reverse=True)",       "sort descending"),
    "uniq":   (lambda r: list(dict.fromkeys(r)),        "r = list(dict.fromkeys(r))",        "drop duplicates keeping first occurrence"),
    "dec":    (lambda r: [x - 1 for x in r],            "r = [x - 1 for x in r]",            "subtract one from each"),
}

# name -> (fn: list->scalar, return line, phrase). Empty-safe.
REDUCE = {
    "sum": (lambda r: sum(r),               "return sum(r)",               "return their sum"),
    "max": (lambda r: max(r) if r else 0,   "return max(r) if r else 0",   "return the maximum (0 if empty)"),
    "len": (lambda r: len(r),               "return len(r)",               "return how many remain"),
    "cnt": (lambda r: sum(1 for _ in r),    "return sum(1 for _ in r)",    "return the count"),
}

INPUTS = [[1, 2, 3, 4], [], [-2, -1, 0, 1, 2], [5, 5, 5], [3, 1, 2], [10], [2, 4, 6], [-7]]

_C = list(CHAIN)
_R = list(REDUCE)


def _build(keys, split):
    chain = [k for k in keys if k in CHAIN]
    red = next((k for k in keys if k in REDUCE), None)
    name = "op_" + "_".join(keys)

    body = ["def %s(xs):" % name, "    r = list(xs)"]
    body += ["    " + CHAIN[k][1] for k in chain]
    body.append("    " + (REDUCE[red][1] if red else "return r"))

    def fn(xs):
        r = list(xs)
        for k in chain:
            r = CHAIN[k][0](r)
        return REDUCE[red][0](r) if red else r

    tests = "\n".join(f"assert {name}({inp!r}) == {fn(inp)!r}" for inp in INPUTS)
    phrase = ", then ".join([CHAIN[k][2] for k in chain] + ([REDUCE[red][2]] if red else []))
    return {
        "id": name, "entry_point": name, "primitives": list(keys),
        "difficulty": len(keys) - 1, "split": split,
        "prompt": "Take a list of integers; " + phrase + ".",
        "solution": "\n".join(body), "tests": tests,
    }


def build_bank():
    """Return (train, heldout). Held-out is validated, not assumed."""
    train = [_build([k], "train") for k in _C]            # chain singletons
    train += [_build([k], "train") for k in _R]           # reducer singletons
    train += [_build([_C[i], _C[i + 1]], "train") for i in range(len(_C) - 1)]            # consecutive 2-chains
    train += [_build([_C[i], _R[i % len(_R)]], "train") for i in range(len(_C))]          # chain+reduce
    train += [_build([_C[i], _C[i + 1], _R[i % len(_R)]], "train") for i in range(0, len(_C) - 1, 2)]  # 2-chain+reduce

    cand = [_build([_C[i], _C[i + 2]], "heldout") for i in range(len(_C) - 2)]            # skip-one 2-chains
    cand += [_build([_C[i], _R[(i + 2) % len(_R)]], "heldout") for i in range(len(_C))]   # cross chain+reduce
    bad = {tid for tid, _, _ in check_heldout_novelty(train, cand)}
    held = [t for t in cand if t["id"] not in bad]
    return train, held, len(cand) - len(held)
