"""Typed primitive registry v2 — the scaled language (TRAINING_DATA_PLAN §1-2).

Each primitive declares in/out types so the composer only builds well-typed
chains. Every primitive carries BOTH a Python and a JS rendering (multi-target
is the "not a cipher" invariant) plus executable semantics (`fn`) used to derive
solutions and tests by execution — never hand-written expected values.

v1 (taskgen.py) stays untouched; this module generates the scaled bank.
"""
from dataclasses import dataclass, field

# types: IL=int list, SL=str list, I=int, S=str, B=bool


@dataclass(frozen=True)
class Prim:
    key: str
    tin: str            # input type
    tout: str           # output type
    py: str             # one code line operating on r / returning
    js: str
    en: list            # paraphrase phrasings (variant 0 = canonical)
    fn: object          # executable semantics
    ends: bool = field(default=False)  # reducer/terminal (emits return)


P = []


def _p(key, tin, tout, py, js, en, fn, ends=False):
    P.append(Prim(key, tin, tout, py, js, en, fn, ends))


# ---- int-list -> int-list (v1 family, re-declared with types) ----------------
_p("evens", "IL", "IL", "r = [x for x in r if x % 2 == 0]", "r = r.filter(x => x % 2 === 0);",
   ["keep the even numbers", "keep only even values", "drop the odd numbers"],
   lambda r: [x for x in r if x % 2 == 0])
_p("odds", "IL", "IL", "r = [x for x in r if x % 2 != 0]", "r = r.filter(x => x % 2 !== 0);",
   ["keep the odd numbers", "keep only odd values", "drop the even numbers"],
   lambda r: [x for x in r if x % 2 != 0])
_p("pos", "IL", "IL", "r = [x for x in r if x > 0]", "r = r.filter(x => x > 0);",
   ["keep the positive numbers", "keep values above zero", "drop non-positive values"],
   lambda r: [x for x in r if x > 0])
_p("neg", "IL", "IL", "r = [x for x in r if x < 0]", "r = r.filter(x => x < 0);",
   ["keep the negative numbers", "keep values below zero", "drop non-negative values"],
   lambda r: [x for x in r if x < 0])
_p("double", "IL", "IL", "r = [x * 2 for x in r]", "r = r.map(x => x * 2);",
   ["double each", "multiply each by two", "make each twice as big"],
   lambda r: [x * 2 for x in r])
_p("halve", "IL", "IL", "r = [x // 2 for x in r]", "r = r.map(x => Math.floor(x / 2));",
   ["halve each (integer division)", "divide each by two rounding down", "floor-divide each by 2"],
   lambda r: [x // 2 for x in r])
_p("square", "IL", "IL", "r = [x * x for x in r]", "r = r.map(x => x * x);",
   ["square each", "multiply each value by itself", "raise each to the power of two"],
   lambda r: [x * x for x in r])
_p("inc", "IL", "IL", "r = [x + 1 for x in r]", "r = r.map(x => x + 1);",
   ["add one to each", "increment each value", "bump each up by one"],
   lambda r: [x + 1 for x in r])
_p("dec", "IL", "IL", "r = [x - 1 for x in r]", "r = r.map(x => x - 1);",
   ["subtract one from each", "decrement each value", "reduce each by one"],
   lambda r: [x - 1 for x in r])
_p("negate", "IL", "IL", "r = [-x for x in r]", "r = r.map(x => -x);",
   ["negate each", "flip the sign of each", "multiply each by -1"],
   lambda r: [-x for x in r])
_p("absval", "IL", "IL", "r = [abs(x) for x in r]", "r = r.map(x => Math.abs(x));",
   ["take the absolute value of each", "make every value non-negative", "strip the signs"],
   lambda r: [abs(x) for x in r])
_p("rev", "IL", "IL", "r = list(reversed(r))", "r = r.slice().reverse();",
   ["reverse the order", "flip the list around", "put the elements backwards"],
   lambda r: list(reversed(r)))
_p("sorta", "IL", "IL", "r = sorted(r)", "r = r.slice().sort((a, b) => a - b);",
   ["sort ascending", "sort smallest to largest", "arrange in increasing order"],
   lambda r: sorted(r))
_p("sortd", "IL", "IL", "r = sorted(r, reverse=True)", "r = r.slice().sort((a, b) => b - a);",
   ["sort descending", "sort largest to smallest", "arrange in decreasing order"],
   lambda r: sorted(r, reverse=True))
_p("uniq", "IL", "IL", "r = list(dict.fromkeys(r))", "r = [...new Set(r)];",
   ["drop duplicates keeping first occurrence", "remove repeated values", "deduplicate the list"],
   lambda r: list(dict.fromkeys(r)))
_p("first3", "IL", "IL", "r = r[:3]", "r = r.slice(0, 3);",
   ["keep only the first three", "take the first 3 elements", "truncate to three items"],
   lambda r: r[:3])
_p("dropfirst", "IL", "IL", "r = r[1:]", "r = r.slice(1);",
   ["drop the first element", "remove the first item", "skip the first one"],
   lambda r: r[1:])
_p("clamp10", "IL", "IL", "r = [min(x, 10) for x in r]", "r = r.map(x => Math.min(x, 10));",
   ["cap each value at 10", "clamp each to at most ten", "limit every value to 10"],
   lambda r: [min(x, 10) for x in r])

# ---- int-list -> int (reducers) ----------------------------------------------
_p("sum", "IL", "I", "return sum(r)", "return r.reduce((a, b) => a + b, 0);",
   ["return their sum", "add them all up", "give back the total"],
   lambda r: sum(r), ends=True)
_p("max", "IL", "I", "return max(r) if r else 0", "return r.length ? Math.max(...r) : 0;",
   ["return the maximum (0 if empty)", "give the largest value (0 if none)", "find the max, defaulting to 0"],
   lambda r: max(r) if r else 0, ends=True)
_p("min", "IL", "I", "return min(r) if r else 0", "return r.length ? Math.min(...r) : 0;",
   ["return the minimum (0 if empty)", "give the smallest value (0 if none)", "find the min, defaulting to 0"],
   lambda r: min(r) if r else 0, ends=True)
_p("len", "IL", "I", "return len(r)", "return r.length;",
   ["return how many remain", "return the length", "count the elements"],
   lambda r: len(r), ends=True)
_p("range_", "IL", "I", "return (max(r) - min(r)) if r else 0",
   "return r.length ? Math.max(...r) - Math.min(...r) : 0;",
   ["return the range (max minus min, 0 if empty)", "give the spread between largest and smallest",
    "return max minus min, defaulting to 0"],
   lambda r: (max(r) - min(r)) if r else 0, ends=True)

# ---- str-list -> str-list -----------------------------------------------------
_p("lower", "SL", "SL", "r = [s.lower() for s in r]", "r = r.map(s => s.toLowerCase());",
   ["lowercase each string", "convert each to lower case", "make every string lowercase"],
   lambda r: [s.lower() for s in r])
_p("upper", "SL", "SL", "r = [s.upper() for s in r]", "r = r.map(s => s.toUpperCase());",
   ["uppercase each string", "convert each to upper case", "make every string uppercase"],
   lambda r: [s.upper() for s in r])
_p("strip", "SL", "SL", "r = [s.strip() for s in r]", "r = r.map(s => s.trim());",
   ["trim whitespace from each", "strip spaces around each string", "remove surrounding whitespace"],
   lambda r: [s.strip() for s in r])
_p("nonempty", "SL", "SL", "r = [s for s in r if s]", "r = r.filter(s => s.length > 0);",
   ["drop empty strings", "keep only non-empty strings", "remove blanks"],
   lambda r: [s for s in r if s])
_p("sortlen", "SL", "SL", "r = sorted(r, key=len)", "r = r.slice().sort((a, b) => a.length - b.length);",
   ["sort by length, shortest first", "order strings from shortest to longest", "arrange by string length ascending"],
   lambda r: sorted(r, key=len))
_p("revstr", "SL", "SL", "r = [s[::-1] for s in r]", "r = r.map(s => [...s].reverse().join(''));",
   ["reverse each string", "flip every string's characters", "spell each string backwards"],
   lambda r: [s[::-1] for s in r])
_p("uniqs", "SL", "SL", "r = list(dict.fromkeys(r))", "r = [...new Set(r)];",
   ["drop duplicate strings keeping the first", "remove repeated strings", "deduplicate the strings"],
   lambda r: list(dict.fromkeys(r)))
_p("firstchar", "SL", "SL", "r = [s[0] for s in r if s]", "r = r.filter(s => s.length).map(s => s[0]);",
   ["keep the first character of each (dropping empties)", "take each string's first letter",
    "reduce each string to its first character"],
   lambda r: [s[0] for s in r if s])

# ---- str-list -> int / str (reducers) -----------------------------------------
_p("lens", "SL", "I", "return sum(len(s) for s in r)",
   "return r.reduce((a, s) => a + s.length, 0);",
   ["return the total number of characters", "sum the lengths of all strings", "count characters across all strings"],
   lambda r: sum(len(s) for s in r), ends=True)
_p("counts", "SL", "I", "return len(r)", "return r.length;",
   ["return how many strings remain", "count the strings", "return the number of strings"],
   lambda r: len(r), ends=True)
_p("joinc", "SL", "S", "return ','.join(r)", "return r.join(',');",
   ["join them with commas", "concatenate with commas between", "return them as one comma-separated string"],
   lambda r: ",".join(r), ends=True)
_p("longest", "SL", "S", "return max(r, key=len) if r else ''",
   "return r.reduce((a, s) => s.length > a.length ? s : a, '');",
   ["return the longest string (empty if none)", "give back the lengthiest string",
    "find the longest one, defaulting to empty"],
   lambda r: max(r, key=len) if r else "", ends=True)

BY_KEY = {p.key: p for p in P}
INPUTS = {
    "IL": [[1, 2, 3, 4], [], [-2, -1, 0, 1, 2], [5, 5, 5], [3, 1, 2], [10], [2, 4, 6], [-7, 12, -7]],
    "SL": [["Hello", "world"], [], ["  a ", "", "BC", "bc"], ["one", "one", "Two"], ["x"], ["abc", "de", ""]],
}


def compose(keys):
    """Type-check a chain; returns the chain's input type or None if ill-typed."""
    prims = [BY_KEY[k] for k in keys]
    for i, p in enumerate(prims):
        if i and prims[i - 1].tout != p.tin:
            return None
        if p.ends and i != len(prims) - 1:
            return None                      # reducer only at the end
    return prims[0].tin


def solution_py(keys, name="solve"):
    lines = [f"def {name}(xs):", "    r = list(xs)"]
    for p in (BY_KEY[k] for k in keys):
        lines.append("    " + p.py)
        if p.ends:
            return "\n".join(lines)
    return "\n".join(lines + ["    return r"])


def solution_js(keys):
    lines = ["function solve(xs) {", "  let r = [...xs];"]
    for p in (BY_KEY[k] for k in keys):
        lines.append("  " + p.js)
        if p.ends:
            return "\n".join(lines + ["}"])
    return "\n".join(lines + ["  return r;", "}"])


def run_chain(keys, value):
    r = list(value)
    for p in (BY_KEY[k] for k in keys):
        r = p.fn(r)
        if p.ends:
            return r
    return r


def english(keys, variant=0):
    parts = [BY_KEY[k].en[variant % len(BY_KEY[k].en)] for k in keys]
    tin = BY_KEY[keys[0]].tin
    noun = "a list of integers" if tin == "IL" else "a list of strings"
    return f"Take {noun}; " + ", then ".join(parts) + "."
