"""Reference decoder: glyph message → code, via the language's defined semantics.

The glyph language is formally specified (each symbol = one operation). This
deterministic decoder is the language's "compiler" — 100% correct given a valid
message. It targets MULTIPLE languages from the same glyph message, which is the
proof the symbols carry intent, not Python syntax (not "just a Python cipher").
The *model* decoding the language (Builder, cold model) is the separate research
result; this is what a production demo runs so the code output is always correct.
"""
from glyph.channel import glyph
from glyph.seed import PRIM_ORDER
from glyph.taskgen import CHAIN, REDUCE

_SYM2PRIM = {glyph(i): p for i, p in enumerate(PRIM_ORDER)}

# JavaScript rendering of the same primitives (semantics identical to taskgen's Python)
JS = {
    "evens":  "r = r.filter(x => x % 2 === 0);",
    "pos":    "r = r.filter(x => x > 0);",
    "double": "r = r.map(x => x * 2);",
    "square": "r = r.map(x => x * x);",
    "inc":    "r = r.map(x => x + 1);",
    "negate": "r = r.map(x => -x);",
    "absval": "r = r.map(x => Math.abs(x));",
    "rev":    "r = r.slice().reverse();",
    "sorta":  "r = r.slice().sort((a, b) => a - b);",
    "sortd":  "r = r.slice().sort((a, b) => b - a);",
    "uniq":   "r = [...new Set(r)];",
    "dec":    "r = r.map(x => x - 1);",
    "sum":    "return r.reduce((a, b) => a + b, 0);",
    "max":    "return r.length ? Math.max(...r) : 0;",
    "len":    "return r.length;",
    "cnt":    "return r.length;",
}


def _prims(message):
    return [_SYM2PRIM[c] for c in message if c in _SYM2PRIM]


def decode(message: str) -> str:
    """Glyph message → Python."""
    lines = ["def solve(xs):", "    r = list(xs)"]
    for p in _prims(message):
        if p in CHAIN:
            lines.append("    " + CHAIN[p][1])        # r = ...
        else:
            lines.append("    " + REDUCE[p][1])        # return ...
            return "\n".join(lines)                    # reducer ends the function
    return "\n".join(lines + ["    return r"])


def decode_js(message: str) -> str:
    """The SAME glyph message → JavaScript (symbols encode intent, not syntax)."""
    lines = ["function solve(xs) {", "  let r = [...xs];"]
    for p in _prims(message):
        line = JS[p]
        lines.append("  " + line)
        if line.startswith("return"):
            return "\n".join(lines + ["}"])
    return "\n".join(lines + ["  return r;", "}"])
