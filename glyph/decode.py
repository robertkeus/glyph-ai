"""Reference decoder: glyph message → Python, via the language's defined semantics.

The glyph language is formally specified (each symbol = one operation = one code
line, from taskgen). This deterministic decoder is the language's "compiler" — 100%
correct given a valid message. The *model* decoding the language (Builder, cold
model) is the separate research result; this is what a production demo runs so the
code output is always correct.
"""
from glyph.channel import glyph
from glyph.seed import PRIM_ORDER
from glyph.taskgen import CHAIN, REDUCE

_SYM2PRIM = {glyph(i): p for i, p in enumerate(PRIM_ORDER)}


def decode(message: str) -> str:
    prims = [_SYM2PRIM[c] for c in message if c in _SYM2PRIM]
    lines = ["def solve(xs):", "    r = list(xs)"]
    reduced = False
    for p in prims:
        if p in CHAIN:
            lines.append("    " + CHAIN[p][1])       # r = ...
        else:
            lines.append("    " + REDUCE[p][1])       # return ...
            reduced = True
            break                                     # reducer ends the function
    if not reduced:
        lines.append("    return r")
    return "\n".join(lines)
