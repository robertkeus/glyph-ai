"""Seeded vocabulary (PLAN §A fallback — from-scratch won't bootstrap).

Ground each primitive to a fixed symbol, so a task's canonical message is the
sequence of its primitives' symbols. SFT-warming BOTH agents to this shared code
breaks the cold-start chicken-and-egg: the Builder can decode from step one, the
Speaker has a target to emit. RL then refines/compresses on top.

Crucially the held-out compositional tasks reuse these SAME primitive symbols in
novel ORDERS — so passing them tests per-symbol (compositional) decoding, not
sequence memorization (PLAN test 2).
"""
from glyph.channel import glyph
from glyph.taskgen import CHAIN, REDUCE

PRIM_ORDER = list(CHAIN) + list(REDUCE)  # stable primitive→symbol-index map (16)


def prim_symbol(name: str) -> str:
    return glyph(PRIM_ORDER.index(name))


def canonical_message(task) -> str:
    """The grounded message for a task: its primitives' symbols, in order."""
    return "".join(prim_symbol(p) for p in task["primitives"])
