import unittest

from glyph.channel import Native
from glyph.seed import PRIM_ORDER, canonical_message, prim_symbol
from glyph.tasks import load_tasks


class TestSeed(unittest.TestCase):
    def test_symbols_distinct(self):
        syms = [prim_symbol(p) for p in PRIM_ORDER]
        self.assertEqual(len(set(syms)), len(PRIM_ORDER))

    def test_canonical_message_is_one_symbol_per_primitive(self):
        n = Native()
        t = load_tasks(split="heldout")[0]
        cm = canonical_message(t)
        self.assertEqual(len(cm), len(t["primitives"]))
        self.assertTrue(all(n.is_symbol(c) for c in cm))

    def test_heldout_reuses_train_symbols_in_novel_order(self):
        # per-symbol grounding: every held-out symbol is seen in train (the COMBO
        # is novel, not the symbols) — this is what makes test 2 compositional
        train_syms = set("".join(canonical_message(t) for t in load_tasks(split="train")))
        held_syms = set("".join(canonical_message(t) for t in load_tasks(split="heldout")))
        self.assertTrue(held_syms <= train_syms)


if __name__ == "__main__":
    unittest.main()
