"""Argument slots (v3): opcode + operand digit glyphs on the wire."""
import unittest

from glyph.lang import compose, english, parse, run_chain, solution_py
from glyph.seed2 import decode_items, message


class TestSlots(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse("gtn:7"), ("gtn", 7))
        self.assertEqual(parse("evens"), ("evens", None))

    def test_compose_arg_arity(self):
        self.assertEqual(compose(["gtn:7", "sum"]), "IL")
        self.assertIsNone(compose(["gtn", "sum"]))       # missing operand
        self.assertIsNone(compose(["evens:3"]))          # spurious operand
        self.assertIsNone(compose(["minlenn:2", "sum"])) # SL -> IL mismatch

    def test_semantics_match_generated_code(self):
        keys = ["gtn:3", "muln:10", "nthn:1"]
        ns = {}
        exec(solution_py(keys), ns)
        for inp in ([1, 2, 3, 4, 5], [], [4]):
            self.assertEqual(ns["solve"](inp), run_chain(keys, inp))
        self.assertEqual(run_chain(keys, [1, 4, 5, 2]), 50)

    def test_english_interpolates_operand(self):
        e = english(["gtn:7", "sum"])
        self.assertIn("greater than 7", e)

    def test_wire_roundtrip_multidigit(self):
        for keys in (["gtn:7", "sum"], ["addn:100", "rev"], ["minagen:18", "names"],
                     ["evens", "double"]):
            self.assertEqual(decode_items(message(keys)), keys)

    def test_field_operands_two_slots(self):
        self.assertEqual(parse("fminlen:name:3"), ("fminlen", ("name", 3)))
        self.assertEqual(compose(["fcontains:email:@", "fplucks:name"]), "RL")
        self.assertIsNone(compose(["fminlen:name", "countrl"]))    # missing 2nd operand
        recs = [{"name": "Ada", "age": 36, "email": "ada@x.io"},
                {"name": "Bo", "age": 12, "email": "box"}]
        keys = ["fcontains:email:@", "fplucks:name"]
        self.assertEqual(run_chain(keys, recs), ["Ada"])
        ns = {}
        exec(solution_py(keys), ns)
        self.assertEqual(ns["solve"](recs), ["Ada"])
        for k in (keys, ["fgt:age:18", "fplucki:age"], ["fminlen:email:5", "countrl"]):
            self.assertEqual(decode_items(message(k)), k)
        e = english(["fminlen:email:5"])
        self.assertIn("email", e)
        self.assertIn("5", e)

    def test_string_operands(self):
        self.assertEqual(parse("prefixs:#"), ("prefixs", "#"))
        self.assertEqual(run_chain(["prefixs:#", "joins:,"], ["a", "b"]), "#a,#b")
        ns = {}
        exec(solution_py(["containss:e", "suffixs:!"]), ns)
        self.assertEqual(ns["solve"](["he", "ox", "we"]), ["he!", "we!"])
        for keys in (["prefixs:#", "joins:;"], ["startss:a", "counts"],
                     ["containss:e", "endss:o"]):
            self.assertEqual(decode_items(message(keys)), keys)
        self.assertIn('with "#"', english(["prefixs:#"]))

    def test_operand_glyphs_disjoint_from_opcodes(self):
        msg = message(["addn:100"])
        self.assertEqual(len(msg), 4)                    # opcode + 3 digit glyphs
        self.assertEqual(decode_items(msg), ["addn:100"])


if __name__ == "__main__":
    unittest.main()
