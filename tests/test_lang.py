import json
import random
import shutil
import subprocess
import unittest

from glyph.agents import grade
from glyph.bankgen import ZEROSHOT, build
from glyph.lang import INPUTS, compose, run_chain, solution_js

TASKS, _ = build()
RNG = random.Random(1)
SAMPLE = RNG.sample(TASKS, 120)


class TestLang(unittest.TestCase):
    def test_type_checker_rejects_ill_typed(self):
        self.assertIsNone(compose(["lower", "double"]))   # SL -> IL mismatch
        self.assertIsNone(compose(["sum", "double"]))     # reducer not at end
        self.assertEqual(compose(["evens", "sum"]), "IL")
        self.assertEqual(compose(["strip", "joinc"]), "SL")

    def test_sampled_solutions_pass_and_stubs_fail(self):
        for t in SAMPLE[:60]:
            self.assertTrue(grade(t["solution"].replace(t["entry_point"], "solve", 1), t)["passed"], t["id"])
        stub = "def solve(*a, **k):\n    return None"
        for t in SAMPLE[:10]:
            self.assertFalse(grade(stub, t)["passed"], t["id"])

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_sampled_js_matches_python(self):
        for t in SAMPLE[:25]:
            keys = t["primitives"]
            want = [run_chain(keys, list(i)) for i in INPUTS[t["type"]]]
            js = solution_js(keys) + \
                f"\nconsole.log(JSON.stringify({json.dumps(INPUTS[t['type']])}.map(i => solve(i))));"
            out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=10)
            self.assertEqual(out.returncode, 0, t["id"] + out.stderr[:150])
            self.assertEqual(json.loads(out.stdout), want, t["id"])

    def test_splits_sound(self):
        train_combos = {tuple(t["primitives"]) for t in TASKS if t["split"] == "train"}
        for t in TASKS:
            if t["split"] == "heldout_comp":
                self.assertNotIn(tuple(t["primitives"]), train_combos, t["id"])
                self.assertFalse(any(k in ZEROSHOT for k in t["primitives"]))
            if t["split"] == "train":
                self.assertFalse(any(k in ZEROSHOT for k in t["primitives"]), t["id"])
            if t["split"] == "heldout_zeroshot":
                self.assertTrue(any(k in ZEROSHOT for k in t["primitives"]), t["id"])


if __name__ == "__main__":
    unittest.main()
