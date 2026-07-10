import json
import shutil
import subprocess
import unittest

from glyph.agents import grade
from glyph.decode import decode, decode_js
from glyph.seed import canonical_message
from glyph.tasks import load_tasks

INPUTS = [[1, 2, 3, 4], [], [-2, -1, 0, 1, 2], [5, 5, 5], [3, 1, 2], [-7]]


class TestDecode(unittest.TestCase):
    def test_reference_decoder_solves_every_task(self):
        # deterministic decode of the canonical message must pass every task's tests
        for t in load_tasks():
            code = decode(canonical_message(t))
            self.assertTrue(grade(code, t)["passed"], t["id"])

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_js_decode_matches_python_semantics(self):
        # the SAME glyph message must compute the same results in JavaScript
        for t in load_tasks():
            msg = canonical_message(t)
            py, ns = decode(msg), {}
            exec(py, ns)
            want = [ns["solve"](list(i)) for i in INPUTS]
            js = decode_js(msg) + f"\nconsole.log(JSON.stringify({json.dumps(INPUTS)}.map(i => solve(i))));"
            out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=10)
            self.assertEqual(out.returncode, 0, t["id"] + ": " + out.stderr[:200])
            self.assertEqual(json.loads(out.stdout), want, t["id"])


if __name__ == "__main__":
    unittest.main()
