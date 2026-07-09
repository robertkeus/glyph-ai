import unittest

from glyph.agents import grade
from glyph.decode import decode
from glyph.seed import canonical_message
from glyph.tasks import load_tasks


class TestDecode(unittest.TestCase):
    def test_reference_decoder_solves_every_task(self):
        # deterministic decode of the canonical message must pass every task's tests
        for t in load_tasks():
            code = decode(canonical_message(t))
            self.assertTrue(grade(code, t)["passed"], t["id"])


if __name__ == "__main__":
    unittest.main()
