import unittest

from glyph.taskgen import build_bank
from glyph.tasks import load_tasks


class TestBank(unittest.TestCase):
    def test_committed_matches_generator(self):
        train, held, _ = build_bank()
        self.assertEqual({t["id"] for t in load_tasks()},
                         {t["id"] for t in train + held},
                         "tasks/seed.jsonl drifted from taskgen — rerun scripts/gen_tasks.py")

    def test_every_task_well_formed(self):
        for t in load_tasks():
            self.assertIn(t["entry_point"], t["solution"])
            self.assertIn("assert", t["tests"])
            self.assertEqual(t["difficulty"], len(t["primitives"]) - 1)

    def test_easy_end_wide_for_cold_start(self):       # PLAN §A: variance needs a wide easy end
        easy = [t for t in load_tasks() if t["difficulty"] == 0]
        self.assertGreaterEqual(len(easy), 8)

    def test_heldout_is_nonempty_and_compositional(self):
        held = load_tasks(split="heldout")
        self.assertTrue(held)
        self.assertTrue(all(len(t["primitives"]) >= 2 for t in held))


if __name__ == "__main__":
    unittest.main()
