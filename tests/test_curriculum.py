import unittest

from glyph.curriculum import check_heldout_novelty, train_primitives
from glyph.tasks import load_tasks

TRAIN = [{"id": "a", "primitives": ["map"]}, {"id": "b", "primitives": ["filter"]}]


class TestCurriculum(unittest.TestCase):
    def test_compositional_split_is_sound(self):
        held = [{"id": "c", "primitives": ["map", "filter"]}]
        self.assertEqual(check_heldout_novelty(TRAIN, held), [])

    def test_unknown_primitive_flagged(self):
        held = [{"id": "c", "primitives": ["map", "recurse"]}]
        issues = check_heldout_novelty(TRAIN, held)
        self.assertEqual(issues[0][1], "unknown-primitive")

    def test_seen_combo_flagged(self):
        train = [{"id": "a", "primitives": ["map", "filter"]}]
        held = [{"id": "c", "primitives": ["map", "filter"]}]
        issues = check_heldout_novelty(train, held)
        self.assertEqual(issues[0][1], "combo-seen")

    def test_train_combos_include_singletons(self):
        seen, combos = train_primitives(TRAIN)
        self.assertIn(frozenset(["map"]), combos)
        self.assertNotIn(frozenset(["map", "filter"]), combos)  # never co-occur

    def test_real_seed_split_sound(self):
        self.assertEqual(
            check_heldout_novelty(load_tasks(split="train"),
                                  load_tasks(split="heldout")), [])


if __name__ == "__main__":
    unittest.main()
