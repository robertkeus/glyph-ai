import unittest

from glyph.paraphrase import HELDOUT_VARIANT, N_VARIANTS, V, english
from glyph.tasks import load_tasks


class TestParaphrase(unittest.TestCase):
    def test_variant0_matches_taskbank_prompt(self):
        for t in load_tasks():
            self.assertEqual(english(t, 0), t["prompt"], t["id"])

    def test_all_primitives_have_all_variants(self):
        for p, vs in V.items():
            self.assertEqual(len(vs), N_VARIANTS, p)
            self.assertEqual(len(set(vs)), N_VARIANTS, p)  # distinct

    def test_heldout_variant_differs_from_trained(self):
        for t in load_tasks(split="heldout")[:5]:
            held = english(t, HELDOUT_VARIANT)
            for v in range(HELDOUT_VARIANT):
                self.assertNotEqual(held, english(t, v))


if __name__ == "__main__":
    unittest.main()
