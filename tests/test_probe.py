import unittest

from glyph.probe import group_advantages, has_signal, probe, probe_robust


class TestProbe(unittest.TestCase):
    def test_no_variance_no_signal(self):
        self.assertFalse(has_signal([0, 0, 0, 0]))
        self.assertFalse(has_signal([1, 1, 1]))
        self.assertFalse(has_signal([0.5]))  # singleton has no group signal

    def test_variance_has_signal(self):
        self.assertTrue(has_signal([0, 1, 0, 1]))

    def test_advantages_sum_to_zero(self):
        self.assertAlmostEqual(sum(group_advantages([0, 1, 1, 0])), 0.0)

    def test_dead_when_all_fail(self):
        r = probe(lambda: 0.0, groups=8, group_size=4)
        self.assertEqual(r["signal_fraction"], 0.0)
        self.assertEqual(r["mean_reward"], 0.0)

    def test_alive_with_variance(self):
        seq = iter([0, 1] * 100)  # deterministic alternation → every group varies
        r = probe(lambda: next(seq), groups=8, group_size=4)
        self.assertEqual(r["signal_fraction"], 1.0)

    def test_robust_gates_on_worst_run(self):
        seq = iter([0, 1] * 2000)
        r = probe_robust(lambda: next(seq), runs=3, groups=8, group_size=4)
        self.assertEqual(r["signal_fraction"]["min"], 1.0)
        self.assertEqual(r["easy_pass_rate"]["mean"], 0.5)
        self.assertEqual(r["verdict"], "from-scratch viable")

    def test_robust_dead_seeds_vocab(self):
        r = probe_robust(lambda: 0.0, runs=3, groups=8, group_size=4)
        self.assertEqual(r["signal_fraction"]["max"], 0.0)
        self.assertEqual(r["verdict"], "seed the vocabulary")


if __name__ == "__main__":
    unittest.main()
