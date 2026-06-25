import unittest

from glyph.channel import Native, glyph
from glyph.forge import forge_run, forge_step, lam_for, reward, should_stop
from glyph.tasks import load_tasks


class LearningPolicy:
    """Fake two-adapter policy. Competence `k` = how many of the group's messages
    decode to the reference solution. Shorter messages as competence rises, then
    a floor (simulates compression then plateau). learn() advances k only when an
    update arrives (i.e. the group had a gradient)."""
    def __init__(self, task, floor_len=2):
        self.task, self.k, self.floor_len = task, 1, floor_len

    def sample(self, _prompt, n):
        length = max(self.floor_len, n - self.k)        # bytes shrink as k grows
        return [glyph(i) * length for i in range(n)]

    def build(self, builder_prompt):                    # message i = glyph(i)*len
        passes = any(glyph(i) in builder_prompt and i < self.k for i in range(8))
        return f"```python\n{self.task['solution']}\n```" if passes else "no"

    def learn(self, _prompt, _messages, _advantages):
        self.k = min(self.k + 1, 8)


class TestForge(unittest.TestCase):
    def test_reward_penalizes_bytes(self):
        t = load_tasks()[0]
        ch = Native()
        good = f"```python\n{t['solution']}\n```"
        r, ok = reward(glyph(0) * 3, good, t, ch, lam=0.01)
        self.assertTrue(ok)
        self.assertAlmostEqual(r, 1.0 - 0.01 * 3 * 2)   # 3 symbols × 2 bytes

    def test_step_skips_update_without_signal(self):
        t = load_tasks()[0]

        class Flat:                                     # all builds fail → no variance
            def sample(self, p, n): return [glyph(0)] * n
            def build(self, p): return "no"
            def learn(self, *a): raise AssertionError("must not update")

        m = forge_step(t, Flat(), Native(), lam=1e-4, group_size=4)
        self.assertFalse(m["had_signal"])
        self.assertEqual(m["pass_rate"], 0.0)

    def test_lambda_floor_then_ramp(self):
        self.assertEqual(lam_for(0.0), 1e-4)            # never zero (PLAN §D)
        self.assertEqual(lam_for(0.9), 1e-2)

    def test_should_stop_on_plateau_not_during_improvement(self):
        self.assertFalse(should_stop([10, 9, 8, 7, 6, 5], window=3))  # still falling
        self.assertTrue(should_stop([10, 9, 5, 5, 5, 5], window=3))   # flat tail

    def test_run_converges_and_stops(self):
        t = load_tasks()[0]
        hist = forge_run([t], LearningPolicy(t), steps=100, group_size=8,
                         knee=0.4, stop_window=5)
        self.assertLess(len(hist), 100, "should stop early on plateau")
        self.assertGreaterEqual(hist[-1]["pass_rate"], 0.4, "should reach competence")


if __name__ == "__main__":
    unittest.main()
