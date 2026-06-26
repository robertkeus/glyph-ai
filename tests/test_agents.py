import unittest

from glyph.agents import _extract_code, episode, run, solve_solution
from glyph.channel import Native
from glyph.tasks import load_tasks


def fake_model(task):
    """Builder returns the reference solution (as neutral `solve`); Speaker echoes."""
    def g(prompt):
        if prompt.startswith("You are the Builder"):
            return f"```python\n{solve_solution(task)}\n```"
        return task["prompt"]
    return g


class TestAgents(unittest.TestCase):
    def test_builder_solves_from_message_and_no_test_leak(self):
        for t in load_tasks():
            e = episode(t, fake_model(t))
            self.assertTrue(e["passed"], t["id"])
            self.assertNotIn("assert", e["message"])  # hidden tests never leak
            self.assertGreater(e["message_bytes"], 0)

    def test_summary_pass_rate_partial(self):
        tasks = load_tasks(split="train")
        # one task's solution for all → only that task passes
        _, s = run(tasks, fake_model(tasks[0]))
        self.assertEqual(s["passed"], 1)
        self.assertAlmostEqual(s["pass_rate"], 1 / len(tasks))

    def test_native_channel_bytes_and_name(self):
        t = load_tasks()[0]
        n = Native()
        msg = n.glyphs[0] * 3

        def g(prompt):
            if prompt.startswith("You are the Builder"):
                return f"```python\n{solve_solution(t)}\n```"
            return msg

        e = episode(t, g, channel=n)
        self.assertEqual(e["channel"], "native")
        self.assertEqual(e["message_bytes"], 3 * 2)
        self.assertTrue(e["passed"])

    def test_extract_code_fence_variants(self):
        for fence in ("python", "py", "Python", ""):  # any/no language tag
            self.assertEqual(
                _extract_code(f"sure:\n```{fence}\ndef f(): return 1\n```\ndone"),
                "def f(): return 1", fence)

    def test_extract_code_unfenced_returns_raw(self):
        self.assertEqual(_extract_code("def f(): return 1"), "def f(): return 1")

    def test_empty_run_summary(self):
        _, s = run([], fake_model({"solution": ""}))
        self.assertEqual(s["pass_rate"], 0.0)
        self.assertIsNone(s["bytes_per_solved"])


if __name__ == "__main__":
    unittest.main()
