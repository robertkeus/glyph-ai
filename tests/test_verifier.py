import unittest

from glyph.verifier import run_tests


class TestVerifier(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(run_tests("def f(): return 1", "assert f() == 1")["passed"])

    def test_fail_reports_detail(self):
        r = run_tests("def f(): return 0", "assert f() == 1")
        self.assertFalse(r["passed"])
        self.assertTrue(r["detail"])

    def test_timeout(self):
        r = run_tests("def f():\n    while True: pass", "f()", timeout=1.0)
        self.assertFalse(r["passed"])
        self.assertEqual(r["detail"], "timeout")

    def test_syntax_error(self):
        self.assertFalse(run_tests("def f( : pass", "f()")["passed"])

    def test_isolation_no_argv_leak(self):
        # `-I` mode: candidate cannot import the harness's environment
        self.assertTrue(run_tests("import sys\ndef f(): return len(sys.argv)",
                                  "assert f() == 1")["passed"])


if __name__ == "__main__":
    unittest.main()
