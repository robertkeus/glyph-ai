import subprocess, sys, tempfile, os


def run_tests(candidate: str, tests: str, timeout: float = 5.0) -> dict:
    """Run candidate code against hidden tests in an isolated subprocess.

    Returns {"passed": bool, "detail": str}. `passed` IS the RL reward signal —
    no judge model, ever.

    NB: trusted-task use only. subprocess+timeout is not a sandbox; model-
    generated code in Phase 1 RL needs real isolation (nsjail / container).
    """
    src = f"{candidate}\n\n{tests}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout
        )
        return {"passed": p.returncode == 0,
                "detail": p.stderr.strip()[-500:] if p.returncode else ""}
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "timeout"}
    finally:
        os.unlink(path)
