import os
import resource
import subprocess
import sys
import tempfile

_MEM_BYTES = 512 * 1024 * 1024  # 512 MB address-space cap
_CPU_SECONDS = 10               # hard CPU cap (backs up the wall-clock timeout)


def _limits():
    """preexec hook: bound CPU and memory of the child (POSIX). Best-effort —
    some platforms ignore RLIMIT_AS; the wall-clock timeout is the backstop."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_BYTES, _MEM_BYTES))
    except (ValueError, OSError):
        pass


def run_tests(candidate: str, tests: str, timeout: float = 5.0) -> dict:
    """Run candidate code against hidden tests in an isolated subprocess.

    Returns {"passed": bool, "detail": str}. `passed` IS the RL reward signal —
    no judge model, ever.

    Hardening: separate process, `-I` isolated interpreter, wall-clock timeout,
    plus CPU/memory rlimits on POSIX. NB still NOT a security sandbox — model-
    generated code in Phase 1 RL needs real isolation (nsjail / container /
    gVisor). These bounds stop runaway loops and OOM, not malice.
    """
    src = f"{candidate}\n\n{tests}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limits if os.name == "posix" else None,
        )
        return {"passed": p.returncode == 0,
                "detail": p.stderr.strip()[-500:] if p.returncode else "",
                "stdout": p.stdout.strip()[-500:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "timeout"}
    finally:
        os.unlink(path)
