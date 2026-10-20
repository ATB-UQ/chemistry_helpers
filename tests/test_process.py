"""run_with_timeout: no survivor on timeout, exception or KeyboardInterrupt."""
import signal
import subprocess
import sys

import pytest

from chemistry_helpers.process import run_with_timeout
from process_helpers import (GRANDCHILD_SCRIPT, PYTHON, alive, kill_leftover, wait_dead,
                             wait_for_pid)


def _spawner(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    return [PYTHON, "-c", GRANDCHILD_SCRIPT, str(pidfile)], pidfile


def test_timeout_leaves_no_grandchild(tmp_path):
    argv, pidfile = _spawner(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        run_with_timeout(argv, timeout=2)
    pid = wait_for_pid(pidfile)
    try:
        assert wait_dead(pid)
    finally:
        kill_leftover(pid)


def test_exception_in_wait_leaves_no_grandchild(tmp_path, monkeypatch):
    argv, pidfile = _spawner(tmp_path)

    def broken(self, *a, **k):
        wait_for_pid(pidfile)
        raise RuntimeError("simulated failure in the parent's wait")

    monkeypatch.setattr(subprocess.Popen, "communicate", broken)
    with pytest.raises(RuntimeError, match="simulated"):
        run_with_timeout(argv, timeout=30)
    monkeypatch.undo()
    pid = wait_for_pid(pidfile)
    try:
        assert wait_dead(pid)
    finally:
        kill_leftover(pid)


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_base_exception_leaves_no_grandchild(tmp_path, exc):
    argv, pidfile = _spawner(tmp_path)

    def interrupt(signum, frame):
        raise exc()

    old = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 1.5)
    try:
        with pytest.raises(exc):
            run_with_timeout(argv, timeout=60)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    pid = wait_for_pid(pidfile)
    try:
        assert wait_dead(pid)
    finally:
        kill_leftover(pid)


def test_output_returncode_and_stdin():
    r = run_with_timeout([PYTHON, "-c",
                          "import sys; d = sys.stdin.read(); print(d.upper()); "
                          "sys.stderr.write('err!'); sys.exit(3)"], timeout=20, stdin="abc")
    assert r.returncode == 3
    assert r.stdout == b"ABC\n"
    assert r.stderr == b"err!"


def test_text_mode_and_cwd(tmp_path):
    r = run_with_timeout(["pwd"], timeout=20, cwd=tmp_path, text=True)
    assert r.stdout.strip() == str(tmp_path.resolve())


def test_check_raises_with_stderr_tail():
    code = "import sys; sys.stderr.write('x' * 10000 + 'END'); sys.exit(2)"
    with pytest.raises(subprocess.CalledProcessError) as info:
        run_with_timeout([PYTHON, "-c", code], timeout=20, check=True)
    assert info.value.returncode == 2
    assert len(info.value.stderr) == 4096 and info.value.stderr.endswith(b"END")


def test_log_path_receives_stderr_and_tail_returned(tmp_path):
    log = tmp_path / "child.log"
    r = run_with_timeout([PYTHON, "-c", "import sys; sys.stderr.write('to the log')"],
                         timeout=20, log_path=log)
    assert log.read_bytes() == b"to the log"
    assert r.stderr == b"to the log"


def test_timeout_carries_stderr_tail(tmp_path):
    code = "import sys, time; sys.stderr.write('last words'); sys.stderr.flush(); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired) as info:
        run_with_timeout([PYTHON, "-c", code], timeout=1.5)
    assert info.value.stderr == b"last words"


def test_shell_string():
    r = run_with_timeout("echo hi && echo oops >&2", shell=True, timeout=20, text=True)
    assert (r.stdout, r.stderr) == ("hi\n", "oops\n")


def test_missing_binary_raises_oserror():
    with pytest.raises(FileNotFoundError):
        run_with_timeout(["/nonexistent/binary"], timeout=5)


def test_no_timeout_waits_for_completion():
    assert run_with_timeout([PYTHON, "-c", "print(1)"]).stdout == b"1\n"


def test_stdout_path(tmp_path):
    out = tmp_path / "out.txt"
    r = run_with_timeout([PYTHON, "-c", "print('to file')"], timeout=20, stdout_path=out, text=True)
    assert out.read_text() == "to file\n" and r.stdout == ""
