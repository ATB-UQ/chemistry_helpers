"""Run a child process with a timeout, killing its whole process group on every exit path.

The platform rule (duplication plan D3): a child runs in its own session, and when the
parent leaves for ANY reason -- timeout, an ordinary exception, or a ``BaseException``
(``KeyboardInterrupt``, ``SystemExit``, Celery's ``SoftTimeLimitExceeded``) -- the whole
group gets SIGTERM, a short grace, then SIGKILL, and is reaped before the exception
propagates.  Killing only the child leaves grandchildren (mpirun ranks, shell
pipelines, ``nice`` wrappers) burning cores with nothing left to reap them.

Pure stdlib.  stderr is never discarded: it is captured (and its tail kept on failure) or,
with *log_path*, written to that file.
"""
import os
import signal
import subprocess
from typing import Dict, Optional, Sequence, Union

STDERR_TAIL_BYTES = 4096
KILL_GRACE_SECONDS = 0.5
REAP_SECONDS = 10


def kill_process_group(proc: subprocess.Popen, grace: float = KILL_GRACE_SECONDS) -> None:
    """SIGTERM *proc*'s group, wait *grace* seconds for the leader, SIGKILL the group, reap.

    *proc* must have been started with ``start_new_session=True`` (its pid is then the
    group id).  SIGKILL is sent even when the leader exited on SIGTERM: a grandchild may
    have ignored it.  Best effort and never raises.
    """
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=grace)
    except BaseException:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=REAP_SECONDS)
    except BaseException:
        pass


def _tail(data: Optional[bytes]) -> bytes:
    return (data or b"")[-STDERR_TAIL_BYTES:]


def _file_tail(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - STDERR_TAIL_BYTES))
            return handle.read()
    except OSError:
        return b""


def run_with_timeout(argv: Union[str, Sequence], timeout: Optional[float] = None,
                     cwd=None, env: Optional[Dict[str, str]] = None,
                     stdin: Optional[Union[bytes, str]] = None, log_path=None,
                     check: bool = False, text: bool = False,
                     shell: bool = False, stdout_path=None) -> subprocess.CompletedProcess:
    """Run *argv* in its own process group; return a ``CompletedProcess``.

    ``stdout`` is captured.  ``stderr`` is captured, or appended to *log_path* when given
    (``CompletedProcess.stderr`` is then the last ~4 kB of that file).  *timeout* None
    means wait forever (the caller's choice; none is invented).  *stdin* (bytes or str) is
    fed to the child; None leaves stdin inherited.  *shell* is for callers that pass a
    command string.  *stdout_path* sends stdout to that file (truncated) instead of capturing it
    (``CompletedProcess.stdout`` is then empty).  ``text=True`` decodes stdout/stderr (replacing bad bytes).

    On timeout, any ``Exception`` or any ``BaseException`` the whole process group is
    killed and reaped, then the exception is re-raised (``subprocess.TimeoutExpired``
    for a timeout, carrying ``output`` and the stderr tail).  ``check=True`` raises
    ``subprocess.CalledProcessError`` (stderr tail attached) on a non-zero exit.
    """
    if isinstance(stdin, str):
        stdin = stdin.encode()
    cmd = argv if shell else [str(a) for a in argv]
    log_handle = open(log_path, "ab") if log_path else None
    out_handle = None
    proc = None
    try:
        out_handle = open(stdout_path, "wb") if stdout_path else None
        proc = subprocess.Popen(
            cmd, cwd=None if cwd is None else str(cwd), env=env, shell=shell,
            stdin=subprocess.PIPE if stdin is not None else None,
            stdout=out_handle if out_handle else subprocess.PIPE,
            stderr=log_handle if log_handle else subprocess.PIPE,
            start_new_session=True)
        try:
            out, err = proc.communicate(input=stdin, timeout=timeout)
        except BaseException as error:
            kill_process_group(proc)
            if isinstance(error, subprocess.TimeoutExpired):
                try:
                    out, err = proc.communicate(timeout=REAP_SECONDS)
                except BaseException:
                    out, err = error.output, error.stderr
                error.output = out
                error.stderr = _file_tail(str(log_path)) if log_handle else _tail(err)
            raise
    except BaseException:
        # Popen itself failed (missing binary...): nothing to kill.
        if proc is not None and proc.poll() is None:
            kill_process_group(proc)
        raise
    finally:
        for handle in (log_handle, out_handle):
            if handle:
                handle.close()
    if log_handle:
        err = _file_tail(str(log_path))
    if text:
        out = (out or b"").decode("utf-8", "replace")
        err = (err or b"").decode("utf-8", "replace")
    result = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=out,
            stderr=err[-STDERR_TAIL_BYTES:] if err else err)
    return result
