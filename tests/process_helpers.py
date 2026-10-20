"""Shared by the process-group tests (here and, copied by value, in the consumers)."""
import os
import sys
import time

# A child that starts a grandchild ignoring SIGTERM, records its pid, then sleeps.
GRANDCHILD_SCRIPT = (
    "import subprocess, sys, time\n"
    "g = subprocess.Popen([sys.executable, '-c', "
    "\"import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)\"])\n"
    "open(sys.argv[1], 'w').write(str(g.pid))\n"
    "time.sleep(60)\n")


def wait_for_pid(path, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            text = open(path).read().strip()
            if text:
                return int(text)
        except OSError:
            pass
        time.sleep(0.05)
    raise AssertionError("grandchild never started")


def alive(pid):
    """True while *pid* exists and is not a zombie."""
    try:
        with open("/proc/{0}/stat".format(pid)) as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def wait_dead(pid, seconds=5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


def kill_leftover(pid):
    try:
        os.kill(pid, 9)
    except OSError:
        pass


PYTHON = sys.executable
