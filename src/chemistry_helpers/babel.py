import os
import signal
import sys
from ctypes import CDLL, c_int, c_ulong
from ctypes.util import find_library
from tempfile import TemporaryFile, gettempdir
from os import environ, makedirs
from subprocess import Popen, PIPE, TimeoutExpired
from os.path import join, abspath, dirname, exists
from hashlib import md5
from typing import Optional, Union

from chemistry_helpers.io import encode_if_necessary


class BabelTimeoutError(Exception):
    pass


class BabelFailure(Exception):
    pass


class Babel_Screw_Up(Exception):
    '''Known cases where Babel completely screws up: breaks bonds, modifies formula, deletes atoms, etc.'''
    pass


PR_SET_PDEATHSIG = 1

# Resolved once, at import time, with the signature pinned, so that the child
# does not have to dlopen anything or infer argument types between fork() and
# exec() -- the less work done there the better, since only async-signal-safe
# operations are strictly legal in a forked child of a threaded process.
try:
    _LIBC = CDLL(find_library('c') or 'libc.so.6', use_errno=True)
    _LIBC.prctl.argtypes = [c_int, c_ulong, c_ulong, c_ulong, c_ulong]
    _LIBC.prctl.restype = c_int
except Exception:
    _LIBC = None


def _isolate_child() -> None:
    '''Runs in the child between fork() and exec().

    Babel is a CPU-bound converter that can wedge on a pathological molecule.
    Two things have to be true for it never to outlive the request that started
    it: it must be signallable as a group, and it must not survive its parent.

    Orphaned babel processes spinning at ~100% CPU for hours were traced to the
    second condition. babel_output() waits up to `timeout` seconds, but a web
    worker is killed well before that by Apache's FcgidIOTimeout (and by
    `make reload`, or any restart), so the code that would have killed babel
    never ran and the child was reparented to PID 1 and left running.
    PR_SET_PDEATHSIG makes the kernel do that cleanup for us.'''
    os.setpgid(0, 0)
    if _LIBC is not None:
        _LIBC.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        # Closes the race where the parent died between fork() and the prctl
        # above, which would leave the death signal armed against nobody.
        if os.getppid() == 1:
            os._exit(1)


def _terminate_process_tree(proc: Popen, grace: float = 5.0) -> None:
    '''Stop `proc` and anything it spawned, then reap it.

    Signals the process group rather than just proc.pid so a converter that
    forked helpers cannot leave stragglers, and escalates to SIGKILL for
    anything that ignores SIGTERM. The reap matters: the previous
    implementation called proc.kill() and never waited, leaving a zombie and
    the pipes open.'''
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            # Already gone, or it never reached _isolate_child() and so is not
            # in a group of its own -- fall back to signalling it directly.
            try:
                proc.send_signal(sig)
            except (ProcessLookupError, OSError):
                return
        try:
            proc.communicate(timeout=grace)
            return
        except TimeoutExpired:
            continue
        except (ValueError, OSError):
            return


def babel_output(
        in_data: str,
        in_format: Optional[str] = None,
        out_format: Optional[str] = None,
        dont_add_H: bool = False,
        debug: bool = False,
        babel_executable: str = '/usr/local/bin/babel',
        babel_libdir: str = None,
        timeout: int = 60,
        title: Optional[str] = None,
        extra_args: str = '',
        pH: Optional[float] = None,
) -> str:
    assert exists(babel_executable), 'Error: Babel executable "{0}" do not exist'.format(babel_executable)
    assert type(in_data) in (str, bytes), 'Error: Invalid in_data type: {0} (expected str or bytes)'.format(
        type(in_data))

    args = [babel_executable, "-i" + in_format, "-o" + out_format] + (['--title', title] if title else []) + (
        ['-p', str(pH)] if pH else []) + extra_args.split()

    if debug:
        print(' '.join(args))

    custom_env = environ.copy()

    if dont_add_H:
        custom_env["DONT_FIX_H_INCHI"] = "1"

    if babel_libdir is not None:
        custom_env['BABEL_LIBDIR'] = babel_libdir

    tmp_file = TemporaryFile(buffering=0)
    try:
        tmp_file.write(encode_if_necessary(in_data))
        tmp_file.seek(0)
        proc = Popen(args, stdin=tmp_file, stdout=PIPE, stderr=PIPE, env=custom_env, preexec_fn=_isolate_child)
    finally:
        tmp_file.close()

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except TimeoutExpired:
        _terminate_process_tree(proc)
        dump_babel_failure(in_data, ' '.join(args))
        raise BabelTimeoutError('Running Babel timed out after {0}s (args="{1}")'.format(timeout, ' '.join(args)))
    except BaseException:
        # Celery's SoftTimeLimitExceeded, KeyboardInterrupt, a cancelled thread:
        # kill the whole group on every exit path (PDEATHSIG only fires when the
        # forking thread exits).
        _terminate_process_tree(proc)
        raise

    if b'ERROR: not a valid' in stderr:
        dump_babel_failure(in_data, ' '.join(args))
        raise BabelFailure(stderr.decode())

    try:
        babel_stdout = stdout.strip().decode()
    except UnicodeDecodeError:
        dump_babel_failure(in_data, ' '.join(args))
        dump_babel_failure(stdout, ' '.join(args))
        raise
    if len(babel_stdout) > 0:
        return babel_stdout
    else:
        dump_babel_failure(in_data, ' '.join(args))
        raise BabelFailure(stderr.decode())


# Failure dumps hold the full (possibly private) input, so they live under the
# runtime-state root, not inside the installed package, and are capped.
# Override with ATB_CHEMISTRY_HELPERS_BABEL_FAILURE_DIR (legacy alias:
# BABEL_FAILURE_LOG_DIR). Tests may set this module attribute directly.
BABEL_FAILURE_LOG_DIR: Optional[str] = None
BABEL_FAILURE_MAX_FILES = 400   # dump = .log + .sh pair, so ~200 failures
BABEL_FAILURE_MAX_BYTES = 50 * 1024 * 1024


def babel_failure_dir() -> str:
    """Resolve the dump directory: module attribute, env override, then
    ``atb_server_settings.project_workdir('chemistry_helpers')/babel_failures``.

    atb_server_settings is imported lazily and only as a soft dependency: this
    package is deliberately dependency-free (it is imported by ~12 siblings and
    atb_server_settings pulls in pydantic-settings), so it is not declared. If it
    is absent the ATB_WORKDIR env var (the same variable that setting reads) is
    used, and finally the system temp dir."""
    if BABEL_FAILURE_LOG_DIR:
        return BABEL_FAILURE_LOG_DIR
    override = environ.get('ATB_CHEMISTRY_HELPERS_BABEL_FAILURE_DIR') or environ.get('BABEL_FAILURE_LOG_DIR')
    if override:
        return override
    try:
        from atb_server_settings import project_workdir
        return join(str(project_workdir('chemistry_helpers')), 'babel_failures')
    except Exception:
        root = environ.get('ATB_WORKDIR') or join(gettempdir(), 'atb_workdir')
        return join(root, 'chemistry_helpers', 'babel_failures')


def _prune_failure_dir(directory: str) -> None:
    """Delete the oldest files until the directory is within the count and size caps."""
    entries = []
    for name in os.listdir(directory):
        path = join(directory, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        entries.append((st.st_mtime, path, st.st_size))
    entries.sort()
    total = sum(e[2] for e in entries)
    count = len(entries)
    for _, path, size in entries:
        if count <= BABEL_FAILURE_MAX_FILES and total <= BABEL_FAILURE_MAX_BYTES:
            break
        try:
            os.remove(path)
        except OSError:
            continue
        count -= 1
        total -= size


def dump_babel_failure(in_data: Union[str, bytes], babel_command: str) -> bool:
    '''Record the input that made Babel fail, for later reproduction.

    Strictly best effort. This is only ever called while another exception is
    being raised, so it must not raise one of its own: the log directory did not
    exist on the deployed host, and the resulting FileNotFoundError propagated
    in place of the BabelTimeoutError/BabelFailure that callers catch -- turning
    a handled "conversion timed out" into an unhandled 500.'''
    log_path = '(unresolved)'
    try:
        directory = babel_failure_dir()
        log_path = join(
            directory,
            md5(babel_command.encode() + encode_if_necessary(in_data)).hexdigest() + '.log',
        )
        makedirs(directory, exist_ok=True)
        with open(log_path, 'w' + ('t' if isinstance(in_data, str) else 'b')) as fh:
            fh.write(in_data)
        with open(log_path[:-len('.log')] + '.sh', 'wt') as fh:
            fh.write(babel_command + '\n')
        _prune_failure_dir(directory)
        return True
    except Exception as e:
        print('WARNING: could not record Babel failure to {0}: {1}'.format(log_path, e), file=sys.stderr)
        return False


if __name__ == '__main__':
    print(
        babel_output('HETATM    1   F3 W60R    0       0.917  -1.314  -1.374  1.00  0.00           F', in_format='pdb',
                     out_format='inchi', debug=True))
    print(
        babel_output('HETATM    1   F3 W60R    0       0.917  -1.314  -1.374  1.00  0.00           F', in_format='pdb',
                     out_format='inchi', dont_add_H=True))
    print(babel_output('CCC', in_format='smiles', out_format='inchi'))
    print(babel_output('ACC', in_format='smiles', out_format='inchi'))
