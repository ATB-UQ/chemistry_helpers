"""Timeout / process-group-kill path of chemistry_helpers.babel, exercised with fake
executables (shell scripts) so that babel itself is not needed."""
import os
import signal
import subprocess
import time

import pytest

import chemistry_helpers.babel as babel
from chemistry_helpers.babel import BabelFailure, BabelTimeoutError, babel_output

from conftest import make_script

PDB3 = ('HETATM    1  C1  RES     1       0.000   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    2  H1  RES     1       1.000   0.000   0.000  1.00  0.00           H  \n'
        'HETATM    3  O1  RES     1       0.000   1.200   0.000  1.00  0.00           O  \n')


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); read its state
    try:
        with open('/proc/{0}/stat'.format(pid)) as fh:
            return fh.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def wait_dead(pids, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end and any(alive(p) for p in pids):
        time.sleep(0.05)
    return [p for p in pids if alive(p)]


def read_pid(path, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError('no pid file')


def test_success_passes_stdin_through_and_strips(tmp_path):
    exe = make_script(tmp_path, 'cat; echo; echo')
    assert babel_output('CCC', in_format='smiles', out_format='inchi', babel_executable=exe) == 'CCC'


def test_args_are_built_from_formats_title_ph_and_extra(tmp_path):
    exe = make_script(tmp_path, 'echo "$@"')
    out = babel_output('x', in_format='pdb', out_format='mol2', babel_executable=exe,
                       title='T', pH=7.4, extra_args='-xC -xd')
    assert out == '-ipdb -omol2 --title T -p 7.4 -xC -xd'


def test_dont_add_h_and_libdir_go_through_the_environment(tmp_path):
    exe = make_script(tmp_path, 'echo "${DONT_FIX_H_INCHI:-unset}:${BABEL_LIBDIR:-unset}"')
    assert babel_output('x', in_format='a', out_format='b', babel_executable=exe) == 'unset:unset'
    assert babel_output('x', in_format='a', out_format='b', babel_executable=exe,
                        dont_add_H=True, babel_libdir='/lib/x') == '1:/lib/x'


def test_bytes_input_is_accepted(tmp_path):
    exe = make_script(tmp_path, 'cat')
    assert babel_output(b'abc', in_format='a', out_format='b', babel_executable=exe) == 'abc'


def test_missing_executable_is_an_assertion_error():
    with pytest.raises(AssertionError, match='do not exist'):
        babel_output('x', in_format='a', out_format='b', babel_executable='/nonexistent/babel')


def test_wrong_input_type_is_an_assertion_error(tmp_path):
    exe = make_script(tmp_path, 'cat')
    with pytest.raises(AssertionError, match='Invalid in_data type'):
        babel_output(123, in_format='a', out_format='b', babel_executable=exe)


def test_missing_format_is_a_typeerror_not_a_babel_error(tmp_path):
    # REVIEW (P3): in_format/out_format default to None but "-i" + None raises TypeError.
    exe = make_script(tmp_path, 'cat')
    with pytest.raises(TypeError):
        babel_output('x', babel_executable=exe)


def test_empty_output_raises_babelfailure_and_dumps_input(tmp_path, failure_log_dir):
    exe = make_script(tmp_path, 'cat >/dev/null; echo "boom" >&2')
    with pytest.raises(BabelFailure, match='boom'):
        babel_output('INPUT', in_format='a', out_format='b', babel_executable=exe)
    logs = sorted(os.listdir(failure_log_dir))
    assert len(logs) == 2 and {os.path.splitext(n)[1] for n in logs} == {'.log', '.sh'}
    log = [n for n in logs if n.endswith('.log')][0]
    assert (failure_log_dir / log).read_text() == 'INPUT'
    assert (failure_log_dir / log.replace('.log', '.sh')).read_text().startswith(exe)


def test_not_a_valid_on_stderr_raises_babelfailure_even_with_stdout(tmp_path):
    exe = make_script(tmp_path, 'echo out; echo "ERROR: not a valid input" >&2')
    with pytest.raises(BabelFailure, match='not a valid'):
        babel_output('x', in_format='a', out_format='b', babel_executable=exe)


def test_undecodable_stdout_is_reraised_and_both_dumped(tmp_path, failure_log_dir):
    exe = make_script(tmp_path, "printf '\\377\\376'")
    with pytest.raises(UnicodeDecodeError):
        babel_output('x', in_format='a', out_format='b', babel_executable=exe)
    assert any(n.endswith('.log') for n in os.listdir(failure_log_dir))


def test_dump_failure_never_raises(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / 'a_file'
    blocker.write_text('x')
    monkeypatch.setattr(babel, 'BABEL_FAILURE_LOG_DIR', str(blocker / 'sub'))
    assert babel.dump_babel_failure('data', 'cmd') is False
    assert 'WARNING: could not record Babel failure' in capsys.readouterr().err


def test_dump_name_is_five_hex_chars_of_md5(tmp_path, failure_log_dir):
    assert babel.dump_babel_failure('data', 'cmd') is True
    names = os.listdir(failure_log_dir)
    stems = {os.path.splitext(n)[0] for n in names}
    assert len(stems) == 1 and len(stems.pop()) == 5  # REVIEW (P3): 20-bit key, collisions overwrite


def test_timeout_raises_and_kills_whole_process_group(tmp_path, failure_log_dir):
    pidfile = tmp_path / 'child.pid'
    exe = make_script(tmp_path, 'sleep 300 &\necho $! > {0}\nwait'.format(pidfile))
    t0 = time.time()
    with pytest.raises(BabelTimeoutError, match='timed out after 1s'):
        babel_output('x', in_format='a', out_format='b', babel_executable=exe, timeout=1)
    assert time.time() - t0 < 20
    grandchild = read_pid(pidfile)
    assert wait_dead([grandchild]) == [], 'grandchild survived the timeout kill'
    assert any(n.endswith('.log') for n in os.listdir(failure_log_dir))


def test_timeout_kills_sigterm_ignoring_child_by_escalating_to_sigkill(tmp_path):
    pidfile = tmp_path / 'child.pid'
    exe = make_script(tmp_path, "trap '' TERM\necho $$ > {0}\nwhile :; do sleep 1; done".format(pidfile))
    proc = subprocess.Popen([exe], preexec_fn=babel._isolate_child,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    read_pid(pidfile)
    babel._terminate_process_tree(proc, grace=0.5)
    assert proc.poll() == -signal.SIGKILL


def test_isolate_child_puts_child_in_its_own_group(tmp_path):
    exe = make_script(tmp_path, 'ps -o pgid= -p $$')
    out = babel_output('x', in_format='a', out_format='b', babel_executable=exe)
    assert int(out) != os.getpgid(0)


def test_keyboard_interrupt_during_wait_does_not_kill_the_child(tmp_path, monkeypatch):
    """REVIEW (P2): only TimeoutExpired is handled. Any other exception out of communicate()
    (KeyboardInterrupt, Celery's SoftTimeLimitExceeded, a worker thread being cancelled) leaves
    babel running until PR_SET_PDEATHSIG fires, which needs the parent *thread* to exit. The
    repo CLAUDE.md rule is 'kill by process group on every exit path, BaseException included'.
    This pins the current behaviour: the child outlives the exception."""
    pidfile = tmp_path / 'child.pid'
    exe = make_script(tmp_path, 'echo $$ > {0}\nsleep 300'.format(pidfile))
    started = []
    real_popen = babel.Popen

    class Boom(real_popen):
        def communicate(self, *a, **kw):
            started.append(self)
            read_pid(pidfile)
            raise KeyboardInterrupt

    monkeypatch.setattr(babel, 'Popen', Boom)
    try:
        with pytest.raises(KeyboardInterrupt):
            babel_output('x', in_format='a', out_format='b', babel_executable=exe, timeout=30)
        pid = started[0].pid
        assert alive(pid), 'child was killed on KeyboardInterrupt: update this test and the REVIEW note'
    finally:
        for p in started:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            real_popen.wait(p)


@pytest.mark.skipif(not os.path.exists('/usr/local/bin/babel'), reason='needs the real babel')
def test_real_babel_tiny_pdb_to_xyz():
    out = babel_output(PDB3, in_format='pdb', out_format='xyz', timeout=30)
    assert out.splitlines()[0].strip() == '3'
