from signal import signal, alarm, SIGALRM
from tempfile import TemporaryFile
from os import environ
from subprocess import Popen, PIPE
from traceback import format_exc
from os.path import join, abspath, dirname, exists
from hashlib import md5
from typing import Optional, List, Union

from chemistry_helpers.io import encode_if_necessary

class BabelTimeoutError(Exception):
    pass

class BabelFailure(Exception):
    pass

class Babel_Screw_Up(Exception):
    '''Known cases where Babel completely screws up: breaks bonds, modifies formula, deletes atoms, etc.'''
    pass

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
    assert type(in_data) in (str, bytes), 'Error: Invalid in_data type: {0} (expected str or bytes)'.format(type(in_data))

    args = [babel_executable, "-i" + in_format, "-o" + out_format] + (['--title', title] if title else []) + (['-p', str(pH)] if pH else []) + extra_args.split()

    def _handle_babel_timeout(signum, frame):
        raise BabelTimeoutError('Running Babel timed out after: {0}s (args="{1}")'.format(timeout, ' '.join( args)))

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
        proc = Popen(args, stdin=tmp_file, stdout=PIPE, stderr=PIPE, env=custom_env)
    finally:
        tmp_file.close()

    signal(SIGALRM, _handle_babel_timeout)
    alarm(timeout)
    try:
        stdout, stderr = proc.communicate()
    except BabelTimeoutError as e:
        try:
            proc.terminate()
        except:
            trace = format_exc()
        dump_babel_failure(in_data, ' '.join(args))
        raise e
    finally:
        alarm(0)

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

def dump_babel_failure(in_data: Union[str, bytes], babel_command: str) -> bool:
    module_dir = dirname(abspath(__file__))
    LOG_DIR = 'logs'
    log_path = join(
        module_dir,
        LOG_DIR,
        md5(babel_command.encode() + encode_if_necessary(in_data)).hexdigest()[:5] + '.log',
    )

    with open(log_path, 'w' + ('t' if isinstance(in_data, str) else 'b')) as fh:
        fh.write(in_data)
    with open(log_path.replace('.log', '.sh'), 'wt') as fh:
        fh.write(babel_command + '\n')
    return True

if __name__ == '__main__':
    print(babel_output('HETATM    1   F3 W60R    0       0.917  -1.314  -1.374  1.00  0.00           F', in_format='pdb', out_format='inchi', debug=True))
    print(babel_output('HETATM    1   F3 W60R    0       0.917  -1.314  -1.374  1.00  0.00           F', in_format='pdb', out_format='inchi', dont_add_H=True))
    print(babel_output('CCC', in_format='smiles', out_format='inchi'))
    print(babel_output('ACC', in_format='smiles', out_format='inchi'))
