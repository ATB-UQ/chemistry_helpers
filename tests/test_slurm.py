"""chemistry_helpers.slurm against the three consumer parsers it replaced.

The OLD_* functions below are the consumers' parsers copied verbatim before migration
(compute_tasks slurm_fleet.poll loop, dihedral_scan_service remote/setonix.py
parse_squeue and _parse_job_id, protein_validation_pipeline queues.py parse_squeue).
Fixtures in tests/data/slurm are real captures from helium and Setonix (2026-10-20).
"""
import re
from pathlib import Path

import pytest

from chemistry_helpers import slurm

DATA = Path(__file__).parent / 'data' / 'slurm'
BANNER = (DATA / 'setonix_login_banner_only.txt').read_text()

LIVE = frozenset(['PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'REQUEUED',
                  'REQUEUE_HOLD', 'REQUEUE_FED', 'RESIZING', 'SIGNALING', 'STAGE_OUT', 'STOPPED',
                  'SPECIAL_EXIT', 'RESV_DEL_HOLD'])
_JOB_NAME = re.compile(r'^atb(\d+)$')


def OLD_fleet(out):
    live, ended = set(), {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        (name, state) = fields
        match = _JOB_NAME.match(name)
        if match is None:
            continue
        molid = int(match.group(1))
        if state.upper() in LIVE:
            live.add(molid)
        else:
            ended[molid] = state.upper()
    return {'live': live, 'ended': {m: s for (m, s) in ended.items() if m not in live}}


def NEW_fleet(out):
    live, ended = set(), {}
    for row in slurm.parse_squeue(out, ('name', 'state')):
        match = _JOB_NAME.match(row['name'])
        if match is None:
            continue
        molid = int(match.group(1))
        if slurm.is_live(row['state']):
            live.add(molid)
        else:
            ended[molid] = row['state'].upper()
    return {'live': live, 'ended': {m: s for (m, s) in ended.items() if m not in live}}


def OLD_setonix(text):
    states = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        job_id, state = parts[0], parts[1]
        if not job_id[:1].isdigit():
            continue
        states[job_id] = state
        if '_' in job_id:
            states.setdefault(job_id.split('_')[0], state)
    return states


def NEW_setonix(text):
    states = {}
    for row in slurm.parse_squeue(text, ('job_id', 'state'), required=2, extra='ignore', id_field='job_id'):
        job_id, state = row['job_id'], row['state']
        states[job_id] = state
        if '_' in job_id:
            states.setdefault(job_id.split('_')[0], state)
    return states


def OLD_pipe(text):
    entries = {}
    for line in text.splitlines():
        parts = line.strip().split('|')
        if len(parts) < 2 or not parts[0]:
            continue
        job_id = parts[0].strip()
        entry = (job_id, parts[1].strip(), parts[2].strip() if len(parts) > 2 else '',
                 parts[3].strip() if len(parts) > 3 else '')
        entries[job_id] = entry
        entries.setdefault(job_id.split('_')[0], entry)
    return entries


def NEW_pipe(text):
    entries = {}
    for r in slurm.parse_squeue(text, ('job_id', 'state', 'reason', 'name'), sep='|',
                                required=2, extra='ignore', id_field='job_id'):
        entry = (r['job_id'], r['state'], r['reason'], r['name'])
        entries[r['job_id']] = entry
        entries.setdefault(r['job_id'].split('_')[0], entry)
    return entries


def OLD_job_id(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ''
    job_id = lines[-1].split(';')[0].strip()
    return job_id if job_id[:1].isdigit() else ''


FLEET_TEXTS = [
    (DATA / 'helium_fleet_fmt_adhoc_all_states.txt').read_text(),
    (DATA / 'helium_fleet_fmt_qm0_empty.txt').read_text(),
    BANNER,
    '',
    '\n\n',
    'atb12 RUNNING\natb13 COMPLETED\natb14 COMPLETING\natb14 COMPLETED\natb15 CANCELLED\n',
    'atb12   PENDING   \nsomeoneelse RUNNING\natb7x RUNNING\natb99\natb5 RUNNING extra\n',
    'atb1 running\natb2 Failed\n',
    'atb2051339 REQUEUE_HOLD\natb8 SPECIAL_EXIT\natb9 TIMEOUT\natb10 OUT_OF_MEMORY\n',
    BANNER + 'atb3 RUNNING\n',
]

SETONIX_TEXTS = [
    (DATA / 'helium_id_state.txt').read_text(),
    BANNER, '', '123 RUNNING\n',
    '48051331_5 RUNNING\n48051331_6 PENDING\n48051332 COMPLETED\n',
    '123_[0-9] PENDING\n124_[0-3%2] PENDING\n',
    '  77 COMPLETING  \n\n78\nabc RUNNING\n79 PENDING (Priority) extra tokens\n',
    '123+1 RUNNING\n',
    BANNER + '55 RUNNING\n',
]

PIPE_TEXTS = [
    (DATA / 'helium_pipe_fmt_empty.txt').read_text(),
    '', '\n',
    '10|RUNNING|None|md_run_1\n11|PENDING|launch failed requeued held|md_run_1\n',
    '12|PENDING|Resources|name with spaces\n',
    '13_4|RUNNING|None|arr\n13_5|PENDING|JobArrayTaskLimit|arr\n',
    '14_[0-9]|PENDING|Priority|arr\n',
    '15|COMPLETED\n16|FAILED|NonZeroExitCode\n',
    '|RUNNING|x|y\n  17 | RUNNING | None | n \n',
    '18|RUNNING|a|b|c|d\n',
    'justonefield\n',
    BANNER,
]


@pytest.mark.parametrize('text', FLEET_TEXTS)
def test_fleet_equivalent(text):
    assert NEW_fleet(text) == OLD_fleet(text)


@pytest.mark.parametrize('text', SETONIX_TEXTS)
def test_setonix_equivalent(text):
    assert NEW_setonix(text) == OLD_setonix(text)


@pytest.mark.parametrize('text', PIPE_TEXTS)
def test_pipe_equivalent(text):
    assert NEW_pipe(text) == OLD_pipe(text)


def test_real_capture_values():
    assert NEW_fleet(FLEET_TEXTS[0]) == {'live': set(), 'ended': {}}  # names are not atb<digits>
    rows = slurm.parse_squeue(FLEET_TEXTS[0], ('name', 'state'))
    assert [r['state'] for r in rows] == ['COMPLETED'] * 5 and rows[0]['name'] == 'rl57_fd6f7e82bff0'
    assert slurm.parse_squeue(BANNER, ('job_id', 'state'), required=2, extra='ignore', id_field='job_id') == []


def test_pipe_banner_with_bars_dropped_by_id_field():
    # Intended difference from queues.py: a banner line containing '|' used to become an entry.
    text = '+-----|--------+\n| NOTICE | to users |\n10|RUNNING|None|x\n'
    assert list(NEW_pipe(text)) == ['10']
    assert '+-----' in OLD_pipe(text)


@pytest.mark.parametrize('output', [
    '', '\n', '12345\n', '12345;helium\n', '  12345 ; helium \n', BANNER + '777\n',
    'Welcome\nmodule: loaded\n4242;setonix\n', 'Submitted batch job 12345\n', 'sbatch: error: x\n',
    '12345\n\n\n', 'abc\n', '12345\ntrailing text\n', ';cluster\n',
])
def test_sbatch_equivalent(output):
    assert slurm.parse_sbatch_parsable(output)[0] == OLD_job_id(output)


def test_sbatch_cluster():
    assert slurm.parse_sbatch_parsable('banner\n99;c1\n') == ('99', 'c1')
    assert slurm.parse_sbatch_parsable('99\n') == ('99', None)
    assert slurm.parse_sbatch_parsable('99;\n') == ('99', None)
    assert slurm.parse_sbatch_parsable('') == ('', None)


def test_narrow_width_is_refused():
    # Real capture: '%8T' prints COMPLETED as COMPLETE.
    with pytest.raises(slurm.SlurmFormatError):
        slurm.parse_squeue((DATA / 'helium_narrow_8T.txt').read_text(), ('state', 'job_id'))
    with pytest.raises(slurm.SlurmFormatError):
        slurm.parse_squeue('atb1 COMPLETI\n', ('name', 'state'))
    # whole state names, however odd, are never refused
    assert slurm.parse_squeue('atb1 COMPLETING\natb2 SOMEFUTURESTATE\n', ('name', 'state'))[1]['state'] == 'SOMEFUTURESTATE'


def test_state_normalisation():
    assert slurm.normalise_state('cancelled by 1234') == 'CANCELLED'
    assert slurm.normalise_state('CANCELLED+') == 'CANCELLED'
    assert slurm.normalise_state('') == ''
    assert slurm.is_live('completing') and not slurm.is_live('COMPLETED') and not slurm.is_live('')
    assert slurm.LIVE_STATES == LIVE
