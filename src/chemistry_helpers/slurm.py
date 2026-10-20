"""Pure parsers for Slurm's text output: ``squeue`` rows and ``sbatch --parsable``.

Parsing only.  Nothing here runs a command, opens an ssh connection or knows a cluster:
transports (ssh, paramiko, shims, the ``sbatch`` invocation) stay with each consumer
(duplication plan group 9, owner decision 2026-10-20).  Pure stdlib.

**The width trap.**  ``squeue --format`` pads *and truncates* to the column width,
silently: ``%8T`` prints COMPLETING as ``COMPLETI`` (and COMPLETED as ``COMPLETE``), and
``%10j`` clips a job name.  A truncated state matches no entry in a live-state set, so a
job that is still finishing reads as terminal and gets recovered alongside itself (seen
on the helium qm0 route).  Use a bare ``%T`` or a wide ``%20T``; names ``%64j``.
:func:`parse_squeue` cannot see a clipped name, but it *can* see a clipped state (a value
that is a strict prefix of a known state) and refuses it with :class:`SlurmFormatError`:
the consumers all treat a raised error as "could not read the queue, try next tick", the
safe reading, whereas a silently wrong state is the dangerous one.
"""
from typing import Dict, List, Optional, Sequence, Tuple

#: States in which Slurm still owns the job.  Everything else ``squeue --states=all``
#: reports is terminal (COMPLETED, FAILED, TIMEOUT, CANCELLED, NODE_FAIL, OUT_OF_MEMORY,
#: BOOT_FAIL, DEADLINE, PREEMPTED, ...).  Listed the safe way round: an unknown state read
#: as live merely waits, read as terminal it gets a running job resubmitted.
LIVE_STATES = frozenset([
    'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'REQUEUED',
    'REQUEUE_HOLD', 'REQUEUE_FED', 'RESIZING', 'SIGNALING', 'STAGE_OUT', 'STOPPED',
    'SPECIAL_EXIT', 'RESV_DEL_HOLD',
])

#: Every state name ``%T`` can print (live and terminal); used to spot a truncated value.
KNOWN_STATES = LIVE_STATES | frozenset([
    'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY',
    'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED',
])


class SlurmFormatError(ValueError):
    """The squeue output was produced with a format that cannot be trusted."""


def normalise_state(state: str) -> str:
    """Upper-case bare state name: ``'cancelled by 1234'`` -> ``CANCELLED``, ``'CANCELLED+'`` -> ``CANCELLED``."""
    parts = state.strip().upper().split()
    return parts[0].rstrip('+') if parts else ''


def is_live(state: str) -> bool:
    return normalise_state(state) in LIVE_STATES


def is_truncated_state(state: str) -> bool:
    """A strict prefix of a known state: what a too-narrow ``%<n>T`` prints."""
    s = state.strip().upper()
    return bool(s) and s not in KNOWN_STATES and any(k.startswith(s) for k in KNOWN_STATES)


def parse_squeue(text: str, fields: Sequence[str], sep: Optional[str] = None,
                 required: Optional[int] = None, extra: str = 'skip',
                 id_field: Optional[str] = None, state_field: str = 'state') -> List[Dict[str, str]]:
    """Rows of ``squeue --noheader`` output as ``{field: value}`` dicts, in order.

    *fields* names the columns of the ``--format`` string, in order.  *sep* is ``None``
    for whitespace-separated columns (``'%i %T'``, ``'%64j %20T'``; a value therefore
    cannot contain a space) or the delimiter (``'|'`` for ``'%i|%T|%r|%j'``, whose
    reason column may contain spaces).  Lines that do not fit are skipped, never raised
    on, because the stream can carry a login banner or a blank line:

    * fewer than *required* columns (default: all of *fields*) -> skipped; the columns
      after *required* but missing are ``''``;
    * more columns than *fields*: ``extra='skip'`` drops the line (whitespace mode, where
      an unexpected extra token means the row is not what we think), ``extra='ignore'``
      keeps the first columns;
    * *id_field*, if given, drops rows whose value does not start with a digit (a
      banner or a stray ``#`` line; job ids, ``123``, ``123_4``, ``123_[0-9]``, ``123+1``,
      always do).

    Raises :class:`SlurmFormatError` when a *state_field* value is a truncated state
    (see the module docstring); values are otherwise returned verbatim (stripped).
    """
    n = len(fields)
    need = n if required is None else required
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(sep) if sep is not None else line.split()
        if len(parts) < need or (len(parts) > n and extra == 'skip'):
            continue
        parts = [p.strip() for p in parts[:n]]
        parts += [''] * (n - len(parts))
        row = dict(zip(fields, parts))
        if id_field is not None and not row[id_field][:1].isdigit():
            continue
        if not row[id_field or fields[0]] and sep is not None:
            continue
        if state_field in row and is_truncated_state(row[state_field]):
            raise SlurmFormatError(
                'squeue state {0!r} looks truncated by a narrow %<n>T width '
                '(COMPLETING prints as COMPLETI): use %T or %20T'.format(row[state_field]))
        rows.append(row)
    return rows


def parse_sbatch_parsable(text: str) -> Tuple[str, Optional[str]]:
    """``(job id, cluster)`` from ``sbatch --parsable`` output; ``('', None)`` if there is none.

    ``--parsable`` prints ``<jobid>`` or ``<jobid>;<cluster>`` (federation).  A login
    shell may have said anything first (a banner, a ``module`` message), so the LAST
    non-empty line is the answer; it must start with a digit.  The cluster is ``None``
    when absent.  ``squeue`` reports the id alone, so callers use the first element.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return '', None
    job_id, _, cluster = lines[-1].partition(';')
    job_id = job_id.strip()
    if not job_id[:1].isdigit():
        return '', None
    return job_id, (cluster.strip() or None)
