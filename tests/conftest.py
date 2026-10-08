"""Shared fixtures. Nothing here may write into the package's own ``logs/`` directory:
``chemistry_helpers.babel.dump_babel_failure`` defaults to ``<package dir>/logs``
(110 MB of failure dumps accumulated there in production, review 2026-10 G9)."""
import os

import pytest

import chemistry_helpers.babel as babel

PACKAGE_LOGS = os.path.join(os.path.dirname(os.path.abspath(babel.__file__)), 'logs')


def _listing():
    return sorted(os.listdir(PACKAGE_LOGS)) if os.path.isdir(PACKAGE_LOGS) else None


@pytest.fixture(autouse=True)
def failure_log_dir(tmp_path, monkeypatch):
    """Redirect failure dumps to tmp and fail the test if the package logs/ dir changed."""
    before = _listing()
    target = tmp_path / 'babel_failures'
    monkeypatch.setattr(babel, 'BABEL_FAILURE_LOG_DIR', str(target))
    yield target
    assert _listing() == before, 'test wrote into the package logs/ directory'


def make_script(tmp_path, body, name='fake_babel'):
    path = tmp_path / name
    path.write_text('#!/bin/sh\n' + body + '\n')
    path.chmod(0o755)
    return str(path)
