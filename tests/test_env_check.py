"""`benchmarks/env_check.py`: the first thing run on a new machine."""
import subprocess
import sys

import pytest

ec = pytest.importorskip('benchmarks.env_check')


def test_a_failed_check_is_reported_and_fails_the_exit():
    text = ec.report([('python', None, '3.x'), ('square solve', False, 'no LAPACK')])
    assert '[FAIL] square solve' in text and 'FAILED: square solve' in text
    assert 'all checks passed' in ec.report([('chladni', True, '0.0.1')])


@pytest.mark.slow
def test_env_check_passes_here():
    """One tiny solve; a couple of seconds."""
    p = subprocess.run([sys.executable, '-m', 'benchmarks.env_check'], capture_output=True,
                       text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    for name in ('python', 'chladni', 'gsvd4py', 'square solve', 'douse sha', 'chladni sha'):
        assert name in p.stdout
