"""`|rho|` doubling as `delta` halves means `F` is discontinuous at the iterate.

The one diagnostic in the optimizer that needs no calibrated constant. `rho = dF/gain` with
`gain` proportional to `delta`, so `|rho|` doubling on every halving says `dF` is not going to
zero with the step -- impossible for a continuous objective, and the fingerprint of a tracked
spectrum that disagrees with the one a fresh solve finds at the same shape.

Three of the eleven stalled rows in the 18-cell grid are this failure. Their Weyl deficits (0.13
to 0.48) never reached the 0.5 gate, and one has an endpoint `F` gap of exactly zero, so nothing
else fires on it at all.

THE TRIGGER IS TESTED AS ARITHMETIC, and the end-to-end evidence comes from re-running the three
REAL cases (`benchmarks/run/ratio_multistart/detector`). Three attempts at a synthetic end-to-end
reproduction each failed for a different reason -- the poison landed on the start point, then on
a run with 6% of headroom left to outrun it, then on a call that never became the carried `F` --
and each failure was the test's, not the detector's. A contrived reproduction of a phenomenon we
hold three real instances of is the wrong instrument.
"""
import numpy as np
import pytest

from douse import SpectrumEvaluator, VertexPolygon, param_vector
from douse.cluster_opt import is_discontinuous, maximize_ratio

PERIM = 4.0


def test_constant_dF_while_the_radius_shrinks_is_the_signature():
    run = [(1e-2, -0.05), (5e-3, -0.05), (2.5e-3, -0.05), (1.25e-3, -0.05)]
    assert is_discontinuous(run, 1.25e-3)


def test_dF_shrinking_with_the_radius_is_healthy_and_must_not_trigger():
    """The ordinary case: a rejected step that is simply too long. `dF` falls with `delta`."""
    run = [(1e-2, -0.04), (5e-3, -0.02), (2.5e-3, -0.01), (1.25e-3, -0.005)]
    assert not is_discontinuous(run, 1.25e-3)


def test_a_short_run_of_rejections_never_triggers():
    run = [(1e-2, -0.05), (5e-3, -0.05), (2.5e-3, -0.05)]
    assert not is_discontinuous(run, 2.5e-3)


def test_the_radius_must_actually_have_shrunk():
    """Four rejections at the SAME radius are not the signature -- the trust region has to have
    been halving for `dF` staying put to mean anything."""
    run = [(1e-2, -0.05)]*4
    assert not is_discontinuous(run, 1e-2)


@pytest.mark.parametrize('ratio', [1.0, 1.9, 2.0])
def test_dF_within_the_ratio_still_counts_as_stationary(ratio):
    run = [(1e-2, -0.05), (5e-3, -0.05*ratio), (2.5e-3, -0.05), (1.25e-3, -0.05*ratio)]
    assert is_discontinuous(run, 1.25e-3)


def test_a_zero_dF_is_not_a_discontinuity():
    """An exactly-zero measured change divides nothing by nothing; it is not evidence."""
    run = [(1e-2, 0.0), (5e-3, 0.0), (2.5e-3, 0.0), (1.25e-3, 0.0)]
    assert not is_discontinuous(run, 1.25e-3)


@pytest.mark.slow
def test_a_healthy_run_never_triggers_the_repair():
    """It is evaluated on every rejection, so a false positive would cost a solve on every hard
    step of every run."""
    fam = VertexPolygon(4, perim=PERIM)
    p = param_vector(np.array([0, 1.3, 1.3 + 0.7j, 0.7j], dtype=complex), PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    res = maximize_ratio(ev, p, k=1, max_iter=60, delta0=0.05)
    assert res.n_discontinuity_repairs == 0 and res.success
