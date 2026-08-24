"""Objectives on a spectrum, and their gradients. Milestone 2.

Two layers, tested separately.

The ARITHMETIC layer (`normalized_reciprocals`, `l2_loss`) is a function of the eigenvalues
alone -- no geometry, no solver -- so it is differenced directly and cheaply, and a failure there
cannot be confused with a shape-derivative problem.

The COMPOSED gradient (`ObjectiveFunction`) is then checked against a finite difference in `p`,
which is the thing an optimizer will actually consume. Its finite-difference reference needs the
same care as Milestone 1's: a tight plan target so the tension floor does not limit how well
lambda can be located, and continuation seeding so the tracker never jumps a full `2h`. See
`test_derivative.py` for the 3.7e-04 that cost.
"""
import warnings

import numpy as np
import pytest

from douse.polygon import VertexPolygon, param_vector
from douse.objective import (normalized_reciprocals, l2_loss, EigenvalueObjective,
                             SpectralMatch)
from douse.evaluate import SpectrumEvaluator, ObjectiveFunction

PERIM = 4.0
SEED_SHAPE = np.array([0.0, 1.3, 1.1 + 0.9j, 0.15 + 0.8j])
H = 1e-3
LAMS = np.array([19.3, 48.1, 49.4, 78.6])          # a plausible spectrum, no solver needed


def _fd_jac(f, x, h=1e-6):
    """Central-difference Jacobian of a vector function."""
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        cols.append((np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e)))/(2*h))
    return np.array(cols).T


# ── The arithmetic layer ────────────────────────────────────────────────────────

def test_normalized_reciprocals_jacobian():
    nu, J = normalized_reciprocals(LAMS, jac=True)
    assert nu.shape == (len(LAMS) - 1,)
    assert J.shape == (len(LAMS) - 1, len(LAMS))
    assert np.max(np.abs(J - _fd_jac(normalized_reciprocals, LAMS))) < 1e-8


def test_normalized_reciprocals_are_scale_invariant():
    """The property the whole inverse problem rests on: a dilated domain has eigenvalues scaled
    by 1/c^2, and these coordinates must not move."""
    for c in (0.25, 1.0, 7.0):
        assert np.allclose(normalized_reciprocals(LAMS/c**2), normalized_reciprocals(LAMS),
                           rtol=1e-14)


def test_normalized_reciprocals_are_bounded_by_one():
    """lam_1 <= lam_j, so every coordinate lies in (0, 1] and no mode dominates the loss just
    for being large."""
    nu = normalized_reciprocals(LAMS)
    assert np.all(nu > 0) and np.all(nu <= 1.0 + 1e-15), nu


def test_l2_loss_gradient():
    nu = normalized_reciprocals(LAMS)
    target = nu*1.05
    _, g = l2_loss(nu, target, jac=True)
    fd = _fd_jac(lambda x: l2_loss(x, target), nu).ravel()
    assert np.max(np.abs(g - fd)) < 1e-8

    _, gl = l2_loss(nu, target, jac=True, log=True)
    fdl = _fd_jac(lambda x: l2_loss(x, target, log=True), nu).ravel()
    assert np.max(np.abs(gl - fdl)) < 1e-7


def test_the_log_loss_is_refused_at_an_exact_match():
    """Its gradient is the plain one divided by the loss; at zero that is an infinity, and
    returning one silently would poison a line search."""
    nu = normalized_reciprocals(LAMS)
    with pytest.raises(ValueError, match='undefined at an exact match'):
        l2_loss(nu, nu, log=True)


def test_a_spectral_match_is_zero_on_a_rescaled_copy_of_itself():
    """The end of the inverse problem: the target's SIZE is unknown and must not matter."""
    match = SpectralMatch(LAMS)
    dlams = np.zeros((len(LAMS), 4))
    for c in (0.3, 1.0, 5.0):
        val, _ = match(LAMS/c**2, dlams)
        assert val < 1e-25, (c, val)


def test_objectives_report_what_they_need():
    assert EigenvalueObjective(0).n_eigs == 1
    assert EigenvalueObjective(2).n_eigs == 3
    assert SpectralMatch(LAMS).n_eigs == len(LAMS)
    with pytest.raises(ValueError, match='at least two target eigenvalues'):
        SpectralMatch(LAMS[:1])


def test_an_objective_refuses_a_short_spectrum():
    with pytest.raises(ValueError, match='needs lam_3'):
        EigenvalueObjective(2)(LAMS[:2], np.zeros((2, 4)))
    with pytest.raises(ValueError, match='expected 4 eigenvalues'):
        SpectralMatch(LAMS)(LAMS[:3], np.zeros((3, 4)))


def test_the_evaluator_refuses_an_objective_it_cannot_feed():
    ev = SpectrumEvaluator(VertexPolygon(4, perim=PERIM), n_eigs=1)
    with pytest.raises(ValueError, match='needs 3 eigenvalues'):
        ObjectiveFunction(ev, EigenvalueObjective(2))


# ── The composed gradient ───────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def p0():
    return param_vector(SEED_SHAPE, PERIM)


def _fd_scalar(obj, p0, direction, h, lams0):
    """Five-point difference of an objective along `direction`, by continuation.

    Each stencil point is seeded from its neighbour: the evaluator tracks from whatever it saw
    last, so walking outward from the centre keeps every step short.
    """
    vals, ev = {}, obj.evaluator
    for signed in ((h, 2*h), (-h, -2*h)):
        ev._last = (np.asarray(p0, dtype=float), np.asarray(lams0, dtype=float))
        for t in signed:
            vals[t] = obj(p0 + t*direction)[0]
    return (vals[-2*h] - 8*vals[-h] + 8*vals[h] - vals[2*h])/(12*h)


@pytest.mark.slow
def test_a_single_eigenvalue_objective_gradient(p0):
    """`min lam_1` -- the objective Milestone 3 optimizes."""
    ev = SpectrumEvaluator(VertexPolygon(4, perim=PERIM), n_eigs=1)
    obj = ObjectiveFunction(ev, EigenvalueObjective(0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val, g = obj(p0)
        lams0 = ev._last[1]
        d = np.array([0.4, -0.55, 0.6, -0.42])
        d = d/np.linalg.norm(d)
        fd = _fd_scalar(obj, p0, d, H, lams0)
    assert 15.0 < val < 30.0, val
    assert abs(g @ d - fd) < 1e-6, (g @ d, fd)


@pytest.mark.slow
def test_a_spectral_match_gradient(p0):
    """The full composition: p -> lam (3) -> scale-invariant ratios -> squared loss.

    Targets are the spectrum of a DIFFERENT shape, so the loss is well away from zero and its
    gradient is not degenerate.
    """
    ev = SpectrumEvaluator(VertexPolygon(4, perim=PERIM), n_eigs=3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams_here, _ = ev(p0)
    targets = lams_here*np.array([1.0, 1.06, 0.95])       # perturbed, so nu_target differs
    obj = ObjectiveFunction(ev, SpectralMatch(targets))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val, g = obj(p0)
        lams0 = ev._last[1]
        d = np.array([0.31, 0.47, -0.58, 0.58])
        d = d/np.linalg.norm(d)
        fd = _fd_scalar(obj, p0, d, H, lams0)
    assert val > 1e-8, f'the target should not already be matched, loss={val:.2e}'
    assert abs(g @ d - fd) < 1e-6*max(1.0, abs(fd)), (g @ d, fd)


@pytest.mark.slow
def test_the_frozen_plan_gives_one_basis_size_across_the_sweep(p0):
    """The architectural claim the evaluator rests on. If the basis size moved between iterates
    the objective would be discontinuous and every gradient above would be measuring the
    discretization as much as the shape."""
    ev = SpectrumEvaluator(VertexPolygon(4, perim=PERIM), n_eigs=1)
    sizes = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for t in (-0.02, -0.01, 0.0, 0.01, 0.02):
            p = p0.copy()
            p[0] += t
            _, solver = ev.eigenvalues(p)
            sizes.append(len(solver.basis))
    assert len(set(sizes)) == 1, sizes


@pytest.mark.slow
def test_the_evaluator_tracks_instead_of_rescanning(p0):
    """Continuation is the cheap path and the evaluator should be taking it: one full solve, then
    tracking. A silent fallback to rescanning every iterate would still be correct but would cost
    several times as much, so it is worth pinning."""
    ev = SpectrumEvaluator(VertexPolygon(4, perim=PERIM), n_eigs=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for t in (0.0, 0.005, 0.01, 0.015):
            p = p0.copy()
            p[0] += t
            ev.eigenvalues(p)
    assert ev.n_solves == 1, f'{ev.n_solves} full solves; continuation is not being used'
    assert ev.n_tracks == 3, ev.n_tracks
