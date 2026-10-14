"""The `chladni` backend: `SpectrumEvaluator` (alias `ChladniEvaluator`), douse's default.

`chladni` plans its own basis (`chladni.plan`), solves to a requested `prec` with per-level bounds
and flags, and supplies Hadamard matrices (`Spectrum.hadamard`). This checks the seam against
ground truth: the rectangle's closed form; the Jacobian against central differences of
`chladni`'s own eigenvalues; the cluster block's diagonal against the members' own simple
gradients; and the mapping of `chladni`'s shortfall flags onto `douse`'s refusal exceptions, so
an optimizer shrinks rather than trusting a flagged spectrum. The one cross-backend check, against
the deprecated lappy evaluator, is marked `lappy`.
"""
import warnings

import numpy as np
import pytest

from douse.chladni_backend import ChladniEvaluator, refusal_for
from douse.evaluate import (DegenerateSpectrum, LappySpectrumEvaluator, UnresolvedSpectrum,
                            SolverRefusal)
from douse.polygon import VertexPolygon, param_vector

PREC = 1e-10
pytestmark = pytest.mark.filterwarnings("ignore")

QUAD = np.array([0.0, 1.1, 1.25 + 0.95j, 0.15 + 0.8j])      # a generic convex quadrilateral


def _p(vertices, perim):
    return param_vector(np.asarray(vertices, dtype=complex), perim)


def test_a_rectangles_eigenvalues_match_the_closed_form():
    from benchmarks.rectangle_exact import spectrum
    L, H = 1.2, 0.8
    fam = VertexPolygon(4, perim=2 * (L + H))
    ev = ChladniEvaluator(fam, n_eigs=4, prec=PREC)
    lams, _ = ev.eigenvalues(_p([0, L, L + 1j * H, 1j * H], fam.perim))
    np.testing.assert_allclose(lams, spectrum(L, H, 6)[:4], rtol=1e-9)


@pytest.mark.slow
@pytest.mark.lappy
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_both_backends_agree_on_values_and_the_jacobian():
    fam = VertexPolygon(4, perim=float(np.sum(np.abs(np.diff(np.append(QUAD, QUAD[0]))))))
    p = _p(QUAD, fam.perim)
    lams_c, J_c = ChladniEvaluator(fam, n_eigs=3, prec=PREC)(p)
    lams_l, J_l = LappySpectrumEvaluator(fam, n_eigs=3)(p)
    np.testing.assert_allclose(lams_c, lams_l, rtol=1e-9)
    np.testing.assert_allclose(J_c, J_l, rtol=1e-6, atol=1e-6 * np.abs(J_l).max())


@pytest.mark.slow
def test_the_jacobian_matches_central_differences():
    fam = VertexPolygon(4, perim=float(np.sum(np.abs(np.diff(np.append(QUAD, QUAD[0]))))))
    p = _p(QUAD, fam.perim)
    ev = ChladniEvaluator(fam, n_eigs=2, prec=PREC)
    _, J = ev(p)
    # On the same evaluator, so the FROZEN plan's basis is the same at p +- h. The values carry
    # ~1e-10 relative error, so the difference carries ~lam * 1e-10 / h ~ 1e-6 relative at
    # h = 1e-4, and the truncation is h^2 ~ 1e-8. (At prec 1e-12 this shape is refused --
    # margin 1.3 -- which is the backend reporting, not failing.)
    h = 1e-4
    for i in range(len(p)):
        e = np.zeros_like(p)
        e[i] = h
        fd = (ev.eigenvalues(p + e)[0] - ev.eigenvalues(p - e)[0]) / (2 * h)
        np.testing.assert_allclose(J[:, i], fd, rtol=1e-5, atol=1e-5 * np.abs(J).max())


@pytest.mark.slow
def test_a_degenerate_level_refuses_the_simple_gradient():
    """The square: lam_2 = lam_3. douse's policy -- a confident wrong gradient is worse than a
    raise -- applies to this backend too; `cluster_matrices` is the path for a cluster."""
    fam = VertexPolygon(4, perim=4.0)
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    with pytest.raises(DegenerateSpectrum):
        ev(_p([0, 1, 1 + 1j, 1j], 4.0))


class _Spec:
    def __init__(self, flags, declined=False):
        self.flags, self.declined = tuple(flags), declined


@pytest.mark.parametrize("flags, declined, want", [
    ((), False, None),
    (("count audit: unaudited (not a prefix window)",), False, None),
    (("unmet: 2 of 4 levels miss prec=1e-10, the first at lam = 31.1 (margin 0.4; 2 is needed)",),
     False, UnresolvedSpectrum),
    (("count audit: short (mean Weyl deficit 0.9)",), False, UnresolvedSpectrum),
    (("short: found 3 of the 4 eigenvalues asked for",), False, UnresolvedSpectrum),
    (("planner: no rung met prec=1e-10; returning rung 3 of 3",), False, UnresolvedSpectrum),
    (("declined: the slope never settled",), True, SolverRefusal),
])
def test_chladni_flags_map_onto_douses_refusals(flags, declined, want):
    exc = refusal_for(_Spec(flags, declined))
    assert (exc is None) if want is None else isinstance(exc, want), exc


@pytest.mark.slow
def test_cluster_matrices_near_a_double_carry_the_members_own_gradients():
    """A nearly square rectangle: lam_2, lam_3 split slightly but resolvably, so each is a simple
    level. Built from the members' OWN eigenfunctions (`lams=`), the cluster block's diagonal is
    each member's simple gradient, which the same backend computes by its simple path -- the
    same Cauchy data, so the two paths agree to the orthonormalization's roundoff.

    Not checked against `lappy`'s `cluster_matrices`, which evaluates both members' Cauchy data at
    the cluster's MEAN eigenvalue: its own diagonal is off its own simple gradients by ~1e-3 here
    (measured 2026-10-12), first order in the split. `chladni` evaluates each member at its own.
    (Until 2026-10-14 this compared against the lappy backend's simple gradients; the two
    backends' simple gradients agree to ~2e-12.)"""
    from douse.derivative import cluster_matrices, dlams_dp
    L, H = 1.0, 1.002
    fam = VertexPolygon(4, perim=2 * (L + H))
    p = _p([0, L, L + 1j * H, 1j * H], fam.perim)
    lams_c, solver_c = ChladniEvaluator(fam, n_eigs=3, prec=PREC,
                                        degenerate_tol=0.0).eigenvalues(p)
    Mc = cluster_matrices(fam, p, solver_c, float(np.mean(lams_c[1:3])), 2, lams=lams_c[1:3])
    Js = dlams_dp(fam, p, solver_c, lams_c[1:3])
    np.testing.assert_allclose(np.stack([Mc[:, 0, 0], Mc[:, 1, 1]]), Js, rtol=1e-8,
                               atol=1e-9 * np.abs(Js).max())


@pytest.mark.slow
def test_the_acceptance_test_passes_on_chladni():
    """`tests/test_cluster_opt.py`'s acceptance test, on this backend: maximize lam_2/lam_1 over
    quadrilaterals from a perturbed square, through the lam_2 = lam_3 degeneracy, to the square's
    2.5. Measured 2026-10-12: 2.4999999991, gap 7.2e-10, 9 cluster steps, 30 s, no refusals."""
    from douse.cluster_opt import maximize_ratio
    perim, N = 4.0, 4
    rng = np.random.default_rng(0)
    v = perim / N / (2 * np.sin(np.pi / N)) * np.exp(2j * np.pi * np.arange(N) / N)
    v = v + 0.06 * (rng.normal(size=N) + 1j * rng.normal(size=N)) * np.abs(v).mean()
    fam = VertexPolygon(N, perim=perim)
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC, degenerate_tol=0.0)
    res = maximize_ratio(ev, param_vector(v, perim), k=1, max_iter=120)
    assert abs(res.fun - 2.5) < 1e-6, (res.fun, res.message)
    assert res.gap < 1e-5 and res.n_cluster_steps > 0


def _perturbed(N, jitter, perim=4.0, seed=0):
    rng = np.random.default_rng(seed)
    v = perim / N / (2 * np.sin(np.pi / N)) * np.exp(2j * np.pi * np.arange(N) / N)
    return v + jitter * (rng.normal(size=N) + 1j * rng.normal(size=N)) * np.abs(v).mean()


@pytest.mark.slow
def test_a_pentagon_run_never_reports_above_the_ppw_bound():
    """`test_cluster_opt.py`'s PPW check on this backend: every reported lam_2/lam_1 on a
    5-gon run stays at or below the Payne-Polya-Weinberger bound (j_11/j_01)^2."""
    from douse.cluster_opt import maximize_ratio
    PPW = (3.8317059702 / 2.4048255577) ** 2
    fam = VertexPolygon(5, perim=4.0)
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC, degenerate_tol=0.0)
    seen = []
    res = maximize_ratio(ev, param_vector(_perturbed(5, 0.06), 4.0), k=1, max_iter=40,
                         callback=lambda p, F, model, delta, rho: seen.append(F))
    assert seen and max(seen + [res.fun]) <= PPW + 1e-9


@pytest.mark.slow
def test_the_model_switches_to_the_cluster_as_the_gap_closes():
    from douse.cluster_opt import maximize_ratio
    fam = VertexPolygon(4, perim=4.0)
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC, degenerate_tol=0.0)
    modes = []
    maximize_ratio(ev, param_vector(_perturbed(4, 0.10), 4.0), k=1, max_iter=60,
                   callback=lambda p, F, model, delta, rho: modes.append(model["mode"]))
    assert "smooth" in modes and "cluster" in modes, set(modes)
    assert modes.index("cluster") > modes.index("smooth")


@pytest.mark.slow
@pytest.mark.parametrize("L", [1.4, 1.6])
def test_a_frozen_plan_that_drifts_out_of_reach_is_replanned_not_refused(L):
    """A plan made on the square, realized on a 1.4 x 0.6 or 1.6 x 0.4 rectangle, no longer
    meets 1e-10 there (the audit reads excess; at 1.6 the search declines). That is plan DRIFT,
    routine over an optimization run -- `lappy`'s backend re-plans on it (`replan_tol`) -- not a
    shape the solver cannot handle. Re-plan at the shape and solve again; refuse only if that
    fails too. (Without this, 4 of the first 5 comparison runs stopped short, 2026-10-13.)"""
    from benchmarks.rectangle_exact import spectrum
    fam = VertexPolygon(4, perim=4.0)
    square = _p([0, 1, 1 + 1j, 1j], 4.0)
    H = 2.0 - L
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC, degenerate_tol=0.0, plan_at=square)
    lams, _ = ev.eigenvalues(_p([0, L, L + 1j * H, 1j * H], 4.0))
    np.testing.assert_allclose(lams, spectrum(L, H, 8)[:3], rtol=1e-9)
    assert ev.n_replans >= 1


# --- warm start (chladni `solve(..., near=prev)`), 2026-10-13 ---------------------------------
# The contract of `SpectrumEvaluator`'s tracking (`douse/evaluate.py`): seed from `_last`, a
# `(p, lams)` pair that `cluster_opt` also writes to reseed from the accepted point; count
# `n_solves`, `n_tracks`, `n_track_refusals`; `reset_continuation` forces a cold solve.

def _rect_family():
    L, H = 1.2, 0.8
    fam = VertexPolygon(4, perim=2 * (L + H))
    return fam, _p([0, L, L + 1j * H, 1j * H], fam.perim)


@pytest.mark.slow
def test_nearby_shapes_are_tracked_after_one_cold_solve():
    fam, p = _rect_family()
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    rng = np.random.default_rng(0)
    for _ in range(4):
        lams, _ = ev.eigenvalues(p)
        assert np.all(np.diff(lams) >= 0)
        p = p + 1e-4 * rng.normal(size=p.size)
    assert (ev.n_solves, ev.n_tracks, ev.n_track_refusals) == (1, 3, 0)
    assert isinstance(ev._last, tuple) and len(ev._last) == 2


@pytest.mark.slow
def test_reset_continuation_forces_a_cold_solve():
    fam, p = _rect_family()
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    ev.eigenvalues(p)
    ev.reset_continuation()
    assert ev._last is None
    ev.eigenvalues(p + 1e-4)
    assert (ev.n_solves, ev.n_tracks) == (2, 0)


@pytest.mark.slow
def test_a_cluster_opt_reseed_tracks_from_the_accepted_point():
    """`cluster_opt` rejects a trial and writes `_last = (p, lams)` for the accepted p; the next
    trial must track from p's spectrum, not from the rejected trial's."""
    fam, p = _rect_family()
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    lams_p, _ = ev.eigenvalues(p)
    ev.eigenvalues(p + 3e-4)                         # a trial, then rejected
    ev._last = (p.copy(), lams_p.copy())
    lams, _ = ev.eigenvalues(p - 1e-4)
    assert (ev.n_solves, ev.n_tracks) == (1, 2)
    assert ev._last_spec.warm


@pytest.mark.slow
def test_a_seed_with_no_spectrum_behind_it_solves_cold():
    """A `_last` written by hand for a p never solved here has no spectrum to warm from."""
    fam, p = _rect_family()
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    lams, _ = ev.eigenvalues(p)
    ev._last = (p + 1e-3, lams)
    ev.eigenvalues(p + 1e-3)
    assert (ev.n_solves, ev.n_tracks) == (2, 0)


# ── a multiple level is never cut by the request ──────────────────────────────────────────────

class _Levels:
    """A `Spectrum` stand-in: levels with multiplicities, as `ChladniSolver` reads them."""
    def __init__(self, lam, mult):
        self.lam, self.mult = np.asarray(lam, float), np.asarray(mult, int)


def test_a_cluster_that_straddles_the_request_resolves_to_the_solvers_levels():
    """Levels `[l1 x1, a x1, b x2]` with `n_eigs=3`: the unfolded array is `[l1, a, b, b]`, four
    values, and an eps-active cluster over `[a, b]` holds THREE eigenvalues by the solver's
    count. Asked for exactly those levels, the solver must say so; asked for two where it holds
    three, it refuses rather than building a block that is not the cluster's."""
    from douse.chladni_backend import ChladniSolver
    sol = ChladniSolver(_Levels([10.0, 20.0, 30.0], [1, 1, 2]), [0, 1, 1 + 1j, 1j])
    assert sol._cluster_levels(25.0, 3, lams=[20.0, 30.0]) == [1, 2]
    assert sol._cluster_levels(30.0, 2, lams=None) == [2]
    with pytest.raises(UnresolvedSpectrum, match="does not match the solver's levels"):
        sol._cluster_levels(25.0, 2, lams=[20.0, 30.0])


def test_the_evaluator_returns_a_top_level_whole():
    """The unit square's lam_2 = lam_3: asked for 2 eigenvalues, the evaluator returns 3, the
    double whole. (Its simple gradient is still refused, by policy: that is what
    `cluster_matrices` is for.) `ObjectiveFunction` hands its objective exactly the count the
    objective declared, however many the evaluator returned."""
    from douse.evaluate import ObjectiveFunction
    from douse.objective import EigenvalueObjective
    fam = VertexPolygon(4, perim=4.0)
    ev = ChladniEvaluator(fam, n_eigs=2, prec=PREC, degenerate_tol=0.0, warn_gap=0.0)
    lams, _ = ev.eigenvalues(_p([0, 1, 1 + 1j, 1j], 4.0))
    assert len(lams) == 3 and abs(lams[1] - lams[2]) < 1e-8 * lams[1]
    np.testing.assert_allclose(lams, np.pi**2 * np.array([2, 5, 5]), rtol=1e-8)

    L, H = 1.2, 0.8
    fam = VertexPolygon(4, perim=2 * (L + H))
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    obj = ObjectiveFunction(ev, EigenvalueObjective(0))
    val, grad = obj(_p([0, L, L + 1j * H, 1j * H], fam.perim))
    lam_1 = np.pi**2 * (1 / L**2 + 1 / H**2)
    assert abs(val - lam_1) < 1e-8 * lam_1 and obj.history[-1]['lams'].shape == (1,)
    assert grad.shape == (4,)


# ── warnings reach the caller ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.filterwarnings("default")
def test_the_gap_warning_reaches_a_maximize_ratio_caller():
    """`maximize_ratio` used to silence every warning inside its loop. On a nearly square
    rectangle lam_2 and lam_3 sit 4e-3 apart, so `warn_gap=0.1` fires on the first solve; the
    caller sees it once, and the result counts it."""
    from douse.cluster_opt import maximize_ratio
    L, H = 1.0, 1.002
    fam = VertexPolygon(4, perim=2 * (L + H))
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC, degenerate_tol=0.0, warn_gap=0.1)
    with pytest.warns(UserWarning, match="within .* of each other"):
        res = maximize_ratio(ev, _p([0, L, L + 1j * H, 1j * H], fam.perim), k=1, max_iter=1)
    assert res.n_warnings >= 1


@pytest.mark.slow
def test_lam3_over_lam1_on_a_hexagon_survives_the_lam4_lam5_pair():
    """The standard problem's own geometry: maximizing lam_3/lam_1 drives lam_3 into the
    regular hexagon's lam_4 = lam_5 pair. With `n_eigs = k + 2 = 4` the request ends INSIDE that
    double, and a truncated copy made douse's eps-active cluster and the solver's level count
    disagree -- a `ValueError` out of `cluster_matrices` that killed the run (2026-10-14). The
    level is now returned whole; a short run must finish, by any honest ending."""
    from douse.cluster_opt import maximize_ratio
    from douse import ClosingBudget, ConstraintSet, MinEdgeLength
    fam = VertexPolygon(6, perim=4.0)
    ev = ChladniEvaluator(fam, n_eigs=4, prec=PREC, degenerate_tol=0.0)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    res = maximize_ratio(ev, param_vector(_perturbed(6, 0.03), 4.0), k=2, max_iter=12,
                         constraints=cons)
    assert np.isfinite(res.fun) and res.fun > 1.0, res.message
    assert all(len(l) >= 4 for l in [res.lams])


def test_the_evaluator_keeps_the_runs_health_and_cost():
    """The ledger reads `worst_sigma`, `worst_weyl_deficit`, `n_weyl_flags` and
    `n_tension_evals` off the evaluator; each must exist and move with the solves."""
    L, H = 1.2, 0.8
    fam = VertexPolygon(4, perim=2 * (L + H))
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    assert ev.worst_sigma == -np.inf and ev.n_tension_evals == 0
    ev.eigenvalues(_p([0, L, L + 1j * H, 1j * H], fam.perim))
    n1 = ev.n_tension_evals
    assert 0 < ev.worst_sigma <= 1e-9 and ev.worst_sigma == ev.worst_sigma_any == ev.last_sigma
    assert n1 > 0 and np.isfinite(ev.worst_weyl_deficit)
    ev.eigenvalues(_p([0, L, L + 1j * H * 1.001, 1j * H], fam.perim))
    assert ev.n_tension_evals > n1
