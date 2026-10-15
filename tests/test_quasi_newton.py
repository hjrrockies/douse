"""The second-order machinery: the exact trust-region solve, SR1, and `model_gain2`.

Every test here is against something independent -- brute force, a quadratic whose Hessian is
known in closed form, or central differences -- rather than against a previously recorded number
from this same code.
"""
import zlib

import numpy as np
import pytest
from scipy.optimize import minimize

from douse.cluster_opt import model_gain, model_gain2, model_gain2_grad
from douse.quasi_newton import SR1, BFGS, ScalarSecant, tr_subproblem


def _seed(*parts):
    """A stable seed from the parametrize arguments.

    NOT `hash(...)`. Python randomizes `hash` of strings per process unless `PYTHONHASHSEED` is
    set, so seeding from it draws a different matrix on every run -- which turned these into
    tests that pass or fail depending on the day. One of them did exactly that: it passed alone
    and failed in the full suite, on a draw that was never the same twice.
    """
    return zlib.crc32(repr(parts).encode()) & 0xffffffff


def _brute_tr(g, B, delta, n_start=40, seed=0):
    """`min_{||d||<=delta} g.d + 0.5 d^T B d` by multi-start SLSQP -- the independent reference."""
    rng = np.random.default_rng(seed)
    n = len(g)
    f = lambda d: g @ d + 0.5*d @ (B @ d)
    cons = [dict(type='ineq', fun=lambda d: delta**2 - d @ d, jac=lambda d: -2*d)]
    r = rng.normal(size=(n_start, n))
    starts = [np.zeros(n)] + list(delta*r/np.linalg.norm(r, axis=1, keepdims=True))
    best = np.inf
    for d0 in starts:
        res = minimize(f, d0, jac=lambda d: g + B @ d, method='SLSQP', constraints=cons,
                       options=dict(maxiter=500, ftol=1e-16))
        d = np.asarray(res.x)
        nd = np.linalg.norm(d)
        if nd > delta:
            d = d*delta/nd
        best = min(best, f(d))
    return best


def _matrix(kind, n, rng):
    A = rng.normal(size=(n, n))
    B = 0.5*(A + A.T)
    if kind == 'pd':
        return B @ B.T + 0.5*np.eye(n)
    if kind == 'nd':
        return -(B @ B.T) - 0.5*np.eye(n)
    if kind == 'singular':
        w, V = np.linalg.eigh(B)
        w[0] = 0.0
        return V @ np.diag(w) @ V.T
    return B                                                          # indefinite


@pytest.mark.parametrize('kind', ['pd', 'indefinite', 'nd', 'singular'])
@pytest.mark.parametrize('n', [2, 6])
@pytest.mark.parametrize('delta', [1e-3, 0.5, 10.0])
def test_tr_subproblem_matches_brute_force(kind, n, delta):
    """The More--Sorensen solve is exact, including for an INDEFINITE model Hessian.

    That case is the reason for solving the subproblem exactly rather than by a dogleg: `lam_1`
    and the cluster mean are not convex along the path a run takes, so `SR1` will hand this an
    indefinite matrix routinely, and a dogleg is not defined there.
    """
    rng = np.random.default_rng(_seed(kind, n, delta))
    B, g = _matrix(kind, n, rng), rng.normal(size=n)
    d, dec = tr_subproblem(g, B, delta)
    assert np.linalg.norm(d) <= delta*(1 + 1e-9)
    got = g @ d + 0.5*d @ (B @ d)
    assert got <= _brute_tr(g, B, delta) + 1e-9*max(1.0, abs(got))
    assert dec == pytest.approx(max(0.0, -got), abs=1e-12)


def test_tr_subproblem_hard_case():
    """`g` orthogonal to the lowest eigenspace: the solution needs a component the formula misses.

    The textbook failure of a naive secular-equation solve. Here the boundary is not reached as
    `sigma -> -lam_min`, so the step has to pick up `tau*v_0` to fill out the radius.
    """
    rng = np.random.default_rng(7)
    n = 5
    A = rng.normal(size=(n, n))
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    g = V[:, 1:] @ rng.normal(size=n - 1)              # exactly no component along V[:, 0]
    for delta in (0.01, 0.5, 5.0):
        d, _ = tr_subproblem(g, B, delta)
        assert np.linalg.norm(d) == pytest.approx(delta, rel=1e-9)
        got = g @ d + 0.5*d @ (B @ d)
        assert got <= _brute_tr(g, B, delta) + 1e-9


def test_tr_subproblem_takes_the_interior_newton_step():
    """A positive definite `B` with a short Newton step must NOT be dragged to the boundary."""
    B = np.diag([1.0, 4.0, 9.0])
    g = np.array([0.1, 0.2, 0.3])
    d, _ = tr_subproblem(g, B, 10.0)
    assert d == pytest.approx(-np.linalg.solve(B, g), rel=1e-10)


def test_sr1_recovers_an_indefinite_hessian_exactly():
    """SR1's defining property, and the reason it is used instead of BFGS.

    On a quadratic, `n` linearly independent pairs reproduce the Hessian EXACTLY -- sign and all.
    BFGS cannot: it is constrained to stay positive definite, so it cannot represent this `H`.
    """
    rng = np.random.default_rng(1)
    n = 6
    A = rng.normal(size=(n, n))
    H = 0.5*(A + A.T)                                  # indefinite by construction
    assert np.linalg.eigvalsh(H)[0] < 0
    b = rng.normal(size=n)
    grad = lambda p: H @ p + b

    sr1 = SR1(n)
    p = np.zeros(n)
    for i in range(3*n):
        q = p + rng.normal(size=n)
        sr1.update(q - p, grad(q) - grad(p))
        p = q
    assert sr1.B == pytest.approx(H, abs=1e-8)

    bfgs = BFGS(n)
    p = np.zeros(n)
    for i in range(3*n):
        q = p + rng.normal(size=n)
        bfgs.update(q - p, grad(q) - grad(p))
        p = q
    assert np.linalg.norm(bfgs.B - H) > 1e-3           # it structurally cannot get there


def test_sr1_observe_breaks_and_keeps_the_chain():
    """`chain_broken` must forget the ITERATE without forgetting the curvature."""
    rng = np.random.default_rng(2)
    n = 4
    A = rng.normal(size=(n, n))
    H = 0.5*(A + A.T)
    grad = lambda p: H @ p
    sr1 = SR1(n)
    p = np.zeros(n)
    for _ in range(2*n):
        q = p + rng.normal(size=n)
        sr1.observe(q, grad(q))
        p = q
    B_before, updates = sr1.B.copy(), sr1.n_updates
    sr1.chain_broken()
    assert sr1.B == pytest.approx(B_before)            # curvature survives
    assert sr1.observe(p + np.ones(n), grad(p + np.ones(n))) is False   # no pair to form
    assert sr1.n_updates == updates
    sr1.reset()
    assert sr1.B == pytest.approx(np.zeros((n, n)))
    assert not sr1.ready


def _fake_model(mode, m, j, n_p, rng):
    lam_1, lam_k = 20.3, 64.1
    g1 = rng.normal(size=n_p)
    if mode == 'smooth':
        gk = rng.normal(size=n_p)
        return dict(mode='smooth', lam_1=lam_1, lam_k=lam_k, g1=g1, gk=gk, m=1, j=0,
                    cluster=(1, 2), grad=(gk*lam_1 - lam_k*g1)/lam_1**2)
    Ms = rng.normal(size=(n_p, m, m))
    Ms = 0.5*(Ms + np.swapaxes(Ms, 1, 2))
    return dict(mode='cluster', lam_1=lam_1, lam_k=lam_k, g1=g1, Ms=Ms,
                Lam=np.diag(np.sort(rng.normal(size=m)*1e-3)), m=m, j=j, cluster=(1, 1 + m))


@pytest.mark.parametrize('mode,m,j', [('smooth', 1, 0), ('cluster', 2, 0),
                                      ('cluster', 2, 1), ('cluster', 3, 1)])
def test_model_gain2_agrees_with_model_gain_to_first_order(mode, m, j):
    """The second-order model must not move the first-order one, or the stopping certificate
    -- which `maximize_ratio` still measures against `model_gain` -- would stop meaning anything."""
    rng = np.random.default_rng(_seed(mode, m, j))
    model = _fake_model(mode, m, j, 6, rng)
    u = rng.normal(size=6)
    u /= np.linalg.norm(u)

    # THE RADII STOP AT 1e-5, and going smaller does NOT make this a sharper test -- it makes it a
    # measurement of roundoff. The two models differ by `O(d^2)` while each is `O(d)`, so the
    # difference is computed by cancelling two nearly equal numbers: at `||d|| = 1e-7` the true
    # difference is ~1e-17 relative to quantities of size ~1e-7, which is below the ~1e-16
    # absolute floor of the subtraction. Measured, on one draw: `rel` fell 3.2e-09 at 1e-6 and
    # then ROSE to 3.8e-08 at 1e-7, which is the floor and not the model.
    prev = None
    for eps in (1e-3, 1e-4, 1e-5):
        d = eps*u
        base = abs(model_gain(model, d))
        rel = abs(model_gain2(model, d) - model_gain(model, d))/max(base, 1e-30)
        assert rel < 1e-2
        if prev is not None:
            assert rel <= prev*0.5                     # shrinking with the radius, not constant
        prev = rel


def test_model_gain2_quotient_is_exact_for_a_singleton():
    """With `m = 1` the difference from `model_gain` is EXACTLY the quotient's own curvature,
    so `|gain2 - gain|/||d||^2` is a constant rather than merely bounded."""
    rng = np.random.default_rng(3)
    model = _fake_model('smooth', 1, 0, 6, rng)
    u = rng.normal(size=6)
    u /= np.linalg.norm(u)
    vals = [abs(model_gain2(model, eps*u) - model_gain(model, eps*u))/eps**2
            for eps in (1e-2, 1e-3, 1e-4)]
    assert vals[0] == pytest.approx(vals[1], rel=1e-3)
    assert vals[1] == pytest.approx(vals[2], rel=1e-3)


@pytest.mark.parametrize('mode,m,j', [('smooth', 1, 0), ('cluster', 2, 0),
                                      ('cluster', 2, 1), ('cluster', 4, 2)])
def test_model_gain2_gradient_matches_central_differences(mode, m, j):
    """The analytic gradient is what makes the SLSQP polish affordable; verify it independently."""
    rng = np.random.default_rng(_seed(mode, m, j, 'g'))
    n_p = 6
    model = _fake_model(mode, m, j, n_p, rng)
    H1 = rng.normal(size=(n_p, n_p))
    H1 = 0.5*(H1 + H1.T)
    Hmu = rng.normal(size=(n_p, n_p))
    Hmu = 0.5*(Hmu + Hmu.T)
    d = 0.05*rng.normal(size=n_p)
    ga = model_gain2_grad(model, d, H1, Hmu)
    h = 1e-7
    gn = np.array([(model_gain2(model, d + h*e, H1, Hmu) - model_gain2(model, d - h*e, H1, Hmu))
                   /(2*h) for e in np.eye(n_p)])
    assert ga == pytest.approx(gn, rel=1e-5, abs=1e-8)


def test_model_gain2_refuses_a_step_that_drives_lam_1_negative():
    """`den <= 0` is not a small-`d` regime, it is a step out of the model's validity. `-inf`
    keeps the maximizer away from it rather than letting the quotient flip sign and look good."""
    rng = np.random.default_rng(4)
    model = _fake_model('smooth', 1, 0, 6, rng)
    H1 = -1e6*np.eye(6)                                # enough curvature to sink the denominator
    assert model_gain2(model, 0.5*np.ones(6), H1, None) == -np.inf


def _zigzag_pairs(H, cubic, n_pairs, rng):
    """Gradient pairs of `0.5 p.Hp + cubic*sum(p^3)/3` along small steps that zigzag inside a
    two-dimensional subspace, the pattern a trust-region crawl along a ridge produces."""
    n = H.shape[0]
    grad = lambda p: H @ p + cubic*p**2
    hess = lambda p: H + 2*cubic*np.diag(p)
    u, v = np.zeros(n), np.zeros(n)
    u[0], v[1] = 1.0, 1.0
    p = np.zeros(n)
    for i in range(n_pairs):
        s = 6e-3*(u*(1 + 0.05*rng.normal()) + ((-1)**i)*0.3*v*(1 + 0.05*rng.normal()))
        yield s, grad(p + s) - grad(p), hess(p + 0.5*s)
        p = p + s


def test_scalar_secant_stays_inside_the_hessian_range():
    """THE PROPERTY SR1 LACKED on 2026-10-15, when its matrix for `lam_1` grew to 1e7 against a
    true Hessian of 2e2 and collapsed the trust radius (module docstring; the instrumented run
    is `benchmarks/run/diagnostics/sr1_blowup_seed0_prefix_2026_10_15.jsonl`, `H1.normB` per
    iterate). That blow-up needs the real run's pairs -- on this synthetic zigzag with a cubic
    term SR1 learns the Hessian correctly -- so what is pinned here is the guarantee: the
    scalar secant is a Rayleigh quotient of the average Hessian over the step, and cannot leave
    the range of that Hessian's eigenvalues."""
    rng = np.random.default_rng(3)
    n = 8
    A = rng.normal(size=(n, n))
    H = 100*0.5*(A + A.T)
    cubic = 2e3
    sr1, sc = SR1(n), ScalarSecant(n)
    for s, y, H_mid in _zigzag_pairs(H, cubic, 80, rng):
        sr1.update(s, y)
        sc.update(s, y)
        lo, hi = np.linalg.eigvalsh(H_mid)[[0, -1]]      # the Hessian's range along THIS step
        assert lo - 1e-6*abs(lo) <= sc.gamma <= hi + 1e-6*abs(hi), (sc.gamma, lo, hi)
    assert sr1.n_updates == 80                          # SR1 is fine HERE; the run is the evidence


def test_scalar_secant_is_exact_along_the_step_on_a_quadratic():
    rng = np.random.default_rng(4)
    n = 5
    A = rng.normal(size=(n, n))
    H = 0.5*(A + A.T)
    grad = lambda p: H @ p
    sc = ScalarSecant(n)
    p = np.zeros(n)
    for _ in range(6):
        q = p + rng.normal(size=n)
        sc.observe(q, grad(q))
        s = q - p
        if sc.ready:
            assert sc.gamma == pytest.approx(float(s @ H @ s)/float(s @ s), rel=1e-12)
            assert sc.B == pytest.approx(sc.gamma*np.eye(n))
        p = q
    sc.chain_broken()
    assert sc.observe(p + 1, grad(p + 1)) is False and sc.ready
    sc.reset()
    assert not sc.ready and sc.gamma == 0.0
