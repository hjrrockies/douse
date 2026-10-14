"""A golden trace of `maximize_ratio`, on a deterministic fake backend.  (2026-10-14)

`maximize_ratio` is being refactored into stages. Equivalence is proved here rather than
argued: every callback tuple and the result dict of a 30-iteration run are recorded once (the
JSON beside this file, written by `python -m tests.test_cluster_opt_trace --record`) and every
later run must reproduce them. The backend is a FAKE, so the trace carries no solver noise:
the spectrum of the "rectangle-ized" quadrilateral -- L the mean of edges 0 and 2, H of 1 and 3
-- by the closed form `pi^2 (m^2/L^2 + n^2/H^2)`, differentiated by central differences in the
vertex coordinates. `lam_2/lam_1` is maximized at L = H, where `lam_2 = lam_3` is a double, so
the run crosses into cluster mode and exercises the cluster block (built from the BRANCH
gradients, `lam_{2,1}` and `lam_{1,2}`, which are the exact directional derivatives there).

Floats are compared at 1e-10 relative (BLAS may differ in the last ulps between machines);
counters, modes and messages exactly.
"""
import json
import pathlib
import sys

import numpy as np

from douse import ClosingBudget, ConstraintSet, MinEdgeLength, VertexPolygon, param_vector
from douse.cluster_opt import maximize_ratio
from douse.polygon import edge_lengths

GOLDEN = pathlib.Path(__file__).parent / "data" / "cluster_opt_trace.json"
MODES = [(m, n) for m in range(1, 5) for n in range(1, 5)]
H_FD = 1e-6


class FakeRectSolver:
    """What `douse.derivative` needs: the chladni-backend dispatch names, on the fake model."""

    def __init__(self, ev, verts, labels):
        self.ev, self.verts, self.labels = ev, verts, labels

    def _branch_grad(self, label):
        """d lam_{m,n} / d [x_0..x_{N-1}, y_0..y_{N-1}], central differences."""
        n = len(self.verts)
        g = np.empty(2 * n)
        for i in range(2 * n):
            e = np.zeros(2 * n)
            e[i] = H_FD
            g[i] = (self.ev.branch(self.verts + _as_complex(e, n), label)
                    - self.ev.branch(self.verts - _as_complex(e, n), label)) / (2 * H_FD)
        return g

    def _label_of(self, lam):
        i = int(np.argmin(np.abs(self.ev.values(self.verts) - lam)))
        return MODES[i]

    def chladni_dlams_dvertices(self, lams, n_vert):
        return np.array([self._branch_grad(self._label_of(l)) for l in lams])

    def chladni_cluster_matrices(self, family, p, lam, mult, lams=None):
        vals = self.ev.values(self.verts)
        if lams is None:
            idx = np.argsort(np.abs(vals - lam))[:mult]
        else:
            idx = [int(np.argmin(np.abs(vals - l))) for l in lams]
        G = np.array([self._branch_grad(MODES[i]) for i in idx])            # (m, 2N)
        Jv = family.d_vertices_dp(p)                                        # (2N, n_p)
        Gp = G @ Jv                                                         # (m, n_p)
        return np.array([np.diag(Gp[:, i]) for i in range(Gp.shape[1])])   # (n_p, m, m)


def _as_complex(e, n):
    return e[:n] + 1j * e[n:]


class FakeRectEvaluator:
    """The evaluator surface `maximize_ratio` uses, on the rectangle-ized spectrum."""

    def __init__(self, family, n_eigs=3):
        self.family, self.n_eigs = family, int(n_eigs)
        self.degenerate_tol = 0.0
        self.n_replans, self.last_sigma = 0, np.nan
        self._last, self._plan_at, self._plan_built_at = None, None, None
        self.refusals, self.worst_sigma, self.worst_weyl_deficit, self.n_weyl_flags = [], 1e-12, 0.0, 0
        self.n_calls = 0

    @staticmethod
    def _LH(verts):
        e = edge_lengths(verts)
        return 0.5 * (e[0] + e[2]), 0.5 * (e[1] + e[3])

    def branch(self, verts, label):
        L, H = self._LH(verts)
        m, n = label
        return np.pi**2 * (m**2 / L**2 + n**2 / H**2)

    def values(self, verts):
        return np.array([self.branch(verts, lab) for lab in MODES])

    def eigenvalues(self, p):
        self.n_calls += 1
        verts = np.asarray(self.family.vertices(p), dtype=complex)
        vals = self.values(verts)
        order = np.argsort(vals)[:self.n_eigs]
        lams = vals[order]
        self._last = (np.array(p, float), lams.copy())
        return lams, FakeRectSolver(self, verts, [MODES[i] for i in order])

    def reset_continuation(self):
        self._last = None

    def replan(self, p=None):
        self.n_replans += 1


def run(max_iter=30):
    fam = VertexPolygon(4, perim=4.0)
    ev = FakeRectEvaluator(fam)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    v0 = np.array([0, 1.25, 1.25 + 0.75j, 0.75j]) + np.array([0.03 + 0.02j, -0.02 + 0.01j, 0.01 - 0.03j, 0.02j])
    p0 = param_vector(v0, 4.0)
    trace = []

    def cb(p, F, model, delta, rho):
        trace.append(dict(p=[float(t) for t in p], F=float(F), mode=model["mode"],
                          cluster=[int(t) for t in model["cluster"]], delta=float(delta),
                          rho=(None if not np.isfinite(rho) else float(rho))))
    res = maximize_ratio(ev, p0, k=1, max_iter=max_iter, constraints=cons,
                         rng=np.random.default_rng(0), callback=cb)
    result = {k: _jsonable(v) for k, v in res.items()}
    result["n_calls"] = ev.n_calls
    return trace, result


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return [float(t) for t in v.ravel()]
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer, int, bool, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(t) for t in v]
    return str(v)


def _compare(got, want, path=""):
    if isinstance(want, float) and isinstance(got, (float, int)):
        assert abs(got - want) <= 1e-10 * max(1.0, abs(want)), (path, got, want)
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (path, got, want)
        for i, (g, w) in enumerate(zip(got, want)):
            _compare(g, w, f"{path}[{i}]")
    elif isinstance(want, dict):
        assert set(got) == set(want), (path, set(got) ^ set(want))
        for key in want:
            _compare(got[key], want[key], f"{path}.{key}")
    else:
        assert got == want, (path, got, want)


def test_maximize_ratio_reproduces_its_golden_trace():
    trace, result = run()
    golden = json.loads(GOLDEN.read_text())
    assert len(trace) == len(golden["trace"]), (len(trace), len(golden["trace"]))
    _compare(trace, golden["trace"], "trace")
    _compare(result, golden["result"], "result")


def test_the_fake_reaches_the_square_through_the_cluster():
    """The trace is only worth keeping if it covers cluster mode and converges."""
    trace, result = run()
    assert any(t["mode"] == "cluster" for t in trace)
    assert abs(result["fun"] - 2.5) < 1e-6 and result["n_cluster_steps"] > 0


if __name__ == "__main__" and "--record" in sys.argv:
    trace, result = run()
    GOLDEN.write_text(json.dumps(dict(trace=trace, result=result), indent=1))
    print(f"recorded {len(trace)} callbacks; fun={result['fun']:.12f}, "
          f"message={result['message']!r}, cluster steps={result['n_cluster_steps']}")
