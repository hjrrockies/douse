"""The backend seam (`douse/backend.py`): both evaluators satisfy the protocols, and the health
record is a value the ledger can write.  (2026-10-14)"""
import numpy as np
import pytest

from douse.backend import RunHealth, SolveTrace, SolvedShape, SpectrumBackend


class _Fam:
    def is_valid(self, p):
        return True


class FakeBackend:
    family, n_eigs, degenerate_tol = _Fam(), 3, 0.0

    def eigenvalues(self, p):
        return np.array([1.0, 2.0, 3.0]), FakeSolved()

    def reseed(self, p, lams, solved=None):
        pass

    def reset_continuation(self):
        pass

    def replan(self, p=None):
        pass

    def health(self):
        return RunHealth()


class FakeSolved:
    def dlams_dvertices(self, lams, n_vert):
        return np.zeros((len(lams), 2 * n_vert))

    def cluster_matrices_dvertices(self, family, p, lam, mult, lams=None):
        return np.zeros((4, mult, mult))


def test_a_fake_satisfies_the_protocols():
    assert isinstance(FakeBackend(), SpectrumBackend)
    assert isinstance(FakeSolved(), SolvedShape)
    assert not isinstance(object(), SpectrumBackend)


def test_the_chladni_evaluator_and_its_solved_shape_satisfy_the_protocols():
    from douse import VertexPolygon
    from douse.chladni_backend import ChladniSolver, SpectrumEvaluator
    ev = SpectrumEvaluator(VertexPolygon(4, perim=4.0), n_eigs=2, degenerate_tol=0.0)
    assert isinstance(ev, SpectrumBackend)
    assert all(hasattr(ChladniSolver, m) for m in ("dlams_dvertices", "cluster_matrices_dvertices",
                                                   "spectrum"))
    h = ev.health()
    assert isinstance(h, RunHealth) and h.n_solves == 0 and h.trace == ()
    row = h.as_row()
    assert row["worst_bound"] is None and row["n_replans"] == 0 and row["trace"] == []


def test_the_lappy_evaluator_class_satisfies_the_protocol_by_shape():
    """Instantiating it needs lappy; the class shape is what the protocol checks."""
    from douse.evaluate import LappySpectrumEvaluator
    for name in ("eigenvalues", "reseed", "reset_continuation", "replan", "health"):
        assert callable(getattr(LappySpectrumEvaluator, name))


def test_health_serialises_its_trace():
    h = RunHealth(worst_bound=1e-10, n_solves=2, trace=(SolveTrace("cold", None, 1, 190, 2.5),
                                                        SolveTrace("warm", None, 1, 80, 1.0)))
    row = h.as_row()
    assert row["trace"][1] == dict(kind="warm", reason=None, rung=1, n_evals=80, seconds=1.0)
    assert row["worst_bound"] == 1e-10 and row["plan_at"] is None
