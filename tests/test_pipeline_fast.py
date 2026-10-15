"""The whole pipeline, end to end, in the DEFAULT tier.  (2026-10-14)

Until this file existed the only test that ran `maximize_ratio` on a real solve was the slow
tier, fifteen minutes of it, so a change to the optimizer or the backend could pass the fast
tier and break the pipeline. This is the cheapest real run that exercises every seam: a chladni
plan and solve, the Hadamard Jacobian, the trust-region step under constraints, a trial solve,
the callback, and the result's health fields. Two iterations on a 1.2 x 0.8 rectangle, no final
verification; about 8 s.
"""
import numpy as np

from douse import ClosingBudget, ConstraintSet, MinEdgeLength, SpectrumEvaluator, VertexPolygon
from douse import param_vector
from douse.cluster_opt import maximize_ratio

L, H = 1.2, 0.8


def test_two_iterations_of_the_pipeline_on_a_rectangle():
    fam = VertexPolygon(4, perim=2 * (L + H))
    ev = SpectrumEvaluator(fam, n_eigs=3, prec=1e-10, degenerate_tol=0.0, warn_gap=0.0)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    seen = []
    res = maximize_ratio(ev, param_vector(np.array([0, L, L + 1j * H, 1j * H]), fam.perim), k=1,
                         max_iter=2, constraints=cons, verify_final=False,
                         callback=lambda p, F, model, delta, rho: seen.append((F, delta, rho)))
    assert len(seen) == 2 and res.nit == 2
    assert np.isfinite(res.fun) and 1.0 < res.fun < 2.5387340 + 1e-9      # under the PPW bound
    assert isinstance(res.n_warnings, int)
    for key in ("worst_sigma", "worst_weyl_deficit", "resolution"):
        assert key in res
    assert np.isfinite(res.worst_sigma) and 0 < res.worst_sigma < 1e-8
    assert ev.n_tension_evals > 0 and ev.n_solves >= 1
    # The trial solves were WINDOWED: the step bounded each level's move and chladni re-located
    # the levels in their own windows (2026-10-14), traced as warm solves.
    assert res.n_windowed >= 1
    assert any(t.kind == "warm" for t in ev.trace), [t.kind for t in ev.trace]
