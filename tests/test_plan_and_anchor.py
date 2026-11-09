"""Two ways the run's bookkeeping, not its shape, stopped row 10.  (2026-11-09)

A flattened vertex left the iterate on a plan with one corner fewer than any trial's, so every
trial was void (`_resync`). And a nearly flat `v_0` made `x_0` ill-determined (`_reanchor_stage`),
which is a property of the labelling: the polygon only matters up to isometry.
"""
import dataclasses

import numpy as np

from douse import cluster_opt as co
from douse.cluster_opt import Outcome, _trial_step
from douse.errors import UnresolvedSpectrum
from douse.polygon import VertexPolygon, edge_lengths, gauge_vector
from test_resync import PlanEv, opts, state


# ---- across a change of plan -----------------------------------------------------------------

def decisive(gain=1.0):
    st = state(gain=gain)
    st.eps_f = 1e-6
    return st


def test_a_decisive_gain_is_judged_across_plans_and_the_step_is_taken():
    st, ev = decisive(), PlanEv([([1.0, 2.5], 'B'), ([1.0, 2.2], 'C')])
    assert _trial_step(st, opts(), ev, {}) is Outcome.CONTINUE
    assert abs(st.rho - 0.3) < 1e-12 and st.F == 2.5 and np.allclose(st.p, [0.1, 0.0])
    assert ev.current_plan == 'B' and st.solver == 'solved on B', 'the trial is on its own plan'
    assert np.allclose(ev.reseeded[-1], [0.1, 0.0]) and st.back_plan is None


def test_a_rejected_step_across_plans_leaves_the_iterate_on_its_own_plan():
    st, ev = decisive(), PlanEv([([1.0, 2.1], 'B'), ([1.0, 2.3], 'C')])
    assert _trial_step(st, opts(), ev, {}) is Outcome.CONTINUE
    assert abs(st.rho + 0.2) < 1e-12 and st.delta == 0.2
    assert np.array_equal(st.p, np.zeros(2)) and st.F == 2.3 and st.solver == 'solved on C'
    assert ev.current_plan == 'C' and st.back_plan is None
    assert np.array_equal(ev.reseeded[-1], np.zeros(2))


def test_a_gain_the_solves_cannot_resolve_is_still_void():
    st, ev = state(gain=1e-5), PlanEv([([1.0, 2.5], 'B'), ([1.0, 2.2], 'C')])
    st.eps_f = 1e-6                           # 10 eps_f, inside the re-plan's own jump
    assert _trial_step(st, opts(), ev, {}) is Outcome.NEXT_ITER
    assert np.array_equal(st.p, np.zeros(2)) and ev.current_plan == 'C' and st.F == 2.2


def test_without_a_known_error_the_gain_is_never_decisive():
    st, ev = state(), PlanEv([([1.0, 2.5], 'B'), ([1.0, 2.2], 'C')])
    assert np.isnan(st.eps_f) and _trial_step(st, opts(), ev, {}) is Outcome.NEXT_ITER


def test_an_iterate_that_refuses_the_new_plan_keeps_its_value_when_the_gain_is_decisive():
    st, ev = decisive(), PlanEv([([1.0, 2.5], 'B'), UnresolvedSpectrum('not on this plan')])
    assert _trial_step(st, opts(), ev, {}) is Outcome.CONTINUE
    assert abs(st.rho - 0.5) < 1e-12 and np.allclose(st.p, [0.1, 0.0]) and st.n_unresolved == 0
    assert ev.current_plan == 'B'


def test_and_a_rejection_there_puts_the_old_plan_back():
    st, ev = decisive(), PlanEv([([1.0, 1.9], 'B'), UnresolvedSpectrum('not on this plan')])
    _trial_step(st, opts(), ev, {})
    assert np.array_equal(st.p, np.zeros(2)) and st.F == 2.0 and ev.current_plan == 'A'
    assert st.solver == 'solved on A' and st.back_plan is None


def test_the_shared_plan_path_needs_nothing_restored():
    st, ev = decisive(), PlanEv([([1.0, 2.1], 'B'), ([1.0, 2.3], 'B')])
    _trial_step(st, opts(), ev, {})
    assert ev.restored == [] and ev.current_plan == 'B'


# ---- the choice of anchor --------------------------------------------------------------------

class AnchorEv(PlanEv):
    def __init__(self, script):
        super().__init__(script)
        self.replanned = []

    def replan(self, p=None):
        self.replanned.append(np.array(p, float))
        self.current_plan = 'replanned'


def hexagon(flat_by_degrees=None):
    v = np.exp(2j*np.pi*np.arange(6)/6)
    if flat_by_degrees is not None:           # pull v[3] onto the chord of its neighbours
        chord = (v[2] + v[4])/2
        half = abs(v[2] - v[4])/2
        v[3] = chord + (v[3] - chord)/abs(v[3] - chord)*half*np.tan(np.radians(flat_by_degrees)/2)
    return v


def near_flat_start():
    """A convex hexagon whose flattest corner is `v_0` of its labelling."""
    fam = VertexPolygon(6)
    return fam, hexagon(2.0), gauge_vector(hexagon(2.0), fam.perim, anchor=3)


def anchored(p, fam, script=None, constraints=None, ev=None):
    st = state()
    st.p = p
    st.eps_f = 1e-8
    ev = ev if ev is not None else AnchorEv(script)
    o = dataclasses.replace(opts(), fam=fam, constraints=constraints)
    assert co._reanchor_stage(st, o, ev, {}) is Outcome.CONTINUE
    return st, ev


def cyclic_match(a, b):
    return any(np.allclose(np.roll(a, s), b, atol=1e-10) for s in range(len(a)))


def test_a_nearly_flat_v0_is_relabelled_to_the_same_polygon_in_better_coordinates():
    fam, v, p = near_flat_start()
    cond = co._anchor_cond(fam, p)
    assert cond > co.REANCHOR_COND
    st, ev = anchored(p, fam, [([1.0, 2.0], 'B')])
    assert st.n_reanchors == 1 and not np.allclose(st.p, p)
    assert co._anchor_cond(fam, st.p) < cond/co.REANCHOR_GAIN
    assert cyclic_match(edge_lengths(fam.vertices(st.p)), edge_lengths(fam.vertices(p)))
    assert np.array_equal(ev.calls[0], st.p) and np.array_equal(ev.replanned[0], st.p)
    assert st.previous is None and st.solver == 'solved on B'


def test_a_new_labelling_that_reads_a_different_value_is_not_the_same_shape_to_the_solver():
    fam, _, p = near_flat_start()
    st, ev = anchored(p, fam, [([1.0, 2.5], 'B')])
    assert st.n_reanchors == 0 and st.reanchor_tries == 1 and np.array_equal(st.p, p)
    assert st.F == 2.0 and ev.current_plan == 'A', 'the old plan is back'
    assert np.array_equal(ev.reseeded[-1], p)


def test_a_refusal_on_the_new_labelling_keeps_the_old_one():
    fam, _, p = near_flat_start()
    st, ev = anchored(p, fam, [UnresolvedSpectrum('no')])
    assert st.n_reanchors == 0 and np.array_equal(st.p, p) and ev.current_plan == 'A'


def test_a_well_conditioned_anchor_costs_no_solve():
    fam = VertexPolygon(6)
    p = gauge_vector(hexagon(), fam.perim, anchor=0)
    st, ev = anchored(p, fam, [])
    assert ev.calls == [] and st.reanchor_tries == 0 and np.array_equal(st.p, p)


def test_the_tries_are_bounded():
    fam, _, p = near_flat_start()
    st, ev = state(), AnchorEv([])
    st.p, st.reanchor_tries = p, co.MAX_REANCHOR_TRIES
    assert co._reanchor_stage(st, dataclasses.replace(opts(), fam=fam), ev, {}) is Outcome.CONTINUE
    assert ev.calls == []


def test_a_labelling_the_constraints_would_reject_is_not_offered():
    class Never:
        def is_feasible(self, p, tol=0.0):
            return False

    fam, _, p = near_flat_start()
    st, ev = anchored(p, fam, [], constraints=Never())
    assert ev.calls == [] and st.reanchor_tries == 0


def test_other_families_are_left_alone():
    st, ev = state(), AnchorEv([])
    assert co._reanchor_stage(st, opts(), ev, {}) is Outcome.CONTINUE and ev.calls == []
