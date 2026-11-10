"""What a run does when `F` can no longer judge a step.  (2026-11-09)

Steps are accepted on a measured change in `F`, so `F` known to `eps_f` stops the climb near
`crit = sqrt(2 L eps_f)` however accurate the gradient is. The square reached 2.5 to 1.6e-12 and
was called `not_converged` at crit 8.2e-7. Two answers, chosen by `floor_policy`.
"""
import dataclasses

import numpy as np
import pytest

from douse import cluster_opt as co
from douse.cluster_opt import Outcome, _finish_iteration, _floor_stop, _floor_tol, _trial_step
from test_resync import PlanEv, opts, state


def at_floor(crit=8e-7, eps_f=2.5e-11, eps_g=5e-9, policy='value_limit'):
    st = state(delta=3e-6, gain=crit*3e-6)
    st.crit, st.gain1, st.eps_f, st.eps_g = crit, crit*3e-6, eps_f, eps_g
    st.floor_run = co.FLOOR_REJECTIONS_STEPS if policy == 'accept_steps' else co.FLOOR_REJECTIONS
    return st, dataclasses.replace(opts(), floor_policy=policy)


def test_the_value_limit_passes_what_f_cannot_improve_on_and_says_so():
    st, o = at_floor()
    assert _floor_stop(st, o) and st.success
    assert st.floor_reason == 'eps_f_value_limit' and 'as far as F resolves' in st.message
    assert st.resolution == 8e-7


def test_a_row_inside_the_strict_test_does_not_claim_the_value_limit():
    st, o = at_floor(crit=5e-8)
    assert _floor_stop(st, o) and st.success and st.floor_reason == 'eps_f'


@pytest.mark.parametrize('crit', [1e-5, 1e-2, 1.0])
def test_a_real_stall_at_the_floor_is_still_not_converged(crit):
    st, o = at_floor(crit=crit)
    assert _floor_stop(st, o) and not st.success
    assert st.floor_reason == 'eps_f' and 'not certified' in st.message


def test_the_value_limit_never_passes_the_plateau_cap():
    st, o = at_floor(crit=1e-3, eps_f=1.0)
    assert _floor_tol(st, o) == pytest.approx(o.plateau_max - st.eps_g)
    assert _floor_stop(st, o) and not st.success


@pytest.mark.parametrize('eps_f,eps_g', [(np.nan, 5e-9), (2.5e-11, np.nan)])
def test_without_both_error_bars_the_value_limit_adds_nothing(eps_f, eps_g):
    st, o = at_floor(eps_f=eps_f, eps_g=eps_g)
    assert _floor_tol(st, o) == co._crit_tol(st, o)


def test_accept_steps_keeps_the_strict_certificate():
    st, o = at_floor(policy='accept_steps')
    assert _floor_stop(st, o) and not st.success and st.floor_reason == 'eps_f'


def test_accept_steps_tolerates_the_halvings_a_step_needs():
    st, o = at_floor(policy='accept_steps')
    st.floor_run = co.FLOOR_REJECTIONS
    assert not _floor_stop(st, o)


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match='floor_policy'):
        co.maximize_ratio(type('Ev', (), {'family': None, 'degenerate_tol': 0.0})(), np.zeros(2),
                          floor_policy='nope')


# ---- taking the step ------------------------------------------------------------------------

@pytest.fixture
def new_crit(monkeypatch):
    """The criticality the model at the trial reports: `_floor_step_accepted`'s last test."""
    value = {'crit': 4e-7, 'ok': True}
    monkeypatch.setattr(co, 'local_model', lambda *a, **k: {})
    monkeypatch.setattr(co, '_subproblem', lambda model, delta, rng=None, lin=None:
                        (None, value['crit']*delta, value['ok']))
    return value


def trial(st, o, dF=1e-12, plan='A'):
    ev = PlanEv([([1.0, 2.0 + dF], plan)] + ([([1.0, 2.0], plan)] if plan != 'A' else []))
    assert _trial_step(st, o, ev, {}) in (Outcome.CONTINUE, Outcome.NEXT_ITER)
    return ev


def test_a_step_f_cannot_judge_is_taken_when_the_gradient_improved(new_crit):
    st, o = at_floor(policy='accept_steps')
    st.floor_run = 2
    trial(st, o, dF=-1e-12)
    assert st.F == 2.0 - 1e-12 and st.n_floor_steps == 1 and st.floor_run == 0
    assert st.delta == 3e-6, 'not grown: rho said nothing'


def test_the_default_policy_never_takes_one(new_crit):
    st, o = at_floor()
    trial(st, o, dF=-1e-12)
    assert st.F == 2.0 and st.n_floor_steps == 0 and st.delta == 1.5e-6


def test_no_step_if_the_gradient_did_not_improve(new_crit):
    new_crit['crit'] = 0.95*8e-7
    st, o = at_floor(policy='accept_steps')
    trial(st, o, dF=-1e-12)
    assert st.F == 2.0 and st.n_floor_steps == 0


def test_no_step_if_the_subproblem_failed(new_crit):
    new_crit['ok'] = False
    st, o = at_floor(policy='accept_steps')
    trial(st, o, dF=-1e-12)
    assert st.F == 2.0


def test_no_step_where_f_has_a_verdict(new_crit):
    st, o = at_floor(policy='accept_steps')
    trial(st, o, dF=-1e-10)
    assert st.F == 2.0 and st.n_floor_steps == 0


def test_no_step_while_the_model_still_promises_a_resolvable_gain(new_crit):
    st, o = at_floor(policy='accept_steps')
    st.gain1 = 1e-9
    trial(st, o, dF=-1e-12)
    assert st.F == 2.0


def test_errors_cannot_add_up(new_crit):
    st, o = at_floor(policy='accept_steps')
    st.floor_F = 2.0 + 2e-11                  # already down 2e-11 of the 2.5e-11 allowed
    trial(st, o, dF=-1e-11)
    assert st.F == 2.0 and st.n_floor_steps == 0


def test_the_steps_are_bounded(new_crit):
    st, o = at_floor(policy='accept_steps')
    st.n_floor_steps = co.MAX_FLOOR_STEPS
    trial(st, o, dF=-1e-12)
    assert st.F == 2.0


def test_never_across_a_change_of_plan(new_crit, monkeypatch):
    monkeypatch.setattr(co, '_resync', lambda *a: None)
    st, o = at_floor(policy='accept_steps')
    trial(st, o, dF=-1e-12, plan='B')
    assert st.F == 2.0 and st.n_floor_steps == 0


def test_finish_iteration_reaches_the_floor_stop():
    st, o = at_floor()
    assert _finish_iteration(st, o, PlanEv([]), {}) is Outcome.STOP and st.success
