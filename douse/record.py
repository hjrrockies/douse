"""A run's own record: everything it computed, one JSON line at a time, so a run can be
followed while it runs and examined afterwards WITHOUT RE-SOLVING anything.

A ledger row is a summary: the endpoint, its `lams`, the refused trials and a capped list of
solve kinds. Diagnostics used to recompute the rest -- the rc1 analyses re-solved endpoints cold
for minutes per shape, on a plan that is not guaranteed to be the one the run used. A
`RunRecorder` keeps it as the run goes, beside the row, flushed after every line, so a task killed
at its time limit still leaves its history (`docs/todo.md`, "Every run keeps the spectrum it
computed", phase 1; rebuilding eigenfunctions from a record is phase 2).

THE RECORDER ONLY READS. It never solves, never draws from the optimizer's random stream and
never changes an object the run uses, so a run with a recorder produces the same row and the
same trace as one without (`tests/test_record.py`, and a real run compared byte for byte).
It is attached in two places, both off by default:

* `SpectrumEvaluator.recorder` -- one ``eval`` line per call to `eigenvalues`, accepted, trial,
  refused, re-solve or precision change alike: `p`, every level's value, multiplicity and
  certified bound, the precision asked, the solves it took (cold / warm / plan / replan, rung,
  tension evaluations, seconds), and the plan it was solved on. One ``plan`` line per plan the
  run builds: where it was built, at what precision, and its fields (`chladni.plan.Plan` is a
  frozen dataclass of choices, not of the basis).
* `maximize_ratio(recorder=)` -- one ``iter`` line per iteration of the loop, whatever the
  iteration became: the first-order model (`g1`; and `Ms`, `Lam` for a cluster, or `gk`, `grad`
  when smooth -- the whole first-order information of a nonsmooth `F`, from which criticality,
  the ascent direction and the active branches can be recomputed), the step and its predicted
  and actual gains, `rho`, the radius before and after, criticality against the error floors
  `eps_f` and `eps_g`, the Hadamard accuracy the model was built on, the subproblem's case, the
  geometry and constraint slack at the iterate it ended on, and the run's counters. One
  ``result`` line at the end.

`read(path)` gives the lines back grouped by kind; `benchmarks/diagnostics/plot_record.py`
draws them.
"""
import dataclasses
import json
import time
import warnings
import weakref

import numpy as np

SCHEMA = 1

#: The `_State` counters copied onto every ``iter`` line, so a reader sees what each iteration
#: did without diffing results.
COUNTERS = ('n_cluster_steps', 'n_interior_steps', 'n_infeasible', 'n_unresolved',
            'n_solver_refused', 'n_discontinuity_repairs', 'n_resyncs', 'n_degeneracy_moves',
            'n_qn_steps', 'n_windowed', 'n_window_caps', 'n_corrected', 'n_reanchors',
            'n_floor_steps', 'refusal_streak', 'resync_streak', 'floor_run')


def to_json(x):
    """`json.dumps` default: arrays and numpy scalars as plain lists and numbers, complex as
    `[re, im]`, anything else by `repr` rather than failing a run over a log line."""
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return [x.real.tolist(), x.imag.tolist()]
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    return repr(x)


def _floats(a):
    return None if a is None else np.asarray(a, dtype=float)


def _guarded(method):
    """A record that cannot be written must not change the run: the failure is counted and
    written as an ``error`` line if the file still takes one, never raised and never warned
    (a warning would land in the run's own tally, `cluster_opt._watch_warnings`)."""
    def wrapper(self, *args, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return method(self, *args, **kwargs)
        except Exception as exc:                                      # noqa: BLE001
            self.n_errors += 1
            try:
                self._write(dict(kind='error', where=method.__name__,
                                 error=f'{type(exc).__name__}: {exc}'[:500]))
            except Exception:                                         # noqa: BLE001
                pass
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper


class RunRecorder:
    """Writes one run's record to `path` (appending, so a resumed run continues its file after a
    new ``head`` line). `header` is any JSON-able dict for the ``head`` line: the run's identity
    and code SHAs."""

    def __init__(self, path, header=None):
        self.path = str(path)
        self._fh = open(self.path, 'a')
        self._t0 = time.perf_counter()
        self._plans = []                  # weakrefs, index = plan number in this file
        self._solves = []                 # solves since the last ``eval`` line
        self._n_eval = 0
        self._replans_seen = None
        self._model_at = None             # (p, F, attained) where this iteration's model was built
        self.n_errors = 0
        self._write(dict(kind='head', schema=SCHEMA, wall=time.time(), **(header or {})))

    # -- plumbing ------------------------------------------------------------------------------
    def _write(self, rec):
        rec.setdefault('t', time.perf_counter() - self._t0)
        self._fh.write(json.dumps(rec, default=to_json) + '\n')
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _plan_index(self, plan, at=None, prec=None):
        """The number of `plan` in this file, writing a ``plan`` line the first time it is seen.
        By identity, as the optimizer compares plans (`cluster_opt._plan_of`)."""
        if plan is None:
            return None
        for i, ref in enumerate(self._plans):
            if ref() is plan:
                return i
        try:
            self._plans.append(weakref.ref(plan))
        except TypeError:                 # not weak-referenceable: hold it (chladni's Plan is)
            self._plans.append(lambda plan=plan: plan)
        fields = {}
        if dataclasses.is_dataclass(plan):
            for f in dataclasses.fields(plan):
                fields[f.name] = getattr(plan, f.name)
        self._write(dict(kind='plan', index=len(self._plans) - 1, plan_at=_floats(at),
                         prec=prec, fields=fields))
        return len(self._plans) - 1

    # -- the evaluator's side --------------------------------------------------------------------
    @_guarded
    def plan_built(self, plan, at, prec):
        self._plan_index(plan, at=at, prec=prec)

    @_guarded
    def solve(self, kind, reason, plan, spec, seconds):
        d = None if spec is None else getattr(spec, 'diagnostics', None)
        self._solves.append(dict(kind=kind, reason=reason, rung=getattr(plan, 'rung', None),
                                 plan=self._plan_index(plan),
                                 n_evals=int(getattr(d, 'evaluations', 0) or 0),
                                 seconds=float(seconds)))

    @_guarded
    def evaluation(self, ev, p, lams=None, spec=None, error=None):
        """One call to `SpectrumEvaluator.eigenvalues`, at its end."""
        self._n_eval += 1
        n_replans = int(getattr(ev, 'n_replans', 0))
        rec = dict(kind='eval', index=self._n_eval, p=_floats(p), prec=float(ev.prec),
                   solves=self._solves, n_replans=n_replans,
                   replanned=(self._replans_seen is not None and n_replans != self._replans_seen))
        self._solves, self._replans_seen = [], n_replans
        try:
            rec['v'] = np.asarray(ev.family.vertices(p), dtype=complex)
        except Exception:                                             # noqa: BLE001
            rec['v'] = None
        if error is not None:
            rec['error'] = error
        if spec is not None:
            # `spec` is chladni's `SolveRecord` (plan, warm start, diagnostics) around the
            # `Spectrum` value (levels, flags, audit); a bare `Spectrum` serves for both.
            value = getattr(spec, 'spectrum', spec)
            levels = getattr(value, 'levels', ())
            audit = getattr(value, 'count_audit', None)
            rec.update(
                plan=self._plan_index(getattr(spec, 'plan', None)),
                lam=_floats(value.lam), mult=np.asarray(value.mult, dtype=int),
                bound=_floats(value.bound),
                gap=[_num(getattr(L, 'gap', None)) for L in levels],
                cluster=[getattr(L, 'cluster', None) for L in levels],
                flags=list(getattr(value, 'flags', ()) or ()),
                warm=bool(getattr(spec, 'warm', False)),
                warm_note=getattr(spec, 'warm_note', None),
                audit=(None if audit is None else
                       dict(verdict=getattr(audit, 'verdict', None),
                            statistic=_num(getattr(audit, 'statistic', None)))),
                sigma=_num(getattr(ev, 'last_sigma', None)))
        elif lams is not None:
            rec['lams'] = _floats(lams)
        self._write(rec)

    # -- the optimizer's side --------------------------------------------------------------------
    @_guarded
    def model_built(self, st):
        """`cluster_opt._build_model`, after the model is built: where it was built and the
        Hadamard accuracy it rests on (the inputs of `_error_floor`'s `eps_g`). Pure reads."""
        attained = getattr(st.solver, 'hadamard_attained', None)
        acc = None
        if attained is not None and st.model is not None:
            lo, hi = (int(i) for i in st.model['cluster'])
            acc = dict(lam_1=_num(attained(0, 1)), cluster=_num(attained(lo, hi)))
        self._model_at = (np.array(st.p, dtype=float), float(st.F), acc)

    @_guarded
    def iteration(self, st, o, ev, outcome, delta_in):
        """One pass of `maximize_ratio`'s loop, after its last stage."""
        p_model, F_model, acc = self._model_at if self._model_at is not None else (None, None, None)
        self._model_at = None
        rec = dict(kind='iter', it=int(st.it), outcome=getattr(outcome, 'name', str(outcome)),
                   delta_in=float(delta_in), delta=float(st.delta), rho=float(st.rho),
                   p=_floats(st.p), F=float(st.F), p_model=p_model, F_model=F_model,
                   hadamard_attained=acc,
                   moved=(p_model is not None and not np.array_equal(p_model, st.p)),
                   d=_floats(st.d), gain=float(st.gain), gain1=float(st.gain1),
                   crit=float(st.crit), eps_f=float(st.eps_f), eps_g=float(st.eps_g),
                   sub_ok=bool(st.sub_ok), prec=float(getattr(ev, 'prec', np.nan)),
                   model=_model(st.model),
                   final_case=(st.final_case if p_model is not None
                               and st.final_case_p is not None
                               and np.array_equal(st.final_case_p, p_model) else None),
                   counters={c: int(getattr(st, c, 0)) for c in COUNTERS},
                   n_replans=int(getattr(ev, 'n_replans', 0)))
        if outcome is not None and getattr(outcome, 'name', '') == 'STOP':
            rec.update(message=st.message, success=bool(st.success),
                       floor_reason=st.floor_reason, resolution=float(st.resolution))
        rec['geometry'] = _geometry(o.fam, st.p, o.constraints)
        self._write(rec)

    @_guarded
    def result(self, res):
        out = {}
        for key, val in dict(res).items():
            if key in ('health', 'refusals'):
                continue
            out[key] = val
        self._write(dict(kind='result', **out))


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x


def _model(model):
    """The first-order model as stored: every array the subproblem used."""
    if model is None:
        return None
    keep = ('mode', 'm', 'j', 'cluster', 'resolved', 'lam_1', 'lam_k', 'ratio', 'eps_abs',
            'scale', 'mu', 'split', 'aligned', 'fallback', 'exact_constrained', 'g1', 'gk',
            'grad', 'gmu', 'Ms', 'Lam')
    out = {}
    for key in keep:
        if key not in model:
            continue
        val = model[key]
        if key == 'Ms':
            val = np.asarray(val, dtype=float)
        elif key == 'Lam':
            val = np.diag(np.asarray(val, dtype=float))   # diagonal by construction
        out[key] = val
    return out


def _geometry(fam, p, constraints):
    """Angles, edges and constraint slack at `p`: the early warning for a spike or a notch."""
    from . import degeneracy as dg
    out = {}
    try:
        v = np.asarray(fam.vertices(p), dtype=complex)
        ang = np.degrees(dg.interior_angles(v))
        edges = dg.edge_lengths(v)
        out.update(v=v, min_angle=float(ang.min()), max_angle=float(ang.max()),
                   min_edge=float(np.min(edges)))
    except Exception:                                                 # noqa: BLE001
        pass
    x1 = getattr(fam, 'x1_share', None)
    if x1 is not None:
        try:
            out['x1_share'] = float(x1(p))
        except Exception:                                             # noqa: BLE001
            pass
    if constraints is not None:
        try:
            from .runs import active_constraints
            vals = [np.asarray(c.value(p), dtype=float) for c in constraints]
            out['slack'] = {type(c).__name__: float(np.min(val)) if val.size else None
                            for c, val in zip(constraints, vals)}
            out['active'] = active_constraints(constraints, p)
        except Exception:                                             # noqa: BLE001
            pass
    return out


def read(path):
    """The record at `path`, by kind: ``head`` (a list: one per start or resume), ``plan``,
    ``eval``, ``iter`` and ``result`` lists. A last line cut off by a killed task is skipped."""
    out = dict(head=[], plan=[], eval=[], iter=[], result=[])
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(rec.get('kind'), []).append(rec)
    return out
