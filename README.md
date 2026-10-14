# douse

**D**omains **o**f **U**nknown **S**hape, from **E**igenvalues — shape optimization and inverse
spectral problems, built on [`chladni`](../chladni).

The forward solver owns *a* domain and its conventions: geometry, normals, eigenvalues with
certified bounds, L²-orthonormal eigenfunctions, and Hadamard boundary integrals. `douse` owns
*families* of domains: the map from a parameter vector to a domain, its derivative, the
shape-derivative chain rule, and the optimizers.

**Backend (2026-10-14): chladni.** `SpectrumEvaluator` solves on chladni: a frozen basis plan
(`chladni.plan`), warm-started from the previous iterate, with certified per-level bounds and a
Weyl count audit, refusing what it cannot certify (`douse/chladni_backend.py`). The earlier
backend, [`lappy`](../lappy), is a **deprecated** optional extra (`pip install douse[lappy]`):
`LappySpectrumEvaluator` reproduces older results, and the tests of its internals are marked
`lappy` and skip without it. lappy-only options (`replan_tol`, `lam_max_window`, `track_ttol`,
...) are accepted by `SpectrumEvaluator` and ignored, with a one-time warning.

## Status: Milestones 1-3 complete

**The stack runs end to end and recovers the known optima.** Minimizing `lam_1` over polygons of
fixed perimeter from a generic starting shape:

| family | result | exact | error | shape |
|---|---|---|---|---|
| triangle | 29.608813203268 | `3 pi^2` | −9.6e-14 | sides equal to 2.5e-08, angles 60° ± 1.9e-06 |
| quadrilateral | 19.739208802178 | `2 pi^2` | −9.3e-13 | sides equal to 2.1e-07, angles 90° ± 2.2e-05 |

Those are the two cases where the answer is a theorem (Pólya–Szegő); `n >= 5` is open.
`benchmarks/known_minimizers.py` reproduces them. Benchmarks that drive the deprecated lappy
backend live in `benchmarks/lappy/` and need an environment of their own (see its README).

Minimizing `lam_2` instead does **not** give the square — it gives a rectangle with
`L = 2*4^(1/3)/(1+4^(1/3)) = 1.227023580871` and `lam_2 = 42.7396543166`, reached to 3.8e-11.

### Spectral ratios: `lam_k/lam_1`

The polygon analogue of Osting's Fourier-boundary work, and the reason `douse.certify` exists.
Over all planar domains the disk maximizes `lam_2/lam_1` at 2.5387340 (Payne–Pólya–Weinberger);
over regular N-gons the ratio climbs monotonically towards it, and **at every one of them
`lam_2 = lam_3` exactly** — so the conjectured optimum sits on a degeneracy, where no gradient
method converges.

```sh
uv run python -m benchmarks.regular_ngon_ratio
```

| N | n_p | λ₂/λ₁ | rank Q | null dim | worst curvature |
|---|---|---|---|---|---|
| 3 | 2 | 2.3333333 (= 7/3) | 2 | **0** | — first order is decisive |
| 4 | 4 | 2.5000000 (= 5/2) | 2 | 2 | −1.565 |
| 5 | 6 | 2.5268387 | 2 | 4 | −1.814 |
| 6 | 8 | 2.5340068 | 2 | 6 | −1.756 |
| 7 | 10 | 2.5365339 | 2 | 8 | −1.445 |

### Optimizing through the degeneracy

`douse.cluster_opt.maximize_ratio` is a trust-region method whose local model is the m×m cluster
matrix rather than a gradient, so it remains valid where `lam_k` is only directionally
differentiable.

| | descent stalls at | cluster method | in cluster mode |
|---|---|---|---|
| N=4 (square) | 5.70e-04 short | **5.4e-09 short**, angles within 1e-4° | 57 of 58 steps |
| N=5 (pentagon) | 1.15e-02 short | **reaches the pentagon**, angles within 0.06° | 230 of 234 |

N=5 used to end 1.97e-04 short with the gap still open at 0.75 and only 14 of 300 steps in
cluster mode. Two things were wrong, and neither was a second local maximum. The eps-active
cluster width was pinned at `1e-3` instead of being tied to the trust radius, so the run barely
entered cluster mode; and the concave subproblem, posed as an LMI in `(d, t)`, had a degenerate
determinant constraint at its own start point and stopped there — **45% short of the true
optimum** at `Delta = 0.05`, worse as the radius shrank. Recast as linear-minus-a-norm it agrees
with brute force to 1e-13.

A control run separates them: fixing only the subproblem leaves N=5 at 2.63e-05 short with 13 of
300 steps in cluster mode — better by 7.5×, same failure. The active-set width is what makes it
converge. See `docs/todo.md`.

**Every `k` is answered now.** The position of `lam_k` inside its cluster decides the subproblem:
lowest is concave (SDP; a 2×2 fast path), highest is convex and reduces exactly to a search over
the unit sphere `S^{m-1}`, and an interior position is neither, so the exact model — an `m×m`
`eigvalsh`, free next to an eigensolve — is maximized directly. `m > 2` is handled throughout.

And the convex branch immediately says something. At the exact square the model gain for
`lam_2/lam_1` is 0, as it must be if the square is the maximizer — but for `lam_3/lam_1` it is
4.4e-02, so **the square is not a local maximizer of `lam_3/lam_1`**, though the two ratios are
equal there (5/2). Same at the equilateral triangle.

**The first-order condition is automatic, not evidence.** At these points `grad lam_1 = 0` (the
regular N-gon minimizes `lam_1`) and `tr M = 0`, which forces `C = -A` and hence
`det M(d) = -(A·d)² - (B·d)² ≤ 0` identically. So `Q = -AAᵀ - BBᵀ` is negative semidefinite of
rank ≤ 2 whatever `n_p` is — measured rank exactly 2 in every case. The content is second order,
on the `n_p - 2` null directions, and there the ratio falls quadratically. That is evidence for a
local maximum, not a proof; `N = 3` is the sole case where first order settles it, because
`n_p = 2` leaves no null space.

### Animations

```sh
uv run python -m benchmarks.animate_runs            # writes benchmarks/animations/*.mp4
```

Three panels — the shape against the exact minimizer in grey dashes, the eigenvalues against
their exact values in black dashes, and the error on a log axis, since these runs converge
through thirteen orders and a linear axis shows none of it. The `min_lam2_quad` one is the most
informative: `lam_2` and `lam_3` stay far apart the whole way, which is the measurement that said
a cluster optimizer is not yet needed.

```python
import numpy as np
from douse import (VertexPolygon, param_vector, SpectrumEvaluator, ObjectiveFunction,
                   EigenvalueObjective, descend)

fam = VertexPolygon(4, perim=4.0)        # perimeter-4 quadrilaterals, 4 parameters
p0  = param_vector(np.array([0, 1.3, 1.1+0.9j, 0.15+0.8j]), 4.0)

ev  = SpectrumEvaluator(fam, n_eigs=1)   # chladni: frozen plan, warm-started, certified
obj = ObjectiveFunction(ev, EigenvalueObjective(0))      # minimize lam_1
res = descend(obj, p0)

print(res.fun, 2*np.pi**2)               # -> the square
```

For the spectrum and its Jacobian alone, without an optimizer:

```python
lams, J = ev(p0)                         # (K,) and (K, 2N-4)
```

The whole gradient costs about **one** eigenvalue evaluation, not `2N` of them: the
eigenfunction's Cauchy data is computed once and each partial derivative is a weighted integral
against a different velocity field. That is the reason to use the Hadamard formula rather than
differencing the solver.

## What is here

| module | contents |
|---|---|
| `douse/polygon.py` | `VertexPolygon`, the perimeter-constrained vertex parametrization, and `d_vertices_dp` |
| `douse/derivative.py` | the Hadamard shape derivative and the chain rule to `p` |
| `douse/objective.py` | `EigenvalueObjective` (forward), `SpectralMatch` (inverse, scale-invariant), and the arithmetic under them |
| `douse/chladni_backend.py` | `SpectrumEvaluator` on chladni — frozen plan, warm start, certified bounds, re-plan on drift — and the Hadamard handle douse's derivatives use |
| `douse/evaluate.py` | `ObjectiveFunction`, and `LappySpectrumEvaluator` (deprecated: the lappy backend) |
| `douse/errors.py` | the refusals every backend raises: `ShapeRefusal`, `SolverRefusal`, `UnresolvedSpectrum`, `DegenerateSpectrum` |
| `douse/backend.py` | the seam: `SpectrumBackend` and `SolvedShape` protocols, `RunHealth` (the backend's statistics as a value, chladni's names) and `SolveTrace` |
| `douse/ledger.py` | the ledger's schema (`SCHEMA = 2`): `normalize` reads rows of any era under both name sets, `read_rows`/`write_row`, `result_row` |
| `douse/runs.py` | one place that sets a run up: `make_evaluator`, `regular`, `perturbed`, `start_shape`, `convex_start_shape`, `constraints_for` |
| `douse/spectral_guard.py` | the Weyl-count deficit (`weyl_deficit`, `weyl_report`, `DEFICIT_TOL`): is a returned set consecutive? |
| `douse/sampling.py` | start shapes: `random_convex` and the samplers the multistart grid draws from |
| `douse/optimize.py` | `descend`, a backtracking gradient method that respects the parametrization's boundary; `gauss_newton`, a Levenberg-Marquardt driver on `SpectralMatch.residual` for the inverse problem |
| `douse/certify.py` | cluster analysis at a degenerate optimum: `first_order_report`, `second_order_probe` |
| `douse/cluster_opt.py` | `maximize_ratio` — trust-region ascent valid through a degeneracy; `model_gain` (first-order, and the stopping certificate) and `model_gain2` (second-order) |
| `douse/quasi_newton.py` | `SR1`, `BFGS`, and an exact Moré–Sorensen `tr_subproblem` — the curvature the second-order model carries |
| `douse/constraints.py` | `MinEdgeLength`, `ClosingBudget`, `ConstraintSet` — linearized into the trust-region subproblem |
| `douse/degeneracy.py` | turning angles, effective vertices, flatten/collapse candidates — how a shape loses a vertex |

## Three things to know before using it

**Only ever go `p -> vertices`.** `polygon_vertices` is smooth. `param_vector` is not — it picks
the longest side and normalizes orientation, and both choices jump when sides tie. A regular
polygon ties everywhere, and regular polygons are what these problems converge to, so the
canonical map has a branch cut sitting exactly on the target. Use `param_vector` once to build a
starting `p`, then never again.

**Read the health fields before quoting a number.** (Schema 2 rows, written since 2026-10-14, name them `worst_bound`, `worst_deficit`, `n_audit_flags`, ...; older rows are read under both names by `douse.ledger.normalize`, and `summarize_grid` prints the schema mix.) Every `maximize_ratio` result and every
ledger row carries `worst_sigma`, `worst_weyl_deficit`, `n_replans`, `n_warnings` and the
refusals the run met. On chladni `worst_sigma` is the worst *certified relative bound* over the
returned levels, a different quantity from lappy's tension, and sits near `prec` on a healthy
run; `summarize_grid` still gates at `1e-6`. A row with no statistic is reported as
**uncertified**, never dropped (`benchmarks/summarize_grid.py`). Rows record `backend`, `prec`
and the git SHAs of douse and chladni, because a cell is not always written by one version.

**Use `solver.hadamard_quad`, never `solver.bdry_quad` (lappy backend).** Every vertex
perturbation moves a corner, and `bdry_quad` is the node set for *normalizing* eigenfunctions,
not for integrating a corner-moving velocity. `douse.derivative` defaults correctly; the
difference on an L-shape's reentrant vertex is 6.1 digits against 9.3. On chladni the Hadamard
rule is chosen by `Spectrum.hadamard` itself.

**A tracked eigenvalue is verified before it is trusted.** On chladni, a warm solve is accepted
only if continuity checks hold (count, no index shift, top level clear of the window, audit), and
goes cold otherwise and every 10th solve. On the lappy backend: `lappy`'s `track` refuses a minimum
whose tension exceeds `ttol`, but the default `ttol = 1e-3` is a threshold for deciding whether a
tension minimum is a mode *at all*, not whether a located value has converged. The first
end-to-end run converged 2.8e-03 **below** `2 pi^2`, which is impossible — `track` had returned a
point with tension 2.0e-04, inside `ttol`, where the true eigenvalue was 2.9e-03 away.
`SpectrumEvaluator` checks against `track_ttol = 1e-8` and falls back to a full solve. Tightening
the solver's own `ttol` instead is wrong and was tried: it made `solve` return *zero* eigenvalues
on an ordinary triangle.

**Multiple eigenvalues are refused, on purpose.** The shape derivative of a degenerate eigenvalue
is not a gradient — it is the spectrum of an `m x m` matrix, and `lam` is only directionally
differentiable. `dlam_dvertices` raises with that explanation rather than returning a number.
Degenerate optima are the *rule* for these problems, so this needs a real decision about which
optimizer consumes them, not a patch.

## Next

`docs/todo.md` has the full list. Done since this list was written: the constraints
(`douse/constraints.py`, linearized into the trust-region subproblem rather than used as a
barrier), whole-set tracking (`lappy.Eigenproblem.track_set`), the multi-start harness
(`benchmarks/ratio_multistart.py`), and a second-order model under the optimizer
(`douse/quasi_newton.py`). The critical path now:

1. **Globality.** The multi-start harness exists and fingerprints its endpoints by a congruence
   invariant, but the runs are still accumulating. Every statement here about the regular N-gon
   is local until they land.
2. **The exact rectangle oracle should be used as an invariant, not just a benchmark.**
   `benchmarks/rectangle_exact.py` gives `max lam_k/lam_1` over rectangles in closed form for
   every `k`, and rectangles are a subfamily of quadrilaterals, so it is a rigorous lower bound
   on the N=4 answer at every `k` — the only independent check that exists beyond k=3.
3. **`N` up to 16 is not solver-limited.** `benchmarks/ngon_basis_probe.py` resolves the regular
   N-gon to 10.0–11.7 digits at N = 5, 8, 12 and 16, so the scope decision to run to N = 12–16 is
   not blocked by the forward solver. It also shows `lappy`'s `n_cap` is completely inert there —
   the Fourier–Bessel budget alone exceeds every cap and the sources sit at their per-arc floor,
   so exactly one basis size is reachable per `(N, target)`. **Pass `prec` when measuring any of
   this:** at the solver default the same square basis reads 8.84 digits where `prec=1e-13` reads
   13.28, against an exactly known spectrum. An earlier version of that probe omitted it and
   concluded the whole N range was out of reach.
4. **The high-`k` conjecture.** `max lam_k/lam_1` is the regular polygon at k=2, provably not at
   k=3, and — on the exactly solvable rectangle subfamily — returns toward it as `k` grows while
   oscillating forever. Whether that survives on the full N-gon family is open and testable.

The inverse problem runs end to end since 2026-10-14: `gauss_newton` recovers a 1.2 x 0.8
rectangle from six eigenvalues on chladni in 12 iterations (`tests/test_inverse_smoke.py`).
Recovery is second order at a rectangle (the shear is a null direction of the ratio Jacobian,
by reflection symmetry), so a residual `r` leaves edge errors of order `sqrt(r)`; and `K - 1`
ratios must cover the family's `2N - 4` parameters, so three eigenvalues do not determine a
quadrilateral.

## Tests

```sh
uv sync --extra test
uv run pytest                  # the fast tier (default): ~230 tests, ~35 s, one real end-to-end run
uv run pytest -m slow          # the slow tier: full optimization runs, ~15 min
uv run pytest -m ""            # everything
```

Tests of the deprecated lappy backend are marked `lappy` and skip unless a *working* lappy is
installed (`tests/conftest.py` checks that its gsvd4py still takes `tola`; set
`DOUSE_NO_LAPPY=1` to force the skip). The slow tier is dominated by a few full
`maximize_ratio` runs on chladni; if that starts discouraging you from running it, trim their
iteration counts before dropping the cases — the acceptance value is in *reaching* the square,
not in its last three digits.

## Documentation

- [`docs/spectral_optimization.tex`](docs/spectral_optimization.tex) → `.pdf` — the mathematics:
  parametrization and its implicit Jacobian, the Hadamard derivative, what replaces it at a
  multiple eigenvalue, both trust-region subproblems, and why the first-order optimality
  condition at a regular N-gon is automatic rather than informative. Build with
  `latexmk -pdf spectral_optimization.tex`.
- [`docs/todo.md`](docs/todo.md) — what is open, what is settled, and the claims that were
  retracted along the way.
- [`lappy/docs/polygon_contract.md`](../lappy/docs/polygon_contract.md) — what the forward solver
  promises, its validated envelope, and its known defects. Read it before trusting a number.
- [`lappy/docs/scope_and_downstream.md`](../lappy/docs/scope_and_downstream.md) — where lappy
  stops and this package begins.
