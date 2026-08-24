# douse

**D**omains **o**f **U**nknown **S**hape, from **E**igenvalues — shape optimization and inverse
spectral problems, built on [`lappy`](../lappy).

`lappy` owns *a* domain and its conventions: geometry, normals, eigenvalues, L²-orthonormal
eigenfunctions, and Cauchy-data boundary integrals. `douse` owns *families* of domains: the map
from a parameter vector to a domain, its derivative, the shape-derivative formulas, and the
optimizers. See `lappy/docs/scope_and_downstream.md` for why the line is drawn there.

## Status: Milestone 1 complete

**`dlam/dp` is implemented and verified against finite differences.** Nothing else is built yet.

```python
import numpy as np
from lappy import Eigenproblem, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver
from douse import VertexPolygon, param_vector, dlam_dp

fam = VertexPolygon(4, perim=4.0)                       # perimeter-4 quadrilaterals, 4 parameters
p   = param_vector(np.array([0, 1.3, 1.1+0.9j, 0.15+0.8j]), 4.0)
dom = fam(p)                                            # a lappy.Polygon

plan   = BP.plan_basis(dom, weyl_est(6, dom), target=1e-14)
solver = MPSEigensolver.from_domain(dom, basis=BP.realize(plan, dom), prec=1e-13)
lam    = float(Eigenproblem(dom, eval_solver=solver, precision=1e-13).solve(1)[0])

g = dlam_dp(fam, p, solver, lam)                         # (4,) gradient
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
| `douse/opt.py` | **not ported.** Pre-existing objective code (normalized reciprocals, ℓ² loss) against an old lappy API. It does not import. Milestone 2 starts here. |

## Three things to know before using it

**Only ever go `p -> vertices`.** `polygon_vertices` is smooth. `param_vector` is not — it picks
the longest side and normalizes orientation, and both choices jump when sides tie. A regular
polygon ties everywhere, and regular polygons are what these problems converge to, so the
canonical map has a branch cut sitting exactly on the target. Use `param_vector` once to build a
starting `p`, then never again.

**Use `solver.hadamard_quad`, never `solver.bdry_quad`.** Every vertex perturbation moves a
corner, and `bdry_quad` is the node set for *normalizing* eigenfunctions, not for integrating a
corner-moving velocity. `douse.derivative` defaults correctly; the difference on an L-shape's
reentrant vertex is 6.1 digits against 9.3, with both answers looking plausible.

**Multiple eigenvalues are refused, on purpose.** The shape derivative of a degenerate eigenvalue
is not a gradient — it is the spectrum of an `m x m` matrix, and `lam` is only directionally
differentiable. `dlam_dvertices` raises with that explanation rather than returning a number.
Degenerate optima are the *rule* for these problems, so this needs a real decision about which
optimizer consumes them, not a patch.

## Next

- **Milestone 2** — the objective (scale-invariant eigenvalue ratios + a loss) with its gradient, FD-verified.
- **Milestone 3** — minimize `lam_1` over perimeter-constrained polygons. Triangles and
  quadrilaterals have known answers (equilateral, square), so the same code validates itself;
  `n >= 5` is open.

## Tests

```sh
uv run pytest             # 31 tests, ~12 s
```
