# douse

**D**omains **o**f **U**nknown **S**hape, from **E**igenvalues — shape optimization and inverse
spectral problems, built on [`lappy`](../lappy).

`lappy` owns *a* domain and its conventions: geometry, normals, eigenvalues, L²-orthonormal
eigenfunctions, and Cauchy-data boundary integrals. `douse` owns *families* of domains: the map
from a parameter vector to a domain, its derivative, the shape-derivative formulas, and the
optimizers. See `lappy/docs/scope_and_downstream.md` for why the line is drawn there.

## Status: Milestones 1-3 complete

**The stack runs end to end and recovers the known optima.** Minimizing `lam_1` over polygons of
fixed perimeter from a generic starting shape:

| family | result | exact | error | shape |
|---|---|---|---|---|
| triangle | 29.608813203268 | `3 pi^2` | −9.6e-14 | sides equal to 2.5e-08, angles 60° ± 1.9e-06 |
| quadrilateral | 19.739208802178 | `2 pi^2` | −9.3e-13 | sides equal to 2.1e-07, angles 90° ± 2.2e-05 |

Those are the two cases where the answer is a theorem (Pólya–Szegő); `n >= 5` is open.
`benchmarks/known_minimizers.py` reproduces them.

Minimizing `lam_2` instead does **not** give the square — it gives a rectangle with
`L = 2*4^(1/3)/(1+4^(1/3)) = 1.227023580871` and `lam_2 = 42.7396543166`, reached to 3.8e-11.

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

ev  = SpectrumEvaluator(fam, n_eigs=1)   # frozen plan, tracked eigenvalues, verified
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
| `douse/evaluate.py` | `SpectrumEvaluator` — frozen plan, tracked-and-verified eigenvalues — and `ObjectiveFunction` |
| `douse/optimize.py` | `descend`, a backtracking method that respects the parametrization's boundary |
| `douse/opt.py` | **superseded**, kept only until its loss functions are confirmed to have no remaining use. Does not import. |

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

**A tracked eigenvalue is verified before it is trusted.** `lappy`'s `track` refuses a minimum
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

- **Degenerate optima.** The open question. `lam` is only directionally differentiable at a
  multiple eigenvalue, and these problems optimize *towards* one; `SpectrumEvaluator` warns when
  the gap closes but nothing handles it. Needed before any gap or higher-mode objective.
- **A real constraint, not an `inf` barrier.** The admissible set has a boundary and `descend`
  merely refuses to cross it. `scipy`'s L-BFGS-B stalls at iteration 1 against `inf`. The smooth
  self-intersection constraint sketched in the old `opt.py` plus one for the closing budget
  would let a proper constrained method run.
- **The inverse problem.** `SpectralMatch` is implemented and its gradient is FD-verified, but no
  recovery has been attempted end to end.
- **`n >= 5`.** Where the forward problem stops being a test and starts being a question.

## Tests

```sh
uv run pytest                  # 52 tests, ~3 min
uv run pytest -m "not slow"    # the fast subset, <1 s
```
