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

| | descent stalls at | cluster method | improvement |
|---|---|---|---|
| N=4 (square) | 5.70e-04 short | **6.9e-09 short** | 82,000× |
| N=5 (pentagon) | 1.15e-02 short | 1.97e-04 short* | 58× |

\* N=5 does not converge in 300 iterations and ends with the gap still open (0.75) and angles
7.5° off. N=4 finishes AT the degeneracy (gap 5.3e-09) with angles within 0.004°.

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
| `douse/certify.py` | cluster analysis at a degenerate optimum: `first_order_report`, `second_order_probe` |
| `douse/cluster_opt.py` | `maximize_ratio` — trust-region ascent valid through a degeneracy |
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

`docs/todo.md` has the full list. The critical path:

1. **The convex branch — k=3, and k=5 for N≥5.** There `lam_k` is the *highest* member of its
   cluster, so the local model is `lam_max` of an affine matrix function: convex, and outside
   what `cluster_opt` solves. For m=2 it reduces exactly to a one-dimensional search over a
   circle (`docs/spectral_optimization.tex` §6.2), so it is roughly half a day rather than a
   research project — and it is where the interesting optimizers are expected to be.
2. **Why N=5 does not converge for k=2.** It ends with the gap still open at 0.75 and only 14 of
   300 steps in cluster mode, so it never approached a degeneracy at all. Slow convergence toward
   the pentagon, or a different non-regular local maximum? The second would be a finding.
3. **Multi-start.** Every statement here is local, including that the regular N-gon is optimal.
   Globality is what a paper needs.
4. **k=4**, untested and expected to work as-is — another lowest-of-cluster case.

Deferred deliberately: clusters with m > 2 (D_N gives m=2 generically), a real constraint to
replace the `inf` barrier that confines `descend` to plain backtracking, and the inverse problem,
which is implemented and FD-verified but has never been run end to end.

## Tests

```sh
uv run pytest                  # 77 tests, ~12 min
uv run pytest -m "not slow"    # the fast subset, <1 s
```

Most of the twelve minutes is three full optimization runs. If that starts discouraging you from
running the suite, trim their iteration counts before dropping the cases — the acceptance value
is in *reaching* the square, not in its last three digits.

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
