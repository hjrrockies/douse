# Overview

Open items for `douse`. Same convention as `lappy/docs/todo.md`: one line per open item with a
pointer to wherever the detail lives (a docstring, a commit, `docs/spectral_optimization.pdf`).
Detail is NOT duplicated here — this file is for knowing what is open. An item leaves when it is
done or abandoned; the finding it produced stays in the code it came from.

Numbers appear below only where the number *is* the actionable content.

## Where things stand

`lam_k/lam_1` over N-gons of fixed perimeter — the polygon analogue of Osting's Fourier-boundary
work — is the target. The forward stack (parametrization, shape derivative, objectives,
optimizer) is built and validated; the k=2 optimizer works for N=4 and does not converge for
N=5. The mathematics is written up in `docs/spectral_optimization.tex`.

| | status |
|---|---|
| `min lam_1` | equilateral triangle to −9.6e-14, square to −9.3e-13 |
| `min lam_2` | rectangle `L = 2·4^(1/3)/(1+4^(1/3))` to 3.8e-11, matching closed form |
| `max lam_2/lam_1`, N=4 | 6.9e-09 short of the square, finishing AT the degeneracy |
| `max lam_2/lam_1`, N=5 | 1.97e-04 short, **not converged** |
| `max lam_k/lam_1`, k odd | refused (convex case, not implemented) |

## Spectral ratios — the critical path

- [ ] **The convex branch (k=3, and k=5 for N≥5).** `lam_k` is the HIGHEST member of its cluster
      there, so the model is `lam_max` of an affine matrix function: convex, and maximizing it is
      not the concave program `cluster_opt` solves. For m=2 it reduces EXACTLY to a
      one-dimensional search over a circle — `docs/spectral_optimization.tex` §6.2, derivation
      verified numerically. Estimated half a day, and it is where the interesting optimizers are
      expected to be.
- [ ] **N=5 does not converge for k=2.** Ends 1.97e-04 short after 300 iterations with the gap
      still open at **0.75**, and only **14 of 300** steps in cluster mode — it never approached
      a degeneracy at all. Two candidate explanations, and they differ in kind: slow smooth
      convergence along a flat valley toward the pentagon, or convergence to a DIFFERENT local
      maximum that is not regular. The second would be a finding. Cheapest discriminator: restart
      from the near-pentagon and see whether it stays.
- [ ] **`eps_rel` is fixed at 1e-3** in `cluster_opt.local_model`; the principled choice is
      "whatever a step of radius `Delta` could merge", i.e. tied to the trust radius. Likely part
      of the N=5 story, since that run barely entered cluster mode.
- [ ] **k=4 is untested and should just work** — `lam_4` is a lowest-of-cluster member for N≥5,
      the same concave case as k=2. Half an hour to confirm or refute.
- [ ] **Everything is LOCAL.** No statement in this repo about the regular N-gon is global: the
      second-order probe, the trust-region stop, all local. Multi-start from random N-gons is
      what a paper needs. Mostly compute; `benchmarks/ratio_stall.py`'s checkpointed-ledger
      pattern already supports leaving it running.
- [ ] **Clusters with m > 2.** Both trust-region branches are written for 2×2 blocks. `D_N` gives
      m=2 generically so this is not urgent; the concave case would need an SDP solver (cvxpy),
      and the convex case is genuinely harder because the §6.2 reduction is special to m=2.

## Scope-setting: which k are tractable

Measured at regular N-gons (`lam_1` simple, then alternating doubles). Position within the
cluster is what decides the algorithm — lowest ⇒ concave ⇒ tractable; highest ⇒ convex.

| N | λ₁ | λ₂ | λ₃ | λ₄ | λ₅ | λ₆ | λ₇ |
|---|---|---|---|---|---|---|---|
| 3, 4 | simple | **low** | high | simple | **low** | high | simple |
| 5, 6, 7 | simple | **low** | high | **low** | high | simple | simple |

`lam_2/lam_1` at the regular N-gon: 7/3 (N=3), 5/2 (N=4), then 2.5268387, 2.5340068, 2.5365339,
2.5375926 — climbing monotonically toward the disk's PPW bound 2.5387340.

## Optimizer and infrastructure

- [ ] **A real constraint, not an `inf` barrier.** The admissible set has a boundary (the
      closing-budget condition) and `descend` merely refuses to cross it. scipy's L-BFGS-B stalls
      at iteration 1 against `inf`. A smooth self-intersection constraint — sketched in the
      pre-Milestone-1 `opt.py`, archived — plus one for the closing budget would let a proper
      constrained method run.
- [ ] **The test suite is ~12 minutes**, most of it three full optimization runs. Slow enough to
      discourage running, which is how suites rot. The acceptance value is in REACHING the
      square, not in its last three digits, so trimming iteration counts or marking the heaviest
      opt-in would cost little.
- [ ] **The inverse problem has never been run end to end.** `SpectralMatch` is implemented and
      its gradient is FD-verified, but no recovery has been attempted. It is what the package is
      named for.
- [ ] `douse/opt.py` is superseded and does not import. Kept only until its loss functions are
      confirmed to have no remaining use; `objective.py` reimplements both.

## Settled, and worth not re-deriving

- [x] **The determinant condition certifies nothing.** At a regular N-gon `grad(lam_1) = 0` and
      `tr M = 0` force `C = -A`, hence `det M(d) = -(A·d)² - (B·d)² ≤ 0` IDENTICALLY, with
      `Q = -AAᵀ - BBᵀ` of rank ≤ 2 whatever `n_p` is. An earlier version of this work reported
      that as a certificate on the strength of margins of ~1e-13, which were not small but
      structurally zero. `certify.first_order_report` now returns
      `'first order is automatic'` and the word "certified" does not appear in its output.
      Full derivation: `docs/spectral_optimization.tex` §7.
- [x] **"Degeneracy does not bite" was true for the wrong problem.** `min lam_2` never approaches
      a crossing (gap stays ≥0.38), which is why it converges with the smooth method. `max
      lam_k/lam_1` optimizes TOWARDS one. The measurement was right; the generalization from it
      was not.
- [x] Sorting tracked eigenvalues, and collapse detection — see `evaluate.py`'s docstring and the
      PPW regression test.

## Invariants that have caught real bugs

Kept here because every bug in this work returned a plausible number rather than crashing, and
two were caught by theorems rather than by tests. Tabulated with what each caught in
`docs/spectral_optimization.tex` §8.

- `lam_2/lam_1 ≤ 2.5387340` (Payne–Pólya–Weinberger) — checked at every iterate, not just at the
  end, because a stale objective passed an end-only check.
- The regular N-gon minimizes `lam_1` — a run converging BELOW `2π²` is impossible.
- `rho → 1` as the trust radius shrinks — a first-order model that does not is wrong in
  direction, not merely in magnitude.
- Rigid motions leave `lam` fixed — the per-vertex gradients must sum to zero in each coordinate.
