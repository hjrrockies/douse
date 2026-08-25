# Overview

Open items for `douse`. Same convention as `lappy/docs/todo.md`: one line per open item with a
pointer to wherever the detail lives (a docstring, a commit, `docs/spectral_optimization.pdf`).
Detail is NOT duplicated here — this file is for knowing what is open. An item leaves when it is
done or abandoned; the finding it produced stays in the code it came from.

Numbers appear below only where the number *is* the actionable content.

## Session handoff

`../../HANDOFF.md` (in `projects/python`, alongside both repos) is the resume-from-here note:
scope decisions, results, what was retracted, and the ordered next steps. `../../LAPPY_WISHLIST.md`
is the list of lappy changes that would unlock performance here, each with the measurement behind
it. Not duplicated below.

**Scope, decided 2026-08-25:** Dirichlet only; `k` as high as the solver honestly supports (the
ceiling is a deliverable, and it has now been measured -- see `lappy/docs/todo.md`, and it is
`n_cap` rather than `k` that binds); `N` up to 12-16; symmetric families as a complementary study
rather than a restriction.

**The next thing to build is a quasi-Newton step on the smooth stretches**, with the cluster trust
region kept as the fallback near degeneracies. N=5 k=3 needed more than 250 iterations at
`n_p = 6` and did not converge; N=16 is `n_p = 28`. The optimizer's iteration count, not the
forward solver, is now the binding constraint.

## Where things stand

`lam_k/lam_1` over N-gons of fixed perimeter — the polygon analogue of Osting's Fourier-boundary
work — is the target. The forward stack (parametrization, shape derivative, objectives,
optimizer) is built and validated, and the optimizer now covers every position a `lam_k` can
take inside its cluster, so no `k` is refused. N=4 and N=5 both reach the regular polygon. The
mathematics is written up in `docs/spectral_optimization.tex`; §6.2's m=2 convex reduction has
since been generalized to any m, and §6.1's LMI form is superseded (see below).

| | status |
|---|---|
| `min lam_1` | equilateral triangle to −9.6e-14, square to −9.3e-13 |
| `min lam_2` | rectangle `L = 2·4^(1/3)/(1+4^(1/3))` to 3.8e-11, matching closed form |
| `max lam_2/lam_1`, N=4 | reaches the square; endpoint cold-solved at 5.4e-09 below 5/2 |
| `max lam_2/lam_1`, N=5 | reaches the regular pentagon, stopping on the criticality test |
| `max lam_3/lam_1`, N=4 | **35/11** at the rectangle `L/H = sqrt(8/3)`, from two starts |
| `max lam_k/lam_1`, any k | answered — every cluster position has a subproblem solver |

## Spectral ratios — the critical path

- [x] **The convex branch (k=3, k=5 for N≥5) — DONE, and it generalizes to any m.** Exchanging
      the two maxima in `lam_max(Lam+M(d)) = max_{||u||=1} [u^T Lam u + m(u).d]` makes the
      subproblem `max_{||u||=1} [u^T Lam u lam_1 + Delta*||m(u) lam_1 - lam_k g1||]`, a search
      over `S^{m-1}` — one angle for m=2, a small multi-start for m≥3. Exact: matches a
      brute-force maximization of the model to 1e-13 relative. `cluster_opt._step_convex`.
- [x] **Clusters with m > 2 — DONE.** Concave goes to a `cvxpy` SDP (imported lazily, so the
      common m=2 case never needs it); convex uses the sphere reduction; interior positions
      `0 < j < m-1`, which are neither, maximize the exact model directly by multi-start SLSQP —
      `model_gain` is an m×m `eigvalsh`, free next to the eigensolve that produced `Ms`. Within
      1.3e-05 of a 200-start brute force for m = 3, 4, 5. Bracketing between the two extremes was
      tried first and measured 10–43% short.
- [x] **`eps_rel` is tied to the trust radius** (`eps_rel='auto'`): the active set is as wide as a
      step of radius `Delta` could merge, `Delta * ||(||M^(i)||)_i||`. N=4 went from 81 iterations
      to 58 for the same endpoint.
- [x] **N=5 converges, and it was not a second local maximum.** Was: 1.97e-04 short, gap 0.75,
      14 of 300 steps in cluster mode. Now: the regular pentagon to within the precision
      2.5268387 is quoted to, gap 0, angles within 0.06°, and **230 of 234 steps in cluster
      mode**. Isolated by a control run: with the concave subproblem fixed but `eps_rel` still
      pinned at 1e-3, N=5 ends 2.63e-05 short after 300 iterations with **13 of 300** steps in
      cluster mode and the gap still 0.760 — better than 1.97e-04 by 7.5×, but the same failure.
      Tying the active set to the trust radius is what makes it converge; the concave fix is what
      makes the converged answer sharp.
- [x] **`SpectrumEvaluator` now uses `lappy.Eigenproblem.track_set`** (`use_track_set=True`).
      Per-value tracking was falling back to a full solve on most near-degenerate iterates and
      was occasionally wrong by 1e-08 when it did not. Measured on the N=4 run: full solves
      43 -> 2, endpoint error 5.4e-09 -> 7.3e-10, wall clock 99 s -> 85 s. On N=5 the difference
      is larger: solves 191 -> 2, refusals 54 -> 0, collapses 135 -> 0, wall clock 816 s -> 524 s.

      | run | | solves | refusals | collapses | wall | stop |
      |---|---|---|---|---|---|---|
      | N=4 | per-value | 43 | 12 | 29 | 99 s | criticality |
      | N=4 | track_set | **2** | 0 | 0 | 85 s | criticality |
      | N=5 | per-value | 191 | 54 | 135 | 816 s | radius collapsed |
      | N=5 | track_set | **2** | 0 | 0 | 524 s | **criticality** |

- [x] **Degeneration is now routed, observable, and partly navigable.** `douse/constraints.py`
      (`MinEdgeLength`, `ClosingBudget`, `ConstraintSet`) and `douse/degeneracy.py`
      (`turning_angles`, `effective_vertices`, `report`, `canonicalize`, `collapse_candidates`).
      An N-gon loses a vertex either by collapsing an edge or by flattening one; both reach the
      same domain, but the first makes the parametrization singular and the second is exactly
      null and harmless. So the edge floor forbids the first and leaves the second free, which is
      a GAUGE FIXING rather than a restriction -- every M-gon stays reachable, written with flat
      vertices. Measured cost of the flat representation: the unit square as a 5-, 6- and 8-gon
      solves to 10.3, 11.6 and 10.8 digits against 12.0.

- [x] **RETRACTED: "N=5 k=3 degenerates."** It does not. The stall was a silently degraded
      FROZEN BASIS, and the near-flat vertex was incidental to where the run happened to seize
      up rather than the cause of it. With the tension monitored (`SpectrumEvaluator.replan_tol`)
      both starts walk AWAY from the near-flat vertex to genuine non-degenerate pentagons that
      BEAT the quadrilateral: 3.191229 and 3.187127 against 35/11 = 3.181818, min edge 0.45-0.46,
      no binding constraint, worst sigma 2.6e-04 and 1.4e-04 over 4 and 3 replans. Neither has
      converged (both hit max_iter=250), so those are lower bounds, and their shapes differ.

      Everything below was written before that and is kept as the record of what the degraded
      basis made the runs look like.

- [ ] ~~**N=5 k=3 still does not converge, but it now fails for a different and better reason.**~~
      Sequence, all measured:
      unconstrained -> left the family entirely (edge 1.0e-08, meaningless as a pentagon);
      + edge floor -> pinned against it at 2.973/3.024 with slack 1e-16, unable to cross to the
      flat representation, which is the tunnelling failure predicted in `constraints.py`;
      + degeneracy move -> crosses once, escapes the floor (min edge 0.71, slack 0.63) and
      improves to 3.072/3.078, ending with a vertex 1.4 and 2.0 degrees from flat.
      Still 0.10 short of the quadrilateral's 35/11 = 3.1818, and the trust radius still
      collapses. What is left is a CONVERGENCE problem in the approach to a flat vertex, not a
      representation problem. Prime suspect: the same near-null direction, since the slide is
      exactly null only AT flat and merely ill-conditioned near it.

- [ ] **The degeneracy move fires at most once per run** and only from the constraint boundary,
      so a shape wanting to lose two vertices cannot. Also `_degeneracy_move` disables itself
      after one refusal, which is cruder than it should be.

- [ ] ~~**NOTHING STOPS A RUN LEAVING THE FAMILY, and it happens.**~~ `max lam_3/lam_1` at N=5
      converged to a shape with an edge of length **1.0e-08** out of a perimeter of 4 -- a
      pentagon that has collapsed to a quadrilateral. A second start ended with an edge of
      0.042 and a 152 degree angle. `is_valid` passes both: they are simple, correctly oriented,
      and the perimeter arithmetic is satisfied. This makes the N=5 k=3 numbers meaningless as
      pentagon results and it is the strongest argument yet for `MinEdgeLength` /
      `MinInteriorAngle` constraints, ahead of the self-intersection one.

- [x] **The N=5 collapsed trust radius was the tracking, and `track_set` fixed it.** The run now
      stops on the criticality certificate rather than on a radius ground down to 5.68e-15. The
      mechanism: `rho = (F_q - F)/gain` divides a measured improvement by a predicted one, and
      once the predicted gain falls below the error in `F`, every step is rejected on noise. Per-
      value tracking put ~1e-08 of error into `F` near the degeneracy; the model's gain passes
      through that scale in the endgame, and the radius collapses from there.
- [ ] **No k=4 OPTIMIZATION has been run.** The static structure is pinned — `tests/
      test_cluster_opt.py::test_every_k_gets_the_right_cluster_and_the_exact_ratio` checks the
      exact spectra at the square and the equilateral triangle for k = 2..6, cluster size and
      position both against arithmetic — but that is not a run.
- [ ] **Everything is LOCAL.** No statement in this repo about the regular N-gon is global: the
      second-order probe, the trust-region stop, all local. Multi-start from random N-gons is
      what a paper needs. Mostly compute; `benchmarks/ratio_stall.py`'s checkpointed-ledger
      pattern already supports leaving it running.
- [ ] **`docs/spectral_optimization.tex` is now behind the code** in two places: §6.1's LMI form
      of the concave subproblem is superseded by the linear-minus-a-norm one, and §6.2's convex
      reduction is stated for m=2 where the implementation does any m. §9's "what is not handled"
      list is largely handled.

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

- [x] **A FROZEN PLAN DEGRADES SILENTLY, AND NOTHING WAS WATCHING IT.** `SpectrumEvaluator.plan`
      promised that a shape wandering out of the plan's reach "shows up as a rising tension
      rather than as a silent change of basis, and `sigma_at` is how a caller checks". No caller
      checked, so it was silent after all. Measured at the end of a `max lam_3/lam_1` run at N=5:
      the plan frozen at the START shape gives sigma = 1.6e-04 where a plan built at the END
      shape gives 5.4e-11 -- a factor of **2.9 million** -- and `F` was wrong by 1.6e-06.

      That number is the entire stall, and the mechanism is worth keeping because it will recur:
      `rho = (F(p+d) - F(p))/gain` divides a measured improvement by a predicted one, so once the
      predicted gain falls below the ERROR in `F`, every step is rejected on noise and the radius
      collapses. Measured at the stalled point: rho = 0.99 at radius 1e-03 (the model is fine),
      0.85 at 1e-05, and **-0.52 at 1e-06**. The run then stops at a point whose gradient is
      still O(1) -- the same signature `cluster_opt`'s docstring records for the original smooth
      stall, and it fooled this work twice for the same reason.

      `sigma` costs ~1 ms against a ~300 ms eigensolve. There is no reason not to check it every
      iterate, and `replan_tol=1e-9` now does. `worst_sigma` should be reported with every run.

- [x] **F has a concave KINK at a flat vertex** -- one-sided derivatives +3.413956 and -1.234267
      on the two sides of the sqrt(8/3) rectangle's flat vertex, with `F(0) = 35/11` strictly
      above both neighbours at +-1e-08. The corner exponent `nu = pi/omega` crosses 1 there, so
      the two sides are genuinely different problems meeting at a point. Measured with a FRESH
      plan at every sample (sigma ~1e-13 throughout), so this survives the retraction above --
      it is a true statement about that shape, it just is not where N=5's optimum lives.
      `degeneracy.flatten_candidates`.

- [x] **A frozen basis plan cannot survive a change in CORNER COUNT.** `BasisPlan` carries one
      block per corner, and a flat vertex is not a corner -- so a plan frozen on a 5-corner
      pentagon, realized on the quadrilateral it collapsed to, writes `fb_orders[4]` into an
      array of length 4 and raises `IndexError` inside `lappy.basis_plan.realize`. The
      frozen-plan discipline assumes a constant corner count across the family, and degeneration
      is exactly the act of breaking that. `SpectrumEvaluator.replan` re-freezes at a new shape;
      it is legitimate ONLY at a discrete jump, where the objective moves by O(eps) anyway.

- [x] **Never swallow an exception in a search move.** `_degeneracy_move` began with a bare
      `except Exception: continue`, which hid the `IndexError` above through three complete runs
      -- each reporting `n_degeneracy_moves=0`, indistinguishable from the move being offered and
      declined on merit. It warns now.

- [x] **`douse` and `lappy` index edges oppositely** -- `lappy.utils.edge_lengths` is
      `|v_k - v_{k+1}|`, `douse.polygon.perimeter_gradient` is `|v_k - v_{k-1}|`. Mixing them
      made `collapse_candidates` delete the wrong vertex and emit a "repaired" polygon carrying a
      0.053 edge, the exact defect it exists to remove. `douse.degeneracy.edge_lengths` now
      shadows lappy's with douse's convention.

- [x] **The concave subproblem was 45 per cent short in every run it ever made.** It was
      posed as an LMI in `(d, t)` with `tr >= 0` and `det >= 0`. At `j = 0` the cluster member IS
      `lam_k`, so `l0 = 0`, so the determinant constraint is degenerate at the natural start
      `z = 0` and SLSQP stopped there -- returning 0.00727 where the true optimum was 0.01324 at
      `Delta = 0.05`, and getting worse as the radius shrank. Nothing detected it: a trust-region
      method handed a too-small step still makes progress, just less of it, which reads as slow
      convergence. Recast as `max [a.d + c0 - lam_1 ||L d + b||]` -- linear minus a norm, the
      closed form of `lam_min` for a 2x2 block -- it agrees with brute force to 1e-13. This is
      what the N=5 stall mostly was.
- [x] **A tracked endpoint is not a result.** The N=4 run reported 2.500000005354, which is
      5.4e-09 ABOVE 5/2 and impossible if the square is the maximizer. The shape was a square to
      2.1e-06 in the sides; a cold solve at that same `p` gave 2.499999994645. These runs converge
      ONTO a degeneracy, which is exactly where two seeds merge onto one value and
      `_verify_multiplicity` sees a legitimate double. `maximize_ratio(verify_final=True)` re-
      solves the last iterate from cold; `res.fun_tracked` keeps the old value so the size of the
      discrepancy stays visible.
- [x] **`max lam_3/lam_1` over quadrilaterals is 35/11, at the rectangle with `L/H = sqrt(8/3)`.**
      The convex branch said the square was not optimal; the run says what is. At that rectangle
      `lam_3 = lam_4` (the (1,2) and (3,1) modes cross when `3/H^2 = 8/L^2`), and with the
      perimeter fixed at 4 the ratios are exactly `lam_2/lam_1 = 20/11` and
      `lam_3/lam_1 = 35/11 = 3.1818...`, against 5/2 at the square. Two independent starts reach
      it, to 1.2e-09 and 1.6e-09. The trajectory crosses from `lam_3` being the HIGHEST of
      {lam_2, lam_3} at the square to the LOWEST of {lam_3, lam_4} at the answer, so it ends in
      the concave branch -- but only the convex branch could say the square was worth leaving.

- [x] **The square is NOT a local maximizer of `lam_3/lam_1`**, though it is one of
      `lam_2/lam_1` and the two ratios are equal there (5/2). At the exact square the model gain
      is 0 for k=2 and 4.4e-02 for k=3; at the equilateral triangle, 0 and 2.3e-02. The convex
      branch does not merely run -- it says the conjectured shape is the wrong one. Same for k=6.

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
