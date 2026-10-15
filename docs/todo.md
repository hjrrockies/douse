# Overview

Open items for `douse`. Same convention as `lappy/docs/todo.md`: one line per open item with a
pointer to wherever the detail lives (a docstring, a commit, `docs/spectral_optimization.pdf`).
Detail is NOT duplicated here — this file is for knowing what is open. An item leaves when it is
done or abandoned; the finding it produced stays in the code it came from.

Numbers appear below only where the number *is* the actionable content.

## Session handoff

**2026-10-14: chladni is the backend, and the pipeline runs on it end to end.** The morning's
commits (`ccc85a8`, `266d370`, `e2e039a`) made the fast tier green on the chladni default, gave
the evaluator run-health statistics the ledger can filter on (a row without one is reported
UNCERTIFIED, never dropped), fixed the level-cut hazard (`lam_3/lam_1` on hexagons drives
`lam_3` into the `lam_4 = lam_5` pair, and a truncated copy killed the run), let warnings reach
`maximize_ratio`'s caller, and ran the inverse problem through `descend` for the first time
(`tests/test_inverse_smoke.py`). lappy is a deprecated extra that cannot run beside chladni's
gsvd4py; its scripts are in `benchmarks/lappy/`. Still open from the audit
(`../CODE_AUDIT_2026-10-13.md`): a single backend protocol instead of `hasattr` dispatch and
`evaluator._last` writes; splitting `maximize_ratio`; speed near the k=2 cluster (chladni is
4-9x slower than lappy there; the seed grid is the parallelizable fifth of a cold solve).


`../../HANDOFF.md` (in `projects/python`, alongside both repos) is the resume-from-here note:
scope decisions, results, what was retracted, and the ordered next steps. `../../LAPPY_WISHLIST.md`
is the list of lappy changes that would unlock performance here, each with the measurement behind
it. Not duplicated below.

**Scope, decided 2026-08-25:** Dirichlet only; `k` as high as the solver honestly supports (the
ceiling is a deliverable, and it has now been measured -- see `lappy/docs/todo.md`, and it is
`n_cap` rather than `k` that binds); `N` up to 12-16; symmetric families as a complementary study
rather than a restriction.

**A SECOND-ORDER MODEL IS NOW UNDER THE OPTIMIZER** (`douse/quasi_newton.py`,
`cluster_opt.model_gain2`). What was planned was a quasi-Newton step on the SMOOTH stretches with
the cluster trust region as the fallback. A mode census said that plan was aimed at the wrong
place, so it was built differently -- see the two entries below.

## Where things stand

**2026-10-14, later: the follow-up plan, sessions 1-3 done, 4 in progress**
(`~/.claude/plans/can-you-do-a-compressed-whisper.md` has the full plan). Session 1: a fast
end-to-end run in the default tier (`tests/test_pipeline_fast.py`), a golden trace of
`maximize_ratio` on a fake backend (`tests/test_cluster_opt_trace.py`), and the optimizer split
into nine stages over an explicit `_State` with unit tests on scripted fakes. Session 2:
chladni's `Spectrum` is a tuple of `Level`s with the cluster logic (`level_index`,
`neighbourhood`, `cluster`) and `SolveRecord` carries what produced it; douse has one backend
seam (`douse/backend.py`: `SpectrumBackend`, `SolvedShape`, `RunHealth`, `SolveTrace`), the
eps-active cluster is built on the solver's levels, and `reseed`/`health` replace the `_last`
writes and the `getattr` reads. Session 3: `douse/ledger.py` (schema 2, `normalize` reads any
era), `douse/runs.py` (one evaluator factory and the shape helpers), provenance and the
per-solve trace on every row, the stored-value pin removed from the summary test; the summary
table on the tracked ledgers is identical before and after. Session 4 so far: chladni's planner
aborts rungs early, hands back the accepted rung's solve (one solve per replan saved), and
declines wild geometry in microseconds; `benchmarks/diagnostics/warm_trace.py` measures where a
k=2 run's solves go. The measurement's verdict (`chladni/notes/warm_trace_2026-10-14.md`): replans are 3-9% of a
run and all `unmet`; the warm solve is the gap (65-70% of a run, ~90 evaluations each, because
chladni's warm window is a full prefix). Next levers (decided 2026-10-14): step-controlled per-level warm windows -- the
optimizer bounds each level's move from its Jacobian and the step (`h_i = max(c |g_i . d|,
resolution_i)`), caps the step so windows stay disjoint, and chladni's `_solve_warm` refines
the windows it is given; then cadence 10 -> 20, `log_j` once per distinct `(nu, k r)`. Session 5 (chladni's
`search.solve` in named steps) not started.


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

- [ ] **A reliable experimental setup for the PRODUCTION spectral-ratio runs, for the thesis**
      (added 2026-10-14). The grid that goes in the thesis -- `max lam_k/lam_1` over N-gons of
      fixed perimeter, N up to 12-16, k as high as chladni honestly supports, Dirichlet, with the
      convex family as the complementary study -- should run on Virginia Tech's research
      computing cluster (ARC), not on a laptop. What that needs, none of it built yet:
      - a run specification that is a FILE (cells, starts, seeds, max_iter, prec, backend, the
        constraint set, the commit SHAs of douse and chladni), so a cell is reproducible from
        the spec alone and `summarize_grid` can say which spec wrote each row;
      - a job layout for the scheduler (Slurm on ARC): one start per task, an array job per
        cell, a resumable ledger per cell as now, BLAS threads pinned per task, memory capped
        per task (the N = 6 runs peaked at 0.9-1.4 GB after the pencil cache fix; measure N = 12);
      - an environment that builds on the cluster without lappy (chladni's `gsvd4py` needs
        LAPACK's `?ggsvd3`; off macOS that is the `scipy-openblas32` route in chladni's
        `pyproject.toml` -- verify on ARC's Linux before anything else);
      - a stability protocol: every certified cell re-run under at least two seeds of the start
        sampler and one ulp-level perturbation of the window, so the thesis quotes a value with a
        spread rather than a number;
      - the summary and figures generated from the ledgers by scripts in the repo, never by hand.
      Depends on the speed work (step-controlled warm windows) landing first: at today's
      1000-1600 s per N = 6, k = 2 start, N = 12 is out of reach on any machine.

- [x] **THE RADIUS COLLAPSE WAS THE ACTIVE SET BEING LOST, and it ended 8 of the first 10
      multi-start runs -- including every run that found the BEST answer.** `eps_abs = Delta*scale`
      is right on the way in and wrong at the end. Measured at the N=4 `max lam_4/lam_1` optimum,
      where `lam_4` and `lam_5` are split by 1.16e-08 absolute (1.14e-10 relative, i.e.
      numerically identical against a basis resolving to ~1e-10) and `scale = 123`:

      | delta | eps_abs | m | criticality |
      |---|---|---|---|
      | 1e-06 | 1.23e-04 | 2 | 1.11e-06 |
      | 1e-10 | 1.23e-08 | 2 | 1.10e-06 |
      | 1e-12 | 1.23e-10 | **1** | **1.35e+00** |
      | 5.7e-15 | 6.98e-13 | **1** | **1.35e+00** |

      Below `Delta = split/scale = 9.4e-11` the model reverts to `m = 1` and reports the
      simple-eigenvalue gradient at a degenerate point -- a DIFFERENT QUANTITY, not an
      approximation (`DegenerateSpectrum`). The optimizer is handed a confident O(1) ascent
      direction where the truth is 1e-06, every step fails, and the radius grinds to 5.68e-15.

      Fixed by `local_model(eps_floor_rel=1e-9)`: the active set never narrows below eigenvalue
      resolution. The asymmetry justifies it -- over-clustering is benign because `Lam` carries a
      real splitting exactly, under-clustering is catastrophic. **When in doubt, cluster.**

- [x] **`gain_tol` CERTIFIED ACCORDING TO WHERE EACH SHAPE'S NOISE FLOOR HAPPENED TO FALL.**
      Criticality does not go to zero at these optima; it plateaus, flat across six decades of
      `Delta`, at the accuracy floor of the shape derivatives. Measured at five converged N=4
      endpoints: **8.9e-08, 1.6e-07, 4.2e-07, 8.2e-07, 1.1e-06**. With `gain_tol=1e-7` the first
      two certified and the other three collapsed their radius -- same optimizer, same quality,
      opposite verdicts, and *the two it certified were the worse local maximum*.

      Replaced by plateau detection: when criticality is already small, re-solve the first-order
      model at `Delta*1e-3` and stop if the two agree. `res.resolution` reports the criticality
      actually achieved and **should be quoted with `F`** -- "first-order optimal" with no
      resolution attached is the claim this exists to stop making.

      Verified against the recorded endpoints: the N=4 `lam_4/lam_1` optimum that previously ended
      `trust radius collapsed to 5.68e-15` after 149 iterations now stops at **nit=1** with
      `resolution = 1.112e-06`, matching the plateau measured independently beforehand.

      It also triages for free. The same number separated a junk shape (6.2e+04), an unconverged
      run (2.1e-03) and four converged ones (<= 5.5e-06) with no other information.

- [x] **The multi-start harness runs its starts on a process pool** (`--workers`). Cells are
      independent, so this is near-linear until the cores run out. Per-worker thread counts are
      throttled to match -- `lappy` takes `mps.n_workers_default` threads per solve (8 by default)
      and scipy's BLAS takes its own, so five unthrottled workers on ten cores is a 4x
      oversubscription contending for the memory bandwidth the per-sigma LAPACK calls are already
      limited by. `benchmarks/run_grid.py` drives a `(N, k)` grid over it, resumable at both cell
      and start granularity.


- [x] **`max lam_k/lam_1` OVER RECTANGLES IS EXACTLY SOLVABLE FOR EVERY `k`, AND THE ANSWER IS
      ALWAYS RATIONAL.** `benchmarks/rectangle_exact.py`. On `L x H` the spectrum is
      `pi^2(m^2/L^2 + n^2/H^2)` in closed form, so the maximization is a scalar problem solvable to
      machine precision in milliseconds with no discretization anywhere.

      | k | 2 | 3 | 4 | 6 | 8 | 11 | 13 | 17 | 25 | 28 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | max | 5/2 | **35/11** | 4 | 77/13 | 7 | 28/3 | 11 | 69/5 | 221/11 | 43/2 |

      Rational at all 29 indices k=2..30, and it is not a coincidence: the maximum sits at a
      CROSSING `lam_{m,n} = lam_{m',n'}`, which pins `L^2/H^2 = (m'^2-m^2)/(n^2-n'^2)` rational,
      and then every eigenvalue is `pi^2/L^2` times a rational so the factor cancels out of the
      ratio. The aspect ratio is the square root of a rational; the VALUE is rational outright.
      **35/11 is the first entry of a family, not a special case.**

      The square is optimal at 11 of the 29 -- k = 2, 4, 5, 7, 9, 14, 16, 20, 21, 23, 29 -- and not
      at the others. Which pair `(m, n)` competes for the k-th slot is an arithmetic question about
      the lattice, so the answer never settles down.

      **USE IT AS AN ORACLE.** Rectangles are a subfamily of quadrilaterals, so this is a rigorous
      LOWER bound for the N=4 answer at every `k` -- the only independent check that exists beyond
      k=3, where until now the acceptance tests stopped. Wired into `benchmarks/ratio_multistart.py`
      and pinned in `tests/test_rectangle_oracle.py` against closed forms.

- [x] **The high-`k` question in the handoff has a candidate answer.** `lam_k/lam_1` grows like `k`
      by Weyl, so what is the claim at k=12? With the perimeter fixed,
      `lam_k/lam_1 ~ 4 pi k/(|Omega| lam_1)` -- and the shape-dependence at leading order is
      ENTIRELY in the scale-invariant Faber--Krahn functional `|Omega| lam_1`. So maximizing the
      ratio tends to MINIMIZING it, whose answer among N-gons is the regular N-gon (a theorem for
      N = 3, 4). Prediction: the maximizer returns to the regular N-gon as `k` grows.

      Measured on rectangles, in the predicted direction and slowly: mean `|L/H - 1|` is 0.1370 for
      k<8 against 0.0773 for k>=15, and `ratio/k` falls to 0.78 at k=30 against the limit
      `2/pi = 0.6366`. **A heuristic, not a theorem** -- Weyl's `O(sqrt(lam))` error is not uniform
      over a family, so this is an interchange of limits that has not been justified. Write it up
      as a numerically supported conjecture with the exact small-`k` table underneath.

      The structure that makes it interesting: at k=2 the answer IS the regular polygon, at k=3 it
      provably is not, and as k grows it comes back to it in the mean while oscillating forever.
      Still to do: test this on the full N-gon family rather than only on rectangles.


- [x] **THE SMOOTH STRETCHES ARE NOT WHERE THE ITERATIONS GO, on the problem that motivated
      looking. Measured, per-iteration mode census over the first 80 iterations:**

      | run | smooth | cluster | accepted |
      |---|---|---|---|
      | N=4 `max lam_3/lam_1` | 3 (4%) | 77 | 53 (66%) |
      | N=5 `max lam_3/lam_1` | 55 (69%) | 25 | 33 (41%) |

      N=4 is 96 per cent cluster mode, so a quasi-Newton step confined to the smooth branch would
      have touched three iterations out of eighty. N=5 spends its first ~52 iterations smooth and
      then goes to cluster and STAYS there, so the smooth branch is the approach and the cluster
      branch is the endgame. `|grad|` on N=5's smooth steps runs 2.53 -> 1.05 and never falls
      below 1.01 -- the signature of a steepest-ascent grind, which is what that branch was
      (`d = Delta*g/||g||`, ignoring curvature and always pinned to the trust boundary).

      The conclusion that mattered: curvature had to reach the CLUSTER branch too, or it would
      not touch the endgame of either run. Which required a model that carries curvature through
      a point where `F` has no gradient.

- [x] **`model_gain2`: a quotient of quadratics, replacing the linearization of a quotient.**

          num(d) = lam_k + sigma_j(Lam + M(d)) + (1/2) d^T Hmu d
          den(d) = lam_1 + g1.d                + (1/2) d^T H1  d
          dF(d)  = num/den - lam_k/lam_1

      Two independent improvements, and the first is free. THE QUOTIENT IS EXACT: even with both
      Hessians zero this beats `model_gain`, using no information the first-order branch did not
      already have. At `Delta = 0.5` with `lam_1 ~ 20`, `g1.d` is O(1), so the linearized quotient
      is wrong in the third figure of its own gain.

      THE CURVATURE RIDES ON THE TWO SMOOTH SCALARS AND NOTHING ELSE. `sigma_j` of an affine
      matrix function stays first-order on purpose -- it is the nonsmooth part, the reason
      `cluster_opt` exists, and a second-order model of it would be fiction. What a Hessian can
      honestly describe is the surface the cluster rides on: `lam_1`, and the CLUSTER MEAN
      `mu = (1/m) sum_{i in cluster} lam_i`, which is analytic through a crossing where its
      members are not, with gradient `(1/m) tr M^(i)` read straight off `Ms`. So curvature
      accumulates in cluster mode as well as smooth mode, which is the whole point. `lam_1`'s
      secant chain never breaks on a mode switch at all.

      SR1 rather than BFGS, because neither scalar is convex along the path a run takes and BFGS
      would spend its pairs refusing to learn what it cannot represent; the trust-region
      subproblem is solved EXACTLY (More--Sorensen, `quasi_newton.tr_subproblem`, verified against
      brute force on 192 cases including indefinite, negative-definite, singular and the hard
      case), so an indefinite Hessian needs no taming.

      THE STOPPING CERTIFICATE IS DELIBERATELY LEFT FIRST-ORDER. `maximize_ratio` still measures
      criticality against `model_gain`. A quadratic `gain/delta` is not a criticality measure:
      near a maximum the Newton step goes interior and its gain falls like `|grad|^2` while
      `delta` sits at `delta_max`, so it would certify optimality at `|grad| ~ 3e-04`. The
      quadratic gain is used for `rho`, where it belongs and where it is what makes `rho -> 1`
      hold to second order.

      Measured, `max lam_3/lam_1` over quadrilaterals from the near-regular start, same endpoint
      (35/11 to 6e-10) and same stop (the criticality certificate):

      | | iterations | wall |
      |---|---|---|
      | first-order | 106 | 192 s |
      | second-order | **79** | **143 s** |

      That is the case that is 96 per cent cluster mode, so it is close to the floor of what this
      buys rather than the ceiling.

- [ ] **AND IT DOES NOTHING AT N=5, WHICH IS THE HONEST HALF.** Same problem, same start,
      `max_iter=400`:

      | | F | iterations | worst_sigma |
      |---|---|---|---|
      | first-order | 3.191288451115 | 400 (no convergence) | 2.6e-04 |
      | second-order | 3.191256026241 | 400 (no convergence) | 2.5e-04 |

      `qn_steps=400` and `curv=(311, 309)` confirm the second-order path ran and absorbed
      curvature on nearly every iterate, so it is being used and it is not helping. The two runs
      agree to 3e-05 and neither reaches the criticality certificate.

      **AND `worst_sigma` IS NOT THE CAUSE, though it looks exactly like it.** N=4 ran at
      2.4e-10 with no replans and N=5 at 2.5e-04 with four, which is the same order as the
      1.6e-04 behind the retracted 3.0717 -- so the obvious reading is a noisy objective. The
      discriminator says otherwise. Instrumented per iterate, the N=5 run has `delta = 1.56e-03`
      (min over the run 1.9e-10, so nowhere near collapse), `rho` between +0.24 and +0.97 with
      every value POSITIVE, 130 of 200 steps ACCEPTED, and `F` climbing monotonically to the last
      iterate. **It is not stalled; it is converging slowly.**

      `rho` settling at 0.62-0.70 just under the 0.75 growth trigger looked like a threshold
      artifact. `eta_grow=0.6` was tried and **measured worse** -- 3.1907912 against 3.1910526
      over the same 200 iterations, 94 accepted steps instead of 130 -- because it produces a
      LIMIT CYCLE: accept at 1.56e-03 with rho +0.86, grow to 3.13e-03, reject with rho -0.3,
      shrink back, repeat. So 1.56e-03 is the genuine radius of validity of the local model, and
      `rho` flips from +0.86 to -0.3 on a single doubling -- the sharp breakdown you get when a
      step crosses out of the region its cluster branch assignment holds in. Full record in
      `maximize_ratio`'s docstring under `eta_grow`, which stays at 0.75.

      **So the lever for N=5 is a model valid over a LARGER region** -- more second-order content
      in the cluster branch, or a better parametrization -- and it is neither a threshold tweak
      nor the basis. That is the next thing to build.

      The wall-clock numbers for N=5 (1319 s and 1222 s) were taken with basis probes running
      alongside; the iteration counts are what should be compared. The N=4 numbers were clean.

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

## The search space itself — constraints and starts

- [x] **`Convexity(family, margin=0)` and a `--convex` search mode (2026-08-26).** `turn_k >=
      margin` at every vertex; a RESTRICTION, not a gauge fixing like the edge floor, and meant
      to be -- it asks "the maximizer over CONVEX N-gons" as its own question. The constraint is
      the TURNING ANGLE, not the cross product, which carries `|e_k||e_{k+1}|` and so would mean
      different things at different vertices. `margin = 0` admits flat vertices exactly on the
      boundary, so every convex M-gon stays reachable. It does NOT stop a spike; that end wants
      its own floor. Convex cells write `N<N>_k<k>_convex.jsonl` and are never pooled with the
      unconstrained ones -- `summarize_grid` and `plot_optima` both key on family.
- [x] **THE START SAMPLER DECIDES HOW MUCH OF A SWEEP SURVIVES ITS OWN GUARDS.** Of the 79 stored
      rows, **7 of the 13 begun at a reentrant shape were excluded on `worst_sigma`, against 2 of
      the other 66**, and the excluded rows repeat by start index across k. Measured over 1500
      draws (`benchmarks/sampler_stats.py`), the star-shaped sampler is reentrant on 1.1% of
      draws at N=4, 26% at N=6, **77% at N=8 and 99.7% at N=12**. So part of the certification
      collapse with N is the sampler, not the physics.
- [x] **The legacy sampler cannot reach N=12 or N=16 at all.** Its fixed 0.25 rad gap floor eats
      48% of the circle at N=12 and 64% at N=16; the rejection loop essentially never lands.
      `douse/sampling.py` subtracts the floor first and splits the remainder with a flat
      Dirichlet -- every draw admissible, floor exact and scaling with N.
      `ratio_multistart.start_shape` is left BYTE-IDENTICAL: a sweep resumes by start index and a
      ledger built from two generators is worthless.
- [x] **`random_convex` is Valtr's algorithm** -- uniform over convex N-gons, exact, no rejection,
      so the angle distribution is a fact about that measure rather than an artifact of a filter.
      Two feasibility bands at the start: `min_edge` matched to `MinEdgeLength`'s floor (17.3% of
      raw N=12 convex draws violate it, i.e. are infeasible before the run begins) and
      `min_interior_deg = 15`, clear of the `sharp_ref/alpha` regime where a corner's block
      shrinks toward `M = 1`.
- [ ] **Reentrant corners are IN SCOPE for `lappy`**, and eventually for the inverse problem, so
      the convex mode is a second question rather than a retreat from the first. The measured
      failure is at ~280 degrees; the envelope probe that maps certified digits against reflex
      angle is the next `lappy`-side step. See `../../LAPPY_WISHLIST.md`.

## Scope-setting: which k are tractable

Measured at regular N-gons (`lam_1` simple, then alternating doubles). Position within the
cluster is what decides the algorithm — lowest ⇒ concave ⇒ tractable; highest ⇒ convex.

| N | λ₁ | λ₂ | λ₃ | λ₄ | λ₅ | λ₆ | λ₇ |
|---|---|---|---|---|---|---|---|
| 3, 4 | simple | **low** | high | simple | **low** | high | simple |
| 5, 6, 7 | simple | **low** | high | **low** | high | simple | simple |

`lam_2/lam_1` at the regular N-gon: 7/3 (N=3), 5/2 (N=4), then 2.5268387, 2.5340068, 2.5365339,
2.5375926 — climbing monotonically toward the disk's PPW bound 2.5387340.

## The oracle is a STRICT lower bound, and the reason is a saddle

- [x] **The best rectangle for `lam_4/lam_1` is a SADDLE of the quadrilateral problem, and its one
      ascending direction is exactly "stop being a rectangle."** The exact oracle
      (`benchmarks/rectangle_exact.py`) gives `max_rectangles lam_4/lam_1 = 4` at `L/H = 0.775064`.
      Evaluated there with the full quadrilateral machinery:

      - `F = 4.0000000000`, matching the oracle to every digit printed -- an independent check of
        the whole forward stack against a closed form, at a `k` nothing else reaches;
      - `lam_4` is SIMPLE there (`m = 1`), so `F` is smooth and an ordinary Hessian applies;
      - criticality `3.9e-13`: it IS a stationary point of the quadrilateral problem, not merely a
        constrained maximum within rectangles;
      - Hessian eigenvalues `[-3607, -1353, -531, +0.462]` -- **a saddle**, with the ascending
        direction about 1000x flatter than the descending ones.

      The `+0.462` is not noise (the second-difference floor at `h = 1e-3` is ~1e-4) and was
      confirmed by walking it: `F` rises in BOTH signs by `+2.31e-07`, `+2.05e-06`, `+1.99e-05` at
      `t = 1e-3, 3e-3, 1e-2`, against `t^2 * 0.462/2 = 2.31e-07, 2.08e-06, 2.31e-05`. It turns over
      by `t = 3e-2` and falls by `t = 0.1`. The shape at `t = 0.1` has angles
      97.4/82.6/83.5/96.5 -- off the rectangle family.

      **So the oracle bound is strict, and it says where to go.** Quadrilaterals beat it at
      4.2833985583 (an isosceles trapezoid), and the escape route from the rectangle family is
      this flat ridge. It also explains the basin structure measured in that cell: 8 of 8 starts
      certified, splitting 4 to the trapezoid at 4.2834 and 4 to the `sqrt(5)` rectangle at
      exactly 7/2, with NONE landing on the 4.0 saddle -- which is what a 1000:1 curvature ratio
      predicts.

      Worth repeating at other `k`: the oracle gives an exactly-known rectangle for every one, so
      "is the rectangle optimum a saddle, and is the escape always out of the family?" is a
      cheap question with an exact reference at every `k`.

## The parametrization — measured, and NOT the bottleneck

- [x] **`cond(dv/dp)` is 1.7 to 2.3 at every optimum found so far**, which is essentially ideal,
      so there is no conditioning case for changing the scheme. Measured at the recorded endpoints
      (N=4 k=2,3,4,8 and N=5 k=3). The pole in `polygon_vertices` -- `vertices[0]` is
      `(C^2-|v_1|^2)/(2(C-x_1))`, singular at `C = x_1` -- sits 0.75 to 1.9 away on a perimeter of
      4, i.e. nowhere near.

      **And it is not what caps the trust radius.** At `||d|| = 1.56e-03`, the radius where `rho`
      flips from +0.86 to -0.3, the map `p -> vertices` is affine to 1.0e-03 RELATIVE. That is two
      orders below the model error it would have to explain, so the radius limit is the cluster
      branch structure, as diagnosed, and not the coordinates. Do not re-derive this.

      Relabelling which vertex is 0 changes `cond` by a factor of 1.00 -- `param_vector`
      canonicalizes on the longest side, so every relabelling maps to the same `p`. That is the
      branch cut earning its keep in one place and costing in another.

- [ ] **If degeneracy handling ever becomes the bottleneck again, the change to make is
      (edge length, turning angle) coordinates**, and the payoff is structural rather than
      numerical. With `l_1..l_N` and `theta_1..theta_N`:

      | today | in (l, theta) |
      |---|---|
      | perimeter: an implicit rational solve for `vertices[0]`, with a pole | `sum l_j = P`, LINEAR (a simplex) |
      | `ClosingBudget`, an artifact of one vertex absorbing the constraint | gone |
      | `MinEdgeLength`, linearized into the subproblem | `l_j >= l_min`, a BOX constraint |
      | a flat vertex: a nonlinear condition, `flatten_candidates`, the kink | `theta_j = 0`, a coordinate hyperplane |
      | a collapsed edge: `collapse_candidates` | `l_j = 0`, another hyperplane |
      | vertex 0 special (row norms `[2, 1, 1, 0]`) | fully symmetric |

      The cost is that closure becomes two explicit nonlinear constraints
      (`sum l_j exp(i Phi_j) = 0`) instead of one implicit elimination. That is a good trade for
      this project *only* because degeneration is routine here -- most of `degeneracy.py` and
      `constraints.py` exists to express, in vertex coordinates, conditions that are coordinate
      hyperplanes in these. It buys nothing for the current bottleneck.

- [x] **For a RATIO objective the scale gauge is unnecessary**, since `lam_k/lam_1` is invariant
      under scaling -- the perimeter constraint fixes a gauge the objective cannot see. It still
      earns its place because `min lam_1` is not scale-invariant and shares the parametrization.
      Worth knowing before anyone "simplifies" it away.

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
- [x] **The inverse problem runs end to end** (2026-10-14): `optimize.gauss_newton` on
      `SpectralMatch.residual` recovers a 1.2 x 0.8 rectangle from six eigenvalues on chladni
      (`tests/test_inverse_smoke.py`). Second order at a rectangle (shear is a null direction of
      the ratio Jacobian by reflection symmetry); `K - 1 >= 2N - 4` ratios are needed.
- [x] `douse/opt.py` deleted 2026-10-14 (its loss functions live in `objective.py`);
      `archive/pre_milestone1/` with it.

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

## Settled this session, methodological

- [x] **`prec` DOMINATES ACHIEVED ACCURACY, AND A MEASUREMENT THAT OMITS IT IS WORTHLESS.**
      `benchmarks/ngon_basis_probe.py` and `lappy`'s `cap_probe.py` were both first written
      without passing `prec` to `MPSEigensolver.from_domain`. On the unit square, whose spectrum
      is `pi^2(m^2+n^2)` exactly, the SAME 108-column basis gives

      | | `check_precision` | true error vs the exact spectrum |
      |---|---|---|
      | default `prec` | 8.84 digits | 4.6e-10 (9.34 digits) |
      | `prec = 1e-13` (what `SpectrumEvaluator` uses) | **13.28** | **3.1e-14 (13.51 digits)** |

      4.4 digits from a parameter that has nothing to do with the basis. The consequence was not
      a rounding error: the first run of the probe reported ~8.2 digits for the regular N-gon at
      N = 8, 12 and 16, FLAT against every basis-size lever, and that was written up as "the N
      range in scope is out of reach and the ceiling is not `n_cap`". Re-measured at `prec=1e-13`
      the same N-gons give **10.0 to 11.7 digits** and the range is comfortably reachable.

      The tell was available and was not read: the square is the one domain here with a spectrum
      known in closed form, and checking against it takes seconds. **Any accuracy claim about the
      forward solver should be validated on the square first**, because it is the only place where
      "digits" can be measured against truth rather than against a residual estimate.

- [x] **`n_cap` is inert for every regular N-gon at N >= 8** -- a byte-identical plan across
      `n_cap` in {150 ... 600}, because the Fourier-Bessel budget alone exceeds every cap and the
      sources are already at `min_src_per_arc`. `plan.capped` reads `True` at 240 and `False` at
      400 for the same plan, so it cannot be used to decide anything. Detail and the digits table:
      `../../LAPPY_WISHLIST.md` sections 2 and 2b.

- [x] **`OptimizeResult.__repr__` raised on every `maximize_ratio` result.** It formatted `nfev`
      and `grad_norm` unconditionally, and `maximize_ratio` sets neither -- it has no line search
      and no gradient at a cluster. Simply printing a result failed. It now formats whichever of
      the two producers' fields are present.

## Spectral guards — what is watched, and what still is not

- [x] **A missed mode is invisible to `worst_sigma`, and now has its own detector.**
      `douse/spectral_guard.py`: place a cut in every gap between distinct levels of the tracked
      set and compare Weyl's three-term count against what was found. Measured separation, on the
      square's exact spectrum and the nine certified grid cells — clean worst +0.319, the real
      missed-mode run +1.010, threshold 0.5. Recorded as `worst_weyl_deficit` on every
      `maximize_ratio` result and every `ratio_multistart` ledger row.
      **Known blind spot, pinned by a test:** a mode missing in the TOPMOST gap sits above every
      cut and is not seen. Callers must request more eigenvalues than the objective consumes;
      `ratio_multistart` asks for `k+2` and uses `k+1`. Do not remove that margin.
- [x] **THE DEFICIT IS ENFORCED, NOT MERELY REPORTED (2026-08-26).** It was computed on every
      evaluation, warned once, and the bad spectrum was returned anyway. Reproduced on N=6
      `lam_3/lam_1` start 3 (recorded row matched digit-for-digit): at 284 degrees of reflex
      angle `track_set` returned `[56.90, 209.97, 274.11, 286.67]` at a sigma of **4.7e-12**, the
      optimizer accepted a step at `rho = +1265` because `F` had leapt to a ratio of
      non-consecutive values, and the last forty iterations ground the radius to 5.68e-15 against
      a frozen `F`. The two sigma spikes in that run are single-iterate excursions at TRIAL
      points, repaired by a replan on the next call -- so `worst_sigma` catches the row while
      naming the wrong cause.
      Now: a trip buys a FULL SOLVE (the tracking seed is the usual culprit); only a deficit that
      survives a spectrum computed from scratch raises `UnresolvedSpectrum`. `maximize_ratio`
      treats a refusal at a trial shape as an unreachable trial -- shrink, count `n_unresolved`,
      continue -- and a refusal at the current iterate stays fatal. (Since 2026-08-29 that policy
      covers every kind of refusal, not only this one; see `ShapeRefusal` below.)
      `enforce_weyl=False` reproduces the old behaviour and is pinned by a test.
- [x] **EVERY SOLVER REFUSAL AT A WILD SHAPE IS NOW AN UNREACHABLE TRIAL, not just the Weyl one.**
      `EigensolverFailure: f has too many local minima` (five instances by 2026-08-29) and
      `ValueError: N of M sources landed inside the domain` from the `FundamentalBasis` guard
      (one) had become the dominant failure, and both killed runs OUTRIGHT -- every one of them
      MID-RUN, since a shape wild enough to provoke either is only reached after a run has
      travelled, i.e. they killed exactly the runs that were making progress.

      One base class, `douse.evaluate.ShapeRefusal`, with `UnresolvedSpectrum` and the new
      `SolverRefusal` under it, translated in `SpectrumEvaluator.eigenvalues` -- the one place
      that knows which `p` was being asked for -- rather than at the five call sites in
      `cluster_opt` that were leaking them. `maximize_ratio` catches the base class, shrinks the
      radius, counts, and stops on a shared streak; the counts stay separate
      (`n_unresolved`, `n_solver_refused`) because the two say different things about a cell.

      `lappy.core.SourcesInsideDomain` (new, a `ValueError` SUBCLASS so no existing caller
      changes) is what makes the catch narrow. `except ValueError` around a solve would turn
      every shape bug in the family code into a silently-shrunk trust radius, and a test pins
      that an ordinary `ValueError` still propagates.

      Two call sites that could still lose a run were closed with it: the discontinuity repair's
      fresh re-solve (refusing there now drops the repair, not the run -- the iterate never
      moved) and `verify_final`, which already handled `UnresolvedSpectrum` and now handles both.

- [x] **A RUN WHOSE BASIS CANNOT RESOLVE ITS SHAPE STOPS INSTEAD OF EARNING ITS EXCLUSION.**
      `worst_sigma > 1e-6` excludes a row from every summary and is a running max, so a drifted
      run is buying a row that will be discarded whatever it converges to. Measured cost of not
      having this: one N=3 row spent 78 minutes doing exactly that, and the re-run then spent
      hours on rows whose verdict was already fixed.

      The test is NOT on `worst_sigma`. `SpectrumEvaluator.last_sigma` (new) is the tension AFTER
      the replan the drift itself provokes -- `replan_tol = 1e-9` is three decades below the
      gate, so anything above it has already forced a basis rebuilt AT that shape, and
      `last_sigma` is what the rebuild achieved. `maximize_ratio(sigma_abort=1e-6,
      sigma_abort_streak=3)` stops on three consecutive iterates above it; any evaluation back
      under the gate resets the streak, so a transient excursion at a wild trial costs nothing.
      Cannot change a usable result by construction. `sigma_abort=None` disables it.

- [x] **`run_grid` can drive the CONVEX family** (`--convex`, passed through to
      `ratio_multistart`, which has had it since 2026-08-26 and writes its own
      `N<N>_k<k>_convex.jsonl`). The certification RATE is the interesting part, not the values:
      if the convex family certifies much better, that is a result about the search space rather
      than about the solver.

- [x] **A RETRIED START WAS COUNTED TWICE IN EVERY DENOMINATOR.** `run_grid` retries a crashed
      start on resume and `ratio_multistart` APPENDS, so `N6_k1` holds 9 rows for 8 starts and
      `summarize_grid` read `n_total = 9`. Reporting only -- no value was affected, since the
      filters run per row and a duplicate crash row certifies nothing -- but a wrong denominator
      is read as evidence about the basin structure. `summarize_grid.load` now dedupes by start
      keeping the LAST attempt (the resume semantics: a retry exists because the earlier one
      failed, and the newer row records the guards it ran under), and `plot_optima` imports that
      loader rather than keeping its own, which is the same fix its own docstring already
      records for the same file.

- [ ] **`spectral_guard.weyl_deficit` NOW EXISTS IN `lappy` TOO** (`asymp.weyl_deficit`, branch
      `missed-modes`, unmerged), where it drives `solve`'s own audit. Once that branch lands,
      import it rather than keeping two copies of a calibrated statistic in two repos -- the
      threshold is evidence about a measurement, and two of them will drift. Keep `douse`'s
      DEFICIT_TOL as the douse-side gate; it is the same number today and need not stay so.

- [ ] **`row_deficit`'s FALLBACK IS THE WEAKER STATISTIC, and the retroactive audit inherits it.**
      The recorded field is the worst over every iterate; the recomputation sees only the
      endpoint, so it is a lower bound. Over the 9 rows carrying both: 8.761 -> 0.186 (below
      `DEFICIT_TOL`, i.e. the fallback PASSES a run the live guard failed by 17x), 5.103 -> 1.284,
      6.711 -> 1.022, 2.008 -> 1.328. Only 9 of 80 rows carry the field, so "the audit found 3"
      is a floor. Documented at `row_deficit`, pinned in `tests/test_multistart_summary.py`,
      behaviour deliberately unchanged -- re-auditing properly is a compute decision.
- [ ] **PPW per iterate on the `maximize_ratio` path.** Cheap, it is a theorem, and the invariant
      list claimed it existed for long enough to matter. Only constrains `k=1`.
- [ ] **`res_max` IS SECTION 9'S MISTAKE ONE LAYER UP.** `gain_tol` was replaced by plateau
      detection because a fixed threshold "certified according to where each shape's noise floor
      happened to fall". `summarize_grid`'s `res_max = 1e-4` is now a fixed threshold applied to
      the PLATEAU VALUE, and it does the same thing. Measured 2026-08-27: N=3 `lam_5/lam_1`
      reports **NONE CERTIFIED** while three independent starts agree to 1e-6 on 4.3618801,
      because every clean row plateaued between 6.5e-04 and 1e-03 -- the N=3 shape-derivative
      noise floor, where sigma is ~1e-07 against ~1e-11 at N=6. The values are right and the
      verdict is wrong.
      The fix is not a looser constant: it is to compare each row's resolution against the noise
      floor MEASURED at that shape, the same move section 9 made, or failing that to report the
      floor next to the resolution so a reader can see which of the two a cell failed.

- [x] **THE FEM COUNT SETTLED A STANDING QUESTION ON ITS FIRST REAL USE, and the answer was not
      the one either side expected.** HANDOFF section 13 flagged
      `pre_plateau_fix/N5_k2` start 0 (F = 3.1913293884, deficit +0.683, `worst_sigma` 1.6e-10)
      and guessed it might be a FALSE positive, asking that it not be quoted as corroboration
      until re-solved. It is a TRUE positive: `fem_deficit = +1` at diameter/60, /150 and /300
      (1984 to 49842 dofs), the first three values matching as `O(h^2)` while the recorded fourth
      sits 22% ABOVE the FEM's. The pentagon has a DOUBLE eigenvalue at 68.8536 and the run
      returned one copy, then jumped to `lam_5 = 88.46`. The certified row shows the pair
      outright: `[21.575211, 39.487156, 68.853605, 68.853605]`, gap 8.9e-10.

      **But the value it produced is fine**, because the missing copy is ABOVE the index the
      objective consumes -- `F = lams[2]/lams[0]` uses `lam_3` -- so the `n_eigs = k+2` margin did
      its job. The SET is wrong and the VALUE is not, which is a distinction no gate currently
      draws: `summarize_grid` drops the row on the set. Worth drawing, since the objective only
      ever reads indices 0 and k.

      This is also the certified reproduction `LAPPY_WISHLIST` section 6
      (`estimate_multiplicity`) has been missing, and the first demonstration of the counting
      form catching what the per-index form provably cannot.

- [ ] **A ROW THAT DETECTED A PROBLEM AND FIXED IT IS GATED AS THOUGH IT STILL HAD ONE.**
      Caught live, 2026-08-29, in the convex sweep, and the first reading of it here was WRONG --
      recorded because the wrong reading is the tempting one.

      `N4_k3_convex` starts 3 and 6 reached the IDENTICAL endpoint: same rectangle
      (0.618 x 1.382, aspect 2.236087), the same five eigenvalues to the digit, both certified,
      both `F = 3.4999999995`, the known `sqrt(5)` local maximum at exactly 7/2. Start 3 is kept.
      Start 6 is DROPPED as a missed mode (`n_skipped_mode = 1`, the cell reads 7 certified of 8).
      The only difference is trajectory length, 46 iterations against 212.

      **The endpoints are provably clean.** `N4_k3_convex` start 6 is a rectangle, so closed form
      settles it -- the values match `pi^2(m^2/L^2 + n^2/H^2)` to 8.5e-06 relative against
      consecutive gaps of ~50% -- and `fem_deficit = 0` at 9583 dofs. `N5_k2_convex` start 0 has
      `fem_deficit = 0` at 1987, 12466 and 49840 dofs, with all four per-index errors positive and
      falling as `O(h^2)`; its endpoint carries BOTH copies of the double level at 68.854006.

      **What the flag actually was, which is NOT what it first looked like.** The first draft of
      this entry read the mid-run deficit as a Weyl false positive from a wild intermediate shape,
      by analogy with `deficit_calibration.py`. The row's own counters say otherwise: both rows
      have `n_weyl_flags = 1`, `n_weyl_recovered = 1` and `n_replans >= 1`. The guard fired on a
      REAL indexing failure mid-run, rebuilt the basis at that shape, and REPAIRED it. Note the
      `N5_k2` value: +0.683, the same score the `pre_plateau_fix` row gets for genuinely losing one
      copy of that cell's double eigenvalue -- so it is the signature of the true failure, not of
      elongation.

      **So the guard is working and the GATE is wrong.** `worst_weyl_deficit` is a high-water mark
      that survives a successful recovery, and `summarize_grid` discards the row on it -- treating
      "had a problem, detected it, fixed it" identically to "has a problem". `n_weyl_recovered` is
      already recorded on the row and no gate reads it.

      Related and still true in general: `evaluate.py` updates `worst_weyl_deficit` on EVERY
      `eigenvalues` call, including TRIAL shapes the optimizer rejected, which is the same disease
      already fixed once for `worst_sigma` ("a number belonging to a shape it never optimized").
      That is not what happened in these two rows.

      **The fix is not simply to swap in the endpoint value** -- section 13 chose worst-over-run
      deliberately, because recomputing from `x` and `lams` alone is the WEAKER statistic, and the
      `pre_plateau_fix` row proves the endpoint number can itself sit under threshold while a mode
      really is missing. Gate the reported VALUE on the endpoint verified independently (a fresh
      re-solve, or a FEM count), and use worst-over-run, `n_weyl_flags` and `n_weyl_recovered` to
      say what the TRAJECTORY did.

      **NOT DONE MID-SWEEP ON PURPOSE**: `ratio_multistart` imports `summarize_grid` for its
      gates, so an edit there while the queue runs would have later cells running different code
      than earlier ones. The console line that prints "the tracked set is not consecutive, so this
      is not lam_4/lam_1" about a correct, recovered endpoint is part of the same fix.

- [ ] **A SPURIOUS DUPLICATE PASSES EVERY GUARD, and it is a 42% error in `F`.** Isolated
      2026-08-29 on the `pre_plateau_fix/N5_k2` shape. A full solve on a plan frozen at the
      REGULAR PENTAGON returns `[21.5752, 21.5752, 39.4871, 68.8537]` -- `lam_1` reported DOUBLE
      where a plan built at the shape says it is simple. Every index above it shifts up, so
      `F = lams[2]/lams[0] = 1.8302` against a true 3.1913.

      | guard | reading | flags? |
      |---|---|---|
      | worst tension over all values | 2.9e-08 | no (`spurious_tol = 1e-4`) |
      | `weyl_deficit` | **-0.791** | no (flags only `> +0.5`) |
      | `fem_deficit` | 0 | no (flags only `> 0`) |

      **Every completeness check in the stack is one-sided in the MISSING direction.** An extra
      copy of a genuine eigenvalue has a low tension by construction -- it IS an eigenvalue -- so
      `spurious_tol` cannot see it either, and both count statistics go NEGATIVE rather than
      positive. What catches it is multiplicity verification (`_verify_multiplicity` covers the
      per-mode `track` path but NOT `track_set`, which is the default), or the other side of the
      FEM bound: nonconforming elements give guaranteed LOWER bounds on eigenvalues, hence an
      UPPER bound on the count, which is exactly the missing half.

      Note the optimizer may still survive it -- `F` collapsing by 42% makes the step fail, and
      the discontinuity repair re-solves fresh and adopts that -- but no SPECTRAL guard sees it,
      and a run that starts at such a shape has no earlier state to fall back on.

- [ ] **TRACKING PERPETUATES A BAD SET RATHER THAN ORIGINATING IT.** Same shape, same basis size
      (n=157), only the path varied: `track_set` seeded with the WRONG set reproduces it exactly;
      seeded with the CORRECT set it returns both copies of the double level, even on a stale
      basis; a full solve on a plan built AT the shape is correct. So a single bad iterate
      contaminates every iterate after it, and `verify_final` on the same frozen plan cannot clear
      it -- which is why the escalation ladder's REPLAN rung is the one that actually repairs
      (measured: both flagged convex rows recovered via replan, `n_weyl_escalations = 0`,
      `n_weyl_recovered = 1`).

- [ ] **A CHEAP FEM COUNT IS A RIGOROUS COMPLETENESS DETECTOR, and it should probably replace the
      Weyl one.** `benchmarks/completeness_detectors.py`. Conforming P1 elements on an EXACTLY
      triangulated polygon span a subspace of `H^1_0`, so `lam_j <= lam_j^h` and the FEM counting
      function is a LOWER BOUND on the true count: `N^FEM(cut) > #found(cut)` PROVES a mode is
      missing. No calibrated constant -- the deficit is an integer and the threshold is `> 0`.

      Measured on exact rectangle spectra at k=8 and k=20, mesh `diameter/150`, against the same
      sets the Weyl statistic sees: **FEM refuses no clean set and catches every coverable
      deletion (8/8, 20/20 at every aspect ratio), where Weyl refuses clean sets at 1.633:1, 3:1
      and 5:1.** 4873-7708 dofs, 0.2-0.4 s -- against ~1000 tension evaluations of a 340x340 SVD
      for one MPS solve, so it is affordable at the START shape, on every refusal, and at
      `verify_final` without being on the per-iterate path.

      **It also catches the one failure nothing else does.** Counting WITH MULTIPLICITY sees a
      missing COPY of a degenerate level -- all four such deletions on the square at K=12, each
      `fem_deficit = +1` -- which no per-index value test can see, because the returned values are
      unchanged. That is the N=5 `lam_6/lam_1` failure and wishlist section 6.

      Blocked on `LAPPY_WISHLIST.md` 9b (the mesh floor is clamped at 0.05 and `FEMEigensolver`
      never overrides it, so refinement silently returns the same mesh). Fix that first.

- [ ] **CONTINUITY ACROSS A TRUST STEP IS A FREE SECOND CHECK, and it is decisive.** The optimizer
      already has `dlams_dp` for every tracked mode and already bounds the step by `delta`, so a
      mode cannot move more than `max|grad lam_j|*delta` while a SKIP shifts a value by a whole
      level spacing. Measured at the regular hexagon: spacing/motion = 13.6 at `delta=0.05`
      (the default `delta0`), 67.9 at 1e-2, 679 at 1e-3 for k=3, and 6.2 / 31 / 312 at k=5. Even
      at the largest radius the margin is 6x.

      It needs an ANCHOR -- continuity propagates completeness, it cannot establish it, so a set
      that was already wrong at the start shape stays undetected. That is exactly what one FEM
      count at the start shape supplies. Anchor with FEM, propagate with continuity, adjudicate a
      flag with a finer FEM count.

- [ ] **RECALIBRATING `DEFICIT_TOL` IS NOT POSSIBLE: PAST k ~ 6 NO THRESHOLD SEPARATES THE
      CLASSES.** Measured 2026-08-29 in `benchmarks/deficit_calibration.py`, on EXACT rectangle
      spectra so no eigensolver is involved and every "clean" set is exactly the consecutive
      spectrum. Two independent failures:

      **(a) The guard refuses CLEAN spectra, earlier the more elongated the shape.** A complete,
      exactly-known set scores +0.577 at k=8 on the `sqrt(8/3)` rectangle -- which is the N=4
      `lam_3` OPTIMUM -- and +1.324 at 5:1. Verified cut by cut: the number of values at or below
      every cut equals the true count from the full exact spectrum, so it is WEYL'S COUNT that is
      wrong by up to +1.3, not the set.

      **(b) Clean and missed-mode overlap from k=8 up**, with top-gap deletions excluded:

          k        clean max   deleted min
          5          +0.572      +0.891      a threshold exists
          6          +0.635      +0.715      exists, margin 0.08
          8          +1.324      +0.705      OVERLAP
          20         +1.324      +1.007      OVERLAP

      So no constant works, and raising 0.5 would only trade false positives for false negatives.
      A STEP statistic (a missed mode shifts every cut above it by 1, while Weyl's error drifts)
      was tried and does NOT rescue it: clean steps reach +1.066, deletions fall to +0.148.

      The fix has to calibrate against Weyl's error AT THE SHAPE rather than against a constant --
      the low end of a returned set is known complete, so the error is measurable there.

- [ ] **THE N=5 `lam_6/lam_1` CELL MAY BE DECIDED BY THAT ARTIFACT, and it is the cell the handoff
      says not to quote.** All EIGHT stored rows sit in a single band, 0.485 to 0.546, with
      `DEFICIT_TOL = 0.5` slicing through the middle of it:

          certified   0.485 0.487 0.489 0.491   F = 6.06565408   edge ratio 0.495
          refused     0.504 0.528 0.542 0.546   F = 5.77 to 6.074, edge ratio 0.22 to 0.82

      The deficit tracks the BASIN, not the health of the spectrum -- the four survivors are one
      shape and the four refused are others. A clean 3:1 rectangle scores +0.572 at this same k,
      i.e. squarely inside the refused band. This is consistent with the threshold excluding a
      whole basin rather than four bad runs, which would change what that cell reports.
      NOT YET PROVEN: it needs one of the 0.54 rows re-solved on a fresh plan with more
      eigenvalues, to show directly that no mode is missing.

## Invariants that have caught real bugs

Kept here because every bug in this work returned a plausible number rather than crashing, and
two were caught by theorems rather than by tests. Tabulated with what each caught in
`docs/spectral_optimization.tex` §8.

- `lam_2/lam_1 ≤ 2.5387340` (Payne–Pólya–Weinberger) — **NOT checked per iterate on the current
  path, contrary to what this list said until 2026-08-26.** The per-iterate check lived in the
  older `descend` path; `maximize_ratio` has never had one. What exists today is the regression
  tests (`test_certify.py`, `test_cluster_opt.py`) and an after-the-fact filter in
  `benchmarks/summarize_grid.py` and `benchmarks/plot_optima.py`. The claim mattered: a run
  reporting 2.6203 survived 92 iterations partly because this list said something was watching.
- **The Weyl count of the tracked set** (`douse.spectral_guard`) — the tracked values must be
  CONSECUTIVE, and `worst_sigma` cannot tell that they are not. Recorded per run as
  `worst_weyl_deficit`, and unlike PPW it covers every `k`, not just `k=1`.
- The regular N-gon minimizes `lam_1` — a run converging BELOW `2π²` is impossible.
- `rho → 1` as the trust radius shrinks — a first-order model that does not is wrong in
  direction, not merely in magnitude.
- Rigid motions leave `lam` fixed — the per-vertex gradients must sum to zero in each coordinate.
