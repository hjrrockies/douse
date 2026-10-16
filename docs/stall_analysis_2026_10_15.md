# Why three overnight runs did not converge (stall analysis, 2026-10-15)

A record of measurements made on 2026-10-15, written up on 2026-10-16. It makes no claim
beyond them. Each number names the tool and the command that produced it; the tools are in
`benchmarks/diagnostics/` (README there), and the two per-iterate logs are
`benchmarks/run/diagnostics/sr1_blowup_seed0_prefix_2026_10_15.jsonl` and
`benchmarks/run/diagnostics/scalar_secant_seed0_prefix_2026_10_15.jsonl`. Where a tool's printed
output was not kept, the number comes from the STALL ANALYSIS entry of the session's plan file
(`~/.claude/plans/can-you-do-a-compressed-whisper.md`), and the text says so.

## The problem and the quantities

**The objective.** `F(p) = lam_3(p) / lam_1(p)`: the third over the first Dirichlet eigenvalue of
a hexagon (N = 6) of perimeter 4. The optimizer counts eigenvalues from 0, so this is `k = 2`.
`p` is the hexagon's parameter vector (`douse.polygon.VertexPolygon` via `runs.family_for`),
with `n_p = 2N - 4 = 8` coordinates. The best value known at N = 6, k = 2 is 3.1981367.
`maximize_ratio` (`douse/cluster_opt.py`) maximizes `F` with a trust region, eigenvalues from
chladni.

**The eps-active cluster.** At a maximum `lam_3` usually meets `lam_4` (a double eigenvalue),
where `F` has a kink and no gradient. The optimizer models the eigenvalues within `eps_abs` of
`lam_k` as one cluster, indices `[lo, hi)` (`_active_cluster`). `eps_abs` is tied to the trust
radius, `eps_abs = delta * scale` (`_build_model`), where `scale` is the norm of the cluster's
derivative (`_cluster_scale`). In the logs `cl=[2, 4]` means `lam_3` and `lam_4` are modelled
together. That is **cluster mode**; with one member it is **smooth mode**.

**The split.** `split = lams[hi-1] - lams[lo]`, the spread of the eigenvalues inside the
cluster (`local_model`). It goes to 0 as the iterate reaches the double.

**The cluster block.** In cluster mode the model of how the cluster's eigenvalues move along a
step `d` is the eigenvalues of the `m x m` matrix `Lam + M(d)`, where `M(d) = sum_i d_i M^(i)`
and `M^(i)` is the derivative along coordinate `i` restricted to the cluster's eigenspace
(`local_model`: `Ms`, from the backend's `cluster_matrices`). `Lam` holds the current
splitting. The modelled `lam_k` is the `j`-th eigenvalue of that block. It is directionally
differentiable but not smooth, which is the kink.

**Trust radius, gains, criticality.** `delta` is the trust radius. The **first-order gain**
`gain1` is the model's increase in `F` at the best step `||d|| <= delta` (`trust_region_step`,
gain from `model_gain`). The **criticality** is `crit = gain1 / delta` (`_first_order_step`):
the slope the model promises per unit of radius. It goes to 0 at a smooth maximum and *not* at
a kink. The **second-order gain** `gain2` (`model_gain2`) adds curvature to two smooth scalars,
`lam_1` and the cluster mean `mu`, as `(1/2) d^T H d` terms. `H1` and `Hmu` are secant estimates
of their Hessians, updated from accepted iterates (`_update_curvature`; `douse.quasi_newton`).
When the second-order step is taken, its gain is the prediction.

**rho.** `rho = (F(p + d) - F(p)) / gain` (`_trial_step`), the actual gain over the predicted
one. A step is accepted if `rho > eta = 0.1`. The radius doubles if `rho > 0.75` and halves on a
rejection.

**The stops.** `crit <= gain_tol = 1e-7`, or the **plateau stop**: `crit <= 1e-3` and re-solving
the model at radius `delta * 1e-3` changes `crit` by at most 5% (criticality has stopped
depending on the radius, the signature of a kink; `_first_order_step`). Otherwise the run stops
at `max_iter`, or with "trust radius collapsed" when `delta < 1e-14` (`_second_order_step`).

## The question

The overnight comparison of 2026-10-15 (`benchmarks/diagnostics/overnight_2026_10_15.py`,
ledger `benchmarks/run/diagnostics/overnight_2026_10_15.jsonl`, `--report`) ran seeds 0, 1 and 2
for 400 iterates on both warm paths. On the `windowed` path chladni gets per-level windows
predicted from the step. On the `prefix` path the warm solve searches the whole prefix. All of
these runs used the SR1 secant Hessian. None reached the known optimum:

| seed | path | setting | F | nit | wall (s) | stop |
|---|---|---|---|---|---|---|
| 0 | prefix | `prefix` | 3.1976339806 | 223 | 3396 | trust radius collapsed to 5.68e-15 |
| 0 | windowed | `windowed2` | 3.1981352262 | 400 | 4711 | max_iter |
| 1 | prefix | `prefix` | 3.1978574621 | 400 | 5052 | max_iter |
| 1 | windowed | `windowed2` | 3.1978712517 | 400 | 4433 | max_iter |
| 2 | prefix | `prefix` | 3.1933571937 | 400 | 7385 | max_iter |
| 2 | windowed | `windowed2` | 3.1922335937 | 276 | 5422 | criticality plateaued at 8.28e-4 |

(`windowed2` is the windowed path after the step-cap fix, douse `271e740`. The `windowed` rows
from before that fix are in the ledger, not here.) Seed 2 windowed stopped at a double:
`lam_3 = lam_4` to 1e-10, 273 of 276 steps in cluster mode, with the radius shrinking from 1e-8
to 1e-11 while criticality stayed at 8e-4.

## Findings

### 1. Seed 2 windowed (3.1922336) is a genuine local maximum

*Tool:* `scan_step` (the objective along the model's step direction, fitted by
`dF = s t + c t^2`).
*Command:* `uv run python -m benchmarks.diagnostics.scan_step 2 windowed2`.
*Numbers* (plan file, and REPRODUCED by the tool on 2026-10-16, output in
`benchmarks/run/diagnostics/logs_2026_10_16/scan_step_2_windowed2.out`): fitted slope
`s = 2.406e-4` (the model's own slope along its step is 3.68e-4), curvature `c = -2.04`,
maximum at `t* = 5.9e-5`, residual rms 2.5e-9. That leaves `-s^2 / 4c = 7.1e-9` of gain
along the step, at the level of the solver's noise (eigenvalue bounds ~1e-9 on `F` = 3.19). The plateau stop was right. The
certificate it printed, "criticality 8e-4", is the wrong one to quote at a kink. The gain bound
`s^2 / 4|c|` is the one to quote. This is a second basin, below the 3.1981 optimum.

### 2. Seed 0 prefix ("radius collapsed", 3.1976340) is not an optimum

*Tools:* `probe_stall` (predicted against actual gain at radii 1e-2..1e-7) and `replay`
(restart with fresh state).
*Commands:* `uv run python -m benchmarks.diagnostics.probe_stall 0 prefix`;
`uv run python -m benchmarks.diagnostics.replay 0 prefix --path prefix --qn sr1 --max-iter 30`
and the same with `--path windowed`.
*Numbers* (plan file, and the probe REPRODUCED on 2026-10-16, output in
`benchmarks/run/diagnostics/logs_2026_10_16/probe_stall_0_prefix.out`): at the endpoint the
model is in cluster mode on `cl=[2, 4]` with slope 2.202e-2 at every radius, and `rho` is 0.853,
0.985, 0.998, 1.000, 0.999 at radii 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 (at 1e-2 the step overshoots,
`rho = -0.46`). The model is accurate and promises gain. A replay from that point with fresh
state climbs at `rho ~ 0.6` per step on both warm paths, and the two paths agree to 1e-9 per
iterate over 30 iterates. So the collapse came from state the run had accumulated, not from the
point it was at. The prefix/windowed agreement is the "1e-9 agreement check" the plan asked for,
here on a 30-iterate stretch.

### 3. Seed 1 (both paths, about 3.19786) is a crawl along a curved ridge

*Tools:* the plan file does not record which; the quantities are the ones `probe_stall` and
`replay` print at the seed 1 endpoints (`1 prefix`, `1 windowed2`).
*Numbers* (plan file): split 1e-4, so the iterate sits on the ridge `lam_3 = lam_4`. The slope
along the ridge is about 2e-3. Curvature caps the useful step at about 1e-3, which gives about
1e-6 of gain per iterate. `rho` stays at 0.55-0.75, below the 0.75 needed to grow the radius, so
the radius never grows. A first-order model of the kink is linear in `d` along the ridge, so it
cannot follow a ridge that curves. This one was not fixed and is open (below).

### 4. Seed 2 prefix (3.1933572) is just slow

*Tool:* not recorded in the plan file; the quantities are `probe_stall 2 prefix`'s.
*Numbers* (plan file): slope 4e-2..0.3, `rho ~ 1`, still climbing at `max_iter`, above the
3.1922 basin of finding 1. It is the slowest run (7385 s).

### 5. The root cause of finding 2: the SR1 secant Hessian of `lam_1` blew up

*Tool:* `instrumented_run`, a from-start run logging gains, curvature state and `rho` at every
iterate.
*Command:* `uv run python -m benchmarks.diagnostics.instrumented_run 0 prefix 400 OUT.jsonl --qn sr1`.
The log is `sr1_blowup_seed0_prefix_2026_10_15.jsonl`; `instrumented_run --show` prints it.
*Numbers* (read from the log; `normB` is the Frobenius norm of `B`):

| iterate | F | delta | norm H1 | norm Hmu | gain1 | gain2 | rho |
|---|---|---|---|---|---|---|---|
| 18 | 3.1935320 | 1.3e-2 | 1.5e2 | 3.5e2 | 4.4e-4 | 4.6e-4 | 0.65 |
| 36 | 3.1966762 | 1.3e-2 | 1.0e4 | 6.8e2 | 3.2e-4 | 2.1e-4 | 0.44 |
| 40 | 3.1969783 | 1.3e-2 | 1.7e5 | 5.9e3 | 3.9e-4 | 1.3e-4 | 0.29 |
| 48 | 3.1971725 | 6.3e-3 | 8.2e6 | 9.6e2 | 1.8e-4 | 6.9e-5 | 0.11 |
| 60 | 3.1973681 | 6.3e-3 | 8.2e6 | 1.0e3 | 1.8e-4 | 9.8e-5 | 0.11 |
| 100 | 3.1976097 | 1.6e-3 | 5.6e6 | 1.0e3 | 3.6e-5 | 5.9e-6 | 0.29 |
| 140 | 3.1976340 | 1.5e-6 | 2.2e2 | 1.8e10 | 3.4e-8 | 1.0e-3 | -0.002 |
| 223 | 3.1976340 | 1.1e-14 | 4.4e2 | 6.2e15 | 2.5e-16 | 4.4e-16 | (collapse) |

The `lam_1` estimate reaches a peak norm of 2.5e7 (iterate 120). The true Hessian has norm
2.2e2. That value is from `fd_hessian` (central differences of the Hadamard gradients) at the
seed 0 prefix endpoint, `uv run python -m benchmarks.diagnostics.fd_hessian 0 prefix <h>`,
symmetric to 3e-7 (plan file; the step `h` was not recorded). In the first 140 iterates the
cluster-mean estimate `Hmu` stays at 1e3-1e4. In the frozen phase after iterate 140 it grows to
1.8e10 and then 6.2e15 at the end. (The `quasi_newton` module docstring says "to 6e11". The log
says 6.2e15 at iterate 219-223. The log is the measurement.) 220 of the 223 iterates are in
cluster mode, on `cl=[2, 4]`. The median `rho` per stretch of the log is 0.46 (iterates 1-40),
0.20 (40-80), 0.16 (80-140) and -0.14 after that. `gain2 / gain1` ranges from 0.085 to 2.3e5.
The last value of `F` that changed is reached by iterate 140 (3.1976339800). The radius then
halves to the collapse at iterate 223.

**The mechanism.** SR1 updates `B <- B + w w^T / (s.w)` with `w = y - B s`, from an accepted
step `s` and the change `y` in the gradient. It skips the update only when
`|s.w| < r ||s|| ||w||` with `r = 1e-8` (`quasi_newton.SR1`). Once the steps zigzag inside a
low-dimensional subspace, `s.w` cancels while `|w|` stays at the size of the third-order terms.
Each rank-one update is then enormous, and the 1e-8 rule does not catch it. With an inflated
`H1` in the denominator of `model_gain2`, the second-order gain falls to between a half and a
tenth of the first-order one (iterates 36-100 above). `rho` decays toward `eta = 0.1`.
Rejections halve the radius. Nothing repairs `B`, because rejected trials produce no secant
pairs. Two observations rule out the alternatives. Warm-solve gradients match cold-solve
gradients to 1e-12 relative (`warm_grad 0 prefix`, plan file), so the pairs are accurate and the
blow-up is pair geometry. A synthetic zigzag with a cubic term did *not* reproduce the blow-up:
SR1 learned that Hessian correctly. So the unit test pins the property of the fix (below), and
this log is the evidence for the failure.

## What was cleared

- **The eigensolver and both warm paths.** The prefix and windowed paths agree to 1e-9 per
  iterate over the 30-iterate replay of finding 2.
- **The Hadamard gradients.** Warm against cold to 1e-12 relative on both paths (`warm_grad`).
  The central-difference Hessian built from them is symmetric to 3e-7 (`fd_hessian`).
- **The seed 2 windowed stop.** A genuine local maximum (finding 1).

## Curvature variants, and the fix

Before the fix, four curvature models were compared over 100 iterates from the seed 0 prefix
endpoint, about 2150 s each (`replay 0 prefix --qn ... --max-iter 100 --every 10`; the staging
outputs `variant_*.out`, deleted with the staging directory, are quoted here):

| variant | F after 100 iterates |
|---|---|
| stock SR1 | 3.1981306846 |
| SR1 with a per-update growth cap (skip when `|w|^2/|s.w| > 10 ||B||`) | 3.1981306846 (never fired) |
| no quasi-Newton (`quasi_newton=False`) | 3.1981270413 |
| scalar secant `B = (s.y / s.s) I` (then a monkeypatch, "bb") | 3.1981257276 |

All four crawl at radius 8e-4..1.6e-3 with `rho` 0.1-0.7 and about 3e-7 of gain per iterate at
the end, 6e-6 short of the optimum. On this stretch the curvature model decides robustness (the
collapse), not speed. The crawl comes from the first-order kink model. An earlier 40-iterate
replay gained 4.3e-4 with `quasi_newton=False` against 3.7e-4 with SR1.

**The fix** (douse `eeae9e1`): `maximize_ratio(quasi_newton=True)` now means `'scalar'`,
`quasi_newton.ScalarSecant`. It keeps `B = gamma I` with `gamma = s.y / s.s`, the Rayleigh
quotient of the average Hessian along the last step. That value lies within the true Hessian's
eigenvalue range whatever the pairs do, which is the property SR1 lacks. `'sr1'` is still
selectable. The golden trace was re-recorded in that commit (fake rectangle: 9 iterates instead
of 19).

**Its measured effect** (`instrumented_run 0 prefix 400 OUT.jsonl --qn scalar`, log
`scalar_secant_seed0_prefix_2026_10_15.jsonl`): the from-start seed 0 prefix run no longer
collapses. It converges by the plateau stop at iterate 335 to `F = 3.1981364839` (known optimum
3.1981367) in 5100 s, with criticality 1.2e-4. Over the whole log the largest `H1` norm is 2.8e2
and the largest `Hmu` norm is 9.6e2, both at iterate 8. The median `rho` per stretch is 0.40-0.47
throughout, and `gain2 / gain1` stays in 0.56-1.20. The SR1 run had stopped climbing at
3.1976340 by iterate 140. At iterate 240 the scalar run is at 3.1981338, and it passes the
400-iterate SR1 windowed run's end value (3.1981352) at iterate 266.

## What remains open

- **The crawl along the ridge** (finding 3). A first-order model of the splitting cannot
  follow a curved `lam_3 = lam_4` ridge. The next step proposed on 2026-10-15 is a secant of
  the cluster block between consecutive iterates (the scalar secant lifted to the 2x2 block),
  before any analytic second shape derivative.
- **The stop in objective units.** At a kink, criticality does not go to zero (finding 1). The
  certificate to report is a gain bound such as `s^2 / 4|c|`, in units of `F`.
- **Basins.** Seed 2 converges to a second local maximum (3.1922) below the 3.1981 optimum. The
  summary must count basins rather than report one best value per cell.

## Baseline on the fixed optimizer (2026-10-16)

(Filled in when the reruns end; see below.)
