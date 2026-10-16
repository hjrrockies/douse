# Diagnostics

One-off harnesses from the 2026-08-27/28 investigation, kept because each answers a question that
recurred and would otherwise be rewritten from scratch. They are scripts, not library code: paths
are absolute, and they expect to be run from the `douse` root with `./.venv/bin/python3`.

| script | question it answers |
|---|---|
| `repro_N6_k2_s3.py` | **Instrumented reproduction of one start.** Per-iterate `F`, `delta`, `rho`, mode, cluster size, plus the guard's own sigma per call, without editing `douse`. The template for "why did this run do that". |
| `../lappy/repro_N3_k2_s4.py` (lappy-only, moved 2026-10-14) | **Can a run be replayed to a known bad shape?** Logs every call's shape, spectrum and Weyl deficit, stopping at the first deficit refusal. The answer is **no**: the re-run diverges at iterate 3 on a spurious refusal and converges cleanly to the cell's answer, never revisiting the failure. Read its docstring before planning any test that replays a trajectory. |
| `phase0_triage.py` | **Classify a stalled run.** Re-runs rows with per-iterate logging and evaluates `rho` at five shrinking radii at the endpoint -- the discriminator that separates rate-limited from noise-limited from a discontinuous objective. |
| `noise_floor.py` | **Criticality against the active-set width**, for reading a plateau. |
| `plan_probe.py` | **Fresh plan vs frozen plan at a shape**: is a bad row stale, or out of reach? |
| `qn_ablation.py` | **Ablate `quasi_newton`** on chosen rows, same seed and thread count, to see what the second-order path contributes. |
| `contested_rerun.py` | Re-run specific starts into a side ledger, for before/after comparison without touching the tool of record. |
| `detector_validation.py` | Re-run rows in the DEFAULT configuration and compare against the cell answer -- the form that produces ledger-comparable evidence. |
| `full_rerun.py` | **The re-run recipe.** Backs up every flagged row to `pre_rerun_<date>/`, strips them, and lets `ratio_multistart` resume-refill exactly the gaps, cheapest cell first. Read its docstring before re-running anything in bulk. |

Two things these encode that are easy to get wrong, both learned the hard way:

* **Match the thread count of the run you are comparing against.** The sweeps ran `workers=5` on
  ten cores, i.e. 2 BLAS threads per worker and `mps.n_workers_default = 2`. A comparison run at
  the default thread count is not the same trajectory.
* **Build a fresh `SpectrumEvaluator` per attempt.** Its statistics are running maxima over its
  lifetime, so a reused one carries a discarded shape's numbers into the run that replaces it --
  which cost a certified row before it was noticed.

* **Never solve inside the callback.** `repro_N3_k2_s4.py` briefly re-evaluated each accepted
  iterate to record its spectrum. That doubled the cost of the run, and -- the real damage --
  advanced the evaluator's continuation seed, so the trajectory being "reproduced" was not the
  recorded one. Log from a wrapper around `eigenvalues`, where the spectrum is already in hand.

## 2026-10-15/16: the stall-analysis tools

Written during the 2026-10-15 stall analysis and made reproducible on 2026-10-16. Unlike the
scripts above these are a small package: run them from the douse root as modules
(`uv run python -m benchmarks.diagnostics.<tool> ...`). The ones that start from a ledger
endpoint take `seed setting` (a row's `start` and `setting`, default ledger
`benchmarks/run/diagnostics/overnight_2026_10_15.jsonl`, `--ledger` for another) and build the
evaluator through `endpoint.py`, which uses `douse.ledger` and `douse.runs`. Each core function
runs on the fake rectangle in `tests/test_diagnostics_tools.py`. Each docstring gives the exact
command behind its numbers in `docs/stall_analysis_2026_10_15.md`.

| tool | question it answers |
|---|---|
| `endpoint.py` | (shared) ledger row -> `Endpoint(fam, ev, cons, p, k, row)`; the `prefix` warm path (`PrefixEvaluator`) |
| `probe_stall.py` | **Is a stopped run at an optimum?** Predicted vs actual gain along the model step at radii 1e-2..1e-7 |
| `scan_step.py` | **How much gain is left?** The objective along the step, a parabola fit, the gain bound `s^2/4|c|` |
| `replay.py` | **Does a restart climb?** `maximize_ratio` from an endpoint with fresh state, `--path windowed|prefix`, `--qn scalar|sr1|none` |
| `instrumented_run.py` | **Why did the radius collapse?** A from-start run logging gains, curvature state and `rho` per iterate; `--show LOG` prints one |
| `fd_hessian.py` | **Are the gradients good enough to difference?** Central-difference Hessians of `lam_1` and the cluster mean, with their asymmetry |
| `warm_grad.py` | **Do warm solves change the gradients?** Both warm paths against a cold solve at one point |
| `overnight_2026_10_15.py` | The three-seed N=6 k=2 comparison driver and its `--report` (settings `windowed`, `prefix`, `windowed2` on SR1; `*_ss` on the scalar secant) |

Retired with the staging scripts: the `cap` and `bb` monkeypatches of `SR1.update`. `bb` is
what `douse.quasi_newton.ScalarSecant` now is (`--qn scalar`); `cap` (skip an update when
`|w|^2/|den| > 10 ||B||`) never fired on the 100-iterate seed 0 stretch and tracked stock SR1 to
every printed digit (3.1981306846 for both).
