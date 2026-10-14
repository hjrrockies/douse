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
