# lappy-only benchmarks (deprecated backend)

These scripts drive `douse.evaluate.LappySpectrumEvaluator` or lappy's own solver internals
(`sigma_at`, `mps.n_workers_default`, the tension plateau `sigma_base`, lappy's `rtol`). They need
the `douse[lappy]` extra in an environment of their own: lappy passes `tola`/`tolb` to
`gsvd4py.gsvdvals`, which gsvd4py >= 0.4 (chladni's requirement) no longer accepts, so they do not
run in the default environment. Kept so the lappy-era ledgers under `benchmarks/run/` can be
reproduced and re-examined. Moved here 2026-10-14 when chladni became the backend.

| script | question it answered |
|---|---|
| `replay_refusal.py` | re-solve a recorded refusal off-line: plan on one shape, solve on another |
| `missing_mode_cache.py` | the data behind `jupyter_notebooks/missing_mode.ipynb` |
| `sigma_base_ablation.py` | an ablation of lappy's tension plateau `sigma_base` |
| `tension_slope_universality.py` | is lappy's tension-well slope `C_k` universal across shapes? |
| `rtol_single_point.py`, `plot_rtol_single_point.py` | can lappy's `rtol` be calibrated from the tension at one non-eigenvalue? |
| `repro_N3_k2_s4.py` (+ `.jsonl`) | an instrumented re-run of N=3 k=2 start 4, stopping at the first Weyl refusal |
