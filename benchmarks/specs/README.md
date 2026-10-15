# Run specifications

One TOML file per sweep (`douse/spec.py`): the cells, the start sampler's seed and count, the
optimizer's budget and radius, the solver's precision and backend, the constraint family. The
driver runs a spec with `python -m benchmarks.ratio_multistart --spec <file>`, and every row it
writes carries `spec` (the name) and `spec_sha` (a content hash), so `summarize_grid` can report
which recipes wrote a cell and a reader can refuse to pool different ones. The code's own
identity (`douse_sha`, `chladni_sha`) is recorded beside the spec, not inside it.

| spec | what |
|---|---|
| `thesis_ratio_v0.toml` | the cells the laptop can reach: N = 3..6, k = 2..4, 8 starts, 400 iterations, 1e-10 |
