# The pre-Milestone-1 modules, archived 2026-08-24

These were **untracked** when `douse/polygon.py` and `douse/derivative.py` replaced them, so
deleting them outright would have been unrecoverable. They are kept here for the same reason
`lappy` keeps `benchmarks/archive/mps_heuristics_poc/`: a file that was never committed has no
history to fall back on.

Neither runs. Both target a `lappy` API that no longer exists.

* **`geometry.py`** — the `ParametricDomain` / `VertexPolygon` design sketch. `VertexPolygon`
  had no body past `polygon_vertices(...)` and `ParametricDomain` was stubs, but the *shape* of
  the API was right and `douse/polygon.py` follows it: a class holding `n` and `perim`, called
  with `p` to produce a `lappy` domain. It also carries a `Polgyon` typo in its import, which is
  why it never imported.
* **`evp.py`** — a five-line stub, and not valid Python (`def solve` with no signature or body).
  Nothing was built on it; `Eigenproblem` in `lappy` covers what it was reaching for.

Two other files were removed and are NOT archived, because they are recoverable:

* **`param.py`** is in git history at `719bfb4`. Its parametrization, perimeter gradient and
  implicit-Jacobian idea all survive in `douse/polygon.py`.
* **`param2.py`** was byte-identical to `param.py` except for its import line
  (`from .utils import edge_lengths, polygon_area` in place of
  `from lappy.utils import edge_lengths, edge_`) and some trailing whitespace — an abandoned
  refactor toward a `douse/utils.py` that was never written. Reconstructible from `param.py` if
  it is ever wanted.
