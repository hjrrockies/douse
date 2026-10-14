"""Animate `max lam_2/lam_1` over pentagons, through the degeneracy, to the regular pentagon.

    .venv/bin/python -m benchmarks.animate_lam2_pentagon [--rerun]

N=5 used to be the failure case: 1.97e-04 short after 300 iterations, gap still 0.75, only 14 of
300 steps in cluster mode -- it never approached the degeneracy at all. Two fixes closed that,
both in `douse.cluster_opt`: sizing the eps-active cluster from the trust radius instead of a
fixed `eps_rel`, and switching `SpectrumEvaluator` to `lappy`'s `track_set` so a near-degenerate
iterate does not fall back to a full solve with ~1e-08 of tracking error. The run now ends on the
first-order criticality certificate with 230 of 234 steps in cluster mode (`docs/todo.md`). The
start (`jitter=0.13`) is the one `benchmarks/ratio_stall.py` uses, well off symmetry.
"""
import sys
import warnings

import numpy as np

from benchmarks._cluster_anim import run_and_record, render, _perturbed, _regular, PERIM

NAME = 'lam2_pentagon'


def _exact_ratio(n=5):
    """lam_2/lam_1 at the regular pentagon, solved once at high precision."""
    from douse import VertexPolygon, param_vector
    from lappy import Eigenproblem
    fam = VertexPolygon(n, perim=PERIM)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e = np.asarray(Eigenproblem(fam(param_vector(_regular(n), PERIM)),
                                    precision=1e-11).solve(2), dtype=float)
    return float(e[1]/e[0])


def main():
    force = '--rerun' in sys.argv
    run_and_record(NAME, n=5, k=1, start=_perturbed(5, 0.13, seed=0),
                   target=_regular(5), exact_ratio=_exact_ratio(),
                   n_eigs=3, max_iter=300, force=force)
    render(NAME, title=r'maximize $\lambda_2/\lambda_1$ over pentagons $\to$ regular pentagon')


if __name__ == '__main__':
    main()
