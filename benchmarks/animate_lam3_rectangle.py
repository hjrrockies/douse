"""Animate `max lam_3/lam_1` over quadrilaterals: away from the square, to a rectangle.

    .venv/bin/python -m benchmarks.animate_lam3_rectangle [--rerun]

`lam_3` is the HIGHEST member of its cluster at the square (`{lam_2, lam_3}`, both 5*pi^2), which
the old code refused outright -- maximizing a convex function has no unique local-model optimum.
Exchanging the two maxima in `lam_max(Lambda + M(d))` turns that into a search over a circle
(`cluster_opt._step_convex`), and the answer is a finding: the model gain at the exact square is
O(1e-02), not zero, so the square is NOT a first-order local maximizer of `lam_3/lam_1` even
though it IS one for `lam_2/lam_1` (the two ratios are equal there, 5/2). Two independent starts
instead reach a RECTANGLE with `L/H = sqrt(8/3)`, where `lam_3 = lam_4` -- the (1,2) and (3,1)
modes crossing -- giving `lam_2/lam_1 = 20/11` and `lam_3/lam_1 = 35/11` exactly, against 5/2 at
the square (`docs/todo.md`). The run starts a small, asymmetric perturbation off the square so it
has a direction to move in rather than sitting on the symmetric saddle.
"""
import sys

import numpy as np

from benchmarks._cluster_anim import run_and_record, render, _perturbed

NAME = 'lam3_rectangle'


def _rectangle_target():
    """`L/H = sqrt(8/3)`, `L + H = 2` -- where the (1,2) and (3,1) modes cross."""
    H_over_L = np.sqrt(3/8)
    L = 2/(1 + H_over_L)
    H = 2 - L
    return np.array([0.0, L, L + H*1j, H*1j])


def main():
    force = '--rerun' in sys.argv
    run_and_record(NAME, n=4, k=2, start=_perturbed(4, 0.03, seed=2),
                   target=_rectangle_target(), exact_ratio=35/11,
                   n_eigs=5, max_iter=150, force=force)
    render(NAME, title=r'maximize $\lambda_3/\lambda_1$ over quadrilaterals: square $\to$ rectangle')


if __name__ == '__main__':
    main()
