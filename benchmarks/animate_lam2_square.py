"""Animate `max lam_2/lam_1` over quadrilaterals, through the degeneracy, to the square.

    .venv/bin/python -m benchmarks.animate_lam2_square [--rerun]

The acceptance test for `douse.cluster_opt.maximize_ratio`: backtracking ascent on the same
objective stalls 5.70e-04 short of the square because `lam_2 = lam_3` there and no gradient
exists to vanish (`benchmarks/ratio_stall.py`). The trust-region cluster method reaches it. The
start (`_perturbed(4, 0.10)`) is the one `tests/test_cluster_opt.py` uses to exercise both the
smooth and the cluster branch in one run -- the gap panel below is where that switch shows.
"""
import sys

from benchmarks._cluster_anim import run_and_record, render, _perturbed

NAME = 'lam2_square'


def main():
    force = '--rerun' in sys.argv
    run_and_record(NAME, n=4, k=1, start=_perturbed(4, 0.10, seed=0),
                   target=[0.0, 1.0, 1.0 + 1j, 1j], exact_ratio=2.5,
                   n_eigs=3, max_iter=120, force=force)
    render(NAME, title=r'maximize $\lambda_2/\lambda_1$ over quadrilaterals $\to$ square')


if __name__ == '__main__':
    main()
