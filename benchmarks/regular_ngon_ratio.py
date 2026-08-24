"""Is the regular N-gon a local maximizer of lam_2/lam_1 over N-gons of fixed perimeter?

    .venv/bin/python -m benchmarks.regular_ngon_ratio

The polygon analogue of Osting's spectral-ratio optimization over Fourier-boundary domains. Over
ALL planar domains the disk maximizes lam_2/lam_1 at 2.5387340 (Payne--Polya--Weinberger); over
regular N-gons the ratio climbs monotonically towards it, and at every one of them
lam_2 = lam_3 EXACTLY -- so the conjectured optimum sits on a degeneracy.

Reported per N: the ratio, the first-order algebra, and a second-order probe along the
directions first order cannot see. See `douse/certify.py` for why first order settles nothing.
"""
import warnings

import numpy as np

from lappy import Eigenproblem, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

from douse import VertexPolygon, param_vector
from douse.certify import first_order_report, second_order_probe

PERIM = 4.0
DISK = (3.8317059702/2.4048255577)**2


def regular(N, perim=PERIM):
    return perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)


def main(Ns=(3, 4, 5, 6, 7)):
    print(f'disk (PPW bound): lam_2/lam_1 = {DISK:.7f}\n')
    print(f"{'N':>3}{'n_p':>5}{'lam_2/lam_1':>13}{'gap 2-3':>10}{'|g1|':>9}{'max|trM|':>10}"
          f"{'rank Q':>8}{'null':>6}{'worst curv':>12}  verdict")
    for N in Ns:
        fam = VertexPolygon(N, perim=PERIM)
        p = param_vector(regular(N), PERIM)
        dom = fam(p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plan = BP.plan_basis(dom, weyl_est(6, dom), target=1e-13)

            def ratio(q):
                d = fam(q)
                s = MPSEigensolver.from_domain(d, lam_max=weyl_est(6, d),
                                               basis=BP.realize(plan, d), prec=1e-12)
                e = np.asarray(Eigenproblem(d, eval_solver=s, precision=1e-12).solve(2),
                               dtype=float)
                return e[1]/e[0]

            s0 = MPSEigensolver.from_domain(dom, lam_max=weyl_est(6, dom),
                                            basis=BP.realize(plan, dom), prec=1e-12)
            lams = np.asarray(Eigenproblem(dom, eval_solver=s0, precision=1e-12).solve(3),
                              dtype=float)
            rep = first_order_report(fam, p, s0, lams, k=1)
            probe = second_order_probe(ratio, p, rep['null_basis'])

        h = rep['hypotheses']
        gap = abs(lams[2] - lams[1])/lams[1]
        w = rep['Q_eigenvalues']
        rank = int(np.sum(np.abs(w) > 1e-6*np.abs(w).max()))
        wc = probe['worst_curvature']
        print(f"{N:>3}{fam.n_params:>5}{rep['ratio']:>13.7f}{gap:>10.1e}"
              f"{h['grad_lam1_norm']:>9.1e}{h['max_abs_trace']:>10.1e}{rank:>8}"
              f"{rep['null_dim']:>6}{(wc if wc is not None else float('nan')):>12.3e}"
              f"  {probe['verdict']}")


if __name__ == '__main__':
    main()
