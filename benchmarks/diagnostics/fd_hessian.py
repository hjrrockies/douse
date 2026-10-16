"""Are the Hadamard gradients accurate enough to difference? Central-difference Hessians and their symmetry.

At a ledger endpoint, difference the gradients of `lam_1` (`g1`) and of the cluster mean
(`gmu`) that `cluster_opt.local_model` returns -- the Hadamard shape gradients through the
parametrization -- in each coordinate with step `h`, and report each Hessian's norm, its
eigenvalue range and its relative asymmetry `|H - H^T| / |H|`. A true Hessian is symmetric, so
the asymmetry bounds the gradients' error over the step; it is also the TRUE curvature scale
against which a secant Hessian (`quasi_newton.SR1`, `ScalarSecant`) can be compared.

The 2026-10-15 STALL ANALYSIS (finding 5: true `lam_1` Hessian norm 2.2e2, symmetric to 3e-7,
against an SR1 estimate grown to 1e7) came from a run of this at the seed 0 prefix endpoint,

    uv run python -m benchmarks.diagnostics.fd_hessian 0 prefix 1e-4

(the step `h` used is not recorded; 1e-4 is this tool's default). 2 n solves, n = 8 at N = 6.
"""
import numpy as np

from douse.cluster_opt import local_model

from benchmarks.diagnostics.endpoint import endpoint_from_args, parser


def gradients(fam, ev, q, k):
    lams, solver = ev.eigenvalues(q)
    m = local_model(fam, q, solver, lams, k=k)
    return np.asarray(m['g1'], float), np.asarray(m['gmu'], float)


def fd_hessian(fam, ev, p, k, h=1e-4):
    """`{'H1': {...}, 'Hmu': {...}}`, each with the matrix, its norm, eigenvalue range and
    relative asymmetry."""
    n = len(p)
    H1, Hmu = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        g1p, gmp = gradients(fam, ev, p + e, k)
        g1m, gmm = gradients(fam, ev, p - e, k)
        H1[:, i] = (g1p - g1m)/(2*h)
        Hmu[:, i] = (gmp - gmm)/(2*h)
    out = {}
    for name, H in (('H1', H1), ('Hmu', Hmu)):
        nrm = float(np.linalg.norm(H))
        w = np.linalg.eigvalsh(0.5*(H + H.T))
        out[name] = dict(H=H, norm=nrm, eig_min=float(w[0]), eig_max=float(w[-1]),
                         asymmetry=float(np.linalg.norm(H - H.T)/nrm) if nrm > 0 else 0.0)
    return out


def main():
    ap = parser(__doc__)
    ap.add_argument('h', type=float, nargs='?', default=1e-4)
    args = ap.parse_args()
    e = endpoint_from_args(args)
    for name, r in fd_hessian(e.fam, e.ev, e.p, e.k, args.h).items():
        print(f"{name}: norm {r['norm']:.3e}  eig range [{r['eig_min']:.2e}, {r['eig_max']:.2e}]  "
              f"asymmetry {r['asymmetry']:.1e}")


if __name__ == '__main__':
    main()
