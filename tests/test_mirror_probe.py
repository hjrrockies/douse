"""`certify.antisymmetric_hessian` on synthetic objectives: no eigensolve, no lappy."""
import numpy as np


def test_antisymmetric_hessian_reads_a_quadratic_and_tells_a_kink_apart():
    from douse.certify import antisymmetric_hessian
    from douse.polygon import MirrorPolygon
    fam, q, _ = MirrorPolygon.from_vertices(np.exp(2j*np.pi*np.arange(5)/5), 4.0)
    v = fam.vertices(q)
    B = fam.antisymmetric_basis(q)
    n = len(v)

    def flat(w):
        d = w - v
        return np.concatenate([d.real, d.imag])

    H = np.diag([-1.0, -2.0, -3.0])
    quad = antisymmetric_hessian(lambda w: 7.0 + 0.5*(B.T @ flat(w)) @ H @ (B.T @ flat(w)), v, B)
    np.testing.assert_allclose(quad['hess_eigs'][0], [-3, -2, -1], atol=1e-9)
    np.testing.assert_allclose(quad['scaling'], 1.0, atol=1e-9)
    assert quad['verdict'].startswith('consistent')
    kink = antisymmetric_hessian(lambda w: 7.0 - np.linalg.norm(flat(w)), v, B, ts=(0.01, 0.03))
    np.testing.assert_allclose(kink['scaling'], 3.0, rtol=1e-9)
    up = antisymmetric_hessian(lambda w: 7.0 + (B[:, 0] @ flat(w))**2, v, B)
    assert up['worst'] > 0 and up['verdict'].startswith('refuted')
    assert n == 5
